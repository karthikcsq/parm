from __future__ import annotations

import hashlib
import json
import tempfile
from collections import OrderedDict, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .retrieval import (
    SCOPED_RETRIEVER_CACHE_SIZE,
    ChunkRecord,
    RetrievalHit,
    RetrievalIndex,
    RetrievalResult,
    SpacyCueConceptExtractor,
    TextEmbedder,
    _BM25Corpus,
    _load_spacy_model,
    _parm_observation_regions,
)


PARM_SEMANTIC_JUDGE_MODEL = "gpt-5-mini"
PARM_SEMANTIC_JUDGE_RUBRIC = "parm_pair_admission_v3"
PARM_SEMANTIC_CANDIDATE_DEPTH = 7
PARM_SEMANTIC_METHODS = (
    "sentence_max",
    "sentence_top_three",
    "chunk_max",
    "blend",
    "bm25",
)
PARM_SEMANTIC_JUDGE_INSTRUCTIONS = """\
You are the admission stage of a personal-memory retrieval system.

Each candidate pairs one visible passage with one source conversation from the
user's history. The complete observation is included so you can tell
substantive passages from surrounding archival noise. Admit at most one pair.
Admit it only when:
1. the user's own words in the source support a durable preference, routine,
   relationship, identity, or prior commitment;
2. the visible passage itself states a concrete action, property, schedule,
   subject, or relationship that specifically satisfies that fact; and
3. recalling the fact materially changes what the user would want done in the
   current task.

Do not treat assistant suggestions as user facts. Reject theme overlap,
hypotheticals, generic advice, and connections that require inventing a key
activity or relation. A title, a topic, or a generic evaluative adjective such
as "general audience", "specialized", "narrower", or "premium" is not a
concrete affordance. When no pair clears all three conditions, return
admit=false. Candidate ranking is only a high-recall prefilter and is not
evidence.
"""


class AdmissionCachePolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


class AdmissionCacheMissError(RuntimeError):
    pass


class AdmissionJudge(Protocol):
    model_name: str
    rubric_version: str

    def judge(
        self,
        *,
        prompt: str,
        observation_text: str,
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, Any]: ...

    @property
    def cache_hash(self) -> str | None: ...


class CachedOpenAIAdmissionJudge:
    model_name = PARM_SEMANTIC_JUDGE_MODEL
    rubric_version = PARM_SEMANTIC_JUDGE_RUBRIC

    def __init__(
        self,
        cache_dir: str | Path,
        policy: AdmissionCachePolicy | str,
        *,
        cache_namespace: str,
        client: Any | None = None,
    ) -> None:
        if not cache_namespace.strip():
            raise ValueError("admission cache namespace must be non-empty")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.policy = AdmissionCachePolicy(policy)
        self.cache_namespace = cache_namespace
        self.client = client
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._used_cache_files: list[Path] = []

    def judge(
        self,
        *,
        prompt: str,
        observation_text: str,
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        rendered_candidates = "\n\n".join(
            (
                f"[{candidate['candidate_id']}]\n"
                f"Visible passage:\n{candidate['region_text']}\n\n"
                f"User-history source:\n{candidate['memory_text']}"
            )
            for candidate in candidates
        )
        input_text = (
            f"User task:\n{prompt}\n\n"
            f"Complete observation:\n{observation_text}\n\n"
            f"Candidate pairs:\n{rendered_candidates}"
        )
        request_hash = semantic_admission_cache_key(
            prompt,
            observation_text,
            cache_namespace=self.cache_namespace,
        )
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            self._used_cache_files.append(cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if self.policy is AdmissionCachePolicy.FROZEN:
            raise AdmissionCacheMissError(
                f"missing PARM admission cache entry: {request_hash}"
            )

        response = self.client.responses.create(
            model=self.model_name,
            instructions=PARM_SEMANTIC_JUDGE_INSTRUCTIONS,
            input=input_text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "parm_pair_admission",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "admit": {"type": "boolean"},
                            "candidate_id": {
                                "type": ["string", "null"],
                                "enum": [
                                    *[
                                        candidate["candidate_id"]
                                        for candidate in candidates
                                    ],
                                    None,
                                ],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "admit",
                            "candidate_id",
                            "confidence",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            store=False,
        )
        parsed = json.loads(response.output_text)
        if parsed["admit"] != (parsed["candidate_id"] is not None):
            raise ValueError("judge returned inconsistent admission fields")
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate["candidate_id"] == parsed["candidate_id"]
            ),
            None,
        )
        result = {
            **parsed,
            "selected_page_id": (
                selected["page_id"] if selected is not None else None
            ),
            "selected_region_id": (
                selected["region_id"] if selected is not None else None
            ),
            "selected_region_text": (
                selected["region_text"] if selected is not None else None
            ),
            "resolved_model": response.model,
            "response_id": response.id,
        }
        _atomic_write_json(cache_path, result)
        self._used_cache_files.append(cache_path)
        return result

    @property
    def cache_hash(self) -> str | None:
        paths = sorted(set(self._used_cache_files))
        if not paths:
            return None
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


class PARMSemanticJudgeRetriever:
    retrieval_condition_detail = "parm_semantic_pair_judge_v1"
    evidence_projection_version = "raw_source_and_region_v1"
    admission_policy = "contrastive_llm_judge_v2"

    def __init__(
        self,
        index: RetrievalIndex,
        embedder: TextEmbedder,
        judge: AdmissionJudge,
        *,
        candidate_depth: int = PARM_SEMANTIC_CANDIDATE_DEPTH,
        concept_extractor: Any | None = None,
    ) -> None:
        if candidate_depth < 1:
            raise ValueError("candidate depth must be at least 1")
        if index.sentence_embeddings is None or not index.sentences:
            raise ValueError(
                "semantic PARM retrieval requires sentence-index artifacts"
            )
        if embedder.model_name != index.manifest["embedding_model"]:
            raise ValueError("query embedder model does not match retrieval index")
        if embedder.dimensions != index.manifest["embedding_dimensions"]:
            raise ValueError("query embedder dimensions do not match retrieval index")
        self.index = index
        self.embedder = embedder
        self.judge = judge
        self.candidate_depth = candidate_depth
        self.concept_extractor = (
            concept_extractor
            if concept_extractor is not None
            else SpacyCueConceptExtractor(_load_spacy_model())
        )
        self._scoped_retrievers: OrderedDict[
            str, PARMSemanticJudgeRetriever
        ] = OrderedDict()
        if len(index.corpus_ids) == 1:
            self._prepare_scoped_index()

    def retrieve_observation(
        self,
        prompt: str,
        observation_text: str,
        *,
        top_k: int,
        corpus_id: str | None = None,
    ) -> RetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        resolved_corpus_id = self.index.resolve_corpus_id(corpus_id)
        if len(self.index.corpus_ids) > 1:
            scoped = self._scoped_retrievers.pop(resolved_corpus_id, None)
            if scoped is None:
                scoped = PARMSemanticJudgeRetriever(
                    self.index.scoped(resolved_corpus_id),
                    self.embedder,
                    self.judge,
                    candidate_depth=self.candidate_depth,
                    concept_extractor=self.concept_extractor,
                )
            self._scoped_retrievers[resolved_corpus_id] = scoped
            if len(self._scoped_retrievers) > SCOPED_RETRIEVER_CACHE_SIZE:
                self._scoped_retrievers.popitem(last=False)
            return scoped.retrieve_observation(
                prompt,
                observation_text,
                top_k=top_k,
                corpus_id=resolved_corpus_id,
            )

        regions = _parm_observation_regions(observation_text)
        concepts_by_region = self.concept_extractor.rare_region_concepts(
            [region["description"] for region in regions]
        )
        distinctive = [
            (region, concepts)
            for region, concepts in zip(regions, concepts_by_region)
            if concepts
        ]
        if not distinctive:
            return RetrievalResult(
                (),
                self._trace(
                    resolved_corpus_id,
                    regions,
                    concepts_by_region,
                    [],
                    None,
                ),
            )
        kept_regions = [region for region, _ in distinctive]
        query_vectors = self.embedder.embed(
            [region["description"] for region in kept_regions]
        )
        score_matrices = self._score_matrices(
            kept_regions,
            query_vectors,
        )
        candidates = self._candidate_pairs(
            kept_regions,
            score_matrices,
        )
        if not candidates:
            return RetrievalResult(
                (),
                self._trace(
                    resolved_corpus_id,
                    regions,
                    concepts_by_region,
                    [],
                    None,
                ),
            )
        judgment = self.judge.judge(
            prompt=prompt,
            observation_text=observation_text,
            candidates=candidates,
        )
        selected = next(
            (
                candidate
                for candidate in candidates
                if (
                    candidate["page_id"]
                    == judgment.get("selected_page_id")
                    and candidate["region_id"]
                    == judgment.get("selected_region_id")
                )
            ),
            None,
        )
        if judgment["admit"] and selected is None:
            selected = self._replay_candidate(
                regions,
                judgment,
            )
        hits = (
            (self._hit(selected, judgment),)
            if judgment["admit"] and selected is not None
            else ()
        )
        return RetrievalResult(
            hits,
            self._trace(
                resolved_corpus_id,
                regions,
                concepts_by_region,
                candidates,
                judgment,
            ),
        )

    def _replay_candidate(
        self,
        regions: Sequence[dict[str, Any]],
        judgment: dict[str, Any],
    ) -> dict[str, Any]:
        page_id = judgment.get("selected_page_id")
        region_id = judgment.get("selected_region_id")
        if page_id not in self._page_text:
            raise ValueError(
                "cached judge selected a page outside the admissible corpus"
            )
        region = next(
            (
                item
                for item in regions
                if item["region_id"] == region_id
            ),
            None,
        )
        if region is None:
            raise ValueError(
                "cached judge selected a region absent from the observation"
            )
        if region["text"] != judgment.get("selected_region_text"):
            raise ValueError("cached judge region text does not match observation")
        return {
            "candidate_id": judgment.get("candidate_id"),
            "region_id": region_id,
            "region_text": region["text"],
            "region_description": region["description"],
            "page_id": page_id,
            "memory_text": self._page_text[page_id],
            "methods": ["frozen_judge_replay"],
            "method_ranks": {},
            "method_scores": {},
            "best_rank": 0,
        }

    def _prepare_scoped_index(self) -> None:
        self._page_by_id = {page.page_id: page for page in self.index.pages}
        self._chunks_by_page: dict[
            str, list[tuple[int, ChunkRecord]]
        ] = defaultdict(list)
        for position, chunk in enumerate(self.index.chunks):
            self._chunks_by_page[chunk.page_id].append((position, chunk))
        self._page_ids = tuple(
            page.page_id
            for page in self.index.pages
            if not page.perturbations
        )
        self._page_text = {
            page_id: "\n".join(
                chunk.text for _, chunk in self._chunks_by_page[page_id]
            )
            for page_id in self._page_ids
        }
        self._bm25 = _BM25Corpus(self._page_text)
        self._sentence_positions: dict[str, list[int]] = defaultdict(list)
        for position, sentence in enumerate(self.index.sentences):
            self._sentence_positions[sentence.page_id].append(position)
        self._sentence_matrix = _normalize_rows(
            self.index.sentence_embeddings
        )
        self._chunk_matrix = _normalize_rows(self.index.embeddings)

    def _score_matrices(
        self,
        regions: Sequence[dict[str, Any]],
        query_vectors: np.ndarray,
    ) -> dict[str, np.ndarray]:
        normalized_queries = _normalize_rows(query_vectors)
        sentence_cosines = normalized_queries @ self._sentence_matrix.T
        chunk_cosines = normalized_queries @ self._chunk_matrix.T
        score_rows = []
        for page_id in self._page_ids:
            per_sentence = sentence_cosines[
                :, self._sentence_positions[page_id]
            ]
            ordered_sentences = np.sort(per_sentence, axis=1)
            top_count = min(3, ordered_sentences.shape[1])
            sentence_max = ordered_sentences[:, -1]
            sentence_top_three = np.mean(
                ordered_sentences[:, -top_count:],
                axis=1,
            )
            chunk_positions = [
                position for position, _ in self._chunks_by_page[page_id]
            ]
            chunk_max = np.max(
                chunk_cosines[:, chunk_positions],
                axis=1,
            )
            score_rows.append(
                np.stack(
                    (
                        sentence_max,
                        sentence_top_three,
                        chunk_max,
                        0.6 * sentence_max
                        + 0.2 * sentence_top_three
                        + 0.2 * chunk_max,
                    ),
                    axis=1,
                )
            )
        dense = np.stack(score_rows, axis=1)
        bm25_scores = np.asarray(
            [
                [
                    scores.get(page_id, 0.0)
                    for page_id in self._page_ids
                ]
                for scores in (
                    self._bm25.scores(region["description"])
                    for region in regions
                )
            ],
            dtype=np.float32,
        )
        bm25_max = float(np.max(bm25_scores))
        normalized_bm25 = (
            bm25_scores / bm25_max
            if bm25_max > 0
            else np.zeros_like(bm25_scores)
        )
        return {
            "sentence_max": dense[:, :, 0],
            "sentence_top_three": dense[:, :, 1],
            "chunk_max": dense[:, :, 2],
            "blend": dense[:, :, 3],
            "bm25": normalized_bm25,
        }

    def _candidate_pairs(
        self,
        regions: Sequence[dict[str, Any]],
        score_matrices: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        grouped: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        for method in PARM_SEMANTIC_METHODS:
            matrix = score_matrices[method]
            flat_order = np.argsort(matrix, axis=None)[::-1]
            for global_rank, flat_position in enumerate(
                flat_order[: self.candidate_depth],
                start=1,
            ):
                region_position, page_position = np.unravel_index(
                    flat_position,
                    matrix.shape,
                )
                region = regions[region_position]
                page_id = self._page_ids[page_position]
                candidate = grouped.setdefault(
                    (region["region_id"], page_id),
                    {
                        "region_id": region["region_id"],
                        "region_text": region["text"],
                        "region_description": region["description"],
                        "page_id": page_id,
                        "memory_text": self._page_text[page_id],
                        "methods": [],
                        "method_ranks": {},
                        "method_scores": {},
                        "best_rank": global_rank,
                    },
                )
                candidate["methods"].append(method)
                candidate["method_ranks"][method] = global_rank
                candidate["method_scores"][method] = float(
                    matrix[region_position, page_position]
                )
                candidate["best_rank"] = min(
                    candidate["best_rank"],
                    global_rank,
                )
        candidates = list(grouped.values())
        candidates.sort(
            key=lambda candidate: (
                candidate["best_rank"],
                -len(candidate["methods"]),
                candidate["region_id"],
                candidate["page_id"],
            )
        )
        for position, candidate in enumerate(candidates, start=1):
            candidate["candidate_id"] = f"C{position:02d}"
            candidate["methods"].sort()
        return candidates

    def _hit(
        self,
        candidate: dict[str, Any],
        judgment: dict[str, Any],
    ) -> RetrievalHit:
        page = self._page_by_id[candidate["page_id"]]
        chunks = self._chunks_by_page[page.page_id]
        return RetrievalHit(
            page_id=page.page_id,
            source_id=page.source_id,
            slug=page.slug,
            title=page.title,
            chunk_id=chunks[0][1].chunk_id,
            text=self._page_text[page.page_id],
            score=float(judgment["confidence"]),
            rank=1,
            perturbations=page.perturbations,
            diagnostics={
                "region_id": candidate["region_id"],
                "candidate_id": candidate["candidate_id"],
                "candidate_methods": candidate["methods"],
                "admission_channel": self.admission_policy,
            },
            corpus_id=page.corpus_id,
        )

    def _trace(
        self,
        corpus_id: str,
        regions: Sequence[dict[str, Any]],
        concepts_by_region: Sequence[Sequence[str]],
        candidates: Sequence[dict[str, Any]],
        judgment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "corpus_id": corpus_id,
            "retrieval_condition_detail": self.retrieval_condition_detail,
            "evidence_projection_version": self.evidence_projection_version,
            "admission_policy": self.admission_policy,
            "candidate_depth": self.candidate_depth,
            "candidate_methods": list(PARM_SEMANTIC_METHODS),
            "regions": [
                {
                    "region_id": region["region_id"],
                    "region_kind": region.get("region_kind"),
                    "span": [region["start"], region["end"]],
                    "text": region["text"],
                    "description": region["description"],
                    "rare_concepts": list(concepts),
                }
                for region, concepts in zip(
                    regions,
                    concepts_by_region,
                )
            ],
            "semantic_seeds": [],
            "entity_seeds": [],
            "candidate_pairs": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "memory_text"
                }
                for candidate in candidates
            ],
            "judge": judgment,
            "judge_model": self.judge.model_name,
            "judge_rubric_version": self.judge.rubric_version,
        }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms != 0,
    )


def semantic_admission_cache_namespace(
    index_manifest_hash: str,
    *,
    candidate_depth: int = PARM_SEMANTIC_CANDIDATE_DEPTH,
) -> str:
    if not index_manifest_hash.strip():
        raise ValueError("index manifest hash must be non-empty")
    return (
        f"parm_semantic_pair_judge_v1:"
        f"{index_manifest_hash}:depth={candidate_depth}"
    )


def semantic_admission_cache_key(
    prompt: str,
    observation_text: str,
    *,
    cache_namespace: str,
) -> str:
    payload = {
        "model": PARM_SEMANTIC_JUDGE_MODEL,
        "rubric_version": PARM_SEMANTIC_JUDGE_RUBRIC,
        "cache_namespace": cache_namespace,
        "prompt": prompt,
        "observation_text": observation_text,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
