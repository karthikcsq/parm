from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np


EMBEDDING_MODEL = "openai:text-embedding-3-small"
EMBEDDING_API_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_ENCODING = "cl100k_base"
EMBEDDING_MAX_INPUT_TOKENS = 8192
CANDIDATE_DEPTH = 20
OFFICIAL_TOP_K = 5
RRF_K = 60
RRF_WEIGHT = 0.70
COSINE_WEIGHT = 0.30
GRAPH_MIN_INBOUND = 2
GRAPH_MULTIPLIER = 1.05
EXPANSION_MODEL = "gpt-5-mini"
EXPANSION_PROMPT_VERSION = "retrieval-expansion-v1"

# Raw-cosine diagnostics drift ~1e-4 across runs (embeddings are deterministic
# within a process but not across processes), so they are kept on the in-memory
# RetrievalHit for tests/debugging but not written to the persisted trace, which
# should replay byte-identically as long as the textual content is unchanged.
_NON_REPRODUCIBLE_DIAGNOSTICS = frozenset(
    {"original_query_cosine", "pre_graph_score", "final_score"}
)


_EMBEDDING_ENCODER: Any | None = None


def _embedding_encoder() -> Any:
    global _EMBEDDING_ENCODER
    if _EMBEDDING_ENCODER is None:
        import tiktoken

        _EMBEDDING_ENCODER = tiktoken.get_encoding(EMBEDDING_ENCODING)
    return _EMBEDDING_ENCODER


def _token_windows(text: str, max_tokens: int) -> list[str]:
    """Split text into contiguous windows that each fit the embedding cap.

    A query shorter than the cap yields a single window equal to the input, so
    normal-length retrieval is unchanged. Naive output RAG can hand a whole tool
    observation (up to MAX_CONTEXT_TOKENS) as the query; windowing lets every
    part of it drive dense retrieval instead of dropping the tail.
    """
    encoder = _embedding_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    return [
        encoder.decode(tokens[start : start + max_tokens])
        for start in range(0, len(tokens), max_tokens)
    ]


class RetrievalValidationError(ValueError):
    pass


class ExpansionCacheMissError(LookupError):
    pass


class RetrievalMode(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    ENHANCED = "enhanced"


class ExpansionPolicy(str, Enum):
    POPULATE = "populate"
    FROZEN = "frozen"


@dataclass(frozen=True)
class PageRecord:
    page_id: str
    source_id: str
    slug: str
    title: str
    perturbations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    page_id: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class LinkRecord:
    source_page_id: str
    target_page_id: str
    link_type: str = ""
    provenance: str | None = None


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = OFFICIAL_TOP_K

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("retrieval query must be non-empty")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


@dataclass(frozen=True)
class RetrievalHit:
    page_id: str
    source_id: str
    slug: str
    title: str
    chunk_id: str
    text: str
    score: float
    rank: int
    perturbations: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    hits: tuple[RetrievalHit, ...]
    trace: dict[str, Any]


@dataclass(frozen=True)
class EntitySeed:
    seed_id: str
    surface: str
    normalized_surface: str
    source: str
    span_start: int | None = None
    span_end: int | None = None
    matched_page_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityRetrievalResult:
    seeds: tuple[EntitySeed, ...]
    hits: tuple[RetrievalHit, ...]
    trace: dict[str, Any]


class Retriever(Protocol):
    mode: RetrievalMode

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


class EntityRetriever(Protocol):
    def retrieve_entities(
        self,
        observation_text: str,
        *,
        top_k: int,
    ) -> EntityRetrievalResult: ...


class TextEmbedder(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class QueryExpander(Protocol):
    model: str
    prompt_version: str

    def expand(self, query: str) -> tuple[str, str, str]: ...

    @property
    def cache_hash(self) -> str | None: ...


@dataclass(frozen=True)
class RetrievalIndex:
    path: Path
    manifest: dict[str, Any]
    manifest_hash: str
    pages: tuple[PageRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    embeddings: np.ndarray
    links: tuple[LinkRecord, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model: str = EMBEDDING_MODEL,
        expected_dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> "RetrievalIndex":
        root = Path(path)
        required = (
            "manifest.json",
            "pages.jsonl",
            "chunks.jsonl",
            "embeddings.npy",
            "links.jsonl",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise RetrievalValidationError(
                f"retrieval index is missing: {', '.join(missing)}"
            )

        manifest_bytes = (root / "manifest.json").read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise RetrievalValidationError("manifest.json is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise RetrievalValidationError("manifest.json must contain an object")
        if manifest.get("schema_version") != 1:
            raise RetrievalValidationError("unsupported retrieval index schema")
        if manifest.get("embedding_model") != expected_model:
            raise RetrievalValidationError(
                f"unexpected embedding model: {manifest.get('embedding_model')!r}"
            )
        if manifest.get("embedding_dimensions") != expected_dimensions:
            raise RetrievalValidationError(
                "unexpected embedding dimensions: "
                f"{manifest.get('embedding_dimensions')!r}"
            )

        hashes = manifest.get("artifact_hashes")
        if not isinstance(hashes, dict):
            raise RetrievalValidationError("manifest artifact_hashes must be an object")
        for name in required[1:]:
            expected = hashes.get(name)
            actual = _sha256_file(root / name)
            if expected != actual:
                raise RetrievalValidationError(f"artifact hash mismatch: {name}")
        content_digest = hashlib.sha256()
        for name in required[1:]:
            content_digest.update((root / name).read_bytes())
        if manifest.get("content_hash") != content_digest.hexdigest():
            raise RetrievalValidationError("retrieval index content hash mismatch")

        pages_data = _read_jsonl(root / "pages.jsonl")
        chunks_data = _read_jsonl(root / "chunks.jsonl")
        links_data = _read_jsonl(root / "links.jsonl")
        pages = tuple(
            PageRecord(
                page_id=_required_text(row, "page_id"),
                source_id=_required_text(row, "source_id"),
                slug=_required_text(row, "slug"),
                title=_required_text(row, "title"),
                perturbations=_string_tuple(row, "perturbations"),
            )
            for row in pages_data
        )
        chunks = tuple(
            ChunkRecord(
                chunk_id=_required_text(row, "chunk_id"),
                page_id=_required_text(row, "page_id"),
                chunk_index=_required_int(row, "chunk_index"),
                text=_required_text(row, "text"),
            )
            for row in chunks_data
        )
        links = tuple(
            LinkRecord(
                source_page_id=_required_text(row, "source_page_id"),
                target_page_id=_required_text(row, "target_page_id"),
                link_type=str(row.get("link_type", "")),
                provenance=row.get("provenance"),
            )
            for row in links_data
        )
        _reject_duplicates("page", [page.page_id for page in pages])
        _reject_duplicates("chunk", [chunk.chunk_id for chunk in chunks])
        page_ids = {page.page_id for page in pages}
        for chunk in chunks:
            if chunk.page_id not in page_ids:
                raise RetrievalValidationError(
                    f"chunk {chunk.chunk_id} references missing page {chunk.page_id}"
                )
        chunk_page_ids = {chunk.page_id for chunk in chunks}
        pages_without_chunks = sorted(page_ids - chunk_page_ids)
        if pages_without_chunks:
            raise RetrievalValidationError(
                "pages have no embedded chunks: "
                + ", ".join(pages_without_chunks)
            )
        for link in links:
            if (
                link.source_page_id not in page_ids
                or link.target_page_id not in page_ids
            ):
                raise RetrievalValidationError(
                    "link references a missing page: "
                    f"{link.source_page_id} -> {link.target_page_id}"
                )
        try:
            embeddings = np.load(root / "embeddings.npy", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RetrievalValidationError("embeddings.npy is invalid") from exc
        if embeddings.ndim != 2:
            raise RetrievalValidationError("embeddings must be a two-dimensional matrix")
        if embeddings.shape != (len(chunks), expected_dimensions):
            raise RetrievalValidationError(
                "embedding matrix shape does not match chunks and dimensions"
            )
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise RetrievalValidationError("embeddings must use a floating dtype")
        if not np.isfinite(embeddings).all():
            raise RetrievalValidationError("embeddings contain non-finite values")
        counts = manifest.get("counts", {})
        actual_counts = {
            "pages": len(pages),
            "chunks": len(chunks),
            "links": len(links),
            "vectors": int(embeddings.shape[0]),
        }
        if counts != actual_counts:
            raise RetrievalValidationError("manifest counts do not match artifacts")
        return cls(
            path=root.resolve(),
            manifest=manifest,
            manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
            pages=pages,
            chunks=chunks,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            links=links,
        )


class OpenAIEmbedder:
    model_name = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, client: Any | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        response = self._client.embeddings.create(
            model=EMBEDDING_API_MODEL,
            input=list(texts),
            dimensions=self.dimensions,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        result = np.asarray(
            [item.embedding for item in ordered], dtype=np.float32
        )
        if result.shape != (len(texts), self.dimensions):
            raise ValueError(
                f"embedder returned {result.shape}, expected "
                f"({len(texts)}, {self.dimensions})"
            )
        return result


class CachedOpenAIQueryExpander:
    model = EXPANSION_MODEL
    prompt_version = EXPANSION_PROMPT_VERSION

    def __init__(
        self,
        cache_dir: str | Path,
        policy: ExpansionPolicy | str,
        *,
        client: Any | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.policy = ExpansionPolicy(policy)
        self._client = client
        self._used_cache_files: list[Path] = []

    def expand(self, query: str) -> tuple[str, str, str]:
        normalized = _normalize_query(query)
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        key_payload = {
            "normalized_query": normalized,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RetrievalValidationError(
                    f"invalid expansion cache entry: {path}"
                ) from exc
            alternatives = _validate_expansions(payload.get("alternatives"), normalized)
            if (
                payload.get("query_hash") != query_hash
                or payload.get("model") != self.model
                or payload.get("prompt_version") != self.prompt_version
            ):
                raise RetrievalValidationError(f"invalid expansion cache entry: {path}")
            self._used_cache_files.append(path)
            return alternatives
        if self.policy is ExpansionPolicy.FROZEN:
            raise ExpansionCacheMissError(f"frozen expansion cache miss for {query_hash}")

        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        response = self._client.responses.create(
            model=self.model,
            instructions=(
                "Generate exactly three distinct search queries that preserve the "
                "user's intent while varying wording and likely memory anchors."
            ),
            input=query,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "retrieval_query_expansions",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "alternatives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 3,
                                "maxItems": 3,
                            }
                        },
                        "required": ["alternatives"],
                        "additionalProperties": False,
                    },
                }
            },
            store=False,
        )
        try:
            generated = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise RetrievalValidationError(
                "expansion model returned invalid structured output"
            ) from exc
        alternatives = _validate_expansions(
            generated.get("alternatives"), normalized
        )
        payload = {
            **key_payload,
            "query_hash": query_hash,
            "alternatives": list(alternatives),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, payload)
        self._used_cache_files.append(path)
        return alternatives

    @property
    def cache_hash(self) -> str | None:
        if not self._used_cache_files:
            return None
        digest = hashlib.sha256()
        for path in sorted(set(self._used_cache_files)):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


class IndexRetriever:
    def __init__(
        self,
        index: RetrievalIndex,
        mode: RetrievalMode | str,
        embedder: TextEmbedder,
        *,
        expander: QueryExpander | None = None,
    ):
        self.index = index
        self.mode = RetrievalMode(mode)
        self.embedder = embedder
        self.expander = expander
        if embedder.model_name != index.manifest["embedding_model"]:
            raise ValueError("query embedder model does not match retrieval index")
        if embedder.dimensions != index.manifest["embedding_dimensions"]:
            raise ValueError("query embedder dimensions do not match retrieval index")
        if self.mode is RetrievalMode.ENHANCED and expander is None:
            raise ValueError("enhanced retrieval requires a query expander")
        if self.mode is not RetrievalMode.ENHANCED and expander is not None:
            raise ValueError("query expansion is only valid for enhanced retrieval")
        self._page_by_id = {page.page_id: page for page in index.pages}
        self._chunks_by_page: dict[str, list[int]] = defaultdict(list)
        for position, chunk in enumerate(index.chunks):
            self._chunks_by_page[chunk.page_id].append(position)
        self._body_documents = {
            page.page_id: " ".join(
                index.chunks[position].text
                for position in self._chunks_by_page[page.page_id]
            )
            for page in index.pages
        }
        self._title_documents = {
            page.page_id: page.title for page in index.pages
        }

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        windows = _token_windows(request.query, EMBEDDING_MAX_INPUT_TOKENS)
        cosine, best_chunks, dense_channels = self._dense_channels(windows)
        expansions: tuple[str, ...] = ()
        channel_lists: dict[str, list[str]] = {}
        raw_rrf: dict[str, float] = {}
        normalized_rrf: dict[str, float] = {}
        pre_graph: dict[str, float] = {}
        inbound: dict[str, int] = {}
        multipliers: dict[str, float] = {}

        if self.mode is RetrievalMode.DENSE:
            channel_lists.update(dense_channels)
            if len(dense_channels) == 1:
                dense_original = next(iter(dense_channels.values()))
                final_scores = {
                    page_id: cosine[page_id] for page_id in dense_original
                }
            else:
                raw_rrf = _rrf(dense_channels.values())
                max_rrf = max(raw_rrf.values(), default=1.0)
                normalized_rrf = {
                    page_id: value / max_rrf for page_id, value in raw_rrf.items()
                }
                final_scores = dict(raw_rrf)
        else:
            body_original = _bm25_rank(
                request.query, self._body_documents, CANDIDATE_DEPTH
            )
            channel_lists = {"body_bm25:original": body_original}
            channel_lists.update(dense_channels)
            if self.mode is RetrievalMode.ENHANCED:
                assert self.expander is not None
                expansions = self.expander.expand(request.query)
                title_original = _bm25_rank(
                    request.query, self._title_documents, CANDIDATE_DEPTH
                )
                channel_lists["title_bm25:original"] = title_original
                for position, expansion in enumerate(expansions, start=1):
                    expansion_vector = self.embedder.embed([expansion])[0]
                    expansion_cosine, _ = self._page_cosines(expansion_vector)
                    channel_lists[f"body_bm25:expansion_{position}"] = _bm25_rank(
                        expansion, self._body_documents, CANDIDATE_DEPTH
                    )
                    channel_lists[f"dense:expansion_{position}"] = _rank_scores(
                        expansion_cosine, CANDIDATE_DEPTH
                    )
                    channel_lists[f"title_bm25:expansion_{position}"] = _bm25_rank(
                        expansion, self._title_documents, CANDIDATE_DEPTH
                    )
            raw_rrf = _rrf(channel_lists.values())
            max_rrf = max(raw_rrf.values(), default=1.0)
            normalized_rrf = {
                page_id: value / max_rrf for page_id, value in raw_rrf.items()
            }
            pre_graph = {
                page_id: (
                    RRF_WEIGHT * normalized_rrf[page_id]
                    + COSINE_WEIGHT * cosine[page_id]
                )
                for page_id in raw_rrf
            }
            final_scores = dict(pre_graph)
            if self.mode is RetrievalMode.ENHANCED:
                leading_pool = set(_rank_scores(pre_graph, CANDIDATE_DEPTH))
                inbound = {page_id: 0 for page_id in leading_pool}
                inbound_sources: dict[str, set[str]] = defaultdict(set)
                for link in self.index.links:
                    if (
                        link.source_page_id in leading_pool
                        and link.target_page_id in leading_pool
                        and link.source_page_id != link.target_page_id
                    ):
                        inbound_sources[link.target_page_id].add(link.source_page_id)
                for page_id in leading_pool:
                    inbound[page_id] = len(inbound_sources[page_id])
                multipliers = {
                    page_id: (
                        GRAPH_MULTIPLIER
                        if inbound.get(page_id, 0) >= GRAPH_MIN_INBOUND
                        else 1.0
                    )
                    for page_id in final_scores
                }
                final_scores = {
                    page_id: score * multipliers[page_id]
                    for page_id, score in final_scores.items()
                }

        ordered = _rank_scores(final_scores, request.top_k)
        channel_ranks = {
            name: {page_id: rank for rank, page_id in enumerate(values, start=1)}
            for name, values in channel_lists.items()
        }
        hits: list[RetrievalHit] = []
        for rank, page_id in enumerate(ordered, start=1):
            page = self._page_by_id[page_id]
            chunk_position = best_chunks[page_id]
            chunk = self.index.chunks[chunk_position]
            diagnostics = {
                "channel_ranks": {
                    name: ranks[page_id]
                    for name, ranks in channel_ranks.items()
                    if page_id in ranks
                },
                "raw_rrf": raw_rrf.get(page_id),
                "normalized_rrf": normalized_rrf.get(page_id),
                "original_query_cosine": cosine[page_id],
                "pre_graph_score": pre_graph.get(page_id, final_scores[page_id]),
                "graph_inbound_count": inbound.get(page_id, 0),
                "graph_multiplier": multipliers.get(page_id, 1.0),
                "final_score": final_scores[page_id],
            }
            hits.append(
                RetrievalHit(
                    page_id=page_id,
                    source_id=page.source_id,
                    slug=page.slug,
                    title=page.title,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=final_scores[page_id],
                    rank=rank,
                    perturbations=page.perturbations,
                    diagnostics=diagnostics,
                )
            )
        trace = {
            "retrieval_mode": self.mode.value,
            "original_query": request.query,
            "expansion_queries": list(expansions),
            "expansion_model": (
                self.expander.model if self.expander is not None else None
            ),
            "expansion_prompt_version": (
                self.expander.prompt_version if self.expander is not None else None
            ),
            "expansion_query_hash": (
                hashlib.sha256(
                    _normalize_query(request.query).encode("utf-8")
                ).hexdigest()
                if self.expander is not None
                else None
            ),
            "expansion_cache_hash": (
                self.expander.cache_hash if self.expander is not None else None
            ),
            "candidate_lists": channel_lists,
            "returned_pages": [
                {
                    "page_id": hit.page_id,
                    "source_id": hit.source_id,
                    "slug": hit.slug,
                    "selected_chunk_id": hit.chunk_id,
                    "perturbations": list(hit.perturbations),
                    "final_rank": hit.rank,
                    **{
                        key: value
                        for key, value in hit.diagnostics.items()
                        if key not in _NON_REPRODUCIBLE_DIAGNOSTICS
                    },
                }
                for hit in hits
            ],
        }
        return RetrievalResult(tuple(hits), trace)

    def _dense_channels(
        self, windows: Sequence[str]
    ) -> tuple[dict[str, float], dict[str, int], dict[str, list[str]]]:
        """Embed each query window and rank pages by cosine per window.

        A single window (the normal case) yields the original dense channel and
        its cosine map unchanged. Multiple windows each become their own dense
        channel — the caller RRF-merges them — while the returned cosine map
        max-pools across windows so a page counts as relevant when any window
        matches it.
        """
        vectors = self.embedder.embed(list(windows))
        per_window = [self._page_cosines(vectors[position]) for position in range(len(windows))]
        if len(per_window) == 1:
            cosine, best_chunks = per_window[0]
            return cosine, best_chunks, {
                "dense:original": _rank_scores(cosine, CANDIDATE_DEPTH)
            }
        cosine = {}
        best_chunks = {}
        for window_cosine, window_best in per_window:
            for page_id, score in window_cosine.items():
                if page_id not in cosine or score > cosine[page_id]:
                    cosine[page_id] = score
                    best_chunks[page_id] = window_best[page_id]
        channels = {
            f"dense:window_{position}": _rank_scores(window_cosine, CANDIDATE_DEPTH)
            for position, (window_cosine, _) in enumerate(per_window, start=1)
        }
        return cosine, best_chunks, channels

    def _page_cosines(
        self, query_vector: np.ndarray
    ) -> tuple[dict[str, float], dict[str, int]]:
        query_norm = float(np.linalg.norm(query_vector))
        matrix_norms = np.linalg.norm(self.index.embeddings, axis=1)
        denominators = matrix_norms * query_norm
        scores = np.divide(
            self.index.embeddings @ query_vector,
            denominators,
            out=np.zeros(len(self.index.chunks), dtype=np.float32),
            where=denominators != 0,
        )
        page_scores: dict[str, float] = {}
        best_chunks: dict[str, int] = {}
        for position, chunk in enumerate(self.index.chunks):
            score = float(scores[position])
            previous = page_scores.get(chunk.page_id)
            previous_position = best_chunks.get(chunk.page_id)
            if (
                previous is None
                or score > previous
                or (
                    score == previous
                    and previous_position is not None
                    and chunk.chunk_id
                    < self.index.chunks[previous_position].chunk_id
                )
            ):
                page_scores[chunk.page_id] = score
                best_chunks[chunk.page_id] = position
        return page_scores, best_chunks


class EntitySurfaceExtractor:
    def __init__(
        self,
        index: RetrievalIndex,
        *,
        nlp: Any | None = None,
        automaton_factory: Callable[[], Any] | None = None,
    ):
        self.index = index
        self._nlp = nlp if nlp is not None else _load_spacy_model()
        factory = automaton_factory or _load_automaton_factory()
        self._surface_pages = self._build_surface_pages()
        self._automaton = factory()
        for normalized, page_ids in self._surface_pages.items():
            self._automaton.add_word(normalized, (normalized, tuple(sorted(page_ids))))
        self._automaton.make_automaton()

    def extract(self, observation_text: str) -> tuple[EntitySeed, ...]:
        seeds: list[EntitySeed] = []
        seen: set[tuple[str, str]] = set()
        normalized_observation = observation_text.casefold()
        for end, (normalized, page_ids) in self._automaton.iter(normalized_observation):
            start = end - len(normalized) + 1
            if not _entity_boundary(normalized_observation, start, end + 1):
                continue
            surface = observation_text[start : end + 1]
            key = ("gazetteer", normalized)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(
                EntitySeed(
                    seed_id=f"entity-{len(seeds) + 1}",
                    surface=surface,
                    normalized_surface=normalized,
                    source="gazetteer",
                    span_start=start,
                    span_end=end + 1,
                    matched_page_ids=page_ids,
                )
            )
        for surface, start, end in self._noun_phrases(observation_text):
            normalized = _normalize_entity_surface(surface)
            key = ("noun_phrase", normalized)
            if not normalized or key in seen or ("gazetteer", normalized) in seen:
                continue
            seen.add(key)
            seeds.append(
                EntitySeed(
                    seed_id=f"entity-{len(seeds) + 1}",
                    surface=surface,
                    normalized_surface=normalized,
                    source="noun_phrase",
                    span_start=start,
                    span_end=end,
                )
            )
        return tuple(seeds)

    def _build_surface_pages(self) -> dict[str, set[str]]:
        surfaces: dict[str, set[str]] = defaultdict(set)
        for page in self.index.pages:
            for surface in _page_title_slug_surfaces(page):
                normalized = _normalize_entity_surface(surface)
                if normalized:
                    surfaces[normalized].add(page.page_id)
        for chunk in self.index.chunks:
            for surface, _, _ in self._noun_phrases(chunk.text):
                if not _keep_body_surface(surface):
                    continue
                normalized = _normalize_entity_surface(surface)
                if normalized:
                    surfaces[normalized].add(chunk.page_id)
        return dict(surfaces)

    def _noun_phrases(self, text: str) -> list[tuple[str, int, int]]:
        doc = self._nlp(text)
        phrases = []
        for chunk in doc.noun_chunks:
            surface = chunk.text.strip()
            if surface:
                phrases.append((surface, int(chunk.start_char), int(chunk.end_char)))
        return phrases


class EntityExactRetriever:
    retrieval_condition_detail = "entity_exact_match"

    def __init__(
        self,
        index: RetrievalIndex,
        *,
        extractor: EntitySurfaceExtractor | None = None,
    ):
        self.index = index
        self.extractor = extractor or EntitySurfaceExtractor(index)
        self._page_by_id = {page.page_id: page for page in index.pages}
        self._chunks_by_page: dict[str, list[ChunkRecord]] = defaultdict(list)
        for chunk in index.chunks:
            self._chunks_by_page[chunk.page_id].append(chunk)
        self._body_documents = {
            page.page_id: " ".join(
                chunk.text for chunk in self._chunks_by_page[page.page_id]
            )
            for page in index.pages
        }
        self._title_documents = {
            page.page_id: page.title for page in index.pages
        }

    def retrieve_entities(
        self,
        observation_text: str,
        *,
        top_k: int,
    ) -> EntityRetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        seeds = self.extractor.extract(observation_text)
        hits: list[RetrievalHit] = []
        per_seed = []
        for seed in seeds:
            page_ids = self._exact_page_ids(seed, top_k)
            seed_pages = []
            for rank, page_id in enumerate(page_ids, start=1):
                page = self._page_by_id[page_id]
                chunk, chunk_score = self._best_exact_chunk(seed, page_id)
                score = self._exact_match_score(seed.normalized_surface, page_id)
                hit = RetrievalHit(
                    page_id=page.page_id,
                    source_id=page.source_id,
                    slug=page.slug,
                    title=page.title,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                    perturbations=page.perturbations,
                    diagnostics={
                        "entity_seed_id": seed.seed_id,
                        "entity_surface": seed.surface,
                        "entity_source": seed.source,
                        "exact_match_score": score,
                        "chunk_exact_match_score": chunk_score,
                    },
                )
                hits.append(hit)
                seed_pages.append(
                    {
                        "page_id": page.page_id,
                        "source_id": page.source_id,
                        "slug": page.slug,
                        "selected_chunk_id": chunk.chunk_id,
                        "rank": rank,
                        "score": score,
                        "chunk_score": chunk_score,
                        "match_type": "exact",
                        "perturbations": list(page.perturbations),
                    }
                )
            per_seed.append(
                {
                    "seed_id": seed.seed_id,
                    "surface": seed.surface,
                    "normalized_surface": seed.normalized_surface,
                    "source": seed.source,
                    "span": [seed.span_start, seed.span_end],
                    "matched_page_ids": list(seed.matched_page_ids),
                    "returned_pages": seed_pages,
                }
            )
        trace = {
            "retrieval_condition_detail": self.retrieval_condition_detail,
            "entity_seeds": [seed.__dict__ for seed in seeds],
            "per_seed_retrievals": per_seed,
        }
        return EntityRetrievalResult(seeds, tuple(hits), trace)

    def _exact_page_ids(self, seed: EntitySeed, top_k: int) -> list[str]:
        if seed.source == "gazetteer":
            return sorted(
                page_id
                for page_id in seed.matched_page_ids
                if page_id in self._page_by_id
            )
        scored = {
            page_id: self._exact_match_score(seed.normalized_surface, page_id)
            for page_id in self._page_by_id
        }
        matched = {
            page_id: score for page_id, score in scored.items() if score > 0
        }
        return _rank_scores(matched, top_k)

    def _exact_match_score(self, normalized_surface: str, page_id: str) -> float:
        score = 0.0
        if _normalized_contains(self._title_documents[page_id], normalized_surface):
            score += 2.0
        body = self._body_documents[page_id]
        if _normalized_contains(body, normalized_surface):
            score += 1.0
        return score

    def _best_exact_chunk(
        self, seed: EntitySeed, page_id: str
    ) -> tuple[ChunkRecord, float]:
        chunks = self._chunks_by_page[page_id]
        exact_scores = {
            chunk.chunk_id: (
                1.0 if _normalized_contains(chunk.text, seed.normalized_surface) else 0.0
            )
            for chunk in chunks
        }
        matched = {
            chunk_id: score for chunk_id, score in exact_scores.items() if score > 0
        }
        if matched:
            chunk_id = _rank_scores(matched, 1)[0]
            chunk = next(chunk for chunk in chunks if chunk.chunk_id == chunk_id)
            return chunk, matched[chunk_id]
        scores = _bm25_scores(seed.surface, {chunk.chunk_id: chunk.text for chunk in chunks})
        if scores:
            chunk_id = _rank_scores(scores, 1)[0]
            chunk = next(chunk for chunk in chunks if chunk.chunk_id == chunk_id)
            return chunk, 0.0
        return chunks[0], 0.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _bm25_scores(query: str, documents: dict[str, str]) -> dict[str, float]:
    tokens_by_id = {
        document_id: _tokenize(text) for document_id, text in documents.items()
    }
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {}
    count = len(tokens_by_id)
    average_length = (
        sum(len(tokens) for tokens in tokens_by_id.values()) / count if count else 0
    )
    document_frequency: Counter[str] = Counter()
    for tokens in tokens_by_id.values():
        document_frequency.update(set(tokens))
    scores: dict[str, float] = {}
    k1 = 1.5
    b = 0.75
    for document_id, tokens in tokens_by_id.items():
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (count - df + 0.5) / (df + 0.5))
            length_ratio = len(tokens) / average_length if average_length else 0.0
            score += inverse_frequency * (
                frequency * (k1 + 1)
                / (frequency + k1 * (1 - b + b * length_ratio))
            )
        if score > 0:
            scores[document_id] = score
    return scores


def _bm25_rank(
    query: str, documents: dict[str, str], limit: int
) -> list[str]:
    scores = _bm25_scores(query, documents)
    if not scores and not _tokenize(query):
        return sorted(documents)[:limit]
    return _rank_scores(scores, limit)


def _rrf(rankings: Sequence[Sequence[str]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, page_id in enumerate(ranking, start=1):
            scores[page_id] += 1.0 / (RRF_K + rank)
    return dict(scores)


def _rank_scores(scores: dict[str, float], limit: int) -> list[str]:
    return [
        page_id
        for page_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _normalize_entity_surface(surface: str) -> str:
    normalized = " ".join(_tokenize(surface))
    return normalized if len(normalized) >= 2 else ""


def _normalized_contains(text: str, normalized_surface: str) -> bool:
    if not normalized_surface:
        return False
    normalized_text = _normalize_entity_surface(text)
    return f" {normalized_surface} " in f" {normalized_text} "


def _page_title_slug_surfaces(page: PageRecord) -> list[str]:
    surfaces = [page.title]
    slug_tail = page.slug.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
    if slug_tail != page.title:
        surfaces.append(slug_tail)
    return surfaces


_BODY_STOP_PHRASES = {
    "next steps",
    "market report",
    "weekly review",
    "morning reflection",
    "team 1 1s",
}


def _keep_body_surface(surface: str) -> bool:
    tokens = _tokenize(surface)
    if not 2 <= len(tokens) <= 6:
        return False
    normalized = " ".join(tokens)
    if normalized in _BODY_STOP_PHRASES:
        return False
    has_identifier = any(any(char.isdigit() for char in token) for token in tokens)
    has_properish = any(part[:1].isupper() for part in surface.split())
    return has_identifier or has_properish


def _entity_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not before.isalnum() and not after.isalnum()


def _load_automaton_factory() -> Callable[[], Any]:
    try:
        import ahocorasick
    except ImportError as exc:
        raise RetrievalValidationError(
            "all_entity_output_rag requires pyahocorasick. "
            "Install project dependencies with `python -m pip install -e .`."
        ) from exc
    return ahocorasick.Automaton


def _load_spacy_model() -> Any:
    try:
        import spacy
    except ImportError as exc:
        raise RetrievalValidationError(
            "all_entity_output_rag requires spacy. "
            "Install project dependencies with `python -m pip install -e .`."
        ) from exc
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RetrievalValidationError(
            "all_entity_output_rag requires the spaCy model en_core_web_sm. "
            "Install it with `python -m spacy download en_core_web_sm`."
        ) from exc


def _validate_expansions(
    values: Any, normalized_original: str
) -> tuple[str, str, str]:
    if not isinstance(values, list) or len(values) != 3:
        raise RetrievalValidationError("expansion output must contain exactly three")
    cleaned = tuple(value.strip() for value in values if isinstance(value, str))
    normalized = [_normalize_query(value) for value in cleaned]
    if (
        len(cleaned) != 3
        or any(not value for value in cleaned)
        or len(set(normalized)) != 3
        or normalized_original in normalized
    ):
        raise RetrievalValidationError(
            "expansion alternatives must be non-empty, unique, and distinct"
        )
    return cleaned  # type: ignore[return-value]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalValidationError(
                f"{path.name}:{line_number} is not valid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise RetrievalValidationError(
                f"{path.name}:{line_number} must be an object"
            )
        rows.append(row)
    return rows


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise RetrievalValidationError(f"{key} must be a non-empty string")
    return value


def _required_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrievalValidationError(f"{key} must be an integer")
    return value


def _string_tuple(row: dict[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise RetrievalValidationError(f"{key} must be a list of strings")
    return tuple(value)


def _reject_duplicates(kind: str, values: Sequence[str]) -> None:
    duplicates = sorted(
        value for value, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise RetrievalValidationError(
            f"duplicate {kind} IDs: {', '.join(duplicates)}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
