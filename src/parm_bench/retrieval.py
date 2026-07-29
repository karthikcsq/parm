from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .service_tier import service_tier_kwargs


EMBEDDING_MODEL = "openai:text-embedding-3-small"
EMBEDDING_API_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_ENCODING = "cl100k_base"
EMBEDDING_MAX_INPUT_TOKENS = 8192
EMBEDDING_BATCH_SIZE = 64
SCOPED_RETRIEVER_CACHE_SIZE = 8
CANDIDATE_DEPTH = 20
OFFICIAL_TOP_K = 5
RRF_K = 60
RRF_WEIGHT = 0.70
COSINE_WEIGHT = 0.30
GRAPH_MIN_INBOUND = 2
GRAPH_MULTIPLIER = 1.05
PARM_SEMANTIC_DEPTH = 5
PARM_SEMANTIC_MIN_COSINE = 0.30
PARM_SINGLETON_MIN_COSINE = 0.36
PARM_MIN_CONVERGING_CONCEPTS = 3
PARM_MIN_LEXICAL_CONCEPTS = 2
PARM_DIRECT_NOTE_MIN_PAGE_CONTRAST = 1.25
PARM_DIRECT_NOTE_MIN_REGION_CONTRAST = 1.25
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

_PARM_LISTING_PREFIX = re.compile(
    r"^(?:Session|Workshop|Lead Story|Item|Brief|Editor's Pick|Episode|"
    r"New Release|Vendor|Candidate|Result|Listing|Demo|Webinar|Company|"
    r"Case Study|Article|Automation|Feature|Film|Event|Essay|Chart)\b"
)
_PARM_GENERIC_CONCEPTS = frozenset(
    {
        "block",
        "candidate",
        "discussion",
        "dispatch",
        "episode",
        "host",
        "item",
        "label",
        "listing",
        "minute",
        "name",
        "option",
        "panel",
        "question",
        "reply",
        "result",
        "reviewer",
        "session",
        "story",
        "team",
        "title",
        "vendor",
        "work",
    }
)
_PARM_GENERIC_ANCHORS = _PARM_GENERIC_CONCEPTS | frozenset(
    {"afternoon", "exactly", "nearby", "new", "one", "thursday", "today"}
)
_PARM_GRAPH_LEXICAL_STOPWORDS = frozenset(
    {
        "after",
        "also",
        "and",
        "are",
        "been",
        "before",
        "being",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "into",
        "its",
        "may",
        "might",
        "more",
        "most",
        "not",
        "only",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "same",
        "she",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "too",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_PARM_DURABLE_PREFIXES = frozenset({"doc", "emails", "meetings", "notes"})


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
    corpus_id: str = ""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    page_id: str
    chunk_index: int
    text: str
    corpus_id: str = ""


@dataclass(frozen=True)
class LinkRecord:
    source_page_id: str
    target_page_id: str
    link_type: str = ""
    provenance: str | None = None


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    chunk_id: str
    page_id: str
    sentence_index: int
    text: str
    corpus_id: str = ""


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = OFFICIAL_TOP_K
    corpus_id: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("retrieval query must be non-empty")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.corpus_id is not None and not self.corpus_id.strip():
            raise ValueError("corpus_id must be non-empty when provided")


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
    corpus_id: str = ""


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
        corpus_id: str | None = None,
    ) -> EntityRetrievalResult: ...


class PARMObservationRetriever(Protocol):
    def retrieve_observation(
        self,
        prompt: str,
        observation_text: str,
        *,
        top_k: int,
        corpus_id: str | None = None,
    ) -> RetrievalResult: ...


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


class CueConceptExtractor(Protocol):
    def task_anchors(self, prompt: str) -> tuple[str, ...]: ...

    def rare_region_concepts(
        self, descriptions: Sequence[str]
    ) -> tuple[tuple[str, ...], ...]: ...


@dataclass(frozen=True)
class RetrievalIndex:
    path: Path
    manifest: dict[str, Any]
    manifest_hash: str
    pages: tuple[PageRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    embeddings: np.ndarray
    links: tuple[LinkRecord, ...]
    sentences: tuple[SentenceRecord, ...] = ()
    sentence_embeddings: np.ndarray | None = None

    @property
    def corpus_ids(self) -> tuple[str, ...]:
        return tuple(sorted({page.corpus_id for page in self.pages}))

    def resolve_corpus_id(self, requested: str | None) -> str:
        corpus_ids = self.corpus_ids
        if requested is None:
            if len(corpus_ids) != 1:
                raise RetrievalValidationError(
                    "multi-corpus retrieval requires corpus_id"
                )
            return corpus_ids[0]
        if requested not in corpus_ids:
            raise RetrievalValidationError(
                f"retrieval corpus_id {requested!r} is not present in the index"
            )
        return requested

    def scoped(self, corpus_id: str) -> "RetrievalIndex":
        resolved = self.resolve_corpus_id(corpus_id)
        if self.corpus_ids == (resolved,):
            return self
        page_ids = {
            page.page_id for page in self.pages if page.corpus_id == resolved
        }
        chunk_positions = [
            position
            for position, chunk in enumerate(self.chunks)
            if chunk.corpus_id == resolved
        ]
        chunk_ids = {self.chunks[position].chunk_id for position in chunk_positions}
        sentence_positions = [
            position
            for position, sentence in enumerate(self.sentences)
            if sentence.corpus_id == resolved
        ]
        manifest = dict(self.manifest)
        manifest["corpus_ids"] = [resolved]
        manifest["corpus_id"] = resolved
        return RetrievalIndex(
            path=self.path,
            manifest=manifest,
            manifest_hash=self.manifest_hash,
            pages=tuple(
                page for page in self.pages if page.corpus_id == resolved
            ),
            chunks=tuple(self.chunks[position] for position in chunk_positions),
            embeddings=np.asarray(
                self.embeddings[chunk_positions], dtype=np.float32
            ),
            links=tuple(
                link
                for link in self.links
                if (
                    link.source_page_id in page_ids
                    and link.target_page_id in page_ids
                )
            ),
            sentences=tuple(
                self.sentences[position] for position in sentence_positions
            ),
            sentence_embeddings=(
                np.asarray(
                    self.sentence_embeddings[sentence_positions], dtype=np.float32
                )
                if self.sentence_embeddings is not None
                else None
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model: str = EMBEDDING_MODEL,
        expected_dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> "RetrievalIndex":
        root = Path(path)
        core_artifacts = (
            "manifest.json",
            "pages.jsonl",
            "chunks.jsonl",
            "embeddings.npy",
            "links.jsonl",
        )
        required = core_artifacts
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
        schema_version = manifest.get("schema_version")
        if schema_version not in {1, 2, 3}:
            raise RetrievalValidationError("unsupported retrieval index schema")
        artifact_names = list(core_artifacts[1:])
        if schema_version in {2, 3}:
            artifact_names.extend(("sentences.jsonl", "sentence_embeddings.npy"))
            missing = [name for name in artifact_names if not (root / name).is_file()]
            if missing:
                raise RetrievalValidationError(
                    f"retrieval index is missing: {', '.join(missing)}"
                )
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
        for name in artifact_names:
            expected = hashes.get(name)
            actual = _sha256_file(root / name)
            if expected != actual:
                raise RetrievalValidationError(f"artifact hash mismatch: {name}")
        content_digest = hashlib.sha256()
        for name in artifact_names:
            content_digest.update((root / name).read_bytes())
        if manifest.get("content_hash") != content_digest.hexdigest():
            raise RetrievalValidationError("retrieval index content hash mismatch")

        pages_data = _read_jsonl(root / "pages.jsonl")
        chunks_data = _read_jsonl(root / "chunks.jsonl")
        links_data = _read_jsonl(root / "links.jsonl")
        legacy_corpus_id = (
            _required_text(manifest, "corpus_id")
            if schema_version in {1, 2}
            else None
        )
        manifest_corpus_ids = manifest.get("corpus_ids")
        if schema_version == 3:
            if (
                not isinstance(manifest_corpus_ids, list)
                or not manifest_corpus_ids
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in manifest_corpus_ids
                )
            ):
                raise RetrievalValidationError(
                    "schema-v3 manifest corpus_ids must be a non-empty string list"
                )
            if len(set(manifest_corpus_ids)) != len(manifest_corpus_ids):
                raise RetrievalValidationError(
                    "schema-v3 manifest corpus_ids contains duplicates"
                )
        pages = tuple(
            PageRecord(
                corpus_id=(
                    _required_text(row, "corpus_id")
                    if schema_version == 3
                    else str(legacy_corpus_id)
                ),
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
                corpus_id=(
                    _required_text(row, "corpus_id")
                    if schema_version == 3
                    else str(legacy_corpus_id)
                ),
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
        sentences: tuple[SentenceRecord, ...] = ()
        if schema_version in {2, 3}:
            sentences_data = _read_jsonl(root / "sentences.jsonl")
            sentences = tuple(
                SentenceRecord(
                    corpus_id=(
                        _required_text(row, "corpus_id")
                        if schema_version == 3
                        else str(legacy_corpus_id)
                    ),
                    sentence_id=_required_text(row, "sentence_id"),
                    chunk_id=_required_text(row, "chunk_id"),
                    page_id=_required_text(row, "page_id"),
                    sentence_index=_required_int(row, "sentence_index"),
                    text=_required_text(row, "text"),
                )
                for row in sentences_data
            )
        _reject_duplicates("page", [page.page_id for page in pages])
        _reject_duplicates("chunk", [chunk.chunk_id for chunk in chunks])
        _reject_duplicates(
            "sentence", [sentence.sentence_id for sentence in sentences]
        )
        page_ids = {page.page_id for page in pages}
        page_corpora = {page.page_id: page.corpus_id for page in pages}
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        chunk_corpora = {chunk.chunk_id: chunk.corpus_id for chunk in chunks}
        if schema_version == 3:
            actual_corpus_ids = sorted(set(page_corpora.values()))
            if sorted(manifest_corpus_ids) != actual_corpus_ids:
                raise RetrievalValidationError(
                    "manifest corpus_ids do not match page corpus IDs"
                )
        for chunk in chunks:
            if chunk.page_id not in page_ids:
                raise RetrievalValidationError(
                    f"chunk {chunk.chunk_id} references missing page {chunk.page_id}"
                )
            if chunk.corpus_id != page_corpora.get(chunk.page_id):
                raise RetrievalValidationError(
                    f"chunk {chunk.chunk_id} crosses corpus boundary"
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
            if (
                page_corpora[link.source_page_id]
                != page_corpora[link.target_page_id]
            ):
                raise RetrievalValidationError("link crosses corpus boundary")
        for sentence in sentences:
            if sentence.page_id not in page_ids:
                raise RetrievalValidationError(
                    f"sentence {sentence.sentence_id} references missing page "
                    f"{sentence.page_id}"
                )
            if sentence.chunk_id not in chunk_ids:
                raise RetrievalValidationError(
                    f"sentence {sentence.sentence_id} references missing chunk "
                    f"{sentence.chunk_id}"
                )
            if (
                sentence.corpus_id != page_corpora.get(sentence.page_id)
                or sentence.corpus_id != chunk_corpora.get(sentence.chunk_id)
            ):
                raise RetrievalValidationError(
                    f"sentence {sentence.sentence_id} crosses corpus boundary"
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
        sentence_embeddings: np.ndarray | None = None
        if schema_version in {2, 3}:
            try:
                sentence_embeddings = np.load(
                    root / "sentence_embeddings.npy", allow_pickle=False
                )
            except (OSError, ValueError) as exc:
                raise RetrievalValidationError(
                    "sentence_embeddings.npy is invalid"
                ) from exc
            if sentence_embeddings.ndim != 2:
                raise RetrievalValidationError(
                    "sentence embeddings must be a two-dimensional matrix"
                )
            if sentence_embeddings.shape != (len(sentences), expected_dimensions):
                raise RetrievalValidationError(
                    "sentence embedding matrix shape does not match sentences "
                    "and dimensions"
                )
            if not np.issubdtype(sentence_embeddings.dtype, np.floating):
                raise RetrievalValidationError(
                    "sentence embeddings must use a floating dtype"
                )
            if not np.isfinite(sentence_embeddings).all():
                raise RetrievalValidationError(
                    "sentence embeddings contain non-finite values"
                )
        counts = manifest.get("counts", {})
        actual_counts = {
            "pages": len(pages),
            "chunks": len(chunks),
            "links": len(links),
            "vectors": int(embeddings.shape[0]),
        }
        if schema_version in {2, 3}:
            assert sentence_embeddings is not None
            actual_counts.update(
                {
                    "sentences": len(sentences),
                    "sentence_vectors": int(sentence_embeddings.shape[0]),
                }
            )
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
            sentences=sentences,
            sentence_embeddings=(
                np.asarray(sentence_embeddings, dtype=np.float32)
                if sentence_embeddings is not None
                else None
            ),
        )


def segment_memory_sentences(text: str) -> tuple[str, ...]:
    """Create stable semantic units from a frozen memory chunk."""
    sentences: list[str] = []
    for paragraph in re.split(r"\r?\n\s*\r?\n", text):
        normalized = " ".join(
            line.strip() for line in paragraph.splitlines() if line.strip()
        )
        if not normalized:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", normalized):
            sentence = sentence.strip()
            if len(sentence.split()) >= 3:
                sentences.append(sentence)
    return tuple(sentences)


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
        embeddings = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
            for attempt in range(3):
                try:
                    response = self._client.embeddings.create(
                        model=EMBEDDING_API_MODEL,
                        input=batch,
                        dimensions=self.dimensions,
                        encoding_format="float",
                    )
                    break
                except json.JSONDecodeError:
                    if attempt == 2:
                        raise
            ordered = sorted(response.data, key=lambda item: item.index)
            batch_result = np.asarray(
                [item.embedding for item in ordered], dtype=np.float32
            )
            if batch_result.shape != (len(batch), self.dimensions):
                raise ValueError(
                    f"embedder returned {batch_result.shape}, expected "
                    f"({len(batch)}, {self.dimensions})"
                )
            embeddings.append(batch_result)
        result = np.concatenate(embeddings)
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
            **service_tier_kwargs(),
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
        corpus_id = self.index.resolve_corpus_id(request.corpus_id)
        if len(self.index.corpus_ids) > 1:
            scoped = IndexRetriever(
                self.index.scoped(corpus_id),
                self.mode,
                self.embedder,
                expander=self.expander,
            )
            return scoped.retrieve(
                RetrievalRequest(
                    request.query,
                    top_k=request.top_k,
                    corpus_id=corpus_id,
                )
            )
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
                    corpus_id=corpus_id,
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
            "corpus_id": corpus_id,
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
                    "corpus_id": hit.corpus_id,
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
        self._automaton_factory = factory
        self._surface_pages = self._build_surface_pages()
        self._automaton = factory()
        for normalized, page_ids in self._surface_pages.items():
            self._automaton.add_word(normalized, (normalized, tuple(sorted(page_ids))))
        self._automaton.make_automaton()

    def for_index(self, index: RetrievalIndex) -> "EntitySurfaceExtractor":
        return EntitySurfaceExtractor(
            index,
            nlp=self._nlp,
            automaton_factory=self._automaton_factory,
        )

    def extract(self, observation_text: str) -> tuple[EntitySeed, ...]:
        seeds = list(self.extract_gazetteer(observation_text))
        seen: set[tuple[str, str]] = set()
        seen.update(
            ("gazetteer", seed.normalized_surface)
            for seed in seeds
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

    def extract_gazetteer(
        self, observation_text: str
    ) -> tuple[EntitySeed, ...]:
        seeds: list[EntitySeed] = []
        seen: set[str] = set()
        normalized_observation = observation_text.casefold()
        for end, (normalized, page_ids) in self._automaton.iter(normalized_observation):
            start = end - len(normalized) + 1
            if not _entity_boundary(normalized_observation, start, end + 1):
                continue
            surface = observation_text[start : end + 1]
            if normalized in seen:
                continue
            seen.add(normalized)
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


class SpacyCueConceptExtractor:
    def __init__(self, nlp: Any | None = None):
        self._nlp = nlp if nlp is not None else _load_spacy_model()

    def task_anchors(self, prompt: str) -> tuple[str, ...]:
        task_text = prompt.split("Reply", 1)[0]
        doc = self._nlp(task_text)
        anchors: list[str] = []
        for chunk in list(doc.noun_chunks)[:2]:
            for token in chunk:
                normalized = token.lemma_.casefold()
                if (
                    token.pos_ in {"ADJ", "NOUN", "PROPN"}
                    and token.is_alpha
                    and normalized not in _PARM_GENERIC_ANCHORS
                ):
                    anchors.append(normalized)
        return tuple(dict.fromkeys(anchors))[:4]

    def rare_region_concepts(
        self, descriptions: Sequence[str]
    ) -> tuple[tuple[str, ...], ...]:
        documents = list(self._nlp.pipe(descriptions, batch_size=64))
        concepts = [
            {
                token.lemma_.casefold()
                for token in document
                if (
                    token.pos_ == "NOUN"
                    and token.is_alpha
                    and not token.is_stop
                    and token.lemma_.casefold() not in _PARM_GENERIC_CONCEPTS
                )
            }
            for document in documents
        ]
        document_frequency = Counter(
            concept for region_concepts in concepts for concept in region_concepts
        )
        return tuple(
            tuple(
                sorted(
                    concept
                    for concept in region_concepts
                    if document_frequency[concept] == 1
                )
            )
            for region_concepts in concepts
        )


class PARMConvergenceRetriever:
    retrieval_condition_detail = "parm_convergence_v2"
    evidence_projection_version = "convergent_evidence_v2"
    admission_policy = "convergence_threshold"

    def __init__(
        self,
        index: RetrievalIndex,
        embedder: TextEmbedder,
        *,
        entity_extractor: EntitySurfaceExtractor | None = None,
        concept_extractor: CueConceptExtractor | None = None,
    ):
        if index.sentence_embeddings is None or not index.sentences:
            raise ValueError(
                "PARM convergence retrieval requires sentence-index artifacts"
            )
        if embedder.model_name != index.manifest["embedding_model"]:
            raise ValueError("query embedder model does not match retrieval index")
        if embedder.dimensions != index.manifest["embedding_dimensions"]:
            raise ValueError("query embedder dimensions do not match retrieval index")
        self.index = index
        self.embedder = embedder
        self._scoped_retrievers: OrderedDict[
            str, PARMConvergenceRetriever
        ] = OrderedDict()
        self._scoped_nlp: Any | None = None
        if len(index.corpus_ids) > 1:
            if entity_extractor is None:
                self._scoped_nlp = _load_spacy_model()
            if concept_extractor is None:
                nlp = self._scoped_nlp or _load_spacy_model()
                concept_extractor = SpacyCueConceptExtractor(nlp)
            self.entity_extractor = entity_extractor
            self.concept_extractor = concept_extractor
            return
        if entity_extractor is None or concept_extractor is None:
            nlp = _load_spacy_model()
            entity_extractor = entity_extractor or EntitySurfaceExtractor(
                index, nlp=nlp
            )
            concept_extractor = concept_extractor or SpacyCueConceptExtractor(nlp)
        self.entity_extractor = entity_extractor
        self.concept_extractor = concept_extractor
        self._page_by_id = {page.page_id: page for page in index.pages}
        self._chunks_by_page: dict[str, list[tuple[int, ChunkRecord]]] = defaultdict(
            list
        )
        for position, chunk in enumerate(index.chunks):
            self._chunks_by_page[chunk.page_id].append((position, chunk))
        self._page_text = {
            page.page_id: " ".join(
                chunk.text for _, chunk in self._chunks_by_page[page.page_id]
            )
            for page in index.pages
        }
        self._eligible_pages = {
            page.page_id
            for page in index.pages
            if (
                page.slug.split("/", 1)[0] in _PARM_DURABLE_PREFIXES
                and not page.perturbations
            )
        }
        self._graph_page_tokens = {
            page_id: frozenset(_tokenize(self._page_text[page_id]))
            for page_id in self._eligible_pages
        }
        self._graph_term_document_frequency: Counter[str] = Counter()
        for tokens in self._graph_page_tokens.values():
            self._graph_term_document_frequency.update(tokens)
        self._graph_document_count = len(self._graph_page_tokens)
        self._direct_note_text = {
            page_id: self._page_text[page_id]
            for page_id in self._eligible_pages
            if self._page_by_id[page_id].slug.startswith("notes/")
        }
        self._direct_note_bm25 = _BM25Corpus(self._direct_note_text)
        self._inbound: dict[str, set[str]] = defaultdict(set)
        for link in index.links:
            if link.source_page_id in self._eligible_pages:
                self._inbound[link.target_page_id].add(link.source_page_id)
        sentence_matrix = index.sentence_embeddings
        sentence_norms = np.linalg.norm(sentence_matrix, axis=1, keepdims=True)
        self._sentence_matrix = np.divide(
            sentence_matrix,
            sentence_norms,
            out=np.zeros_like(sentence_matrix),
            where=sentence_norms != 0,
        )
        self._sentence_page_ids = tuple(
            sentence.page_id for sentence in index.sentences
        )
        self._sentences_by_page: dict[str, list[SentenceRecord]] = defaultdict(list)
        for sentence in index.sentences:
            self._sentences_by_page[sentence.page_id].append(sentence)

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
                scoped_index = self.index.scoped(resolved_corpus_id)
                scoped_entity_extractor = (
                    self.entity_extractor.for_index(scoped_index)
                    if self.entity_extractor is not None
                    else EntitySurfaceExtractor(
                        scoped_index,
                        nlp=self._scoped_nlp,
                    )
                )
                scoped = PARMConvergenceRetriever(
                    scoped_index,
                    self.embedder,
                    entity_extractor=scoped_entity_extractor,
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
        anchors = self.concept_extractor.task_anchors(prompt)
        concepts_by_region = self.concept_extractor.rare_region_concepts(
            [region["description"] for region in regions]
        )
        semantic_seeds: list[dict[str, Any]] = []
        for region, concepts in zip(regions, concepts_by_region):
            for concept in concepts:
                for anchor in anchors:
                    semantic_seeds.append(
                        {
                            "seed_id": f"semantic-{len(semantic_seeds) + 1}",
                            "region_id": region["region_id"],
                            "anchor": anchor,
                            "concept": concept,
                            "query": f"{anchor} {concept}",
                        }
                    )
        gazetteer_seeds = self.entity_extractor.extract_gazetteer(observation_text)
        graph_contributions = self._graph_contributions(
            gazetteer_seeds, regions
        )
        graph_region_ids = sorted(
            {
                contribution["region_id"]
                for contributions in graph_contributions.values()
                for contribution in contributions
            }
        )
        graph_queries = [
            next(
                region["text"]
                for region in regions
                if region["region_id"] == region_id
            )
            for region_id in graph_region_ids
        ]
        queries = [seed["query"] for seed in semantic_seeds] + graph_queries
        vectors = self.embedder.embed(queries)
        semantic_vectors = vectors[: len(semantic_seeds)]
        graph_vectors = vectors[len(semantic_seeds) :]
        semantic_contributions = self._semantic_contributions(
            semantic_seeds, semantic_vectors
        )
        graph_cosines = {
            region_id: self._page_cosines(vector)
            for region_id, vector in zip(graph_region_ids, graph_vectors)
        }
        admissions = self._select_graph_admissions(
            graph_contributions,
            graph_cosines,
            {
                region["region_id"]: region["text"]
                for region in regions
            },
        )
        admissions.extend(
            self._select_semantic_admissions(semantic_contributions)
        )
        direct_note_admissions, direct_note_trace = (
            self._select_direct_note_admissions(regions)
        )
        admissions.extend(direct_note_admissions)
        best_admission_by_page: dict[str, dict[str, Any]] = {}
        for admission in admissions:
            previous = best_admission_by_page.get(admission["page_id"])
            if previous is None or (
                admission["priority"],
                admission["score"],
                admission["page_id"],
            ) > (
                previous["priority"],
                previous["score"],
                previous["page_id"],
            ):
                best_admission_by_page[admission["page_id"]] = admission
        ordered = sorted(
            best_admission_by_page.values(),
            key=lambda item: (-item["priority"], -item["score"], item["page_id"]),
        )[:top_k]
        hits = tuple(
            self._hit(admission, rank)
            for rank, admission in enumerate(ordered, start=1)
        )
        trace = {
            "corpus_id": resolved_corpus_id,
            "retrieval_condition_detail": self.retrieval_condition_detail,
            "evidence_projection_version": self.evidence_projection_version,
            "admission_policy": self.admission_policy,
            "task_anchors": list(anchors),
            "regions": [
                {
                    "region_id": region["region_id"],
                    "span": [region["start"], region["end"]],
                    "text": region["text"],
                    "description": region["description"],
                    "rare_concepts": list(concepts),
                }
                for region, concepts in zip(regions, concepts_by_region)
            ],
            "entity_seeds": [seed.__dict__ for seed in gazetteer_seeds],
            "semantic_seeds": semantic_seeds,
            "graph_candidates": _trace_candidate_contributions(
                graph_contributions
            ),
            "semantic_candidates": _trace_candidate_contributions(
                semantic_contributions
            ),
            "direct_note_candidates": direct_note_trace,
            "thresholds": {
                "semantic_depth": PARM_SEMANTIC_DEPTH,
                "semantic_min_cosine": PARM_SEMANTIC_MIN_COSINE,
                "singleton_min_cosine": PARM_SINGLETON_MIN_COSINE,
                "minimum_converging_concepts": PARM_MIN_CONVERGING_CONCEPTS,
                "minimum_lexical_concepts": PARM_MIN_LEXICAL_CONCEPTS,
                "direct_note_min_page_contrast": (
                    PARM_DIRECT_NOTE_MIN_PAGE_CONTRAST
                ),
                "direct_note_min_region_contrast": (
                    PARM_DIRECT_NOTE_MIN_REGION_CONTRAST
                ),
            },
            "returned_pages": [
                {
                    "corpus_id": hit.corpus_id,
                    "page_id": hit.page_id,
                    "source_id": hit.source_id,
                    "slug": hit.slug,
                    "selected_chunk_id": hit.chunk_id,
                    "perturbations": list(hit.perturbations),
                    "final_rank": hit.rank,
                    "score": hit.score,
                    **hit.diagnostics,
                }
                for hit in hits
            ],
        }
        return RetrievalResult(hits, trace)

    def _graph_contributions(
        self,
        seeds: Sequence[EntitySeed],
        regions: Sequence[dict[str, Any]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        contributions: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for seed in seeds:
            if seed.span_start is None:
                continue
            region = next(
                (
                    item
                    for item in regions
                    if item["start"] <= seed.span_start < item["end"]
                ),
                None,
            )
            if region is None:
                continue
            for entity_page_id in seed.matched_page_ids:
                for page_id in sorted(self._inbound.get(entity_page_id, ())):
                    contributions[(region["region_id"], page_id)].append(
                        {
                            "seed_id": seed.seed_id,
                            "surface": seed.surface,
                            "entity_page_id": entity_page_id,
                            "region_id": region["region_id"],
                        }
                    )
        return dict(contributions)

    def _semantic_contributions(
        self,
        seeds: Sequence[dict[str, Any]],
        vectors: np.ndarray,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        contributions: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for seed, vector in zip(seeds, vectors):
            page_scores, best_sentences = self._sentence_page_cosines(vector)
            ordered = _rank_scores(page_scores, PARM_SEMANTIC_DEPTH)
            for rank, page_id in enumerate(ordered, start=1):
                score = page_scores[page_id]
                if score < PARM_SEMANTIC_MIN_COSINE:
                    continue
                sentence = self.index.sentences[best_sentences[page_id]]
                contributions[(seed["region_id"], page_id)].append(
                    {
                        **seed,
                        "rank": rank,
                        "cosine": score,
                        "sentence_id": sentence.sentence_id,
                        "sentence_text": sentence.text,
                    }
                )
        return dict(contributions)

    def _select_graph_admissions(
        self,
        contributions: dict[tuple[str, str], list[dict[str, Any]]],
        cosines: dict[str, dict[str, float]],
        region_texts: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        region_texts = region_texts or {}
        lexical_evidence = {
            region_id: self._graph_lexical_evidence(
                region_texts.get(region_id, ""),
                [
                    page_id
                    for candidate_region_id, page_id in contributions
                    if candidate_region_id == region_id
                ],
            )
            for region_id in {
                candidate_region_id
                for candidate_region_id, _ in contributions
            }
        }
        for (region_id, page_id), values in contributions.items():
            distinct_seeds = {value["seed_id"] for value in values}
            cosine = cosines.get(region_id, {}).get(page_id, -1.0)
            lexical_score, lexical_terms = lexical_evidence[region_id].get(
                page_id, (0.0, ())
            )
            by_region[region_id].append(
                {
                    "page_id": page_id,
                    "region_id": region_id,
                    "channel": "entity_graph",
                    "priority": 3,
                    "score": len(distinct_seeds) + max(cosine, 0.0),
                    "entity_seed_count": len(distinct_seeds),
                    "region_cosine": cosine,
                    "graph_lexical_score": lexical_score,
                    "graph_lexical_terms": list(lexical_terms),
                }
            )
        admissions = []
        for candidates in by_region.values():
            candidates = [
                candidate
                for candidate in candidates
                if candidate["region_cosine"] >= 0.30
            ]
            candidates.sort(
                key=lambda item: (
                    -item["entity_seed_count"],
                    -item["graph_lexical_score"],
                    -len(item["graph_lexical_terms"]),
                    -item["region_cosine"],
                    item["page_id"],
                )
            )
            if candidates:
                admissions.append(candidates[0])
        return admissions

    def _graph_lexical_evidence(
        self,
        region_text: str,
        candidate_page_ids: Sequence[str],
    ) -> dict[str, tuple[float, tuple[str, ...]]]:
        query_terms = []
        for token in _tokenize(region_text):
            singular = token[:-1] if token.endswith("s") else token
            if (
                len(token) < 3
                or token.isdigit()
                or token in _PARM_GRAPH_LEXICAL_STOPWORDS
                or token in _PARM_GENERIC_CONCEPTS
                or singular in _PARM_GENERIC_CONCEPTS
                or token in query_terms
            ):
                continue
            query_terms.append(token)
        evidence = {}
        for page_id in candidate_page_ids:
            page_tokens = self._graph_page_tokens.get(page_id, frozenset())
            matches = [
                (
                    math.log(
                        1
                        + (
                            self._graph_document_count
                            - self._graph_term_document_frequency[token]
                            + 0.5
                        )
                        / (
                            self._graph_term_document_frequency[token]
                            + 0.5
                        )
                    ),
                    token,
                )
                for token in query_terms
                if token in page_tokens
            ]
            matches.sort(key=lambda item: (-item[0], item[1]))
            evidence[page_id] = (
                matches[0][0] if matches else 0.0,
                tuple(token for _, token in matches[:8]),
            )
        return evidence

    def _select_direct_note_admissions(
        self,
        regions: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Admit one durable note when two independent BM25 contrasts agree.

        The page contrast requires one note to explain a region substantially
        better than the next note. The region contrast requires that region to
        stand out from every other listing in the observation. This lets
        durable notes outside the review/reflection naming convention
        participate without opening a broad semantic admission path.
        """
        candidates: list[dict[str, Any]] = []
        if len(self._direct_note_text) < 2:
            return [], candidates
        for region in regions:
            bm25 = getattr(self, "_direct_note_bm25", None)
            scores = (
                bm25.scores(region["text"])
                if bm25 is not None
                else _bm25_scores(region["text"], self._direct_note_text)
            )
            ordered = _rank_scores(scores, 2)
            if len(ordered) < 2:
                continue
            page_id, runner_page_id = ordered
            page_score = scores[page_id]
            runner_page_score = scores[runner_page_id]
            if runner_page_score <= 0:
                continue
            candidates.append(
                {
                    "page_id": page_id,
                    "region_id": region["region_id"],
                    "page_score": page_score,
                    "runner_page_id": runner_page_id,
                    "runner_page_score": runner_page_score,
                    "page_contrast": page_score / runner_page_score,
                    "region_text": region["text"],
                }
            )
        candidates.sort(
            key=lambda item: (
                -item["page_score"],
                item["region_id"],
                item["page_id"],
            )
        )
        trace = [
            {
                key: value
                for key, value in candidate.items()
                if key != "region_text"
            }
            for candidate in candidates[:2]
        ]
        if len(candidates) < 2:
            return [], trace
        best, runner = candidates[:2]
        region_contrast = best["page_score"] / runner["page_score"]
        trace[0]["region_contrast"] = region_contrast
        trace[0]["runner_region_id"] = runner["region_id"]
        if (
            best["page_contrast"] < PARM_DIRECT_NOTE_MIN_PAGE_CONTRAST
            or region_contrast < PARM_DIRECT_NOTE_MIN_REGION_CONTRAST
        ):
            return [], trace
        sentences = {
            sentence.sentence_id: sentence.text
            for sentence in self._sentences_by_page.get(best["page_id"], ())
        }
        evidence_sentence_ids = _bm25_rank(
            best["region_text"], sentences, 3
        )
        evidence_sentences = [
            sentences[sentence_id] for sentence_id in evidence_sentence_ids
        ]
        return [
            {
                "page_id": best["page_id"],
                "region_id": best["region_id"],
                "channel": "direct_note_contrast",
                "priority": 4,
                "score": best["page_score"],
                "page_contrast": best["page_contrast"],
                "region_contrast": region_contrast,
                "runner_page_id": best["runner_page_id"],
                "runner_region_id": runner["region_id"],
                "evidence_sentences": evidence_sentences,
            }
        ], trace

    def _select_semantic_admissions(
        self,
        contributions: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        admissions = []
        for (region_id, page_id), values in contributions.items():
            page = self._page_by_id[page_id]
            page_kind = page.slug.rsplit("/", 1)[-1].casefold()
            if "review" not in page_kind and "reflection" not in page_kind:
                continue
            best_by_concept: dict[str, dict[str, Any]] = {}
            for value in values:
                previous = best_by_concept.get(value["concept"])
                if previous is None or (
                    -value["rank"],
                    value["cosine"],
                    value["query"],
                ) > (
                    -previous["rank"],
                    previous["cosine"],
                    previous["query"],
                ):
                    best_by_concept[value["concept"]] = value
            concepts = set(best_by_concept)
            page_text = self._page_text[page_id]
            lexical_concepts = {
                concept for concept in concepts if _word_present(page_text, concept)
            }
            multi_concept = (
                len(concepts) >= PARM_MIN_CONVERGING_CONCEPTS
                and len(lexical_concepts) >= PARM_MIN_LEXICAL_CONCEPTS
            )
            singleton_values = [
                value
                for value in best_by_concept.values()
                if (
                    value["rank"] == 1
                    and value["cosine"] >= PARM_SINGLETON_MIN_COSINE
                    and _word_present(page_text, value["anchor"])
                    and not _word_present(page_text, value["concept"])
                )
            ]
            if not multi_concept and not singleton_values:
                continue
            evidence_terms = (
                lexical_concepts
                if multi_concept
                else {value["anchor"] for value in singleton_values}
            )
            evidence_sentences = [
                sentence.text
                for sentence in self._sentences_by_page.get(page_id, [])
                if any(
                    _word_present(sentence.text, term)
                    for term in evidence_terms
                )
            ][:3]
            if not evidence_sentences:
                evidence_sentences = list(
                    dict.fromkeys(
                        value["sentence_text"]
                        for value in sorted(
                            best_by_concept.values(),
                            key=lambda item: (-item["cosine"], item["sentence_id"]),
                        )
                    )
                )[:3]
            score = sum(value["cosine"] for value in best_by_concept.values())
            admissions.append(
                {
                    "page_id": page_id,
                    "region_id": region_id,
                    "channel": (
                        "semantic_multi_concept"
                        if multi_concept
                        else "semantic_anchored_singleton"
                    ),
                    "priority": 2 if multi_concept else 1,
                    "score": score,
                    "concept_count": len(concepts),
                    "concepts": sorted(concepts),
                    "lexical_concepts": sorted(lexical_concepts),
                    "evidence_sentences": evidence_sentences,
                    "singleton_queries": [
                        value["query"] for value in singleton_values
                    ],
                }
            )
        return admissions

    def _sentence_page_cosines(
        self, query_vector: np.ndarray
    ) -> tuple[dict[str, float], dict[str, int]]:
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0:
            scores = np.zeros(len(self.index.sentences), dtype=np.float32)
        else:
            scores = self._sentence_matrix @ (query_vector / query_norm)
        page_scores: dict[str, float] = {}
        best_sentences: dict[str, int] = {}
        for position, score_value in enumerate(scores):
            page_id = self._sentence_page_ids[position]
            if page_id not in self._eligible_pages:
                continue
            score = float(score_value)
            if page_id not in page_scores or score > page_scores[page_id]:
                page_scores[page_id] = score
                best_sentences[page_id] = position
        return page_scores, best_sentences

    def _page_cosines(self, query_vector: np.ndarray) -> dict[str, float]:
        query_norm = float(np.linalg.norm(query_vector))
        matrix_norms = np.linalg.norm(self.index.embeddings, axis=1)
        denominators = matrix_norms * query_norm
        chunk_scores = np.divide(
            self.index.embeddings @ query_vector,
            denominators,
            out=np.zeros(len(self.index.chunks), dtype=np.float32),
            where=denominators != 0,
        )
        page_scores: dict[str, float] = {}
        for position, chunk in enumerate(self.index.chunks):
            if chunk.page_id not in self._eligible_pages:
                continue
            page_scores[chunk.page_id] = max(
                page_scores.get(chunk.page_id, -1.0),
                float(chunk_scores[position]),
            )
        return page_scores

    def _hit(self, admission: dict[str, Any], rank: int) -> RetrievalHit:
        page_id = admission["page_id"]
        page = self._page_by_id[page_id]
        _, chunk = self._chunks_by_page[page_id][0]
        text = chunk.text
        if (
            admission["channel"].startswith("semantic_")
            or admission["channel"] == "direct_note_contrast"
        ):
            text = "\n".join(admission.get("evidence_sentences", ())) or text
        diagnostics = {
            key: value
            for key, value in admission.items()
            if key not in {"page_id", "priority"}
        }
        return RetrievalHit(
            corpus_id=page.corpus_id,
            page_id=page_id,
            source_id=page.source_id,
            slug=page.slug,
            title=page.title,
            chunk_id=chunk.chunk_id,
            text=text,
            score=float(admission["score"]),
            rank=rank,
            perturbations=page.perturbations,
            diagnostics=diagnostics,
        )


def _parm_listing_regions(observation_text: str) -> list[dict[str, Any]]:
    regions = []
    offset = 0
    for line in observation_text.splitlines(keepends=True):
        text = line.strip()
        start = offset + len(line) - len(line.lstrip())
        end = start + len(text)
        offset += len(line)
        if not text or not _PARM_LISTING_PREFIX.match(text):
            continue
        parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        description = parts[1] if len(parts) == 2 else ""
        regions.append(
            {
                "region_id": f"region-{len(regions) + 1}",
                "region_kind": "listing",
                "start": start,
                "end": end,
                "text": text,
                "description": description,
            }
        )
    return regions


def _parm_observation_regions(observation_text: str) -> list[dict[str, Any]]:
    """Split either a structured feed or general markdown into cue regions.

    Existing PARMBench feeds keep their one-listing-per-region behavior. When
    the observation is prose or mixed markdown, paragraph-like blocks become
    regions instead. Short headings and export fragments are skipped because
    they do not carry enough semantic evidence to retrieve a personal memory.
    """
    listing_regions = _parm_listing_regions(observation_text)
    if len(listing_regions) >= 10:
        return listing_regions

    regions: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?:\A|(?:\r?\n){2,})(.*?)(?=(?:\r?\n){2,}|\Z)",
        observation_text,
        flags=re.DOTALL,
    ):
        raw = match.group(1)
        text = raw.strip()
        if not text:
            continue
        start = match.start(1) + len(raw) - len(raw.lstrip())
        end = start + len(text)
        semantic_text = _strip_markdown_scaffolding(text)
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", semantic_text)
        if len(words) < 8:
            continue
        regions.append(
            {
                "region_id": f"region-{len(regions) + 1}",
                "region_kind": "semantic_block",
                "start": start,
                "end": end,
                "text": text,
                "description": semantic_text,
            }
        )
    return regions


def _strip_markdown_scaffolding(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = re.sub(
            r"^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+\[[ xX]\]\s+|[-*+]\s+)",
            "",
            line,
        )
        normalized = re.sub(r"<!--.*?-->", " ", normalized)
        normalized = normalized.replace("`", " ").replace("**", "")
        lines.append(normalized.strip())
    return " ".join(line for line in lines if line).strip()


def _word_present(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE) is not None


def _trace_candidate_contributions(
    contributions: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "region_id": region_id,
            "page_id": page_id,
            "contributions": values,
        }
        for (region_id, page_id), values in sorted(contributions.items())
    ]


class EntityExactRetriever:
    retrieval_condition_detail = "entity_exact_match"

    def __init__(
        self,
        index: RetrievalIndex,
        *,
        extractor: EntitySurfaceExtractor | None = None,
    ):
        self.index = index
        self.extractor = extractor
        if self.extractor is None and len(index.corpus_ids) == 1:
            self.extractor = EntitySurfaceExtractor(index)
        self._scoped_retrievers: OrderedDict[
            str, EntityExactRetriever
        ] = OrderedDict()
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
        corpus_id: str | None = None,
    ) -> EntityRetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        resolved_corpus_id = self.index.resolve_corpus_id(corpus_id)
        if len(self.index.corpus_ids) > 1:
            scoped = self._scoped_retrievers.pop(resolved_corpus_id, None)
            if scoped is None:
                scoped_index = self.index.scoped(resolved_corpus_id)
                scoped_extractor = (
                    self.extractor.for_index(scoped_index)
                    if self.extractor is not None
                    else None
                )
                scoped = EntityExactRetriever(
                    scoped_index,
                    extractor=scoped_extractor,
                )
            self._scoped_retrievers[resolved_corpus_id] = scoped
            if len(self._scoped_retrievers) > SCOPED_RETRIEVER_CACHE_SIZE:
                self._scoped_retrievers.popitem(last=False)
            return scoped.retrieve_entities(
                observation_text,
                top_k=top_k,
                corpus_id=resolved_corpus_id,
            )
        assert self.extractor is not None
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
                    corpus_id=page.corpus_id,
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
                        "corpus_id": page.corpus_id,
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
            "corpus_id": resolved_corpus_id,
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


class _BM25Corpus:
    """Pre-tokenized BM25 corpus for repeated queries over fixed documents."""

    def __init__(self, documents: dict[str, str]):
        self._tokens = {
            document_id: _tokenize(text)
            for document_id, text in documents.items()
        }
        self._frequencies = {
            document_id: Counter(tokens)
            for document_id, tokens in self._tokens.items()
        }
        self._count = len(self._tokens)
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens.values()) / self._count
            if self._count
            else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._document_frequency.update(set(tokens))

    def scores(self, query: str) -> dict[str, float]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}
        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        for document_id, tokens in self._tokens.items():
            frequencies = self._frequencies[document_id]
            score = 0.0
            for term in query_tokens:
                frequency = frequencies[term]
                if frequency == 0:
                    continue
                df = self._document_frequency[term]
                inverse_frequency = math.log(
                    1 + (self._count - df + 0.5) / (df + 0.5)
                )
                length_ratio = (
                    len(tokens) / self._average_length
                    if self._average_length
                    else 0.0
                )
                score += inverse_frequency * (
                    frequency * (k1 + 1)
                    / (frequency + k1 * (1 - b + b * length_ratio))
                )
            if score > 0:
                scores[document_id] = score
        return scores


def _bm25_scores(query: str, documents: dict[str, str]) -> dict[str, float]:
    return _BM25Corpus(documents).scores(query)


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
