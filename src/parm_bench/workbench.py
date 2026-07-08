from __future__ import annotations

import json
import time
import webbrowser
from dataclasses import asdict, replace
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Protocol

from .baselines import (
    INPUT_RAG_INSTRUCTIONS,
    BenchmarkInput,
    NaiveOutputRagBaseline,
    OutputRagFlow,
)
from .choice_matching import matches_choice
from .dataset import load_cases, validate_cases
from .models import (
    FINAL_ANSWER_INSTRUCTIONS,
    LanguageModel,
    OpenAIResponsesModel,
)
from .retrieval import (
    CANDIDATE_DEPTH,
    CachedOpenAIQueryExpander,
    ExpansionCacheMissError,
    ExpansionPolicy,
    IndexRetriever,
    OpenAIEmbedder,
    RetrievalIndex,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
    RetrievalValidationError,
    TextEmbedder,
)


WORKBENCH_INSTRUCTIONS = (
    "Answer the user's prompt directly and helpfully. Use retrieved personal "
    "memory when it is supplied and relevant. Do not mention the retrieval "
    "process unless the user asks about it."
)
MAX_REQUEST_BYTES = 1_000_000
MAX_PROMPT_CHARACTERS = 100_000


class RetrievalCondition(str, Enum):
    NO_MEMORY = "no_memory"
    INPUT_RAG = "input_rag"
    NAIVE_OUTPUT_RAG = "naive_output_rag"


class WorkbenchRequestError(ValueError):
    pass


class WorkbenchRunError(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(
            f"{stage} failed: {cause.__class__.__name__}: {cause}"
        )


class RetrieverFactory(Protocol):
    def __call__(self, mode: RetrievalMode) -> IndexRetriever: ...


class WorkbenchService:
    def __init__(
        self,
        index: RetrievalIndex,
        embedder: TextEmbedder,
        *,
        default_model: str,
        expansion_cache: str | Path | None = None,
        expansion_policy: ExpansionPolicy | str = ExpansionPolicy.FROZEN,
        model_factory: Callable[[str], LanguageModel] = OpenAIResponsesModel,
        retriever_factory: RetrieverFactory | None = None,
        cases: list[dict[str, Any]] | None = None,
    ):
        self.index = index
        self.embedder = embedder
        self.default_model = default_model
        self.expansion_cache = (
            Path(expansion_cache) if expansion_cache is not None else None
        )
        self.expansion_policy = ExpansionPolicy(expansion_policy)
        self.model_factory = model_factory
        self._retriever_factory = retriever_factory
        self._cases = tuple(cases or ())
        self._case_by_id = {
            str(case["case_id"]): case for case in self._cases
        }
        if len(self._case_by_id) != len(self._cases):
            raise ValueError("workbench cases contain duplicate case IDs")

    def configuration(self) -> dict[str, Any]:
        return {
            "conditions": [condition.value for condition in RetrievalCondition],
            "output_rag_flows": [flow.value for flow in OutputRagFlow],
            "retrieval_modes": [mode.value for mode in RetrievalMode],
            "enhanced_available": self.expansion_cache is not None,
            "expansion_policy": self.expansion_policy.value,
            "default_model": self.default_model,
            "default_top_k": 5,
            "max_top_k": CANDIDATE_DEPTH,
            "cases": [
                {
                    "case_id": str(case["case_id"]),
                    "base_case_id": str(case["base_case_id"]),
                    "label": _case_label(case),
                    "variant": str(case["variant"]),
                    "prompt": str(case["prompt"]),
                    "observation_kind": str(case["observation"]["kind"]),
                }
                for case in self._cases
            ],
            "index": {
                "path": str(self.index.path),
                "manifest_hash": self.index.manifest_hash,
                "corpus_id": self.index.manifest["corpus_id"],
                "pages": self.index.manifest["counts"]["pages"],
                "chunks": self.index.manifest["counts"]["chunks"],
                "embedding_model": self.index.manifest["embedding_model"],
            },
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        case_id = payload.get("case_id")
        selected_case: dict[str, Any] | None = None
        if case_id not in (None, ""):
            if not isinstance(case_id, str) or case_id not in self._case_by_id:
                raise WorkbenchRequestError("Unknown benchmark case.")
            selected_case = self._case_by_id[case_id]
        prompt = (
            selected_case["prompt"]
            if selected_case is not None
            else payload.get("prompt")
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkbenchRequestError("Enter a prompt before running.")
        prompt = prompt.strip()
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise WorkbenchRequestError(
                f"Prompt exceeds {MAX_PROMPT_CHARACTERS:,} characters."
            )
        try:
            condition = RetrievalCondition(
                payload.get("condition", RetrievalCondition.INPUT_RAG.value)
            )
        except ValueError as exc:
            raise WorkbenchRequestError("Unknown retrieval condition.") from exc
        top_k = payload.get("top_k", 5)
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= CANDIDATE_DEPTH
        ):
            raise WorkbenchRequestError(
                f"Top-k must be between 1 and {CANDIDATE_DEPTH}."
            )
        model_name = payload.get("model") or self.default_model
        if not isinstance(model_name, str) or not model_name.strip():
            raise WorkbenchRequestError("Model name must be non-empty.")

        started = time.perf_counter()
        retrieval: RetrievalResult | None = None
        captured_retrievals: list[RetrievalResult] = []
        retrieval_mode: RetrievalMode | None = None
        output_rag_flow: OutputRagFlow | None = None
        memory_condition = condition is not RetrievalCondition.NO_MEMORY
        if memory_condition:
            try:
                retrieval_mode = RetrievalMode(
                    payload.get("retrieval_mode", RetrievalMode.DENSE.value)
                )
            except ValueError as exc:
                raise WorkbenchRequestError("Unknown retrieval mode.") from exc
            if (
                retrieval_mode is RetrievalMode.ENHANCED
                and self.expansion_cache is None
            ):
                raise WorkbenchRequestError(
                    "Enhanced retrieval is unavailable. Restart the server "
                    "with --expansion-cache."
                )
        model = self.model_factory(model_name.strip())
        observation_kind = (
            str(selected_case["observation"]["kind"])
            if selected_case is not None
            else ""
        )
        observation_text = (
            str(selected_case["observation_text"])
            if selected_case is not None
            else ""
        )
        row: dict[str, Any] | None = None
        if condition is RetrievalCondition.INPUT_RAG:
            assert retrieval_mode is not None
            try:
                retrieval = self._retriever(retrieval_mode).retrieve(
                    RetrievalRequest(prompt, top_k=top_k)
                )
            except (ExpansionCacheMissError, RetrievalValidationError):
                raise
            except Exception as exc:
                raise WorkbenchRunError("retrieval", exc) from exc
            memory_context = _memory_context(retrieval)
            try:
                response = model.generate(
                    prompt=prompt,
                    observation_kind=observation_kind,
                    observation_text=observation_text,
                    instructions=(
                        INPUT_RAG_INSTRUCTIONS
                        if selected_case is not None
                        else WORKBENCH_INSTRUCTIONS
                    ),
                    memory_context=memory_context,
                )
            except Exception as exc:
                raise WorkbenchRunError("model generation", exc) from exc
        elif condition is RetrievalCondition.NAIVE_OUTPUT_RAG:
            try:
                output_rag_flow = OutputRagFlow(
                    payload.get(
                        "output_rag_flow",
                        OutputRagFlow.TOOL_THEN_MODEL_OUTPUT.value,
                    )
                )
            except ValueError as exc:
                raise WorkbenchRequestError("Unknown output-RAG flow.") from exc
            if (
                selected_case is None
                and output_rag_flow is not OutputRagFlow.MODEL_OUTPUT_ONLY
            ):
                raise WorkbenchRequestError(
                    "Choose a benchmark case for output-RAG flows that retrieve "
                    "from observed tool/output text."
                )
            assert retrieval_mode is not None
            retriever = _CapturingRetriever(self._retriever(retrieval_mode))
            baseline = NaiveOutputRagBaseline(
                retrieval_limit=top_k,
                output_rag_flow=output_rag_flow,
            )
            try:
                row = baseline.run(
                    BenchmarkInput(
                        case_id=(
                            str(selected_case["case_id"])
                            if selected_case is not None
                            else "custom"
                        ),
                        prompt=prompt,
                        observation_kind=observation_kind,
                        observation_text=observation_text,
                    ),
                    model,
                    retriever,
                )
            except (ExpansionCacheMissError, RetrievalValidationError):
                raise
            except Exception as exc:
                stage = (
                    "retrieval"
                    if retriever.failed_during_retrieval
                    else "model generation"
                )
                raise WorkbenchRunError(stage, exc) from exc
            captured_retrievals = retriever.results
            retrieval = _merge_retrievals(captured_retrievals, row["trace"])
            response = _RowResponse(row)
        else:
            try:
                response = model.generate(
                    prompt=prompt,
                    observation_kind=observation_kind,
                    observation_text=observation_text,
                    instructions=(
                        FINAL_ANSWER_INSTRUCTIONS
                        if selected_case is not None
                        else WORKBENCH_INSTRUCTIONS
                    ),
                    memory_context=None,
                )
            except Exception as exc:
                raise WorkbenchRunError("model generation", exc) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "prompt": prompt,
            "case": (
                {
                    "case_id": str(selected_case["case_id"]),
                    "base_case_id": str(selected_case["base_case_id"]),
                    "label": _case_label(selected_case),
                    "variant": str(selected_case["variant"]),
                    "observation_kind": observation_kind,
                }
                if selected_case is not None
                else None
            ),
            "condition": condition.value,
            "output_rag_flow": (
                output_rag_flow.value if output_rag_flow is not None else None
            ),
            "retrieval_mode": (
                retrieval_mode.value if retrieval_mode is not None else None
            ),
            "top_k": top_k if memory_condition else None,
            "response_text": response.text,
            "elapsed_ms": elapsed_ms,
            "model": {
                "text": response.text,
                "requested_model": model.model_name,
                "resolved_model": response.resolved_model,
                "response_id": response.response_id,
                "usage": response.usage,
            },
            "evaluation": (
                _evaluate_case(
                    selected_case,
                    condition,
                    response.text,
                    retrieval,
                )
                if selected_case is not None
                else None
            ),
            "retrieval": (
                {
                    "hits": [_serialize_hit(hit) for hit in retrieval.hits],
                    "trace": retrieval.trace,
                }
                if retrieval is not None
                else {"hits": [], "trace": None}
            ),
        }

    def _retriever(self, mode: RetrievalMode) -> IndexRetriever:
        if self._retriever_factory is not None:
            return self._retriever_factory(mode)
        expander = None
        if mode is RetrievalMode.ENHANCED:
            assert self.expansion_cache is not None
            expander = CachedOpenAIQueryExpander(
                self.expansion_cache, self.expansion_policy
            )
        return IndexRetriever(
            self.index,
            mode,
            self.embedder,
            expander=expander,
        )


def serve_workbench(
    *,
    retrieval_index: str | Path,
    dataset_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    model_name: str = "gpt-5-mini",
    expansion_cache: str | Path | None = None,
    expansion_policy: ExpansionPolicy | str = ExpansionPolicy.FROZEN,
    open_browser: bool = True,
) -> None:
    index = RetrievalIndex.load(retrieval_index)
    cases: list[dict[str, Any]] = []
    if dataset_dir is not None:
        cases = load_cases(dataset_dir)
        validate_cases(cases)
    service = WorkbenchService(
        index,
        OpenAIEmbedder(),
        default_model=model_name,
        expansion_cache=expansion_cache,
        expansion_policy=expansion_policy,
        cases=cases,
    )
    server = create_workbench_server(service, host=host, port=port)
    url = f"http://{host}:{server.server_port}/"
    print(f"PARM retrieval workbench: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workbench.")
    finally:
        server.server_close()


def create_workbench_server(
    service: WorkbenchService,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> "WorkbenchHTTPServer":
    class Handler(WorkbenchRequestHandler):
        pass

    Handler.service = service
    return WorkbenchHTTPServer((host, port), Handler)


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class _CapturingRetriever:
    def __init__(self, retriever: IndexRetriever):
        self._retriever = retriever
        self.mode = retriever.mode
        self.results: list[RetrievalResult] = []
        self.failed_during_retrieval = False

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        try:
            result = self._retriever.retrieve(request)
        except Exception:
            self.failed_during_retrieval = True
            raise
        self.results.append(result)
        return result


class _RowResponse:
    def __init__(self, row: dict[str, Any]):
        self.text = str(row["response_text"])
        self.response_id = row["provider_response_id"]
        self.resolved_model = row["resolved_model"]
        self.usage = row["usage"]


def _merge_retrievals(
    retrievals: list[RetrievalResult],
    trace: dict[str, Any],
) -> RetrievalResult:
    hits = []
    admitted = set(trace.get("admitted_source_ids", []))
    seen = set()
    for retrieval in retrievals:
        for hit in retrieval.hits:
            if hit.slug in seen or (admitted and hit.slug not in admitted):
                continue
            seen.add(hit.slug)
            hits.append(replace(hit, rank=len(hits) + 1))
    return RetrievalResult(tuple(hits), trace)


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    service: WorkbenchService
    server_version = "PARMWorkbench/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_bytes(
                HTTPStatus.OK,
                files("parm_bench")
                .joinpath("web/workbench.html")
                .read_bytes(),
                "text/html; charset=utf-8",
            )
            return
        if self.path == "/api/config":
            self._send_json(HTTPStatus.OK, self.service.configuration())
            return
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "Invalid content length."}
            )
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Request body is empty or too large."},
            )
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise WorkbenchRequestError("Request body must be an object.")
            result = self.service.run(payload)
        except (
            ExpansionCacheMissError,
            RetrievalValidationError,
            WorkbenchRequestError,
        ) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "Request body is not valid JSON."}
            )
            return
        except WorkbenchRunError as exc:
            message = str(exc)
            print(f"Workbench run failed: {message}", flush=True)
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": message,
                    "stage": exc.stage,
                    "cause": exc.cause.__class__.__name__,
                },
            )
            return
        except Exception as exc:
            print(
                f"Workbench run failed: {exc.__class__.__name__}: {exc}",
                flush=True,
            )
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": (
                        "The run failed. Check the server terminal for the "
                        "underlying model or retrieval error."
                    )
                },
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)


def _memory_context(retrieval: RetrievalResult | None) -> str | None:
    if retrieval is None or not retrieval.hits:
        return None
    return "\n\n".join(
        f"{hit.rank}. {hit.text}" for hit in retrieval.hits
    )


def _serialize_hit(hit: Any) -> dict[str, Any]:
    payload = asdict(hit)
    payload["perturbations"] = list(hit.perturbations)
    return payload


def _case_label(case: dict[str, Any]) -> str:
    base = str(case["base_case_id"])
    for prefix in ("parm-amara-", "parm-"):
        if base.startswith(prefix):
            base = base.removeprefix(prefix)
            break
    subject = base.replace("-", " ").title()
    subject = subject.replace("Ai ", "AI ")
    variant = case.get("variant")
    if variant == "positive":
        label = "Cue present"
    elif variant == "memory-included":
        label = "Memory included"
    else:
        label = "Cue ablated"
    return f"{subject} — {label}"


def _evaluate_case(
    case: dict[str, Any],
    condition: RetrievalCondition,
    response_text: str,
    retrieval: RetrievalResult | None,
) -> dict[str, Any]:
    decisions = case["decisions"]
    # memory-included injects the gold memory into the prompt, so its expected
    # answer is the memory-conditioned choice under every condition, including
    # no_memory. Other variants keep the condition-based basis.
    expected_basis = (
        "output-only"
        if condition is RetrievalCondition.NO_MEMORY
        and str(case["variant"]) != "memory-included"
        else "memory-conditioned"
    )
    expected_choice = str(decisions[expected_basis.replace("-", "_")]["choice"])
    choices = {
        str(decisions["output_only"]["choice"]),
        str(decisions["memory_conditioned"]["choice"]),
    }
    matched_choices = sorted(
        choice for choice in choices if matches_choice(response_text, choice)
    )
    successful = matched_choices == [expected_choice]

    retrieved = {
        _canonical_source_id(hit.slug)
        for hit in (retrieval.hits if retrieval is not None else ())
    }
    gold = {
        _canonical_source_id(str(source_id))
        for source_id in case["memory"]["gold_source_ids"]
    }
    retrieved_gold = sorted(retrieved & gold)
    variant = str(case["variant"])
    if variant == "memory-included" and condition is RetrievalCondition.NO_MEMORY:
        retrieval_status = "ceiling"
    elif condition is RetrievalCondition.NO_MEMORY:
        retrieval_status = "not-run"
    elif variant == "cue-ablated":
        retrieval_status = "control"
    elif not retrieved_gold:
        retrieval_status = "miss"
    elif len(retrieved_gold) < len(gold):
        retrieval_status = "partial"
    else:
        retrieval_status = "complete"
    if variant == "memory-included":
        reason = (
            "The gold memory was injected into the prompt and the response "
            "selected the expected memory-conditioned choice (ceiling reached)."
            if successful
            else (
                "The gold memory was injected into the prompt, but the "
                "response did not select the expected memory-conditioned "
                "choice (ceiling missed)."
            )
        )
    elif variant == "cue-ablated":
        if successful:
            reason = (
                "The cue-ablated control kept the expected reference choice, "
                "so retrieval did not cause a false intervention."
            )
        else:
            reason = (
                "The cue-ablated control should keep the reference choice, "
                "but the response selected something else or was ambiguous."
            )
    elif condition is RetrievalCondition.NO_MEMORY:
        reason = (
            "The no-memory run selected the expected output-only choice."
            if successful
            else (
                "The no-memory run did not select the expected output-only "
                "choice."
            )
        )
    elif successful and retrieved_gold:
        reason = (
            "The response selected the expected memory-conditioned choice, "
            "and retrieval included the required gold memory."
        )
    elif successful:
        reason = (
            "The response selected the expected memory-conditioned choice, "
            "but retrieval did not include a required gold memory."
        )
    elif retrieved_gold:
        reason = (
            "Retrieval included a required gold memory, but the response did "
            "not select the expected memory-conditioned choice."
        )
    else:
        reason = (
            "The response missed the expected memory-conditioned choice, and "
            "retrieval did not include a required gold memory."
        )
    return {
        "successful": successful,
        "expected_choice": expected_choice,
        "expected_basis": expected_basis,
        "matched_choices": matched_choices,
        "gold_source_ids": sorted(gold),
        "retrieved_gold_source_ids": retrieved_gold,
        "retrieval_status": retrieval_status,
        "retrieved_gold_count": len(retrieved_gold),
        "gold_source_count": len(gold),
        "reason": reason,
    }


def _canonical_source_id(value: str) -> str:
    for stored, canonical in (
        ("notes/", "note/"),
        ("meetings/", "meeting/"),
    ):
        if value.startswith(stored):
            return canonical + value.removeprefix(stored)
    return value
