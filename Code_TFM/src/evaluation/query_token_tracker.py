"""
query_token_tracker.py
======================
Tracking genérico de tokens para la fase de query (no RAGAS).

Componentes:
  - QueryTokenTracker : acumulador agnóstico de cliente
  - LangChainTokenCallback : callback LCEL que alimenta el tracker
  - ITokenTrackable   : protocolo que implementan los baselines compatibles

Uso desde el evaluador:
    if isinstance(rag, ITokenTrackable):
        tracker = QueryTokenTracker()
        rag.attach_token_tracker(tracker)
        tracker.reset()
        answer, contexts = await _query(question)
        stats = tracker.get_stats_for_slice(tracker.request_log[idx:])
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ── Log por llamada ────────────────────────────────────────────────────────────

@dataclass
class RequestLog:
    """Registro de una sola llamada al LLM."""
    timestamp: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── Acumulador genérico ────────────────────────────────────────────────────────

class QueryTokenTracker:
    """
    Acumulador de tokens de query, desacoplado de cualquier cliente LLM.

    Cada baseline compatible llama a .record() tras cada llamada al LLM.
    El evaluador llama a .reset() antes de cada pregunta para aislar
    los tokens por pregunta.
    """

    def __init__(self):
        self.request_log: list[RequestLog] = []
        self._start_time: float = time.time()

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Registra los tokens de una llamada al LLM."""
        self.request_log.append(RequestLog(
            timestamp=time.time(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))

    def reset(self) -> None:
        """Vacía el log y reinicia el cronómetro."""
        self.request_log.clear()
        self._start_time = time.time()

    def get_stats(self) -> dict:
        """Estadísticas globales del tracker desde el último reset()."""
        if not self.request_log:
            return {}
        elapsed_min = max((time.time() - self._start_time) / 60, 1 / 60)
        total_prompt      = sum(r.prompt_tokens     for r in self.request_log)
        total_completion  = sum(r.completion_tokens for r in self.request_log)
        total_tokens      = total_prompt + total_completion
        n_requests        = len(self.request_log)
        return {
            "total": {
                "prompt_tokens":     total_prompt,
                "completion_tokens": total_completion,
                "total_tokens":      total_tokens,
            },
            "rates": {
                "tpm":              round(total_tokens / elapsed_min),
                "rpm":              round(n_requests  / elapsed_min),
                "elapsed_minutes":  round(elapsed_min, 2),
                "n_requests":       n_requests,
            },
        }

    def get_stats_for_slice(self, log_slice: list[RequestLog]) -> dict:
        """Stats para un subconjunto del log (e.g. una sola pregunta)."""
        if not log_slice:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "n_requests": 0,
            }
        return {
            "prompt_tokens":     sum(r.prompt_tokens     for r in log_slice),
            "completion_tokens": sum(r.completion_tokens for r in log_slice),
            "total_tokens":      sum(r.total_tokens      for r in log_slice),
            "n_requests":        len(log_slice),
        }


# ── Callback LangChain ─────────────────────────────────────────────────────────

class LangChainTokenCallback(BaseCallbackHandler):
    """
    Callback LCEL que intercepta on_llm_end y registra tokens en un
    QueryTokenTracker.

    Compatible con ChatGoogleGenerativeAI (langchain-google-genai).
    Busca usage en varios sitios para ser robusto a cambios de versión.
    """

    def __init__(self, tracker: QueryTokenTracker):
        super().__init__()
        self.tracker = tracker

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # 1. Intento principal: llm_output (agregado de todas las generaciones)
        if response.llm_output:
            # Google GenAI ≥ 2.x
            usage = response.llm_output.get("usage_metadata")
            if usage:
                self.tracker.record(
                    prompt_tokens=usage.get("input_tokens", usage.get("prompt_token_count", 0)),
                    completion_tokens=usage.get("output_tokens", usage.get("candidates_token_count", 0)),
                )
                return
            # OpenAI-style fallback
            usage = response.llm_output.get("token_usage", {})
            if usage:
                self.tracker.record(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
                return

        # 2. Fallback: iterar generaciones individuales
        for gen_list in response.generations:
            for gen in gen_list:
                meta = None
                if hasattr(gen, "message"):
                    meta = getattr(gen.message, "usage_metadata", None)
                if meta:
                    self.tracker.record(
                        prompt_tokens=meta.get("input_tokens", meta.get("prompt_token_count", 0)),
                        completion_tokens=meta.get("output_tokens", meta.get("candidates_token_count", 0)),
                    )
                    return


# ── Protocolo ─────────────────────────────────────────────────────────────────

@runtime_checkable
class ITokenTrackable(Protocol):
    """
    Protocolo que implementan los baselines que soportan query token tracking.

    El evaluador detecta si el objeto RAG implementa este protocolo con
    isinstance(rag, ITokenTrackable) y, en ese caso, llama a
    attach_token_tracker antes de empezar las queries.
    """

    def attach_token_tracker(self, tracker: QueryTokenTracker) -> None:
        """Engancha el tracker al LLM interno del baseline."""
        ...
