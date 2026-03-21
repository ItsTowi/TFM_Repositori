"""
metrics.py
==========
Cálculo de métricas RAGAS (v0.4.x) sobre una lista de QAResult.
Incluye tracking de tokens, TPM y RPM para análisis de coste.
"""

import os
import time
import inspect
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from google import genai

from .results import QAResult


# ── Token Tracker ──────────────────────────────────────────────────────────────

@dataclass
class RequestLog:
    """Log de una sola llamada a la API."""
    timestamp: float       # epoch cuando llegó la respuesta
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens


class TrackingAsyncOpenAI(AsyncOpenAI):
    """AsyncOpenAI que intercepta respuestas y registra tokens + timestamps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_log: list[RequestLog] = []
        self._start_time: float = time.time()
        self.chat.completions.create = self._tracked_create(self.chat.completions.create)

    def _tracked_create(self, original_fn):
        async def wrapper(*args, **kwargs):
            response = await original_fn(*args, **kwargs)
            if hasattr(response, "usage") and response.usage:
                self.request_log.append(RequestLog(
                    timestamp=time.time(),
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                ))
            return response
        return wrapper

    def reset(self):
        self.request_log = []
        self._start_time = time.time()

    def get_stats(self) -> dict:
        if not self.request_log:
            return {}

        elapsed_min = (time.time() - self._start_time) / 60
        elapsed_min = max(elapsed_min, 1 / 60)  # mínimo 1 segundo para evitar /0

        total_prompt     = sum(r.prompt_tokens     for r in self.request_log)
        total_completion = sum(r.completion_tokens for r in self.request_log)
        total_tokens     = total_prompt + total_completion
        n_requests       = len(self.request_log)

        return {
            "total": {
                "prompt_tokens":     total_prompt,
                "completion_tokens": total_completion,
                "total_tokens":      total_tokens,
            },
            "rates": {
                "tpm": round(total_tokens / elapsed_min),
                "rpm": round(n_requests  / elapsed_min),
                "elapsed_minutes": round(elapsed_min, 2),
                "n_requests": n_requests,
            },
        }

    def get_stats_for_slice(self, log_slice: list[RequestLog]) -> dict:
        """Stats para un subconjunto de requests (ej: una sola pregunta)."""
        if not log_slice:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_requests": 0}
        return {
            "prompt_tokens":     sum(r.prompt_tokens     for r in log_slice),
            "completion_tokens": sum(r.completion_tokens for r in log_slice),
            "total_tokens":      sum(r.total_tokens      for r in log_slice),
            "n_requests":        len(log_slice),
        }


# ── Build wrappers ─────────────────────────────────────────────────────────────

def build_ragas_wrappers(gemini_model: str = "gemini-2.5-flash-lite"):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    tracking_client = TrackingAsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    llm = llm_factory(
        gemini_model,
        client=tracking_client,
        adapter="instructor",
        max_tokens=8192,
    )
    llm._tracking_client = tracking_client

    genai_client = genai.Client(api_key=api_key)
    embeddings = embedding_factory(
        "google",
        model="gemini-embedding-001",
        client=genai_client,
        interface="modern"
    )

    return llm, embeddings


# ── Score helpers ──────────────────────────────────────────────────────────────

def _get_ascore_params(metric_class):
    sig = inspect.signature(metric_class.ascore)
    return set(sig.parameters.keys()) - {"self"}


async def _score_one(metric, r: QAResult) -> float:
    params = _get_ascore_params(type(metric))
    kwargs = {}
    if "user_input"         in params: kwargs["user_input"]         = r.question
    if "response"           in params: kwargs["response"]           = r.answer
    if "retrieved_contexts" in params: kwargs["retrieved_contexts"] = r.contexts
    if "reference"          in params: kwargs["reference"]          = r.ground_truth
    result = await metric.ascore(**kwargs)
    return float(result.value)


# ── Main evaluation ────────────────────────────────────────────────────────────

async def compute_ragas_scores(
    qa_results: list[QAResult],
    llm,
    embeddings,
) -> dict:
    validos = [r for r in qa_results if r.answer and not r.error]
    if not validos:
        print("  ⚠️ Sin resultados válidos para RAGAS")
        return {}

    client = getattr(llm, "_tracking_client", None)
    if client:
        client.reset()

    metrics = {
        "faithfulness":      Faithfulness(llm=llm),
        "answer_relevancy":  AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall":    ContextRecall(llm=llm),
    }

    scores = {name: [] for name in metrics}
    token_usage_per_question = []

    for i, r in enumerate(validos):
        print(f"  RAGAS [{i+1}/{len(validos)}] {r.question[:50]}...")

        idx_before = len(client.request_log) if client else 0

        for name, metric in metrics.items():
            try:
                val = await _score_one(metric, r)
                scores[name].append(val)
            except Exception as e:
                print(f"    ⚠️ {name}: {e}")

        if client:
            q_slice = client.request_log[idx_before:]
            q_stats = client.get_stats_for_slice(q_slice)
            token_usage_per_question.append({
                "question": r.question[:60],
                **q_stats,
            })

    result = {
        name: round(sum(vals) / len(vals), 4)
        for name, vals in scores.items()
        if vals
    }

    if client:
        stats = client.get_stats()
        result["token_usage"] = {
            **stats,
            "per_question": token_usage_per_question,
        }
        t = stats.get("total", {})
        r2 = stats.get("rates", {})
        print(f"\n  💰 Tokens RAGAS:")
        print(f"     Total  : {t.get('total_tokens', 0):,}  (prompt: {t.get('prompt_tokens', 0):,} | completion: {t.get('completion_tokens', 0):,})")
        print(f"     TPM    : {r2.get('tpm', 0):,} tokens/min")
        print(f"     RPM    : {r2.get('rpm', 0):,} requests/min  ({r2.get('n_requests', 0)} requests en {r2.get('elapsed_minutes', 0)} min)")

    return result