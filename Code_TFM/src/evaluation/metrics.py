import os
import inspect
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


def build_ragas_wrappers(gemini_model: str = "gemini-2.5-flash-lite"):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    llm = llm_factory(
        gemini_model,
        client=async_client,
        adapter="instructor",
        max_tokens=8192,  # suficiente para respuestas largas
    )

    genai_client = genai.Client(api_key=api_key)
    embeddings = embedding_factory(
        "google",
        model="gemini-embedding-001",
        client=genai_client,
        interface="modern"
    )

    return llm, embeddings


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


async def compute_ragas_scores(
    qa_results: list[QAResult],
    llm,
    embeddings,
) -> dict:
    validos = [r for r in qa_results if r.answer and not r.error]
    if not validos:
        print("  ⚠️ Sin resultados válidos para RAGAS")
        return {}

    metrics = {
        "faithfulness":       Faithfulness(llm=llm),
        "answer_relevancy":   AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision":  ContextPrecision(llm=llm),
        "context_recall":     ContextRecall(llm=llm),
    }

    scores = {name: [] for name in metrics}

    for i, r in enumerate(validos):
        print(f"  RAGAS [{i+1}/{len(validos)}] {r.question[:50]}...")
        for name, metric in metrics.items():
            try:
                val = await _score_one(metric, r)
                scores[name].append(val)
            except Exception as e:
                print(f"    ⚠️ {name}: {e}")

    return {
        name: round(sum(vals) / len(vals), 4)
        for name, vals in scores.items()
        if vals
    }