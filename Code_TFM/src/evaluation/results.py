"""
results.py
==========
Dataclasses que representan los resultados de un experimento RAG.
"""

from dataclasses import dataclass, field
from typing import Literal

RAGType = Literal["traditional", "lightrag", "llamaindex", "literag"]


@dataclass
class QAResult:
    """Resultado de una sola pregunta."""
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    latency_s: float
    rag_type: RAGType
    dominio: str
    titulo: str
    error: str = ""
    # Tokens consumidos por el RAG para responder esta pregunta (sin RAGAS)
    query_token_usage: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Resultado completo de un experimento."""
    rag_type: RAGType
    dominio: str
    n_libros: int
    timestamp: str
    qa_results: list[QAResult] = field(default_factory=list)
    ragas_scores: dict = field(default_factory=dict)
    avg_latency_s: float = 0.0
    n_errors: int = 0
    # Tokens consumidos por el RAG en todas las queries (sin RAGAS)
    query_token_usage: dict = field(default_factory=dict)
    # Tokens consumidos por RAGAS durante la evaluación
    token_usage: dict = field(default_factory=dict)