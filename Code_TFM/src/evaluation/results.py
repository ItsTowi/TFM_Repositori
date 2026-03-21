"""
results.py
==========
Dataclasses que representan los resultados de un experimento RAG.
"""

from dataclasses import dataclass, field
from typing import Literal

RAGType = Literal["traditional", "lightrag", "llamaindex"]


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