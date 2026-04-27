"""
experiment.py
=============
Función de alto nivel para lanzar experimentos desde el notebook en una sola línea.
"""

from .results import RAGType, ExperimentResult
from .evaluator import RAGEvaluator
import json
import os


async def run_experiment(
    rag_type: RAGType,
    rag_object,
    dominio: str,
    n_libros: int = 1,
    max_questions: int = None,
    shuffle: bool = False,
    lightrag_mode: str = "hybrid",
    save_path: str = "../../results/",
) -> ExperimentResult:
    """
    Lanza un experimento completo en una sola llamada.

    Args:
        rag_type:       "traditional" | "lightrag" | "llamaindex"
        rag_object:     el objeto RAG ya inicializado (o query_engine para llamaindex)
        dominio:        dominio de UltraDomain, ej: "cs", "biology"
        n_libros:       cuántos libros indexar
        max_questions:  limitar nº de preguntas (útil para pruebas rápidas)
        shuffle:        si True, selecciona libros aleatoriamente
        lightrag_mode:  "local" | "global" | "hybrid" | "naive" (solo para lightrag)
        save_path:      carpeta donde guardar el JSON de resultados

    Returns:
        ExperimentResult con todas las respuestas y métricas RAGAS

    Ejemplos:
        # Traditional RAG
        result = await run_experiment("traditional", rag, dominio="cs", n_libros=1)

        # LightRAG en modo híbrido
        result = await run_experiment("lightrag", rag, dominio="cs", lightrag_mode="hybrid")

        # LlamaIndex GraphRAG
        result = await run_experiment("llamaindex", index.as_query_engine(include_text=True),
                                      dominio="cs", n_libros=1)

        # Prueba rápida con solo 5 preguntas
        result = await run_experiment("traditional", rag, dominio="cs", max_questions=5)
    """
    from src.ultradomain import cargar_experimento

    libros, qas = cargar_experimento(dominio, n_libros=n_libros, shuffle=shuffle)

    evaluator = RAGEvaluator(
        rag_type=rag_type,
        rag_object=rag_object,
        lightrag_mode=lightrag_mode,
    )

    result = await evaluator.run(libros, qas, dominio, max_questions=max_questions)
    evaluator.print_summary(result)
    evaluator.save(result, path=save_path)

    return result


async def run_local_experiment(
    rag_type: str,
    rag_object,
    questions_path: str,
    max_questions: int = None,
    lightrag_mode: str = "hybrid",
    save_path: str = "./results/",
    nombre_experimento: str = "local_test"
) -> ExperimentResult:
    
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    qas_formateadas = []
    for item in data:
        qas_formateadas.append({
            "question": item.get("question", ""),
            "answer": item.get("answer", item.get("ground_truth", "")),
            "context_id": item.get("context_id", "local_doc")
        })
        
    evaluator = RAGEvaluator(
        rag_type=rag_type,
        rag_object=rag_object,
        lightrag_mode=lightrag_mode,
    )

    result = await evaluator.run(
        libros=[], 
        qas=qas_formateadas, 
        dominio=nombre_experimento, 
        max_questions=max_questions
    )

    os.makedirs(save_path, exist_ok=True)
    evaluator.save(result, path=save_path)

    return result