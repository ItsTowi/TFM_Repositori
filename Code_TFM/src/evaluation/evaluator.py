"""
evaluator.py
============
Clase RAGEvaluator: orquesta queries, recoge resultados y calcula métricas.
"""

import os
import json
import time
from dataclasses import asdict
from datetime import datetime

from .results import QAResult, ExperimentResult, RAGType
from .query_adapters import query_traditional, query_lightrag, query_llamaindex
from .metrics import build_ragas_wrappers, compute_ragas_scores


class RAGEvaluator:
    def __init__(
        self,
        rag_type: RAGType,
        rag_object,
        lightrag_mode: str = "hybrid",
        gemini_model: str = "gemini-2.5-flash-lite",
    ):
        self.rag_type = rag_type
        self.rag = rag_object
        self.lightrag_mode = lightrag_mode
        self.llm, self.embeddings = build_ragas_wrappers(gemini_model)

    async def _query(self, question: str) -> tuple[str, list[str]]:
        if self.rag_type == "traditional":
            return await query_traditional(self.rag, question)
        elif self.rag_type == "lightrag":
            return await query_lightrag(self.rag, question, self.lightrag_mode)
        elif self.rag_type == "llamaindex":
            return await query_llamaindex(self.rag, question)
        else:
            raise ValueError(f"rag_type desconocido: {self.rag_type}")

    async def run(
        self,
        libros: list[dict],
        qas: list[dict],
        dominio: str,
        max_questions: int = None,
    ) -> ExperimentResult:
        result = ExperimentResult(
            rag_type=self.rag_type,
            dominio=dominio,
            n_libros=len(libros),
            timestamp=datetime.now().isoformat(),
        )

        titulo_por_cid = {l["context_id"]: l["titulo"] for l in libros}
        qas_a_evaluar = qas[:max_questions] if max_questions else qas

        print(f"\n🔍 Evaluando {self.rag_type.upper()} | {dominio} | {len(qas_a_evaluar)} preguntas")
        print("─" * 60)

        for i, qa in enumerate(qas_a_evaluar):
            print(f"  [{i+1}/{len(qas_a_evaluar)}] {qa['question'][:70]}...")
            t0 = time.time()
            try:
                answer, contexts = await self._query(qa["question"])
                error = ""
            except Exception as e:
                answer, contexts, error = "", [], str(e)
                result.n_errors += 1
                print(f"    ⚠️ Error: {e}")

            result.qa_results.append(QAResult(
                question=qa["question"],
                ground_truth=qa["answer"],
                answer=answer,
                contexts=contexts,
                latency_s=round(time.time() - t0, 2),
                rag_type=self.rag_type,
                dominio=dominio,
                titulo=titulo_por_cid.get(qa["context_id"], ""),
                error=error,
            ))

        latencias = [r.latency_s for r in result.qa_results if not r.error]
        result.avg_latency_s = round(sum(latencias) / len(latencias), 2) if latencias else 0

        print("\n📊 Calculando métricas RAGAS...")
        ragas_output = await compute_ragas_scores(
            result.qa_results, self.llm, self.embeddings
        )

        # Separar scores de token_usage
        result.token_usage = ragas_output.pop("token_usage", {})
        result.ragas_scores = ragas_output

        return result

    def save(self, result: ExperimentResult, path: str = "../../results/") -> str:
        os.makedirs(path, exist_ok=True)
        filename = f"{result.rag_type}_{result.dominio}_{result.timestamp[:10]}.json"
        filepath = os.path.join(path, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"💾 Guardado en: {filepath}")
        return filepath

    def print_summary(self, result: ExperimentResult):
        print("\n" + "=" * 60)
        print(f"📋 RESUMEN: {result.rag_type.upper()} | {result.dominio}")
        print("=" * 60)
        print(f"  Libros indexados : {result.n_libros}")
        print(f"  Preguntas        : {len(result.qa_results)}")
        print(f"  Errores          : {result.n_errors}")
        print(f"  Latencia media   : {result.avg_latency_s}s")
        print("\n  📊 Métricas RAGAS:")
        if result.ragas_scores:
            for metric, score in result.ragas_scores.items():
                if isinstance(score, float):
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    print(f"    {metric:<25} {bar} {score:.4f}")
                else:
                    print(f"    {metric:<25} {score}")
        else:
            print("    (sin métricas)")
        if result.token_usage:
            t = result.token_usage.get("total", {})
            r2 = result.token_usage.get("rates", {})
            print(f"\n  💰 Tokens RAGAS:")
            print(f"    Total      : {t.get('total_tokens', 0):,}")
            print(f"    Prompt     : {t.get('prompt_tokens', 0):,}")
            print(f"    Completion : {t.get('completion_tokens', 0):,}")
            print(f"    TPM        : {r2.get('tpm', 0):,} tokens/min")
            print(f"    RPM        : {r2.get('rpm', 0):,} requests/min")
        print("=" * 60)