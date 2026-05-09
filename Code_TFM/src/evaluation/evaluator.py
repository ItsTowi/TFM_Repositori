import os
import json
import math
import time
import traceback
from dataclasses import asdict
from datetime import datetime

from .results import QAResult, ExperimentResult, RAGType
from .query_adapters import query_traditional, query_lightrag, query_llamaindex, query_advanced, query_msgraphrag_global, query_msgraphrag_local, query_literag
from .metrics import build_ragas_wrappers, compute_ragas_scores
from .query_token_tracker import QueryTokenTracker, ITokenTrackable


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
        elif self.rag_type == "advanced":
            return await query_advanced(self.rag, question)
        elif self.rag_type == "lightrag":
            return await query_lightrag(self.rag, question, self.lightrag_mode)
        elif self.rag_type == "llamaindex":
            return await query_llamaindex(self.rag, question)
        elif self.rag_type == "msgraphrag_local":
            return await query_msgraphrag_local(self.rag, question)
        elif self.rag_type == "msgraphrag_global":
            return await query_msgraphrag_global(self.rag, question)
        elif self.rag_type == "literag":
            return await query_literag(self.rag, question)
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

        # ── Setup query token tracker ──────────────────────────────────────────
        query_tracker: QueryTokenTracker | None = None
        if isinstance(self.rag, ITokenTrackable):
            query_tracker = QueryTokenTracker()
            self.rag.attach_token_tracker(query_tracker)
            print(f"  📡 Query token tracking activado para {self.rag_type}")
        else:
            print(f"  ⚠️  {self.rag_type} no implementa ITokenTrackable — sin query token tracking")

        print(f"\n🔍 Evaluando {self.rag_type.upper()} | {dominio} | {len(qas_a_evaluar)} preguntas")
        print("─" * 60)

        for i, qa in enumerate(qas_a_evaluar):
            print(f"  [{i+1}/{len(qas_a_evaluar)}] {qa['question'][:70]}...")
            t0 = time.time()

            # Snapshot del log antes de la query para aislar tokens de esta pregunta
            idx_before = len(query_tracker.request_log) if query_tracker else 0

            try:
                answer, contexts = await self._query(qa["question"])
                error = ""
            except Exception as e:
                traceback.print_exc()
                answer, contexts, error = "", [], str(e)
                result.n_errors += 1
                print(f"    ⚠️ Error: {e}")

            # Tokens de query para esta pregunta
            q_token_usage = {}
            if query_tracker:
                q_slice = query_tracker.request_log[idx_before:]
                q_token_usage = query_tracker.get_stats_for_slice(q_slice)

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
                query_token_usage=q_token_usage,
            ))

        latencias = [r.latency_s for r in result.qa_results if not r.error]
        result.avg_latency_s = round(sum(latencias) / len(latencias), 2) if latencias else 0

        # Tokens totales de query (sin RAGAS)
        if query_tracker:
            result.query_token_usage = query_tracker.get_stats()

        print("\n📊 Calculando métricas RAGAS...")
        ragas_output = await compute_ragas_scores(
            result.qa_results, self.llm, self.embeddings
        )

        # Separar scores de token_usage (RAGAS)
        result.token_usage = ragas_output.pop("token_usage", {})
        result.ragas_scores = ragas_output

        return result

    def save(self, result: ExperimentResult, path: str = "../../results/") -> str:
        os.makedirs(path, exist_ok=True)
        filename = f"{result.rag_type}_{result.dominio}_{result.timestamp[:10]}.json"
        filepath = os.path.join(path, filename)

        # Convertir NaN a None para serialización JSON válida
        data = asdict(result)
        if "ragas_scores" in data:
            data["ragas_scores"] = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in data["ragas_scores"].items()
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
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
                if isinstance(score, float) and not math.isnan(score):
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    print(f"    {metric:<25} {bar} {score:.4f}")
                else:
                    print(f"    {metric:<25} {'':>20} NaN")
        else:
            print("    (sin métricas)")

        # ── Query token usage ──────────────────────────────────────────────────
        if result.query_token_usage:
            t = result.query_token_usage.get("total", {})
            r2 = result.query_token_usage.get("rates", {})
            estimated = result.query_token_usage.get("estimated", False)
            label = "💰 Tokens Query (estimado):" if estimated else "💰 Tokens Query:"
            print(f"\n  {label}")
            print(f"    Total      : {t.get('total_tokens', 0):,}")
            print(f"    Prompt     : {t.get('prompt_tokens', 0):,}")
            print(f"    Completion : {t.get('completion_tokens', 0):,}")
            if r2:
                print(f"    TPM        : {r2.get('tpm', 0):,} tokens/min")
                print(f"    RPM        : {r2.get('rpm', 0):,} requests/min")
        else:
            print("\n  💰 Tokens Query: no disponible")

        # ── RAGAS token usage ──────────────────────────────────────────────────
        if result.token_usage:
            t = result.token_usage.get("total", {})
            r2 = result.token_usage.get("rates", {})
            print(f"\n  🔬 Tokens RAGAS (evaluación):")
            print(f"    Total      : {t.get('total_tokens', 0):,}")
            print(f"    Prompt     : {t.get('prompt_tokens', 0):,}")
            print(f"    Completion : {t.get('completion_tokens', 0):,}")
            print(f"    TPM        : {r2.get('tpm', 0):,} tokens/min")
            print(f"    RPM        : {r2.get('rpm', 0):,} requests/min")
        print("=" * 60)
