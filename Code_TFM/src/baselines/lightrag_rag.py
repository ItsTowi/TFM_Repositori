"""
lightrag_rag.py
===============
Inicialización de LightRAG con Gemini LLM + Gemini Embeddings.
Incluye tracking de tokens de indexación y reintentos automáticos.

Uso:
    from src.baselines.lightrag_rag import build_lightrag, index_documents

    rag, tracker = await build_lightrag(workspace_dir="./lightrag_ws")
    await index_documents(rag, tracker, libros)
    print(tracker.summary())
"""

import os
import time
import shutil
import asyncio
import numpy as np
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.openai import openai_complete_if_cache


# ── Token Tracker de Indexación ────────────────────────────────────────────────

@dataclass
class IndexingStats:
    """Acumula tokens gastados durante la indexación."""

    llm_prompt_tokens:      int = 0
    llm_completion_tokens:  int = 0
    llm_requests:           int = 0
    llm_retries:            int = 0

    embedding_tokens:       int = 0
    embedding_requests:     int = 0

    start_time:             float = field(default_factory=time.time)
    end_time:               float = 0.0

    def add_llm(self, prompt_tokens: int, completion_tokens: int):
        self.llm_prompt_tokens     += prompt_tokens
        self.llm_completion_tokens += completion_tokens
        self.llm_requests          += 1

    def add_embedding(self, n_texts: int, tokens_per_text: int = 256):
        self.embedding_tokens   += n_texts * tokens_per_text
        self.embedding_requests += n_texts

    def finish(self):
        self.end_time = time.time()

    @property
    def elapsed_min(self):
        end = self.end_time if self.end_time else time.time()
        return max((end - self.start_time) / 60, 1/60)

    @property
    def llm_total_tokens(self):
        return self.llm_prompt_tokens + self.llm_completion_tokens

    def summary(self) -> str:
        lines = [
            "\n📊 Estadísticas de Indexación LightRAG",
            "─" * 45,
            f"  ⏱️  Tiempo total     : {self.elapsed_min:.2f} min",
            "",
            f"  🧠 LLM (extracción de grafo):",
            f"     Requests         : {self.llm_requests:,}",
            f"     Reintentos       : {self.llm_retries:,}",
            f"     Prompt tokens    : {self.llm_prompt_tokens:,}",
            f"     Completion tokens: {self.llm_completion_tokens:,}",
            f"     Total tokens     : {self.llm_total_tokens:,}",
            f"     TPM              : {int(self.llm_total_tokens / self.elapsed_min):,}",
            f"     RPM              : {int(self.llm_requests / self.elapsed_min):,}",
            "",
            f"  🔢 Embeddings:",
            f"     Requests         : {self.embedding_requests:,}",
            f"     Tokens estimados : {self.embedding_tokens:,}  (~256 tok/texto)",
            "─" * 45,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "elapsed_minutes": round(self.elapsed_min, 2),
            "llm": {
                "requests":          self.llm_requests,
                "retries":           self.llm_retries,
                "prompt_tokens":     self.llm_prompt_tokens,
                "completion_tokens": self.llm_completion_tokens,
                "total_tokens":      self.llm_total_tokens,
                "tpm":               int(self.llm_total_tokens / self.elapsed_min),
                "rpm":               int(self.llm_requests / self.elapsed_min),
            },
            "embeddings": {
                "requests":         self.embedding_requests,
                "tokens_estimated": self.embedding_tokens,
            },
        }


# ── Wrappers con tracking y reintentos ────────────────────────────────────────

def _make_llm_wrapper(
    tracker: IndexingStats,
    model: str = "gemini-2.5-flash-lite",
    max_retries: int = 5,
    base_wait: float = 10.0,
):
    async def gemini_llm_wrapper(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        kwargs["api_key"]  = os.getenv("GEMINI_API_KEY")

        last_exc = None
        for attempt in range(max_retries):
            try:
                response = await openai_complete_if_cache(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs
                )
                tracker.llm_requests += 1
                return response

            except Exception as e:
                last_exc = e
                tracker.llm_retries += 1
                wait = base_wait * (2 ** attempt)  # 10s, 20s, 40s, 80s, 160s
                print(f"\n⚠️  LLM error (intento {attempt+1}/{max_retries}): {type(e).__name__}")
                print(f"   Esperando {wait:.0f}s antes de reintentar...")
                await asyncio.sleep(wait)

        print(f"\n❌ LLM fallido tras {max_retries} intentos: {last_exc}")
        raise last_exc

    return gemini_llm_wrapper


def _make_embedding_wrapper(tracker: IndexingStats):
    @wrap_embedding_func_with_attrs(
        embedding_dim=3072,
        max_token_size=2048,
        model_name="gemini-embedding-001"
    )
    async def gemini_embedding_wrapper(texts: list[str]) -> np.ndarray:
        client = AsyncOpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        embeddings = []
        for text in texts:
            response = await client.embeddings.create(
                input=[text],
                model="gemini-embedding-001"
            )
            embeddings.append(response.data[0].embedding)

        tracker.add_embedding(n_texts=len(texts))
        return np.array(embeddings)

    return gemini_embedding_wrapper


# ── Builder principal ──────────────────────────────────────────────────────────

async def build_lightrag(
    workspace_dir: str,
    model: str = "gemini-2.5-flash-lite",
    clean: bool = True,
    max_async: int = 12,        # FIX: bajado de 12 → 4 para evitar sobrecarga
    max_retries: int = 10,
    chunk_token_size: int = 600,        # FIX: reducido del default ~1200
    chunk_overlap_token_size: int = 50, # FIX: reducido del default ~100
    entity_extract_max_gleaning: int = 0,  # FIX: eliminado gleaning (ahorra ~50% tiempo LLM)
) -> tuple[LightRAG, IndexingStats]:
    """
    Inicializa LightRAG con Gemini y devuelve (rag, tracker).

    Args:
        workspace_dir:                carpeta donde LightRAG guarda el grafo
        model:                        modelo Gemini para el LLM
        clean:                        si True, borra el workspace antes de inicializar
        max_async:                    workers paralelos para el LLM
        max_retries:                  reintentos automáticos ante errores o timeouts
        chunk_token_size:             tamaño de chunk en tokens (default LightRAG ~1200)
        chunk_overlap_token_size:     solapamiento entre chunks
        entity_extract_max_gleaning:  pasadas extra de extracción por chunk (0 = desactivado)
    """
    if clean:
        shutil.rmtree(workspace_dir, ignore_errors=True)
    os.makedirs(workspace_dir, exist_ok=True)

    tracker = IndexingStats()

    rag = LightRAG(
        working_dir=workspace_dir,
        llm_model_func=_make_llm_wrapper(tracker, model, max_retries=max_retries),
        llm_model_max_async=max_async,
        embedding_func=_make_embedding_wrapper(tracker),
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
        entity_extract_max_gleaning=entity_extract_max_gleaning,
    )

    await rag.initialize_storages()
    print(f"✅ LightRAG inicializado en: {workspace_dir}")
    print(f"   max_async={max_async} | max_retries={max_retries} | modelo={model}")
    print(f"   chunk_size={chunk_token_size} | gleaning={entity_extract_max_gleaning}")
    return rag, tracker


async def index_documents(
    rag: LightRAG,
    tracker: IndexingStats,
    libros: list[dict],
) -> IndexingStats:
    """
    Indexa una lista de libros en LightRAG con reintentos automáticos por libro.

    Args:
        rag:     instancia de LightRAG
        tracker: objeto IndexingStats
        libros:  salida de cargar_experimento()[0]
    """
    tracker.start_time = time.time()

    for i, libro in enumerate(libros):
        print(f"\n📖 [{i+1}/{len(libros)}] Indexando: {libro['titulo']} ({len(libro['texto']):,} chars)...")
        try:
            await rag.ainsert(libro["texto"])  # FIX: usar ainsert async en vez de insert síncrono
            print(f"   ✅ Completado")
        except Exception as e:
            print(f"   ❌ Error indexando '{libro['titulo']}': {e}")
            print(f"   ⚠️  Continuando con el siguiente libro...")

    tracker.finish()
    print(tracker.summary())
    return tracker