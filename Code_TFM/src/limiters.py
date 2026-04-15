"""
src/limiters.py
===============
Rate limiters para APIs externas.
"""

import time
import asyncio
from llama_index.embeddings.gemini import GeminiEmbedding


class RateLimitedGeminiEmbedding(GeminiEmbedding):
    """GeminiEmbedding con rate limiting de RPM."""

    def __init__(self, *args, max_rpm: int = 500, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval = 60.0 / max_rpm
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def _wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.time()

    async def aget_text_embedding(self, text: str) -> list[float]:
        await self._wait()
        return await super().aget_text_embedding(text)

    async def aget_text_embedding_batch(self, texts, **kwargs):
        await self._wait()
        return await super().aget_text_embedding_batch(texts, **kwargs)

    def get_text_embedding(self, text: str) -> list[float]:
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()
        return super().get_text_embedding(text)


from llama_index.embeddings.ollama import OllamaEmbedding

class TruncatedOllamaEmbedding(OllamaEmbedding):
    """OllamaEmbedding que trunca textos largos antes de embeddear."""
    
    MAX_CHARS: int = 2000  # ~512 tokens aprox
    
    def _truncate(self, texts):
        return [t[:self.MAX_CHARS] for t in texts]
    
    def get_text_embedding_batch(self, texts, **kwargs):
        return super().get_text_embedding_batch(self._truncate(texts), **kwargs)
    
    async def aget_text_embedding_batch(self, texts, **kwargs):
        return await super().aget_text_embedding_batch(self._truncate(texts), **kwargs)
    
    def get_text_embedding(self, text, **kwargs):
        return super().get_text_embedding(text[:self.MAX_CHARS], **kwargs)
    
    async def aget_text_embedding(self, text, **kwargs):
        return await super().aget_text_embedding(text[:self.MAX_CHARS], **kwargs)