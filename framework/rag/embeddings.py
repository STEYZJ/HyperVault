from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from framework.config import Settings


class EmbeddingProvider(ABC):
    model: str
    dimensions: int

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query]))[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dim
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [[float(value) for value in item.embedding] for item in response.data]


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider for tests and offline smoke checks only."""

    def __init__(self, model: str = "deterministic-test", dimensions: int = 64) -> None:
        self.model = model
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimensions)]
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

