from __future__ import annotations

from framework.config import Settings
from framework.rag.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from framework.rag.indexing_service import IndexingService
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.vector_store import QdrantVectorStore


def build_indexing_service(
    settings: Settings,
    embedding_provider: EmbeddingProvider | None = None,
) -> IndexingService:
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    vector_store = QdrantVectorStore(settings)
    provider = embedding_provider or OpenAIEmbeddingProvider(settings)
    return IndexingService(
        settings=settings,
        repository=repository,
        vector_store=vector_store,
        embedding_provider=provider,
    )

