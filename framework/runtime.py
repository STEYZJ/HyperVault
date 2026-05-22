from __future__ import annotations

from framework.config import Settings, get_settings
from framework.rag.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from framework.rag.indexing_service import IndexingService
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.retrieval_service import RetrievalService
from framework.rag.vector_store import QdrantVectorStore


def build_embedding_provider(
    settings: Settings,
    offline_test_embeddings: bool = False,
) -> EmbeddingProvider:
    if offline_test_embeddings:
        return DeterministicEmbeddingProvider(
            model="deterministic-test",
            dimensions=settings.embedding_dim,
        )
    return OpenAIEmbeddingProvider(settings)


def build_repository(settings: Settings | None = None) -> SQLiteKnowledgeRepository:
    active_settings = settings or get_settings()
    return SQLiteKnowledgeRepository(active_settings.sqlite_path)


def build_vector_store(settings: Settings | None = None) -> QdrantVectorStore:
    active_settings = settings or get_settings()
    return QdrantVectorStore(active_settings)


def build_indexing_service(
    settings: Settings | None = None,
    offline_test_embeddings: bool = False,
) -> IndexingService:
    active_settings = settings or get_settings()
    repository = build_repository(active_settings)
    vector_store = build_vector_store(active_settings)
    provider = build_embedding_provider(active_settings, offline_test_embeddings)
    return IndexingService(active_settings, repository, vector_store, provider)


def build_retrieval_service(
    settings: Settings | None = None,
    offline_test_embeddings: bool = False,
) -> RetrievalService:
    active_settings = settings or get_settings()
    repository = build_repository(active_settings)
    vector_store = build_vector_store(active_settings)
    provider = build_embedding_provider(active_settings, offline_test_embeddings)
    return RetrievalService(repository, vector_store, provider)

