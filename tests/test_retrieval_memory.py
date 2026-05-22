from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.config import Settings
from framework.rag.embeddings import DeterministicEmbeddingProvider
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.retrieval_service import RetrievalService
from framework.schemas import ChunkRecord, FileRecord, MetadataFilter, SearchRequest


class FakeVectorStore:
    async def search(self, query_vector, limit, filters=None):
        return [
            SimpleNamespace(id="memory", payload={"chunk_id": "memory"}, score=0.7),
            SimpleNamespace(id="ordinary", payload={"chunk_id": "ordinary"}, score=0.72),
        ][:limit]


@pytest.mark.asyncio
async def test_hybrid_retrieval_filters_and_boosts_memory(tmp_path: Path) -> None:
    settings = Settings(vault_path=tmp_path / "vault", runtime_path=tmp_path / "runtime")
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    await repository.init()
    now = datetime.now(tz=UTC)
    await repository.upsert_file(
        FileRecord(
            path="memory/prefs.md",
            file_hash="a",
            mtime_ns=1,
            size_bytes=10,
            title="Prefs",
            metadata={"tags": ["memory"], "type": "memory", "title": "Prefs"},
            is_memory=True,
        )
    )
    await repository.upsert_file(
        FileRecord(
            path="research/rag.md",
            file_hash="b",
            mtime_ns=2,
            size_bytes=10,
            title="RAG",
            metadata={"tags": ["rag"], "type": "research", "title": "RAG"},
            is_memory=False,
        )
    )
    await repository.replace_chunks(
        "memory/prefs.md",
        [
            ChunkRecord(
                chunk_id="memory",
                file_path="memory/prefs.md",
                ordinal=0,
                chunk_hash="mh",
                text="agent preferences incremental indexing",
                metadata={"tags": ["memory"], "type": "memory", "title": "Prefs"},
                is_memory=True,
                modified_time=now,
            )
        ],
    )
    await repository.replace_chunks(
        "research/rag.md",
        [
            ChunkRecord(
                chunk_id="ordinary",
                file_path="research/rag.md",
                ordinal=0,
                chunk_hash="oh",
                text="incremental indexing qdrant",
                metadata={"tags": ["rag"], "type": "research", "title": "RAG"},
                is_memory=False,
                modified_time=now,
            )
        ],
    )

    service = RetrievalService(
        repository=repository,
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        embedding_provider=DeterministicEmbeddingProvider(dimensions=64),
    )

    memory_response = await service.search(SearchRequest(query="incremental indexing", top_k=2))
    filtered_response = await service.search(
        SearchRequest(
            query="incremental indexing",
            top_k=2,
            filters=MetadataFilter(type="research"),
        )
    )

    assert memory_response.hits[0].is_memory is True
    assert [hit.file_path for hit in filtered_response.hits] == ["research/rag.md"]

