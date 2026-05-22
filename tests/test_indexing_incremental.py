from __future__ import annotations

from pathlib import Path

import pytest

from framework.config import Settings
from framework.rag.embeddings import DeterministicEmbeddingProvider
from framework.rag.indexing_service import IndexingService
from framework.rag.repository import SQLiteKnowledgeRepository


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.deleted_files: list[str] = []

    async def ensure_collection(self) -> None:
        return None

    async def delete_file(self, file_path: str) -> None:
        self.deleted_files.append(file_path)
        self.points = {
            point_id: payload
            for point_id, payload in self.points.items()
            if payload["file_path"] != file_path
        }

    async def upsert_chunks(self, chunks, embeddings) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.points[chunk.chunk_id] = {
                "file_path": chunk.file_path,
                "embedding_dim": len(embedding),
            }


@pytest.mark.asyncio
async def test_incremental_index_skips_unchanged_and_cleans_deleted(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("---\ntags: [ai]\ntype: research\n---\n# Note\n\nBody", encoding="utf-8")

    settings = Settings(vault_path=vault, runtime_path=runtime, embedding_dim=64)
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    vector_store = FakeVectorStore()
    service = IndexingService(
        settings=settings,
        repository=repository,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_provider=DeterministicEmbeddingProvider(dimensions=64),
    )

    first = await service.index_vault()
    second = await service.index_vault()
    note.unlink()
    third = await service.index_vault()

    assert first.indexed_files == 1
    assert first.embedded_chunks == 1
    assert second.skipped_files == 1
    assert second.indexed_files == 0
    assert third.deleted_files == 1
    assert await repository.chunk_count() == 0

@pytest.mark.asyncio
async def test_embedding_cache_reuses_chunk_hash(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Same\n\nStable text", encoding="utf-8")

    settings = Settings(vault_path=vault, runtime_path=runtime, embedding_dim=64)
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    vector_store = FakeVectorStore()
    service = IndexingService(
        settings=settings,
        repository=repository,
        vector_store=vector_store,  # type: ignore[arg-type]
        embedding_provider=DeterministicEmbeddingProvider(dimensions=64),
    )

    first = await service.index_vault()
    note.write_text("# Same\n\nStable text\n", encoding="utf-8")
    second = await service.index_vault()

    assert first.embedded_chunks == 1
    assert second.reused_embeddings == 1

