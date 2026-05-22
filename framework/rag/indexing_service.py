from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from framework.config import Settings
from framework.ingestion.chunking import MarkdownChunker
from framework.ingestion.hashing import sha256_file
from framework.ingestion.markdown_loader import MarkdownLoader
from framework.ingestion.relations import extract_relations
from framework.rag.embeddings import EmbeddingProvider
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.vector_store import QdrantVectorStore
from framework.schemas import ChunkRecord, FileRecord, IndexRunSummary

logger = logging.getLogger(__name__)


class IndexingService:
    """Async incremental indexing pipeline for the Markdown vault."""

    def __init__(
        self,
        settings: Settings,
        repository: SQLiteKnowledgeRepository,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.loader = MarkdownLoader()
        self.chunker = MarkdownChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )

    async def initialize(self) -> None:
        await self.repository.init()
        await self.vector_store.ensure_collection()

    async def index_vault(self) -> IndexRunSummary:
        await self.initialize()
        started_at = datetime.now(tz=UTC)
        summary = IndexRunSummary(run_id=str(uuid.uuid4()), started_at=started_at)
        markdown_paths = sorted(self.settings.vault_path.rglob("*.md"))
        current_rel_paths = {
            path.relative_to(self.settings.vault_path).as_posix() for path in markdown_paths
        }
        summary.scanned_files = len(markdown_paths)

        known_paths = await self.repository.known_paths()
        deleted_paths = sorted(known_paths - current_rel_paths)
        for rel_path in deleted_paths:
            await self._delete_file(rel_path)
            summary.deleted_files += 1

        for path in markdown_paths:
            try:
                indexed = await self.index_file(path)
                if indexed.indexed:
                    summary.indexed_files += 1
                    summary.indexed_chunks += indexed.chunk_count
                    summary.embedded_chunks += indexed.embedded_count
                    summary.reused_embeddings += indexed.reused_count
                else:
                    summary.skipped_files += 1
            except Exception as exc:
                logger.exception("Failed to index %s", path)
                summary.errors.append(f"{path}: {exc}")

        summary.finished_at = datetime.now(tz=UTC)
        logger.info(
            "Index run complete: scanned=%s indexed_files=%s skipped=%s deleted=%s chunks=%s",
            summary.scanned_files,
            summary.indexed_files,
            summary.skipped_files,
            summary.deleted_files,
            summary.indexed_chunks,
        )
        return summary

    async def index_file(self, path: Path) -> IndexedFileResult:
        if not path.exists() or path.suffix.lower() != ".md":
            return IndexedFileResult(indexed=False)
        rel_path = path.relative_to(self.settings.vault_path).as_posix()
        stat = path.stat()
        file_hash = sha256_file(path)
        existing = await self.repository.get_file(rel_path)
        if existing and existing.file_hash == file_hash and existing.mtime_ns == stat.st_mtime_ns:
            logger.debug("Skipping unchanged file %s", rel_path)
            return IndexedFileResult(indexed=False)

        parsed = self.loader.load(path, self.settings.vault_path)
        modified_time = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        chunks = self.chunker.chunk(parsed, modified_time=modified_time)
        relations = extract_relations(parsed.relative_path, parsed.raw_text, parsed.metadata)
        embeddings, embedded_count, reused_count = await self._embeddings_for_chunks(chunks)

        file_record = FileRecord(
            path=parsed.relative_path,
            file_hash=file_hash,
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
            title=parsed.title,
            metadata=parsed.metadata,
            is_memory=parsed.is_memory,
        )

        await self.vector_store.delete_file(parsed.relative_path)
        await self.repository.upsert_file(file_record)
        await self.repository.replace_chunks(parsed.relative_path, chunks)
        await self.repository.replace_relations(parsed.relative_path, relations)
        await self.vector_store.upsert_chunks(chunks, embeddings)
        logger.info(
            "Indexed %s: chunks=%s embedded=%s reused=%s relations=%s",
            parsed.relative_path,
            len(chunks),
            embedded_count,
            reused_count,
            len(relations),
        )
        return IndexedFileResult(
            indexed=True,
            chunk_count=len(chunks),
            embedded_count=embedded_count,
            reused_count=reused_count,
        )

    async def delete_file_by_relative_path(self, relative_path: str) -> None:
        await self.initialize()
        await self._delete_file(relative_path)

    async def _delete_file(self, relative_path: str) -> None:
        try:
            await self.vector_store.delete_file(relative_path)
        except Exception as exc:
            logger.warning("Qdrant cleanup failed for %s: %s", relative_path, exc)
        await self.repository.delete_file_state(relative_path)
        logger.info("Deleted index state for %s", relative_path)

    async def _embeddings_for_chunks(
        self, chunks: list[ChunkRecord]
    ) -> tuple[list[list[float]], int, int]:
        vectors: list[list[float] | None] = [None for _ in chunks]
        missing_positions: list[int] = []
        reused_count = 0
        for index, chunk in enumerate(chunks):
            cached = await self.repository.get_cached_embedding(
                chunk.chunk_hash,
                self.embedding_provider.model,
                self.embedding_provider.dimensions,
            )
            if cached is None:
                missing_positions.append(index)
            else:
                vectors[index] = cached
                reused_count += 1

        embedded_count = 0
        for start in range(0, len(missing_positions), self.settings.index_batch_size):
            batch_positions = missing_positions[start : start + self.settings.index_batch_size]
            texts = [chunks[position].text for position in batch_positions]
            batch_vectors = await self.embedding_provider.embed_texts(texts)
            for position, vector in zip(batch_positions, batch_vectors, strict=True):
                chunk = chunks[position]
                vectors[position] = vector
                await self.repository.cache_embedding(
                    chunk.chunk_hash,
                    self.embedding_provider.model,
                    self.embedding_provider.dimensions,
                    vector,
                )
                embedded_count += 1

        return (
            [[float(value) for value in vector or []] for vector in vectors],
            embedded_count,
            reused_count,
        )


class IndexedFileResult:
    def __init__(
        self,
        indexed: bool,
        chunk_count: int = 0,
        embedded_count: int = 0,
        reused_count: int = 0,
    ) -> None:
        self.indexed = indexed
        self.chunk_count = chunk_count
        self.embedded_count = embedded_count
        self.reused_count = reused_count
