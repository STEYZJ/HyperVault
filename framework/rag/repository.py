from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from framework.schemas import ChunkRecord, FileRecord, RelationRecord

logger = logging.getLogger(__name__)


class SQLiteKnowledgeRepository:
    """SQLite repository for incremental indexing state, FTS and embedding cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        self._init_sync()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_sync(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    is_memory INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    heading_path_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    is_memory INTEGER NOT NULL DEFAULT 0,
                    modified_time TEXT,
                    FOREIGN KEY(file_path) REFERENCES files(path) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
                CREATE INDEX IF NOT EXISTS idx_chunks_is_memory ON chunks(is_memory);

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    chunk_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_hash, model, dimensions)
                );

                CREATE TABLE IF NOT EXISTS relations (
                    source_path TEXT NOT NULL,
                    target TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    context TEXT,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (source_path, target, relation_type)
                );

                CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_path);
                CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target);

                CREATE TABLE IF NOT EXISTS index_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    summary_json TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    file_path UNINDEXED,
                    title,
                    text,
                    tokenize='porter unicode61'
                );
                """
            )

    async def get_file(self, path: str) -> FileRecord | None:
        return self._get_file_sync(path)

    def _get_file_sync(self, path: str) -> FileRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return file_record_from_row(row) if row else None

    async def known_paths(self) -> set[str]:
        return self._known_paths_sync()

    def _known_paths_sync(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM files").fetchall()
        return {str(row["path"]) for row in rows}

    async def upsert_file(self, record: FileRecord) -> None:
        self._upsert_file_sync(record)

    def _upsert_file_sync(self, record: FileRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files(path, file_hash, mtime_ns, size_bytes, title, metadata_json,
                                  is_memory, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_hash=excluded.file_hash,
                    mtime_ns=excluded.mtime_ns,
                    size_bytes=excluded.size_bytes,
                    title=excluded.title,
                    metadata_json=excluded.metadata_json,
                    is_memory=excluded.is_memory,
                    updated_at=excluded.updated_at
                """,
                (
                    record.path,
                    record.file_hash,
                    record.mtime_ns,
                    record.size_bytes,
                    record.title,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    int(record.is_memory),
                    record.updated_at.isoformat(),
                ),
            )

    async def replace_chunks(self, file_path: str, chunks: list[ChunkRecord]) -> None:
        self._replace_chunks_sync(file_path, chunks)

    def _replace_chunks_sync(self, file_path: str, chunks: list[ChunkRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM chunks_fts WHERE file_path = ?", (file_path,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunks(chunk_id, file_path, ordinal, chunk_hash, text,
                                       heading_path_json, metadata_json, token_count,
                                       is_memory, modified_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.file_path,
                        chunk.ordinal,
                        chunk.chunk_hash,
                        chunk.text,
                        json.dumps(chunk.heading_path, ensure_ascii=False),
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        chunk.token_count,
                        int(chunk.is_memory),
                        chunk.modified_time.isoformat() if chunk.modified_time else None,
                    ),
                )
                conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, file_path, title, text) VALUES (?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.file_path,
                        str(chunk.metadata.get("title") or ""),
                        chunk.text,
                    ),
                )

    async def replace_relations(self, source_path: str, relations: list[RelationRecord]) -> None:
        self._replace_relations_sync(source_path, relations)

    def _replace_relations_sync(self, source_path: str, relations: list[RelationRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM relations WHERE source_path = ?", (source_path,))
            for relation in relations:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO relations(source_path, target, relation_type, context,
                                                     metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        relation.source_path,
                        relation.target,
                        relation.relation_type,
                        relation.context,
                        json.dumps(relation.metadata, ensure_ascii=False, sort_keys=True),
                    ),
                )

    async def list_chunk_ids_for_file(self, file_path: str) -> list[str]:
        return self._list_chunk_ids_for_file_sync(file_path)

    def _list_chunk_ids_for_file_sync(self, file_path: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE file_path = ? ORDER BY ordinal", (file_path,)
            ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    async def delete_file_state(self, file_path: str) -> None:
        self._delete_file_state_sync(file_path)

    def _delete_file_state_sync(self, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM relations WHERE source_path = ?", (file_path,))
            conn.execute("DELETE FROM files WHERE path = ?", (file_path,))

    async def get_cached_embedding(
        self, chunk_hash: str, model: str, dimensions: int
    ) -> list[float] | None:
        return self._get_cached_embedding_sync(chunk_hash, model, dimensions)

    def _get_cached_embedding_sync(
        self, chunk_hash: str, model: str, dimensions: int
    ) -> list[float] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT vector_json FROM embedding_cache
                WHERE chunk_hash = ? AND model = ? AND dimensions = ?
                """,
                (chunk_hash, model, dimensions),
            ).fetchone()
        if not row:
            return None
        return [float(value) for value in json.loads(row["vector_json"])]

    async def cache_embedding(
        self, chunk_hash: str, model: str, dimensions: int, vector: list[float]
    ) -> None:
        self._cache_embedding_sync(chunk_hash, model, dimensions, vector)

    def _cache_embedding_sync(
        self, chunk_hash: str, model: str, dimensions: int, vector: list[float]
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache(chunk_hash, model, dimensions, vector_json,
                                                       created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk_hash,
                    model,
                    dimensions,
                    json.dumps(vector),
                    datetime.now(tz=UTC).isoformat(),
                ),
            )

    async def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, ChunkRecord]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        return self._get_chunks_sync(ids)

    def _get_chunks_sync(self, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
            ).fetchall()
        return {str(row["chunk_id"]): chunk_record_from_row(row) for row in rows}

    async def list_chunks_for_file(self, file_path: str) -> list[ChunkRecord]:
        return self._list_chunks_for_file_sync(file_path)

    def _list_chunks_for_file_sync(self, file_path: str) -> list[ChunkRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE file_path = ? ORDER BY ordinal",
                (file_path,),
            ).fetchall()
        return [chunk_record_from_row(row) for row in rows]

    async def lexical_search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        return self._lexical_search_sync(query, limit)

    def _lexical_search_sync(self, query: str, limit: int) -> list[tuple[str, float]]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT chunk_id, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS query failed for %r: %s", query, exc)
            return []
        return [(str(row["chunk_id"]), 1.0 / (1.0 + abs(float(row["rank"])))) for row in rows]

    async def list_relations(self, source_path: str | None = None) -> list[RelationRecord]:
        return self._list_relations_sync(source_path)

    def _list_relations_sync(self, source_path: str | None) -> list[RelationRecord]:
        sql = "SELECT * FROM relations"
        params: tuple[Any, ...] = ()
        if source_path:
            sql += " WHERE source_path = ?"
            params = (source_path,)
        sql += " ORDER BY source_path, relation_type, target"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            RelationRecord(
                source_path=str(row["source_path"]),
                target=str(row["target"]),
                relation_type=row["relation_type"],
                context=row["context"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    async def chunk_count(self) -> int:
        return self._chunk_count_sync()

    def _chunk_count_sync(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])


def file_record_from_row(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        path=str(row["path"]),
        file_hash=str(row["file_hash"]),
        mtime_ns=int(row["mtime_ns"]),
        size_bytes=int(row["size_bytes"]),
        title=str(row["title"]),
        metadata=json.loads(row["metadata_json"]),
        is_memory=bool(row["is_memory"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def chunk_record_from_row(row: sqlite3.Row) -> ChunkRecord:
    modified_time = row["modified_time"]
    return ChunkRecord(
        chunk_id=str(row["chunk_id"]),
        file_path=str(row["file_path"]),
        ordinal=int(row["ordinal"]),
        chunk_hash=str(row["chunk_hash"]),
        text=str(row["text"]),
        heading_path=json.loads(row["heading_path_json"]),
        metadata=json.loads(row["metadata_json"]),
        token_count=int(row["token_count"]),
        is_memory=bool(row["is_memory"]),
        modified_time=datetime.fromisoformat(modified_time) if modified_time else None,
    )


def build_fts_query(query: str) -> str:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_]+", query)]
    tokens = [token for token in tokens if len(token) >= 2]
    return " OR ".join(f"{token}*" for token in tokens[:12])
