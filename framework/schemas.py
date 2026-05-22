from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

JsonDict = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


class MetadataFilter(BaseModel):
    tags: list[str] | None = None
    type: str | None = None
    priority: str | None = None
    is_memory: bool | None = None
    paths: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    filters: MetadataFilter | None = None
    semantic_weight: float = Field(default=0.72, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.22, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.06, ge=0.0, le=1.0)
    memory_boost: float = Field(default=0.18, ge=0.0, le=1.0)


class SearchHit(BaseModel):
    chunk_id: str
    file_path: str
    title: str
    text: str
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    recency_score: float = 0.0
    heading_path: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    is_memory: bool = False
    modified_time: datetime | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class IndexRunSummary(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    scanned_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    indexed_chunks: int = 0
    embedded_chunks: int = 0
    reused_embeddings: int = 0
    errors: list[str] = Field(default_factory=list)


class FileRecord(BaseModel):
    path: str
    file_hash: str
    mtime_ns: int
    size_bytes: int
    title: str
    metadata: JsonDict = Field(default_factory=dict)
    is_memory: bool = False
    updated_at: datetime = Field(default_factory=utc_now)


class ChunkRecord(BaseModel):
    chunk_id: str
    file_path: str
    ordinal: int
    chunk_hash: str
    text: str
    heading_path: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    token_count: int = 0
    is_memory: bool = False
    modified_time: datetime | None = None


class RelationRecord(BaseModel):
    source_path: str
    target: str
    relation_type: Literal["wikilink", "tag", "heading"] = "wikilink"
    context: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


class MemoryConsolidationRequest(BaseModel):
    topic: str
    query: str | None = None
    top_k: int = Field(default=8, ge=1, le=30)


class MemoryConsolidationResponse(BaseModel):
    path: str
    source_count: int
    title: str


@dataclass(frozen=True)
class ParsedMarkdown:
    path: Path
    relative_path: str
    raw_text: str
    body: str
    metadata: JsonDict
    title: str
    is_memory: bool
