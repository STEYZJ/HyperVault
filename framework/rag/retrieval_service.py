from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from framework.rag.embeddings import EmbeddingProvider
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.vector_store import QdrantVectorStore
from framework.schemas import ChunkRecord, MetadataFilter, SearchHit, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        repository: SQLiteKnowledgeRepository,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def search(self, request: SearchRequest) -> SearchResponse:
        await self.repository.init()
        query_vector = await self.embedding_provider.embed_query(request.query)
        semantic_results = await self._semantic_search(query_vector, request)
        lexical_results = await self.repository.lexical_search(
            request.query,
            limit=request.top_k * 8,
        )

        semantic_scores = {chunk_id: score for chunk_id, score in semantic_results}
        lexical_scores = {chunk_id: score for chunk_id, score in lexical_results}
        candidate_ids = list(dict.fromkeys([*semantic_scores.keys(), *lexical_scores.keys()]))
        chunks = await self.repository.get_chunks(candidate_ids)

        hits: list[SearchHit] = []
        for chunk_id in candidate_ids:
            chunk = chunks.get(chunk_id)
            if not chunk or not metadata_matches(chunk, request.filters):
                continue
            semantic_score = semantic_scores.get(chunk_id, 0.0)
            lexical_score = lexical_scores.get(chunk_id, 0.0)
            recency_score = compute_recency_score(chunk)
            score = (
                semantic_score * request.semantic_weight
                + lexical_score * request.lexical_weight
                + recency_score * request.recency_weight
            )
            if chunk.is_memory:
                score += request.memory_boost
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    title=str(chunk.metadata.get("title") or chunk.file_path),
                    text=chunk.text,
                    score=score,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    recency_score=recency_score,
                    heading_path=chunk.heading_path,
                    metadata=chunk.metadata,
                    is_memory=chunk.is_memory,
                    modified_time=chunk.modified_time,
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return SearchResponse(query=request.query, hits=hits[: request.top_k])

    async def _semantic_search(
        self, query_vector: list[float], request: SearchRequest
    ) -> list[tuple[str, float]]:
        try:
            points = await self.vector_store.search(
                query_vector=query_vector,
                limit=request.top_k * 8,
                filters=request.filters,
            )
        except Exception as exc:
            logger.warning("Semantic search unavailable, falling back to lexical only: %s", exc)
            return []
        results: list[tuple[str, float]] = []
        for point in points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point.id)
            results.append((chunk_id, float(point.score or 0.0)))
        return results


def metadata_matches(chunk: ChunkRecord, filters: MetadataFilter | None) -> bool:
    if filters is None:
        return True
    metadata = chunk.metadata
    if filters.is_memory is not None and chunk.is_memory != filters.is_memory:
        return False
    if filters.paths and chunk.file_path not in set(filters.paths):
        return False
    if filters.type and metadata.get("type") != filters.type:
        return False
    if filters.priority and metadata.get("priority") != filters.priority:
        return False
    if filters.tags:
        existing = {str(tag) for tag in metadata.get("tags") or []}
        if not existing.intersection(filters.tags):
            return False
    return True


def compute_recency_score(chunk: ChunkRecord) -> float:
    if not chunk.modified_time:
        return 0.0
    now = datetime.now(tz=UTC)
    modified = chunk.modified_time
    if modified.tzinfo is None:
        modified = modified.replace(tzinfo=UTC)
    age_days = max((now - modified).total_seconds() / 86400.0, 0.0)
    return math.exp(-age_days / 365.0)
