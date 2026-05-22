from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from framework.config import Settings
from framework.schemas import ChunkRecord, MetadataFilter

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection = settings.qdrant_collection
        self.client = build_async_qdrant_client(settings)

    async def ensure_collection(self) -> None:
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", self.collection)
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        indexes = {
            "file_path": PayloadSchemaType.KEYWORD,
            "is_memory": PayloadSchemaType.BOOL,
            "metadata.type": PayloadSchemaType.KEYWORD,
            "metadata.priority": PayloadSchemaType.KEYWORD,
            "metadata.tags": PayloadSchemaType.KEYWORD,
            "metadata.paper_id": PayloadSchemaType.KEYWORD,
            "metadata.venue": PayloadSchemaType.KEYWORD,
            "metadata.year": PayloadSchemaType.INTEGER,
            "metadata.verified": PayloadSchemaType.BOOL,
            "metadata.strategy_dimensions": PayloadSchemaType.KEYWORD,
            "metadata.source": PayloadSchemaType.KEYWORD,
        }
        for field_name, schema in indexes.items():
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # Existing indexes are fine across qdrant versions.
                logger.debug("Payload index %s skipped: %s", field_name, exc)

    async def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=chunk_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        await self.client.upsert(collection_name=self.collection, points=points, wait=True)

    async def delete_file(self, file_path: str) -> None:
        selector = FilterSelector(
            filter=Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))])
        )
        await self.client.delete(
            collection_name=self.collection,
            points_selector=selector,
            wait=True,
        )

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: MetadataFilter | None = None,
    ) -> list[ScoredPoint]:
        qdrant_filter = build_qdrant_filter(filters)
        if hasattr(self.client, "search"):
            return await self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
        response = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    async def collection_info(self) -> dict[str, Any]:
        info = await self.client.get_collection(self.collection)
        return info.model_dump(mode="json") if hasattr(info, "model_dump") else dict(info)

    async def count_points(self) -> int:
        result = await self.client.count(collection_name=self.collection, exact=True)
        return int(result.count)


def chunk_payload(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "file_path": chunk.file_path,
        "ordinal": chunk.ordinal,
        "chunk_hash": chunk.chunk_hash,
        "heading_path": chunk.heading_path,
        "metadata": chunk.metadata,
        "is_memory": chunk.is_memory,
        "modified_time": chunk.modified_time.isoformat() if chunk.modified_time else None,
        "text_preview": chunk.text[:500],
    }


def build_qdrant_filter(filters: MetadataFilter | None) -> Filter | None:
    if filters is None:
        return None
    must: list[FieldCondition] = []
    if filters.is_memory is not None:
        must.append(FieldCondition(key="is_memory", match=MatchValue(value=filters.is_memory)))
    if filters.paths:
        must.append(FieldCondition(key="file_path", match=MatchAny(any=filters.paths)))
    if filters.type:
        must.append(FieldCondition(key="metadata.type", match=MatchValue(value=filters.type)))
    if filters.priority:
        must.append(
            FieldCondition(key="metadata.priority", match=MatchValue(value=filters.priority))
        )
    if filters.paper_id:
        must.append(
            FieldCondition(key="metadata.paper_id", match=MatchValue(value=filters.paper_id))
        )
    if filters.venue:
        must.append(FieldCondition(key="metadata.venue", match=MatchValue(value=filters.venue)))
    if filters.year is not None:
        must.append(FieldCondition(key="metadata.year", match=MatchValue(value=filters.year)))
    if filters.verified is not None:
        must.append(
            FieldCondition(key="metadata.verified", match=MatchValue(value=filters.verified))
        )
    if filters.source:
        must.append(FieldCondition(key="metadata.source", match=MatchValue(value=filters.source)))
    if filters.strategy_dimensions:
        must.append(
            FieldCondition(
                key="metadata.strategy_dimensions",
                match=MatchAny(any=filters.strategy_dimensions),
            )
        )
    if filters.tags:
        must.append(FieldCondition(key="metadata.tags", match=MatchAny(any=filters.tags)))
    return Filter(must=must) if must else None


def build_async_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant_url == ":memory:":
        return AsyncQdrantClient(location=":memory:", timeout=30)
    if settings.qdrant_url.startswith("local:"):
        return AsyncQdrantClient(path=settings.qdrant_url.removeprefix("local:"), timeout=30)
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
    )
