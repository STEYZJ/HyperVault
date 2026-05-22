from __future__ import annotations

from pydantic import BaseModel

from framework.config import Settings
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.vector_store import QdrantVectorStore


class QdrantVerification(BaseModel):
    collection: str
    exists: bool
    points_count: int = 0
    local_chunks_count: int = 0
    vector_size: int | None = None
    distance: str | None = None
    payload_schema_fields: list[str] = []


async def verify_qdrant_collection(settings: Settings) -> QdrantVerification:
    vector_store = QdrantVectorStore(settings)
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    await repository.init()
    exists = await vector_store.client.collection_exists(settings.qdrant_collection)
    if not exists:
        return QdrantVerification(
            collection=settings.qdrant_collection,
            exists=False,
            local_chunks_count=await repository.chunk_count(),
        )
    info = await vector_store.collection_info()
    vectors = info.get("config", {}).get("params", {}).get("vectors", {})
    vector_size = vectors.get("size") if isinstance(vectors, dict) else None
    distance = vectors.get("distance") if isinstance(vectors, dict) else None
    payload_schema = info.get("payload_schema") or {}
    return QdrantVerification(
        collection=settings.qdrant_collection,
        exists=True,
        points_count=await vector_store.count_points(),
        local_chunks_count=await repository.chunk_count(),
        vector_size=vector_size,
        distance=distance,
        payload_schema_fields=sorted(payload_schema.keys()),
    )

