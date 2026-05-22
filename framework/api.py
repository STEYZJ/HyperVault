from __future__ import annotations

from fastapi import FastAPI

from framework.config import get_settings
from framework.memory.service import MemoryService
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.runtime import build_indexing_service, build_retrieval_service
from framework.schemas import (
    MemoryConsolidationRequest,
    MemoryConsolidationResponse,
    MetadataFilter,
    SearchRequest,
    SearchResponse,
)
from framework.tools.verify import QdrantVerification, verify_qdrant_collection

app = FastAPI(title="HyperVault Knowledge API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "vault_path": str(settings.vault_path),
        "collection": settings.qdrant_collection,
    }


@app.post("/index/run")
async def run_index() -> dict:
    settings = get_settings()
    service = build_indexing_service(settings, settings.offline_test_embeddings)
    summary = await service.index_vault()
    return summary.model_dump(mode="json")


@app.get("/index/status")
async def index_status() -> dict:
    settings = get_settings()
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    await repository.init()
    return {
        "sqlite_path": str(settings.sqlite_path),
        "chunks": await repository.chunk_count(),
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    settings = get_settings()
    service = build_retrieval_service(settings, settings.offline_test_embeddings)
    return await service.search(request)


@app.post("/memory/search", response_model=SearchResponse)
async def memory_search(request: SearchRequest) -> SearchResponse:
    settings = get_settings()
    retrieval = build_retrieval_service(settings, settings.offline_test_embeddings)
    memory = MemoryService(settings, retrieval)
    filters = request.filters or MetadataFilter()
    filters.is_memory = True
    return await memory.search_memory(request.model_copy(update={"filters": filters}))


@app.post("/memory/consolidate", response_model=MemoryConsolidationResponse)
async def consolidate_memory(
    request: MemoryConsolidationRequest,
) -> MemoryConsolidationResponse:
    settings = get_settings()
    retrieval = build_retrieval_service(settings, settings.offline_test_embeddings)
    memory = MemoryService(settings, retrieval)
    return await memory.consolidate(request)


@app.get("/relations")
async def relations(source_path: str | None = None) -> dict:
    settings = get_settings()
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    await repository.init()
    rows = await repository.list_relations(source_path)
    return {"relations": [row.model_dump(mode="json") for row in rows]}


@app.get("/verify/qdrant", response_model=QdrantVerification)
async def verify_qdrant() -> QdrantVerification:
    return await verify_qdrant_collection(get_settings())
