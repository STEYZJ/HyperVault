from __future__ import annotations

from fastapi import FastAPI

from framework.config import get_settings
from framework.memory.service import MemoryService
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.runtime import (
    build_indexing_service,
    build_retrieval_service,
    build_strategy_service,
)
from framework.schemas import (
    MemoryConsolidationRequest,
    MemoryConsolidationResponse,
    MetadataFilter,
    SearchRequest,
    SearchResponse,
)
from framework.strategy.schemas import (
    AgentExperienceSubmitRequest,
    AgentExperienceSubmitResponse,
    HyperAgentSummaryRequest,
    HyperAgentSummaryResponse,
    PaperImportRequest,
    PaperImportResponse,
    PaperStrategyCard,
    StrategyConsolidationRequest,
    StrategyConsolidationResponse,
    StrategyExtractionRequest,
    StrategyExtractionResponse,
    StrategySearchRequest,
    StrategySearchResponse,
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


@app.post("/papers/import", response_model=PaperImportResponse)
async def import_paper(request: PaperImportRequest) -> PaperImportResponse:
    settings = get_settings()
    service = build_strategy_service(
        settings,
        settings.offline_test_embeddings,
        fake_strategy_llm=True,
    )
    return await service.import_paper(request)


@app.post("/strategy/extract", response_model=StrategyExtractionResponse)
async def extract_strategy(
    request: StrategyExtractionRequest,
) -> StrategyExtractionResponse:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings)
    return await service.extract_paper_strategy(request)


@app.post("/strategy/search", response_model=StrategySearchResponse)
async def strategy_search(request: StrategySearchRequest) -> StrategySearchResponse:
    settings = get_settings()
    service = build_strategy_service(
        settings,
        settings.offline_test_embeddings,
        fake_strategy_llm=True,
    )
    return await service.search_strategy(request)


@app.post("/strategy/consolidate", response_model=StrategyConsolidationResponse)
async def strategy_consolidate(
    request: StrategyConsolidationRequest,
) -> StrategyConsolidationResponse:
    settings = get_settings()
    service = build_strategy_service(
        settings,
        settings.offline_test_embeddings,
        fake_strategy_llm=True,
    )
    return await service.consolidate_strategy(request)


@app.get("/strategy/cards/{paper_id}", response_model=PaperStrategyCard)
async def strategy_card(paper_id: str) -> PaperStrategyCard:
    settings = get_settings()
    service = build_strategy_service(
        settings,
        settings.offline_test_embeddings,
        fake_strategy_llm=True,
    )
    return await service.strategy_report(paper_id)


@app.post("/agent-experience/submit", response_model=AgentExperienceSubmitResponse)
async def submit_agent_experience(
    request: AgentExperienceSubmitRequest,
) -> AgentExperienceSubmitResponse:
    settings = get_settings()
    service = build_strategy_service(
        settings,
        settings.offline_test_embeddings,
        fake_strategy_llm=True,
    )
    return await service.submit_agent_experience(request)


@app.post("/integrations/hyperagent/summarize", response_model=HyperAgentSummaryResponse)
async def call_hyperagent_summary(
    request: HyperAgentSummaryRequest,
) -> HyperAgentSummaryResponse:
    settings = get_settings()
    service = build_strategy_service(
        settings,
        settings.offline_test_embeddings,
        fake_strategy_llm=True,
    )
    return await service.call_hyperagent_summary(request)
