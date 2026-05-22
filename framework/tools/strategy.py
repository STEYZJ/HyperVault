from __future__ import annotations

from framework.config import get_settings
from framework.runtime import build_strategy_service
from framework.strategy.schemas import (
    AgentExperienceSubmitRequest,
    HyperAgentSummaryRequest,
    PaperImportRequest,
    StrategyConsolidationRequest,
    StrategyExtractionRequest,
    StrategySearchRequest,
)


async def hypervault_import_paper(path: str, paper_id: str | None = None) -> dict:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings, True)
    response = await service.import_paper(PaperImportRequest(path=path, paper_id=paper_id))
    return response.model_dump(mode="json")


async def hypervault_extract_paper_strategy(paper: str, fake_llm: bool = False) -> dict:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings, fake_llm)
    response = await service.extract_paper_strategy(StrategyExtractionRequest(paper=paper))
    return response.model_dump(mode="json")


async def hypervault_strategy_search(
    query: str,
    dimension: str | None = None,
    top_k: int = 8,
) -> dict:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings, True)
    response = await service.search_strategy(
        StrategySearchRequest(query=query, dimension=dimension, top_k=top_k)
    )
    return response.model_dump(mode="json")


async def hypervault_consolidate_strategy(
    topic: str,
    dimension: str | None = None,
    top_k: int = 8,
) -> dict:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings, True)
    response = await service.consolidate_strategy(
        StrategyConsolidationRequest(topic=topic, dimension=dimension, top_k=top_k)
    )
    return response.model_dump(mode="json")


async def hypervault_submit_agent_experience(
    source: str,
    content: str,
    title: str | None = None,
) -> dict:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings, True)
    response = await service.submit_agent_experience(
        AgentExperienceSubmitRequest(source=source, title=title, content=content)
    )
    return response.model_dump(mode="json")


async def hypervault_call_hyperagent_summary(topic: str) -> dict:
    settings = get_settings()
    service = build_strategy_service(settings, settings.offline_test_embeddings, True)
    response = await service.call_hyperagent_summary(HyperAgentSummaryRequest(topic=topic))
    return response.model_dump(mode="json")
