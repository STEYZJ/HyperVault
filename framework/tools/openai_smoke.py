from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from framework.config import Settings
from framework.runtime import build_strategy_service
from framework.strategy.schemas import StrategyExtractionRequest, StrategySearchRequest


class OpenAISmokeResult(BaseModel):
    status: str
    reason: str | None = None
    base_url_configured: bool = False
    embedding_model: str
    strategy_llm_model: str
    qdrant_url: str
    indexed_lessons: int = 0
    insufficient_dimensions: list[str] = Field(default_factory=list)
    card_path: str | None = None
    search_hits: int = 0


async def run_openai_strategy_smoke(settings: Settings) -> OpenAISmokeResult:
    smoke_runtime = settings.runtime_path / "openai-smoke"
    smoke_vault = smoke_runtime / "vault"
    smoke_qdrant = smoke_runtime / "qdrant"
    smoke_settings = settings.model_copy(
        update={
            "runtime_path": smoke_runtime,
            "vault_path": smoke_vault,
            "qdrant_url": f"local:{smoke_qdrant}",
            "offline_test_embeddings": False,
            "strategy_llm_provider": "openai",
        }
    )
    if not smoke_settings.openai_api_key:
        return OpenAISmokeResult(
            status="skipped",
            reason="OPENAI_API_KEY is not configured in local environment or .env.",
            base_url_configured=bool(smoke_settings.openai_base_url),
            embedding_model=smoke_settings.embedding_model,
            strategy_llm_model=smoke_settings.strategy_llm_model,
            qdrant_url=smoke_settings.qdrant_url,
        )

    paper_path = write_smoke_paper(smoke_vault)
    service = build_strategy_service(
        smoke_settings,
        offline_test_embeddings=False,
        fake_strategy_llm=False,
    )
    extraction = await service.extract_paper_strategy(
        StrategyExtractionRequest(paper=str(paper_path))
    )
    search = await service.search_strategy(
        StrategySearchRequest(query="baseline selection", dimension="baseline", top_k=3)
    )
    return OpenAISmokeResult(
        status="ok",
        base_url_configured=bool(smoke_settings.openai_base_url),
        embedding_model=smoke_settings.embedding_model,
        strategy_llm_model=smoke_settings.strategy_llm_model,
        qdrant_url=smoke_settings.qdrant_url,
        indexed_lessons=len(extraction.card.lessons),
        insufficient_dimensions=[
            str(dimension) for dimension in extraction.card.insufficient_dimensions
        ],
        card_path=extraction.card_path,
        search_hits=len(search.hits),
    )


def write_smoke_paper(vault_path: Path) -> Path:
    paper_dir = vault_path / "research" / "papers"
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper = paper_dir / "openai-smoke-paper.md"
    paper.write_text(
        """---
tags: [paper, smoke]
type: paper
paper_id: openai-smoke-paper
title: OpenAI Smoke Paper
venue: Smoke
year: 2026
---
# OpenAI Smoke Paper

## Introduction

However, existing evaluations leave a gap in judging long-running agent experience reuse. The
authors package the contribution as a bounded benchmark and a reproducible failure analysis.

## Experiments

The experiment compares strong baselines, retrieval-only memory, and consolidation variants across
multiple datasets. Table 1 summarizes main results and Figure 1 shows how performance changes with
the memory budget.

## Limitations

The authors explicitly limit the claim to text-heavy workflows and present multimodal coverage as
future work.
""",
        encoding="utf-8",
    )
    return paper
