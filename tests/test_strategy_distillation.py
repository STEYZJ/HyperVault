from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.config import Settings
from framework.rag.embeddings import DeterministicEmbeddingProvider
from framework.rag.indexing_service import IndexingService
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.retrieval_service import RetrievalService
from framework.rag.vector_store import chunk_payload
from framework.strategy.hyperagent_bridge import HyperAgentBridge
from framework.strategy.llm import FakeStrategyLLMProvider
from framework.strategy.paper_import import PaperImportService
from framework.strategy.quality import StrategyQualityError, StrategyQualityGate
from framework.strategy.schemas import (
    EvidenceSpan,
    HyperAgentSummaryRequest,
    PaperImportRequest,
    PaperStrategyCard,
    StrategyConsolidationRequest,
    StrategyExtractionRequest,
    StrategyLesson,
    StrategySearchRequest,
)
from framework.strategy.service import StrategyService


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, dict] = {}

    async def ensure_collection(self) -> None:
        return None

    async def delete_file(self, file_path: str) -> None:
        self.points = {
            point_id: payload
            for point_id, payload in self.points.items()
            if payload["file_path"] != file_path
        }

    async def upsert_chunks(self, chunks, embeddings) -> None:
        for chunk in chunks:
            self.points[chunk.chunk_id] = chunk_payload(chunk)

    async def search(self, query_vector, limit, filters=None):
        return [
            SimpleNamespace(id=point_id, payload=payload, score=0.7)
            for point_id, payload in list(self.points.items())[:limit]
        ]


def build_strategy_service(tmp_path: Path) -> StrategyService:
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    vault.mkdir()
    settings = Settings(
        vault_path=vault,
        runtime_path=runtime,
        embedding_dim=64,
        strategy_llm_provider="fake",
    )
    repository = SQLiteKnowledgeRepository(settings.sqlite_path)
    vector_store = FakeVectorStore()
    provider = DeterministicEmbeddingProvider(dimensions=64)
    indexing = IndexingService(settings, repository, vector_store, provider)  # type: ignore[arg-type]
    retrieval = RetrievalService(repository, vector_store, provider)  # type: ignore[arg-type]
    return StrategyService(settings, indexing, retrieval, FakeStrategyLLMProvider())


def write_fixture_paper(path: Path) -> None:
    path.write_text(
        """---
tags: [paper]
type: paper
paper_id: fixture-paper
title: Fixture Paper
venue: TestConf
year: 2026
---
# Fixture Paper

## Introduction

However, existing work leaves a gap in long-running agent evaluation. To our knowledge, no
benchmark isolates this problem while keeping the protocol simple.

## Experiments

We compare with strong baseline systems and evaluate across multiple datasets. Figure 1 visualizes
the trend and Table 1 reports consistent results.

## Ablation And Limitations

The ablation removes one component at a time. The authors present the multimodal limitation as
future work and keep the central claim bounded to text-heavy workflows.
""",
        encoding="utf-8",
    )


def test_strategy_quality_gate_rejects_summary_only_claim() -> None:
    evidence = EvidenceSpan(
        source_path="paper.md",
        chunk_id="c1",
        excerpt="The paper proposes a new method.",
        confidence=0.7,
    )
    card = PaperStrategyCard(
        paper_id="p",
        title="P",
        source_path="paper.md",
        lessons=[
            StrategyLesson(
                dimension="novelty_construction",
                strategy_claim="The paper proposes a new method",
                why_it_works="It gives the reviewer a method detail.",
                evidence_span=evidence,
                transferable_template="State the method and show a result.",
                risk_or_limit="summary only",
                confidence=0.6,
            )
        ],
    )
    with pytest.raises(StrategyQualityError):
        StrategyQualityGate().validate(card)


@pytest.mark.asyncio
async def test_markdown_strategy_extraction_search_and_consolidation(tmp_path: Path) -> None:
    service = build_strategy_service(tmp_path)
    source = tmp_path / "fixture-paper.md"
    write_fixture_paper(source)

    extraction = await service.extract_paper_strategy(
        StrategyExtractionRequest(paper=str(source))
    )
    assert extraction.card.paper_id == "fixture-paper"
    assert extraction.card.lessons
    assert all(lesson.evidence_span.chunk_id for lesson in extraction.card.lessons)
    assert extraction.card_path.startswith("summaries/paper-strategies/")

    search = await service.search_strategy(
        StrategySearchRequest(
            query="baseline selection",
            dimension="baseline",
            top_k=5,
        )
    )
    assert any(hit.metadata.get("type") == "paper_strategy" for hit in search.hits)

    cards = await service.list_strategy_cards()
    assert [card.paper_id for card in cards] == ["fixture-paper"]

    memory = await service.consolidate_strategy(
        StrategyConsolidationRequest(topic="baseline selection", top_k=5)
    )
    assert memory.path.startswith("memory/research-strategy/")
    assert memory.source_count >= 1


@pytest.mark.asyncio
async def test_agent_experience_submit_and_hyperagent_fake_runner(tmp_path: Path) -> None:
    service = build_strategy_service(tmp_path)
    script = tmp_path / "fake_hyperagent.py"
    script.write_text(
        "#!/usr/bin/env python\n"
        "import sys\n"
        "print('HyperAgent lesson: bound weak claims with explicit evidence.')\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    service.settings.hyperagent_cli = script
    service.bridge = HyperAgentBridge(service.settings)

    response = await service.call_hyperagent_summary(
        HyperAgentSummaryRequest(topic="claim boundary")
    )
    assert "HyperAgent lesson" in response.content
    assert response.submitted_path
    assert response.submitted_path.startswith("research/hyperagent-experience/")


@pytest.mark.asyncio
async def test_pdf_import_extracts_text_and_captions(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    vault = tmp_path / "vault"
    settings = Settings(vault_path=vault, runtime_path=tmp_path / "runtime")
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Figure 1: Evidence flow. The experiment compares baselines.")
    document.save(pdf_path)
    document.close()

    response = await PaperImportService(settings).import_paper(
        PaperImportRequest(path=pdf_path, paper_id="pdf-fixture")
    )
    assert response.source_kind == "pdf"
    assert response.markdown_path == "research/papers/pdf-fixture.md"
    assert response.figure_table_refs


def test_strategy_dimension_aliases_are_normalized() -> None:
    assert StrategySearchRequest(query="gap", dimension="problem_gap").dimension == (
        "problem_gap_framing"
    )
    assert StrategyConsolidationRequest(topic="ablation", dimension="ablation").dimension == (
        "ablation_logic"
    )
