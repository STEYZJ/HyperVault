from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.config import Settings
from framework.rag.embeddings import DeterministicEmbeddingProvider, OpenAIEmbeddingProvider
from framework.rag.indexing_service import IndexingService
from framework.rag.repository import SQLiteKnowledgeRepository
from framework.rag.retrieval_service import RetrievalService
from framework.rag.vector_store import chunk_payload
from framework.strategy.hyperagent_bridge import HyperAgentBridge
from framework.strategy.llm import FakeStrategyLLMProvider, OpenAIStrategyLLMProvider
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
from framework.tools.openai_smoke import run_openai_strategy_smoke
from framework.tools.preflight import docker_preflight
from framework.tools.security import scan_file


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
    assert all(lesson.dimension != "research_taste" for lesson in extraction.card.lessons)
    assert "research_taste" in extraction.card.insufficient_dimensions
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

    with pytest.raises(RuntimeError, match="Research Taste"):
        await service.consolidate_strategy(
            StrategyConsolidationRequest(topic="Fixture Paper", dimension="research_taste")
        )


@pytest.mark.asyncio
async def test_agent_experience_submit_and_hyperagent_fake_runner(tmp_path: Path) -> None:
    service = build_strategy_service(tmp_path)
    script = tmp_path / "fake_hyperagent.py"
    script.write_text(
        "#!/usr/bin/env python\n"
        "import sys\n"
        "print('HyperAgent lesson: bound weak claims with explicit evidence.')\n"
        "print('argv=' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    service.settings.hyperagent_cli = script
    service.bridge = HyperAgentBridge(service.settings)

    response = await service.call_hyperagent_summary(
        HyperAgentSummaryRequest(topic="claim boundary")
    )
    assert "HyperAgent lesson" in response.content
    assert "research search" in response.content
    assert response.submitted_path
    assert response.submitted_path.startswith("research/hyperagent-experience/")


@pytest.mark.asyncio
async def test_pdf_import_extracts_text_and_captions(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    vault = tmp_path / "vault"
    settings = Settings(
        vault_path=vault,
        runtime_path=tmp_path / "runtime",
        export_figure_pages=True,
    )
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
    assert response.figure_table_refs[0].asset_path
    assert (vault / response.figure_table_refs[0].asset_path).exists()


def test_strategy_dimension_aliases_are_normalized() -> None:
    assert StrategySearchRequest(query="gap", dimension="problem_gap").dimension == (
        "problem_gap_framing"
    )
    assert StrategyConsolidationRequest(topic="ablation", dimension="ablation").dimension == (
        "ablation_logic"
    )


def test_openai_base_url_is_passed_to_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "sk" + "-" + ("baseurltest" * 3)
    settings = Settings(
        openai_api_key=key,
        openai_base_url="https://example.invalid/v1",
    )
    captured: list[dict] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr("framework.rag.embeddings.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("framework.strategy.llm.AsyncOpenAI", FakeAsyncOpenAI)

    OpenAIEmbeddingProvider(settings)
    OpenAIStrategyLLMProvider(settings)

    assert captured == [
        {"api_key": key, "base_url": "https://example.invalid/v1"},
        {"api_key": key, "base_url": "https://example.invalid/v1"},
    ]


def test_secret_scan_redacts_findings(tmp_path: Path) -> None:
    key = "sk" + "-" + ("redactedtest" * 3)
    key_name = "OPENAI" + "_API_KEY"
    source = tmp_path / "tracked.txt"
    source.write_text(f"{key_name}={key}\n", encoding="utf-8")

    findings = scan_file(source, tmp_path)

    assert findings
    assert key not in findings[0].excerpt
    assert "***REDACTED***" in findings[0].excerpt


def test_docker_preflight_reports_missing_socket(tmp_path: Path) -> None:
    result = docker_preflight(socket_path=tmp_path / "missing.sock", project_dir=tmp_path)

    assert not result.ok
    assert not result.socket_exists
    assert result.guidance


@pytest.mark.asyncio
async def test_openai_smoke_skips_without_key(tmp_path: Path) -> None:
    settings = Settings(
        vault_path=tmp_path / "vault",
        runtime_path=tmp_path / "runtime",
        openai_api_key=None,
    )

    result = await run_openai_strategy_smoke(settings)

    assert result.status == "skipped"
    assert "OPENAI_API_KEY" in str(result.reason)


@pytest.mark.asyncio
async def test_strategy_synthesis_repairs_after_bad_first_attempt(tmp_path: Path) -> None:
    service = build_strategy_service(tmp_path)
    source = tmp_path / "fixture-paper.md"
    write_fixture_paper(source)

    class RepairingProvider:
        def __init__(self) -> None:
            self.attempts = 0

        async def synthesize(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                evidence = kwargs["evidence_items"][0].evidence_span
                return PaperStrategyCard(
                    paper_id=kwargs["paper_id"],
                    title=kwargs["title"],
                    source_path=kwargs["source_path"],
                    lessons=[
                        StrategyLesson(
                            dimension="novelty_construction",
                            strategy_claim="The paper proposes a new method",
                            why_it_works="It gives a method detail.",
                            evidence_span=evidence,
                            transferable_template="State the method and show a result.",
                            risk_or_limit="summary only",
                            confidence=0.6,
                        )
                    ],
                )
            return await FakeStrategyLLMProvider().synthesize(**kwargs)

    provider = RepairingProvider()
    service.llm_provider = provider

    extraction = await service.extract_paper_strategy(
        StrategyExtractionRequest(paper=str(source))
    )

    assert provider.attempts == 2
    assert extraction.card.lessons
