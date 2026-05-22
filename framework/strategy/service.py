from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from framework.config import Settings
from framework.ingestion.markdown_loader import MarkdownLoader
from framework.rag.indexing_service import IndexingService
from framework.rag.retrieval_service import RetrievalService
from framework.schemas import MetadataFilter, SearchRequest
from framework.strategy.evidence import build_section_map, mine_strategy_evidence
from framework.strategy.hyperagent_bridge import HyperAgentBridge
from framework.strategy.llm import StrategyLLMProvider
from framework.strategy.paper_import import PaperImportService, render_markdown, slugify
from framework.strategy.quality import StrategyQualityError, StrategyQualityGate
from framework.strategy.schemas import (
    DIMENSION_LABELS,
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
    StrategyLesson,
    StrategySearchRequest,
    StrategySearchResponse,
)

logger = logging.getLogger(__name__)


class StrategyService:
    def __init__(
        self,
        settings: Settings,
        indexing_service: IndexingService,
        retrieval_service: RetrievalService,
        llm_provider: StrategyLLMProvider,
    ) -> None:
        self.settings = settings
        self.indexing_service = indexing_service
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.import_service = PaperImportService(settings)
        self.quality_gate = StrategyQualityGate()
        self.bridge = HyperAgentBridge(settings)
        self.loader = MarkdownLoader()

    async def import_paper(self, request: PaperImportRequest) -> PaperImportResponse:
        result = await self.import_service.import_paper(request)
        await self.indexing_service.initialize()
        await self.indexing_service.index_file(self.settings.vault_path / result.markdown_path)
        return result

    async def extract_paper_strategy(
        self,
        request: StrategyExtractionRequest,
    ) -> StrategyExtractionResponse:
        paper = await self._resolve_or_import_paper(request.paper)
        markdown_path = self.settings.vault_path / paper.markdown_path
        await self.indexing_service.initialize()
        await self.indexing_service.index_file(markdown_path)
        chunks = await self.indexing_service.repository.list_chunks_for_file(paper.markdown_path)
        if not chunks:
            raise RuntimeError(f"No indexed chunks found for {paper.markdown_path}")

        metadata = dict(chunks[0].metadata)
        metadata.setdefault("paper_id", paper.paper_id)
        metadata.setdefault("title", paper.title)
        metadata.setdefault("source_pdf", paper.source_pdf)
        section_map = build_section_map(paper.paper_id, chunks)
        evidence = mine_strategy_evidence(
            chunks,
            max_items=self.settings.strategy_max_evidence_items,
        )
        card = await self._synthesize_card_with_repair(
            paper_id=paper.paper_id,
            title=paper.title,
            source_path=paper.markdown_path,
            section_map=section_map,
            evidence_items=evidence,
            metadata=metadata,
        )
        card = enforce_single_paper_research_taste_boundary(card)
        card_path = await self.write_strategy_card(card)
        await self.indexing_service.index_file(self.settings.vault_path / card_path)
        return StrategyExtractionResponse(
            card=card,
            card_path=card_path,
            section_map=section_map,
            mined_evidence_count=len(evidence),
        )

    async def search_strategy(self, request: StrategySearchRequest) -> StrategySearchResponse:
        prefixes = ["summaries/paper-strategies/"]
        if request.include_memory:
            prefixes.append("memory/research-strategy/")
        filters = MetadataFilter(
            path_prefixes=prefixes,
            paper_id=request.paper_id,
            venue=request.venue,
            year=request.year,
            verified=request.verified,
            strategy_dimensions=[request.dimension] if request.dimension else None,
        )
        response = await self.retrieval_service.search(
            SearchRequest(
                query=request.query,
                top_k=request.top_k,
                filters=filters,
                memory_boost=0.28,
                research_strategy_boost=0.28,
            )
        )
        return StrategySearchResponse(query=request.query, hits=response.hits)

    async def consolidate_strategy(
        self,
        request: StrategyConsolidationRequest,
    ) -> StrategyConsolidationResponse:
        search = await self.search_strategy(
            StrategySearchRequest(
                query=request.topic,
                top_k=request.top_k,
                dimension=None if request.dimension == "research_taste" else request.dimension,
                include_memory=False,
            )
        )
        hits = [hit for hit in search.hits if hit.metadata.get("type") == "paper_strategy"]
        if not hits:
            raise RuntimeError(
                "No paper_strategy cards found for consolidation. "
                "Create strategy cards before writing long-term research memory."
            )
        source_cards = list(OrderedDict((hit.file_path, None) for hit in hits).keys())
        source_papers = {
            str(hit.metadata.get("paper_id") or hit.file_path)
            for hit in hits
            if hit.metadata.get("paper_id") or hit.file_path
        }
        if request.dimension == "research_taste" and len(source_papers) < 2:
            raise RuntimeError(
                "Research Taste consolidation requires at least two distinct paper_strategy cards."
            )

        now = datetime.now(tz=UTC)
        title = f"Research Strategy - {request.topic}"
        filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{slugify(request.topic)}.md"
        target = self.settings.research_strategy_memory_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        dimensions = sorted(
            {
                str(dimension)
                for hit in hits
                for dimension in (hit.metadata.get("strategy_dimensions") or [])
            }
        )
        if request.dimension and str(request.dimension) not in dimensions:
            dimensions.append(str(request.dimension))
        frontmatter = {
            "tags": ["memory", "research-strategy", "consolidated"],
            "type": "research_pattern",
            "priority": "high",
            "topic": request.topic,
            "strategy_dimensions": dimensions,
            "verified": False,
            "evidence_level": "multi_paper" if len(source_cards) >= 2 else "single_card",
            "created": now.isoformat(),
            "sources": source_cards,
        }
        body = render_strategy_memory_body(title, request.topic, hits)
        target.write_text(render_markdown(frontmatter, body), encoding="utf-8")
        rel_path = target.relative_to(self.settings.vault_path).as_posix()
        await self.indexing_service.initialize()
        await self.indexing_service.index_file(target)
        return StrategyConsolidationResponse(
            path=rel_path,
            title=title,
            source_count=len(hits),
            source_cards=source_cards,
        )

    async def _synthesize_card_with_repair(
        self,
        *,
        paper_id: str,
        title: str,
        source_path: str,
        section_map,
        evidence_items,
        metadata: dict,
    ) -> PaperStrategyCard:
        feedback: str | None = None
        last_error: Exception | None = None
        max_attempts = max(self.settings.strategy_llm_max_retries, 0) + 1
        for attempt in range(max_attempts):
            try:
                card = await self.llm_provider.synthesize(
                    paper_id=paper_id,
                    title=title,
                    source_path=source_path,
                    section_map=section_map,
                    evidence_items=evidence_items,
                    metadata=metadata,
                    repair_feedback=feedback,
                )
                return self.quality_gate.validate(card)
            except (json.JSONDecodeError, ValueError, StrategyQualityError) as exc:
                last_error = exc
                feedback = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Strategy synthesis attempt %s/%s failed for %s: %s",
                    attempt + 1,
                    max_attempts,
                    paper_id,
                    feedback,
                )
        raise RuntimeError(
            f"Strategy extraction failed after {max_attempts} attempts: {last_error}"
        )

    async def submit_agent_experience(
        self,
        request: AgentExperienceSubmitRequest,
    ) -> AgentExperienceSubmitResponse:
        source_text = request.content
        source_path = None
        if request.path:
            path = request.path.expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            source_path = str(path)
            source_text = path.read_text(encoding="utf-8")
        assert source_text is not None
        now = datetime.now(tz=UTC)
        title = request.title or infer_experience_title(source_text, request.source)
        filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{slugify(request.source)}-{slugify(title)}.md"
        target = self.settings.hyperagent_experience_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = {
            "tags": ["agent-experience", request.source],
            "type": "agent_experience",
            "source": request.source,
            "title": title,
            "verified": False,
            "source_path": source_path,
            "created": now.isoformat(),
            **request.metadata,
        }
        body = f"# {title}\n\n{source_text.strip()}\n"
        target.write_text(render_markdown(frontmatter, body), encoding="utf-8")
        await self.indexing_service.initialize()
        result = await self.indexing_service.index_file(target)
        rel_path = target.relative_to(self.settings.vault_path).as_posix()
        return AgentExperienceSubmitResponse(
            path=rel_path,
            title=title,
            source=request.source,
            indexed=result.indexed,
        )

    async def call_hyperagent_summary(
        self,
        request: HyperAgentSummaryRequest,
    ) -> HyperAgentSummaryResponse:
        response = await self.bridge.summarize(request)
        if not request.submit_to_vault:
            return response
        submitted = await self.submit_agent_experience(
            AgentExperienceSubmitRequest(
                source="hyperagent",
                title=f"HyperAgent Summary - {request.topic}",
                content=response.content,
                metadata={"topic": request.topic, "type": "agent_experience"},
            )
        )
        return response.model_copy(update={"submitted_path": submitted.path})

    async def strategy_report(self, paper_id: str) -> PaperStrategyCard:
        path = self.settings.paper_strategy_path / f"{paper_id}-strategy.md"
        if not path.exists():
            raise FileNotFoundError(path)
        parsed = self.loader.load(path, self.settings.vault_path)
        lessons = parse_lessons_from_card_markdown(parsed.body, parsed.metadata)
        return PaperStrategyCard(
            paper_id=paper_id,
            title=str(parsed.metadata.get("title") or path.stem),
            source_path=str(parsed.metadata.get("source_paper") or ""),
            strategy_dimensions=parsed.metadata.get("strategy_dimensions") or [],
            lessons=lessons,
            insufficient_dimensions=parsed.metadata.get("insufficient_dimensions") or [],
            venue=parsed.metadata.get("venue"),
            year=parse_year(parsed.metadata.get("year")),
            field=parsed.metadata.get("field"),
            verified=bool(parsed.metadata.get("verified")),
            evidence_level=str(parsed.metadata.get("evidence_level") or "chunk"),
            source_pdf=parsed.metadata.get("source_pdf"),
        )

    async def list_strategy_cards(
        self,
        paper_id: str | None = None,
        verified: bool | None = None,
    ) -> list[PaperStrategyCard]:
        if not self.settings.paper_strategy_path.exists():
            return []
        cards: list[PaperStrategyCard] = []
        for path in sorted(self.settings.paper_strategy_path.glob("*-strategy.md")):
            parsed = self.loader.load(path, self.settings.vault_path)
            current_paper_id = str(
                parsed.metadata.get("paper_id") or path.stem.removesuffix("-strategy")
            )
            if paper_id and current_paper_id != paper_id:
                continue
            current_verified = bool(parsed.metadata.get("verified"))
            if verified is not None and current_verified != verified:
                continue
            cards.append(await self.strategy_report(current_paper_id))
        return cards

    async def _resolve_or_import_paper(self, paper: str) -> PaperImportResponse:
        candidate = Path(paper).expanduser()
        if candidate.exists():
            return await self.import_paper(PaperImportRequest(path=candidate))

        vault_candidate = self.settings.vault_path / paper
        if vault_candidate.exists():
            return await self.import_paper(PaperImportRequest(path=vault_candidate))

        by_id = self.settings.paper_markdown_path / f"{paper}.md"
        if by_id.exists():
            parsed = self.loader.load(by_id, self.settings.vault_path)
            return PaperImportResponse(
                paper_id=str(parsed.metadata.get("paper_id") or paper),
                title=parsed.title,
                source_kind="markdown",
                markdown_path=parsed.relative_path,
                source_pdf=parsed.metadata.get("source_pdf"),
                needs_ocr=bool(parsed.metadata.get("needs_ocr", False)),
                pages=parse_year(parsed.metadata.get("pages")) or 0,
            )
        raise FileNotFoundError(f"Could not resolve paper path or paper_id: {paper}")

    async def write_strategy_card(self, card: PaperStrategyCard) -> str:
        target = self.settings.paper_strategy_path / f"{card.paper_id}-strategy.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = {
            "tags": ["research-strategy", "paper-strategy"],
            "type": "paper_strategy",
            "paper_id": card.paper_id,
            "title": f"{card.title} - Strategy Card",
            "source_paper": card.source_path,
            "source_pdf": card.source_pdf,
            "venue": card.venue,
            "year": card.year,
            "field": card.field,
            "strategy_dimensions": card.strategy_dimensions,
            "insufficient_dimensions": card.insufficient_dimensions,
            "verified": card.verified,
            "evidence_level": card.evidence_level,
            "created": card.created_at.isoformat(),
        }
        target.write_text(
            render_markdown(frontmatter, render_strategy_card_body(card)),
            encoding="utf-8",
        )
        rel_path = target.relative_to(self.settings.vault_path).as_posix()
        logger.info("Wrote paper strategy card %s", rel_path)
        return rel_path


def render_strategy_card_body(card: PaperStrategyCard) -> str:
    lines = [
        f"# {card.title} - Strategy Card",
        "",
        "## Strategy Lessons",
        "",
    ]
    for lesson in card.lessons:
        label = DIMENSION_LABELS[lesson.dimension]
        evidence = lesson.evidence_span
        lines.extend(
            [
                f"### {label}",
                "",
                f"- strategy_claim: {lesson.strategy_claim}",
                f"- why_it_works: {lesson.why_it_works}",
                f"- transferable_template: {lesson.transferable_template}",
                f"- risk_or_limit: {lesson.risk_or_limit}",
                f"- confidence: {lesson.confidence:.2f}",
                "- evidence_span:",
                f"  - source_path: {evidence.source_path}",
                f"  - chunk_id: {evidence.chunk_id or 'unknown'}",
                f"  - heading: {evidence.heading or 'unknown'}",
                f"  - page: {evidence.page or 'unknown'}",
                f"  - confidence: {evidence.confidence:.2f}",
                f"  - excerpt: {evidence.excerpt}",
                "",
            ]
        )
    if card.insufficient_dimensions:
        lines.extend(["## Insufficient Evidence", ""])
        for dimension in card.insufficient_dimensions:
            lines.append(f"- {DIMENSION_LABELS[dimension]}: insufficient_evidence")
    return "\n".join(lines).strip() + "\n"


def enforce_single_paper_research_taste_boundary(
    card: PaperStrategyCard,
) -> PaperStrategyCard:
    retained_lessons = [
        lesson for lesson in card.lessons if lesson.dimension != "research_taste"
    ]
    insufficient = list(dict.fromkeys([*card.insufficient_dimensions, "research_taste"]))
    dimensions = [lesson.dimension for lesson in retained_lessons]
    metadata = dict(card.metadata)
    metadata["research_taste_scope"] = "limited_single_paper_signal"
    return card.model_copy(
        update={
            "lessons": retained_lessons,
            "strategy_dimensions": dimensions,
            "insufficient_dimensions": insufficient,
            "metadata": metadata,
        }
    )


def render_strategy_memory_body(title: str, topic: str, hits: list) -> str:
    lines = [
        f"# {title}",
        "",
        "## Transferable Pattern",
        "",
        (
            f"For `{topic}`, prefer lessons that are explicitly backed by paper strategy cards. "
            "Use the templates below as reusable research moves, not as paper summaries."
        ),
        "",
        "## Evidence-Backed Templates",
        "",
    ]
    for hit in hits:
        lines.extend(
            [
                f"### {hit.title}",
                "",
                compact_text(hit.text),
                "",
                f"Source: [[{Path(hit.file_path).stem}]]",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def parse_lessons_from_card_markdown(body: str, metadata: dict[str, Any]) -> list[StrategyLesson]:
    raw_lessons = metadata.get("strategy_lessons") or []
    lessons: list[StrategyLesson] = []
    for raw_lesson in raw_lessons:
        try:
            lessons.append(StrategyLesson.model_validate(raw_lesson))
        except Exception:
            logger.debug("Skipping malformed strategy lesson in card metadata", exc_info=True)
    if lessons:
        return lessons
    label_to_dimension = {label: dimension for dimension, label in DIMENSION_LABELS.items()}
    sections = re.split(r"^###\s+", body, flags=re.MULTILINE)
    for section in sections[1:]:
        header, _, rest = section.partition("\n")
        dimension = label_to_dimension.get(header.strip())
        if not dimension:
            continue
        fields = parse_bullet_fields(rest)
        try:
            lessons.append(
                StrategyLesson(
                    dimension=dimension,
                    strategy_claim=fields["strategy_claim"],
                    why_it_works=fields["why_it_works"],
                    transferable_template=fields["transferable_template"],
                    risk_or_limit=fields["risk_or_limit"],
                    confidence=float(fields.get("confidence", "0.6")),
                    evidence_span={
                        "source_path": fields["source_path"],
                        "chunk_id": normalize_unknown(fields.get("chunk_id")),
                        "heading": normalize_unknown(fields.get("heading")),
                        "page": parse_page(fields.get("page")),
                        "confidence": float(fields.get("evidence_confidence", "0.6")),
                        "excerpt": fields["excerpt"],
                    },
                )
            )
        except Exception:
            logger.debug("Skipping unparsable strategy lesson section %s", header, exc_info=True)
    return lessons


def parse_bullet_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- strategy_claim:"):
            fields["strategy_claim"] = stripped.removeprefix("- strategy_claim:").strip()
        elif stripped.startswith("- why_it_works:"):
            fields["why_it_works"] = stripped.removeprefix("- why_it_works:").strip()
        elif stripped.startswith("- transferable_template:"):
            fields["transferable_template"] = stripped.removeprefix(
                "- transferable_template:"
            ).strip()
        elif stripped.startswith("- risk_or_limit:"):
            fields["risk_or_limit"] = stripped.removeprefix("- risk_or_limit:").strip()
        elif stripped.startswith("- confidence:") and "confidence" not in fields:
            fields["confidence"] = stripped.removeprefix("- confidence:").strip()
        elif stripped.startswith("- source_path:"):
            fields["source_path"] = stripped.removeprefix("- source_path:").strip()
        elif stripped.startswith("- chunk_id:"):
            fields["chunk_id"] = stripped.removeprefix("- chunk_id:").strip()
        elif stripped.startswith("- heading:"):
            fields["heading"] = stripped.removeprefix("- heading:").strip()
        elif stripped.startswith("- page:"):
            fields["page"] = stripped.removeprefix("- page:").strip()
        elif stripped.startswith("- excerpt:"):
            fields["excerpt"] = stripped.removeprefix("- excerpt:").strip()
        elif stripped.startswith("- confidence:"):
            fields["evidence_confidence"] = stripped.removeprefix("- confidence:").strip()
    fields.setdefault("evidence_confidence", fields.get("confidence", "0.6"))
    return fields


def normalize_unknown(value: str | None) -> str | None:
    if value in {None, "", "unknown"}:
        return None
    return value


def parse_page(value: str | None) -> int | None:
    if value and value.isdigit():
        return int(value)
    return None


def infer_experience_title(content: str, source: str) -> str:
    for line in content.splitlines():
        stripped = line.strip("# ").strip()
        if stripped:
            return stripped[:90]
    return f"{source} experience"


def compact_text(text: str, max_chars: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def parse_year(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
