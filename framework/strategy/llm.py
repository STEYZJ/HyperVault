from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Protocol

from openai import AsyncOpenAI

from framework.config import Settings
from framework.strategy.schemas import (
    ALL_STRATEGY_DIMENSIONS,
    DIMENSION_LABELS,
    EvidenceSpan,
    MinedEvidenceItem,
    PaperStrategyCard,
    SectionMap,
    StrategyDimension,
    StrategyLesson,
)

logger = logging.getLogger(__name__)


class StrategyLLMProvider(Protocol):
    async def synthesize(
        self,
        *,
        paper_id: str,
        title: str,
        source_path: str,
        section_map: SectionMap,
        evidence_items: list[MinedEvidenceItem],
        metadata: dict,
    ) -> PaperStrategyCard:
        ...


class FakeStrategyLLMProvider:
    """Deterministic offline extractor for tests and no-key smoke validation."""

    async def synthesize(
        self,
        *,
        paper_id: str,
        title: str,
        source_path: str,
        section_map: SectionMap,
        evidence_items: list[MinedEvidenceItem],
        metadata: dict,
    ) -> PaperStrategyCard:
        grouped: defaultdict[StrategyDimension, list[MinedEvidenceItem]] = defaultdict(list)
        for item in evidence_items:
            grouped[item.dimension].append(item)

        lessons: list[StrategyLesson] = []
        insufficient: list[StrategyDimension] = []
        for dimension in ALL_STRATEGY_DIMENSIONS:
            candidates = grouped.get(dimension, [])
            if not candidates:
                insufficient.append(dimension)
                continue
            item = candidates[0]
            lessons.append(build_fake_lesson(dimension, item.evidence_span))

        return PaperStrategyCard(
            paper_id=paper_id,
            title=title,
            source_path=source_path,
            strategy_dimensions=[lesson.dimension for lesson in lessons],
            lessons=lessons,
            insufficient_dimensions=insufficient,
            venue=metadata.get("venue"),
            year=parse_year(metadata.get("year")),
            field=metadata.get("field"),
            verified=False,
            evidence_level="chunk",
            source_pdf=metadata.get("source_pdf"),
            metadata={"section_count": len(section_map.sections), "llm_provider": "fake"},
        )


class OpenAIStrategyLLMProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when STRATEGY_LLM_PROVIDER=openai. "
                "Use STRATEGY_LLM_PROVIDER=fake for offline validation."
            )
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def synthesize(
        self,
        *,
        paper_id: str,
        title: str,
        source_path: str,
        section_map: SectionMap,
        evidence_items: list[MinedEvidenceItem],
        metadata: dict,
    ) -> PaperStrategyCard:
        payload = {
            "paper_id": paper_id,
            "title": title,
            "source_path": source_path,
            "metadata": metadata,
            "section_map": section_map.model_dump(mode="json"),
            "evidence_items": [
                item.model_dump(mode="json")
                for item in evidence_items[: self.settings.strategy_max_evidence_items]
            ],
            "required_dimensions": list(ALL_STRATEGY_DIMENSIONS),
        }
        response = await self.client.chat.completions.create(
            model=self.settings.strategy_llm_model,
            temperature=self.settings.strategy_extraction_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        data.setdefault("paper_id", paper_id)
        data.setdefault("title", title)
        data.setdefault("source_path", source_path)
        data.setdefault("verified", False)
        data.setdefault("evidence_level", "chunk")
        data.setdefault("source_pdf", metadata.get("source_pdf"))
        data.setdefault("metadata", {})
        data["metadata"]["llm_provider"] = "openai"
        card = PaperStrategyCard.model_validate(data)
        logger.info(
            "OpenAI strategy extraction produced %s lessons for %s",
            len(card.lessons),
            paper_id,
        )
        return card


SYSTEM_PROMPT = """You extract transferable research strategy, not paper summaries.
Return one JSON object matching this shape:
{
  "paper_id": "...",
  "title": "...",
  "source_path": "...",
  "strategy_dimensions": ["contribution_packaging"],
  "lessons": [{
    "dimension": "contribution_packaging",
    "strategy_claim": "How the authors frame, package, prove, persuade, or avoid weakness.",
    "why_it_works": "Why this move satisfies reviewer expectations.",
    "evidence_span": {
      "source_path": "...",
      "chunk_id": "...",
      "heading": "...",
      "page": 1,
      "excerpt": "short source excerpt",
      "confidence": 0.7
    },
    "transferable_template": "A reusable writing or experiment design template.",
    "risk_or_limit": "When this move can fail.",
    "confidence": 0.7
  }],
  "insufficient_dimensions": ["research_taste"],
  "venue": null,
  "year": null,
  "field": null,
  "verified": false,
  "evidence_level": "chunk"
}
Rules:
- Do not say only what method the paper proposes.
- Every lesson must be about author strategy, experiment logic, storytelling, reviewer persuasion,
  credibility building, weakness handling, or research taste.
- Every lesson must reuse one provided evidence item. If evidence is missing, put that dimension in
  insufficient_dimensions instead of inventing.
- Research taste from a single paper must be cautious and marked as limited if used.
"""


def build_strategy_llm_provider(
    settings: Settings,
    force_fake: bool = False,
) -> StrategyLLMProvider:
    provider = settings.strategy_llm_provider.lower()
    if force_fake or provider in {"fake", "offline", "deterministic"}:
        return FakeStrategyLLMProvider()
    if provider == "openai":
        return OpenAIStrategyLLMProvider(settings)
    raise ValueError(f"Unsupported STRATEGY_LLM_PROVIDER={settings.strategy_llm_provider}")


def build_fake_lesson(dimension: StrategyDimension, evidence: EvidenceSpan) -> StrategyLesson:
    label = DIMENSION_LABELS[dimension]
    claim_templates: dict[StrategyDimension, str] = {
        "contribution_packaging": (
            "作者把贡献包装成 reviewer 能快速核验的少数清晰承诺，而不是散乱的方法细节。"
        ),
        "baseline_selection_logic": "作者通过选择强 baseline 和可解释对照来降低公平性疑虑。",
        "novelty_construction": "作者把 novelty 构造成现有路径的明确缺口补位，而不是孤立的新模块。",
        "problem_gap_framing": "作者先框定问题缺口，再把方法定位成对该缺口的直接回应。",
        "figure_table_expression": "作者用图表把 claim 的证据路径前置，让读者先看到可验证结构。",
        "experiment_arrangement": "作者按主结果、控制变量、稳健性递进安排实验，逐步关闭质疑空间。",
        "reviewer_concern_handling": "作者显式处理 reviewer 可能关心的泛化、公平或稳健性问题。",
        "credibility_building": "作者通过跨数据、跨设置或一致性证据建立可信度。",
        "weakness_avoidance": "作者把弱点边界化为 scope 或 future work，减少其对主 claim 的伤害。",
        "storytelling_moves": "作者用递进叙事把动机、方法和证据串成可跟随的说服链。",
        "claim_organization": "作者把 claim 组织成先可观察、再可解释、最后可推广的层次。",
        "ablation_logic": "作者用 ablation 把方法贡献拆成 reviewer 能接受的因果证据。",
        "research_taste": "作者选择的问题呈现出趋势、benchmark 或社区关注度带来的研究时机。",
    }
    return StrategyLesson(
        dimension=dimension,
        strategy_claim=claim_templates[dimension],
        why_it_works=(
            f"{label} 的证据让 reviewer 能把论文判断从“是否喜欢方法”转移到"
            "“作者是否证明了一个明确研究承诺”。"
        ),
        evidence_span=evidence,
        transferable_template=(
            "写作时先写出缺口或质疑点，再安排一个直接回应它的证据单元，"
            "最后把 claim 限定在该证据真正支持的范围内。"
        ),
        risk_or_limit=(
            "如果证据只展示方法效果而没有控制变量或边界说明，这个策略会退化成普通内容总结。"
        ),
        confidence=min(evidence.confidence, 0.74),
    )


def parse_year(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
