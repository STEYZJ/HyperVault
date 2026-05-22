from __future__ import annotations

import re

from framework.strategy.schemas import PaperStrategyCard, StrategyLesson

SUMMARY_ONLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(本文|论文|paper|the paper)\s*(提出|proposes|introduces|presents)", re.I),
    re.compile(r"^\s*(we|authors?)\s*(propose|introduce|present)\s+(a|an|the)\s+\w+", re.I),
)

STRATEGY_VERBS = (
    "package",
    "frame",
    "position",
    "justify",
    "sequence",
    "contrast",
    "bound",
    "hide",
    "convert",
    "说服",
    "包装",
    "构造",
    "组织",
    "规避",
    "淡化",
    "铺垫",
    "建立",
)


class StrategyQualityError(ValueError):
    pass


class StrategyQualityGate:
    """Reject cards that drift into paper summaries or unsupported claims."""

    def validate(self, card: PaperStrategyCard) -> PaperStrategyCard:
        if not card.lessons:
            raise StrategyQualityError("No evidence-backed strategy lessons were produced")
        for lesson in card.lessons:
            self._validate_lesson(lesson)
        return card

    def _validate_lesson(self, lesson: StrategyLesson) -> None:
        evidence = lesson.evidence_span
        if not evidence.excerpt.strip():
            raise StrategyQualityError(f"{lesson.dimension} has empty evidence excerpt")
        if evidence.confidence <= 0:
            raise StrategyQualityError(f"{lesson.dimension} has non-positive evidence confidence")
        claim = lesson.strategy_claim.strip()
        if is_summary_only_claim(claim):
            raise StrategyQualityError(
                f"{lesson.dimension} looks like a content summary, not a research strategy"
            )
        if len(lesson.why_it_works.strip()) < 8:
            raise StrategyQualityError(f"{lesson.dimension} has weak why_it_works")
        if len(lesson.transferable_template.strip()) < 8:
            raise StrategyQualityError(f"{lesson.dimension} has weak transferable_template")


def is_summary_only_claim(claim: str) -> bool:
    lowered = claim.lower()
    if any(verb in lowered for verb in STRATEGY_VERBS):
        return False
    return any(pattern.search(claim) for pattern in SUMMARY_ONLY_PATTERNS)
