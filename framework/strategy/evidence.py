from __future__ import annotations

import re
from collections import defaultdict

from framework.schemas import ChunkRecord
from framework.strategy.schemas import (
    ALL_STRATEGY_DIMENSIONS,
    EvidenceSpan,
    MinedEvidenceItem,
    SectionMap,
    SectionSpan,
    StrategyDimension,
)

SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"\babstract\b", re.I)),
    ("introduction", re.compile(r"\b(introduction|motivation)\b", re.I)),
    ("related_work", re.compile(r"\b(related work|background)\b", re.I)),
    ("method", re.compile(r"\b(method|approach|model|framework)\b", re.I)),
    ("experiment", re.compile(r"\b(experiment|evaluation|result|dataset|benchmark)\b", re.I)),
    ("discussion", re.compile(r"\b(discussion|analysis)\b", re.I)),
    ("limitation", re.compile(r"\b(limitation|failure|threat|future work)\b", re.I)),
    ("conclusion", re.compile(r"\b(conclusion|concluding)\b", re.I)),
)

DIMENSION_CUES: dict[StrategyDimension, tuple[str, ...]] = {
    "contribution_packaging": (
        "contribution",
        "we make",
        "we present",
        "we propose",
        "our work",
        "in summary",
    ),
    "baseline_selection_logic": (
        "baseline",
        "compare",
        "compared with",
        "state-of-the-art",
        "sota",
        "competitive",
    ),
    "novelty_construction": (
        "novel",
        "new",
        "first",
        "unlike",
        "different from",
        "to our knowledge",
    ),
    "problem_gap_framing": (
        "however",
        "challenge",
        "gap",
        "remain",
        "limited",
        "fail to",
        "not yet",
    ),
    "figure_table_expression": (
        "figure",
        "fig.",
        "table",
        "visualize",
        "shown in",
        "caption",
    ),
    "experiment_arrangement": (
        "experiment",
        "evaluate",
        "dataset",
        "metric",
        "protocol",
        "setting",
    ),
    "reviewer_concern_handling": (
        "robust",
        "sensitivity",
        "generalize",
        "fair",
        "statistical",
        "threat",
    ),
    "credibility_building": (
        "significant",
        "consistent",
        "multiple",
        "across",
        "robust",
        "validated",
    ),
    "weakness_avoidance": (
        "limitation",
        "scope",
        "future work",
        "failure",
        "trade-off",
        "only",
    ),
    "storytelling_moves": (
        "motivate",
        "first",
        "then",
        "finally",
        "we begin",
        "this suggests",
    ),
    "claim_organization": (
        "show",
        "demonstrate",
        "evidence",
        "claim",
        "we find",
        "indicate",
    ),
    "ablation_logic": (
        "ablation",
        "remove",
        "variant",
        "component",
        "without",
        "effect of",
    ),
    "research_taste": (
        "trend",
        "emerging",
        "benchmark",
        "community",
        "underexplored",
        "important problem",
    ),
}


def build_section_map(paper_id: str, chunks: list[ChunkRecord]) -> SectionMap:
    grouped: dict[tuple[str, str], SectionSpan] = {}
    for chunk in chunks:
        heading = " / ".join(chunk.heading_path) or str(chunk.metadata.get("title") or "Document")
        section_type = infer_section_type(heading, chunk.text)
        key = (section_type, heading)
        if key not in grouped:
            grouped[key] = SectionSpan(section_type=section_type, heading=heading)
        grouped[key].chunk_ids.append(chunk.chunk_id)
        page = page_from_chunk(chunk)
        if page and page not in grouped[key].pages:
            grouped[key].pages.append(page)
    return SectionMap(paper_id=paper_id, sections=list(grouped.values()))


def mine_strategy_evidence(
    chunks: list[ChunkRecord],
    max_items: int = 80,
) -> list[MinedEvidenceItem]:
    items: list[MinedEvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    per_dimension_count: defaultdict[StrategyDimension, int] = defaultdict(int)
    for chunk in chunks:
        heading = " / ".join(chunk.heading_path) or str(chunk.metadata.get("title") or "Document")
        section_type = infer_section_type(heading, chunk.text)
        for dimension in ALL_STRATEGY_DIMENSIONS:
            if per_dimension_count[dimension] >= 8:
                continue
            match = best_sentence_for_dimension(chunk.text, dimension)
            if not match:
                continue
            cue, sentence = match
            key = (dimension, chunk.chunk_id, sentence)
            if key in seen:
                continue
            seen.add(key)
            page = page_from_chunk(chunk)
            evidence = EvidenceSpan(
                source_path=chunk.file_path,
                chunk_id=chunk.chunk_id,
                heading=heading,
                page=page,
                excerpt=sentence,
                confidence=0.72 if section_type != "unknown" else 0.62,
            )
            items.append(
                MinedEvidenceItem(
                    dimension=dimension,
                    evidence_span=evidence,
                    cue=cue,
                    section_type=section_type,
                )
            )
            per_dimension_count[dimension] += 1
            if len(items) >= max_items:
                return items
    return items


def infer_section_type(heading: str, text: str) -> str:
    haystack = f"{heading}\n{text[:400]}"
    for section_type, pattern in SECTION_PATTERNS:
        if pattern.search(haystack):
            return section_type
    return "unknown"


def best_sentence_for_dimension(text: str, dimension: StrategyDimension) -> tuple[str, str] | None:
    sentences = split_sentences(text)
    cues = DIMENSION_CUES[dimension]
    for sentence in sentences:
        lowered = sentence.lower()
        for cue in cues:
            if cue_matches(lowered, cue):
                return cue, compact_sentence(sentence)
    return None


def cue_matches(lowered_sentence: str, cue: str) -> bool:
    escaped = re.escape(cue.lower())
    if " " in cue:
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered_sentence) is not None
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered_sentence) is not None


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+", normalized)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 24]


def compact_sentence(sentence: str, max_chars: int = 500) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 3].rstrip() + "..."


def page_from_chunk(chunk: ChunkRecord) -> int | None:
    page = chunk.metadata.get("page")
    if isinstance(page, int):
        return page
    text = chunk.text[:80]
    match = re.search(r"<!--\s*page:\s*(\d+)\s*-->", text)
    if match:
        return int(match.group(1))
    for heading in reversed(chunk.heading_path):
        page_match = re.search(r"\bpage\s+(\d+)\b", heading, flags=re.I)
        if page_match:
            return int(page_match.group(1))
    return None
