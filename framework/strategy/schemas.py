from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from framework.schemas import SearchHit, utc_now

StrategyDimension = Literal[
    "contribution_packaging",
    "baseline_selection_logic",
    "novelty_construction",
    "problem_gap_framing",
    "figure_table_expression",
    "experiment_arrangement",
    "reviewer_concern_handling",
    "credibility_building",
    "weakness_avoidance",
    "storytelling_moves",
    "claim_organization",
    "ablation_logic",
    "research_taste",
]

ExperienceKind = Literal[
    "research_pattern",
    "experiment_strategy",
    "scientific_storytelling",
    "research_taste",
]

ALL_STRATEGY_DIMENSIONS: tuple[StrategyDimension, ...] = (
    "contribution_packaging",
    "baseline_selection_logic",
    "novelty_construction",
    "problem_gap_framing",
    "figure_table_expression",
    "experiment_arrangement",
    "reviewer_concern_handling",
    "credibility_building",
    "weakness_avoidance",
    "storytelling_moves",
    "claim_organization",
    "ablation_logic",
    "research_taste",
)

DIMENSION_LABELS: dict[StrategyDimension, str] = {
    "contribution_packaging": "Contribution Packaging",
    "baseline_selection_logic": "Baseline Selection Logic",
    "novelty_construction": "Novelty Construction",
    "problem_gap_framing": "Problem-Gap Framing",
    "figure_table_expression": "Figure/Table Expression",
    "experiment_arrangement": "Experiment Arrangement",
    "reviewer_concern_handling": "Reviewer Concern Handling",
    "credibility_building": "Credibility Building",
    "weakness_avoidance": "Weakness Avoidance",
    "storytelling_moves": "Storytelling Moves",
    "claim_organization": "Claim Organization",
    "ablation_logic": "Ablation Logic",
    "research_taste": "Research Taste",
}


class EvidenceSpan(BaseModel):
    source_path: str
    excerpt: str = Field(min_length=3, max_length=1400)
    chunk_id: str | None = None
    heading: str | None = None
    page: int | None = Field(default=None, ge=1)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ensure_location(self) -> EvidenceSpan:
        if not any([self.chunk_id, self.heading, self.page]):
            raise ValueError("EvidenceSpan requires chunk_id, heading, or page")
        return self


class StrategyLesson(BaseModel):
    dimension: StrategyDimension
    strategy_claim: str = Field(min_length=8, max_length=800)
    why_it_works: str = Field(min_length=8, max_length=1200)
    evidence_span: EvidenceSpan
    transferable_template: str = Field(min_length=8, max_length=1200)
    risk_or_limit: str = Field(min_length=3, max_length=900)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)


class SectionSpan(BaseModel):
    section_type: Literal[
        "abstract",
        "introduction",
        "method",
        "experiment",
        "discussion",
        "limitation",
        "related_work",
        "conclusion",
        "unknown",
    ]
    heading: str
    chunk_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)


class SectionMap(BaseModel):
    paper_id: str
    sections: list[SectionSpan] = Field(default_factory=list)


class MinedEvidenceItem(BaseModel):
    dimension: StrategyDimension
    evidence_span: EvidenceSpan
    cue: str
    section_type: str = "unknown"


class PaperStrategyCard(BaseModel):
    paper_id: str
    title: str
    source_path: str
    strategy_dimensions: list[StrategyDimension] = Field(default_factory=list)
    lessons: list[StrategyLesson] = Field(default_factory=list)
    insufficient_dimensions: list[StrategyDimension] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    field: str | None = None
    verified: bool = False
    evidence_level: str = "chunk"
    source_pdf: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_lessons_are_supported(self) -> PaperStrategyCard:
        if not self.lessons and not self.insufficient_dimensions:
            raise ValueError("PaperStrategyCard requires lessons or insufficient_dimensions")
        lesson_dims = [lesson.dimension for lesson in self.lessons]
        if not self.strategy_dimensions:
            self.strategy_dimensions = sorted(set(lesson_dims))  # type: ignore[assignment]
        missing = set(lesson_dims) - set(self.strategy_dimensions)
        if missing:
            self.strategy_dimensions.extend(sorted(missing))  # type: ignore[arg-type]
        return self


class ResearchPattern(BaseModel):
    kind: Literal["research_pattern"] = "research_pattern"
    lessons: list[StrategyLesson]


class ExperimentStrategy(BaseModel):
    kind: Literal["experiment_strategy"] = "experiment_strategy"
    lessons: list[StrategyLesson]


class ScientificStorytelling(BaseModel):
    kind: Literal["scientific_storytelling"] = "scientific_storytelling"
    lessons: list[StrategyLesson]


class ResearchTaste(BaseModel):
    kind: Literal["research_taste"] = "research_taste"
    lessons: list[StrategyLesson]
    source_paper_count: int = Field(ge=2)


class PaperImportRequest(BaseModel):
    path: Path
    paper_id: str | None = None
    title: str | None = None
    venue: str | None = None
    year: int | None = None
    field: str | None = None


class FigureTableReference(BaseModel):
    label: str
    page: int | None = None
    caption: str
    context: str | None = None


class PaperImportResponse(BaseModel):
    paper_id: str
    title: str
    source_kind: Literal["pdf", "markdown"]
    markdown_path: str
    source_pdf: str | None = None
    needs_ocr: bool = False
    pages: int = 0
    figure_table_refs: list[FigureTableReference] = Field(default_factory=list)


class StrategyExtractionRequest(BaseModel):
    paper: str
    force_reimport: bool = False
    force_reindex: bool = False


class StrategyExtractionResponse(BaseModel):
    card: PaperStrategyCard
    card_path: str
    section_map: SectionMap
    mined_evidence_count: int


class StrategySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    dimension: StrategyDimension | None = None
    paper_id: str | None = None
    venue: str | None = None
    year: int | None = None
    verified: bool | None = None
    include_memory: bool = True


class StrategySearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class StrategyConsolidationRequest(BaseModel):
    topic: str
    dimension: StrategyDimension | None = None
    top_k: int = Field(default=8, ge=1, le=30)


class StrategyConsolidationResponse(BaseModel):
    path: str
    title: str
    source_count: int
    source_cards: list[str]


class AgentExperienceSubmitRequest(BaseModel):
    source: str = "hyperagent"
    title: str | None = None
    content: str | None = None
    path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_content_or_path(self) -> AgentExperienceSubmitRequest:
        if not self.content and not self.path:
            raise ValueError("Agent experience submission requires content or path")
        return self


class AgentExperienceSubmitResponse(BaseModel):
    path: str
    title: str
    source: str
    indexed: bool


class HyperAgentSummaryRequest(BaseModel):
    topic: str
    input_path: Path | None = None
    extra_args: list[str] = Field(default_factory=list)
    submit_to_vault: bool = True


class HyperAgentSummaryResponse(BaseModel):
    topic: str
    content: str
    submitted_path: str | None = None
    return_code: int = 0
