from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from framework.config import Settings
from framework.ingestion.markdown_loader import infer_title, normalize_metadata, parse_frontmatter
from framework.strategy.schemas import FigureTableReference, PaperImportRequest, PaperImportResponse

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - import depends on optional runtime package.
    fitz = None


@dataclass(frozen=True)
class ExtractedPdf:
    title: str
    pages: int
    markdown: str
    needs_ocr: bool
    figure_table_refs: list[FigureTableReference] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PaperImportService:
    """Import PDF or Markdown papers into the decoupled knowledge vault."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def import_paper(self, request: PaperImportRequest) -> PaperImportResponse:
        path = request.path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._import_pdf(path, request)
        if suffix in {".md", ".markdown"}:
            return self._import_markdown(path, request)
        raise ValueError(f"Unsupported paper input type: {path.suffix}")

    def _import_pdf(self, path: Path, request: PaperImportRequest) -> PaperImportResponse:
        extracted = extract_pdf(path)
        paper_id = request.paper_id or slugify(path.stem)
        title = request.title or extracted.title or path.stem
        source_pdf_name = f"{paper_id}.pdf"
        target_pdf = self.settings.paper_assets_path / source_pdf_name
        target_pdf.parent.mkdir(parents=True, exist_ok=True)
        if path != target_pdf:
            shutil.copy2(path, target_pdf)

        frontmatter = base_paper_frontmatter(
            paper_id=paper_id,
            title=title,
            venue=request.venue,
            year=request.year,
            field=request.field,
        )
        frontmatter.update(
            {
                "source_pdf": f"assets/papers/{source_pdf_name}",
                "needs_ocr": extracted.needs_ocr,
                "pages": extracted.pages,
                "figure_table_refs": [
                    ref.model_dump(mode="json") for ref in extracted.figure_table_refs
                ],
            }
        )
        markdown = render_markdown(frontmatter, f"# {title}\n\n{extracted.markdown}".strip())
        target_markdown = self.settings.paper_markdown_path / f"{paper_id}.md"
        target_markdown.parent.mkdir(parents=True, exist_ok=True)
        target_markdown.write_text(markdown + "\n", encoding="utf-8")
        rel_markdown = target_markdown.relative_to(self.settings.vault_path).as_posix()
        rel_pdf = target_pdf.relative_to(self.settings.vault_path).as_posix()
        logger.info("Imported PDF paper %s -> %s", path, rel_markdown)
        return PaperImportResponse(
            paper_id=paper_id,
            title=title,
            source_kind="pdf",
            markdown_path=rel_markdown,
            source_pdf=rel_pdf,
            needs_ocr=extracted.needs_ocr,
            pages=extracted.pages,
            figure_table_refs=extracted.figure_table_refs,
        )

    def _import_markdown(self, path: Path, request: PaperImportRequest) -> PaperImportResponse:
        raw = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw)
        metadata = normalize_metadata(metadata)
        paper_id = request.paper_id or str(metadata.get("paper_id") or slugify(path.stem))
        title = request.title or infer_title(body, path, metadata)
        figure_refs = extract_figure_table_refs_from_text(body)

        try:
            rel_existing = path.relative_to(self.settings.vault_path).as_posix()
        except ValueError:
            rel_existing = None

        if rel_existing and rel_existing.startswith("research/papers/"):
            logger.info("Markdown paper already lives in vault: %s", rel_existing)
            return PaperImportResponse(
                paper_id=paper_id,
                title=title,
                source_kind="markdown",
                markdown_path=rel_existing,
                pages=0,
                figure_table_refs=figure_refs,
            )

        target = self.settings.paper_markdown_path / f"{paper_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = dict(metadata)
        frontmatter.update(
            base_paper_frontmatter(
                paper_id=paper_id,
                title=title,
                venue=request.venue or metadata.get("venue"),
                year=request.year or metadata.get("year"),
                field=request.field or metadata.get("field"),
            )
        )
        frontmatter["figure_table_refs"] = [ref.model_dump(mode="json") for ref in figure_refs]
        markdown = render_markdown(frontmatter, body.strip() or raw.strip())
        target.write_text(markdown + "\n", encoding="utf-8")
        rel_target = target.relative_to(self.settings.vault_path).as_posix()
        logger.info("Imported Markdown paper %s -> %s", path, rel_target)
        return PaperImportResponse(
            paper_id=paper_id,
            title=title,
            source_kind="markdown",
            markdown_path=rel_target,
            pages=0,
            figure_table_refs=figure_refs,
        )


def extract_pdf(path: Path) -> ExtractedPdf:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF import. Install the pymupdf package.")
    document = fitz.open(path)  # type: ignore[union-attr]
    metadata = dict(document.metadata or {})
    title = str(metadata.get("title") or path.stem).strip() or path.stem
    page_markdown: list[str] = []
    refs: list[FigureTableReference] = []
    total_text_chars = 0
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        text = page.get_text("text") or ""
        total_text_chars += len(text.strip())
        page_number = page_index + 1
        captions = extract_figure_table_refs_from_text(text, page=page_number)
        refs.extend(captions)
        page_markdown.append(render_pdf_page(page_number, text, captions))
    needs_ocr = document.page_count > 0 and total_text_chars < max(400, document.page_count * 40)
    if needs_ocr:
        body = (
            "> [!warning] OCR Required\n"
            "> This PDF appears to contain too little extractable text. "
            "HyperVault did not fabricate paper content.\n"
        )
    else:
        body = "\n\n".join(page_markdown)
    return ExtractedPdf(
        title=title,
        pages=document.page_count,
        markdown=body,
        needs_ocr=needs_ocr,
        figure_table_refs=refs,
        metadata=metadata,
    )


def render_pdf_page(
    page_number: int,
    text: str,
    refs: list[FigureTableReference],
) -> str:
    body = text.strip() or "[No extractable text on this page.]"
    caption_block = ""
    if refs:
        lines = "\n".join(f"- {ref.label}: {ref.caption}" for ref in refs)
        caption_block = f"\n\n### Figure/Table Captions\n\n{lines}"
    return f"## Page {page_number}\n\n<!-- page: {page_number} -->\n\n{body}{caption_block}"


def base_paper_frontmatter(
    paper_id: str,
    title: str,
    venue: str | None = None,
    year: int | str | None = None,
    field: str | None = None,
) -> dict[str, Any]:
    normalized_year = int(year) if isinstance(year, str) and year.isdigit() else year
    return {
        "tags": ["paper", "research"],
        "type": "paper",
        "paper_id": paper_id,
        "title": title,
        "venue": venue,
        "year": normalized_year,
        "field": field,
        "verified": False,
        "imported_at": datetime.now(tz=UTC).isoformat(),
    }


def render_markdown(frontmatter: dict[str, Any], body: str) -> str:
    clean_frontmatter = {
        key: value
        for key, value in frontmatter.items()
        if value not in (None, "", [], {})
    }
    return (
        "---\n"
        f"{yaml.safe_dump(clean_frontmatter, sort_keys=False, allow_unicode=True)}"
        "---\n\n"
        f"{body.strip()}\n"
    )


def extract_figure_table_refs_from_text(
    text: str,
    page: int | None = None,
) -> list[FigureTableReference]:
    refs: list[FigureTableReference] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    caption_re = re.compile(
        r"^(?P<label>(?:Fig\.?|Figure|Table)\s*\d+[A-Za-z]?)"
        r"\s*[:.\-]?\s*(?P<caption>.+)$",
        re.I,
    )
    for index, line in enumerate(lines):
        match = caption_re.match(line)
        if not match:
            continue
        label = normalize_label(match.group("label"))
        caption = match.group("caption").strip()
        context = surrounding_context(lines, index)
        refs.append(
            FigureTableReference(
                label=label,
                page=page,
                caption=caption[:900],
                context=context,
            )
        )
    return refs


def surrounding_context(lines: list[str], index: int, window: int = 2) -> str:
    start = max(index - window, 0)
    end = min(index + window + 1, len(lines))
    return " ".join(lines[start:end])[:900]


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("Fig.", "Figure")).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text.lower()).strip("-")
    return slug or "paper"
