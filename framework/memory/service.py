from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from framework.config import Settings
from framework.rag.retrieval_service import RetrievalService
from framework.schemas import (
    MemoryConsolidationRequest,
    MemoryConsolidationResponse,
    MetadataFilter,
    SearchRequest,
    SearchResponse,
)


class MemoryService:
    def __init__(self, settings: Settings, retrieval_service: RetrievalService) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service

    async def search_memory(self, request: SearchRequest) -> SearchResponse:
        filters = request.filters or MetadataFilter()
        filters.is_memory = True
        memory_request = request.model_copy(update={"filters": filters, "memory_boost": 0.35})
        return await self.retrieval_service.search(memory_request)

    async def consolidate(
        self, request: MemoryConsolidationRequest
    ) -> MemoryConsolidationResponse:
        query = request.query or request.topic
        response = await self.retrieval_service.search(
            SearchRequest(query=query, top_k=request.top_k, memory_boost=0.25)
        )
        now = datetime.now(tz=UTC)
        title = f"Memory - {request.topic}"
        filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{slugify(request.topic)}.md"
        target_path = self.settings.vault_memory_path / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        frontmatter = {
            "tags": ["memory", "consolidated"],
            "type": "memory",
            "priority": "high",
            "topic": request.topic,
            "created": now.isoformat(),
            "sources": [hit.file_path for hit in response.hits],
        }
        source_links = "\n".join(
            f"- [[{Path(hit.file_path).stem}]] - score {hit.score:.4f}" for hit in response.hits
        )
        distilled = "\n\n".join(
            f"### {hit.title}\n\n{compact_text(hit.text)}" for hit in response.hits
        )
        content = (
            "---\n"
            f"{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)}"
            "---\n\n"
            f"# {title}\n\n"
            "## Consolidated Memory\n\n"
            f"{distilled or 'No source notes were found for this topic.'}\n\n"
            "## Sources\n\n"
            f"{source_links or '- No sources'}\n"
        )
        target_path.write_text(content, encoding="utf-8")
        return MemoryConsolidationResponse(
            path=target_path.relative_to(self.settings.vault_path).as_posix(),
            source_count=len(response.hits),
            title=title,
        )


def compact_text(text: str, max_chars: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text.lower()).strip("-")
    return slug or "memory"

