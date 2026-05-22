from __future__ import annotations

from framework.memory.service import MemoryService
from framework.rag.indexing_service import IndexingService
from framework.rag.retrieval_service import RetrievalService
from framework.schemas import (
    IndexRunSummary,
    MemoryConsolidationRequest,
    MemoryConsolidationResponse,
    SearchRequest,
    SearchResponse,
)


class KnowledgeAgent:
    """Thin agent adapter that keeps framework code independent from the vault."""

    def __init__(
        self,
        indexing_service: IndexingService,
        retrieval_service: RetrievalService,
        memory_service: MemoryService,
    ) -> None:
        self.indexing_service = indexing_service
        self.retrieval_service = retrieval_service
        self.memory_service = memory_service

    async def index(self) -> IndexRunSummary:
        return await self.indexing_service.index_vault()

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self.retrieval_service.search(request)

    async def memory_search(self, request: SearchRequest) -> SearchResponse:
        return await self.memory_service.search_memory(request)

    async def consolidate_memory(
        self, request: MemoryConsolidationRequest
    ) -> MemoryConsolidationResponse:
        return await self.memory_service.consolidate(request)

