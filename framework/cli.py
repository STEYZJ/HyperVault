from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

import uvicorn

from framework.config import get_settings
from framework.ingestion.watcher import VaultWatcherService
from framework.logging_config import configure_logging
from framework.memory.service import MemoryService
from framework.runtime import build_indexing_service, build_retrieval_service
from framework.schemas import MemoryConsolidationRequest, MetadataFilter, SearchRequest
from framework.tools.verify import verify_qdrant_collection

logger = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    if args.command == "serve":
        uvicorn.run("framework.api:app", host=settings.api_host, port=settings.api_port)
        return
    asyncio.run(run_command(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperVault knowledge layer CLI")
    parser.add_argument(
        "--offline-test-embeddings",
        action="store_true",
        help=(
            "Use deterministic embeddings for local tests only. "
            "Do not use for production indexes."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index-once", help="Run one incremental vault index")
    subparsers.add_parser("watch", help="Watch vault and incrementally index Markdown changes")
    subparsers.add_parser("serve", help="Serve FastAPI app")
    subparsers.add_parser("verify-qdrant", help="Verify Qdrant collection and local chunk state")

    search = subparsers.add_parser("search", help="Hybrid semantic search")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=8)
    add_filter_args(search)

    memory_search = subparsers.add_parser("memory-search", help="Search long-term memory notes")
    memory_search.add_argument("--query", required=True)
    memory_search.add_argument("--top-k", type=int, default=8)

    consolidate = subparsers.add_parser("consolidate-memory", help="Create a memory note")
    consolidate.add_argument("--topic", required=True)
    consolidate.add_argument("--query")
    consolidate.add_argument("--top-k", type=int, default=8)

    return parser


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--type")
    parser.add_argument("--priority")
    parser.add_argument("--memory", action="store_true")


async def run_command(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.command == "index-once":
        service = build_indexing_service(settings, args.offline_test_embeddings)
        summary = await service.index_vault()
        print_json(summary.model_dump(mode="json"))
        return

    if args.command == "watch":
        service = build_indexing_service(settings, args.offline_test_embeddings)
        watcher = VaultWatcherService(service, settings.watch_debounce_seconds)
        await watcher.run()
        return

    if args.command == "verify-qdrant":
        result = await verify_qdrant_collection(settings)
        print_json(result.model_dump(mode="json"))
        return

    if args.command == "search":
        retrieval = build_retrieval_service(settings, args.offline_test_embeddings)
        request = SearchRequest(
            query=args.query,
            top_k=args.top_k,
            filters=filters_from_args(args),
        )
        response = await retrieval.search(request)
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "memory-search":
        retrieval = build_retrieval_service(settings, args.offline_test_embeddings)
        memory = MemoryService(settings, retrieval)
        response = await memory.search_memory(SearchRequest(query=args.query, top_k=args.top_k))
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "consolidate-memory":
        retrieval = build_retrieval_service(settings, args.offline_test_embeddings)
        memory = MemoryService(settings, retrieval)
        response = await memory.consolidate(
            MemoryConsolidationRequest(topic=args.topic, query=args.query, top_k=args.top_k)
        )
        print_json(response.model_dump(mode="json"))
        return

    raise ValueError(f"Unknown command: {args.command}")


def filters_from_args(args: argparse.Namespace) -> MetadataFilter | None:
    filters = MetadataFilter(
        tags=args.tags,
        type=args.type,
        priority=args.priority,
        is_memory=True if args.memory else None,
    )
    if not any([filters.tags, filters.type, filters.priority, filters.is_memory is not None]):
        return None
    return filters


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
