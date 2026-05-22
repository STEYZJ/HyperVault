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
from framework.runtime import (
    build_indexing_service,
    build_retrieval_service,
    build_strategy_service,
)
from framework.schemas import MemoryConsolidationRequest, MetadataFilter, SearchRequest
from framework.strategy.schemas import (
    AgentExperienceSubmitRequest,
    HyperAgentSummaryRequest,
    PaperImportRequest,
    StrategyConsolidationRequest,
    StrategyExtractionRequest,
    StrategySearchRequest,
)
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
    parser.add_argument(
        "--fake-strategy-llm",
        action="store_true",
        help="Use deterministic strategy extraction for tests and no-key smoke validation.",
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

    import_paper = subparsers.add_parser("import-paper", help="Import a PDF or Markdown paper")
    import_paper.add_argument("--path", required=True)
    import_paper.add_argument("--paper-id")
    import_paper.add_argument("--title")
    import_paper.add_argument("--venue")
    import_paper.add_argument("--year", type=int)
    import_paper.add_argument("--field")

    extract = subparsers.add_parser(
        "extract-paper-strategy",
        help="Extract a research strategy card from a paper",
    )
    extract.add_argument("--paper", required=True)

    strategy_search = subparsers.add_parser(
        "strategy-search",
        help="Search paper strategy cards and research strategy memory",
    )
    strategy_search.add_argument("--query", required=True)
    strategy_search.add_argument("--top-k", type=int, default=8)
    strategy_search.add_argument("--dimension")
    strategy_search.add_argument("--paper-id")
    strategy_search.add_argument("--venue")
    strategy_search.add_argument("--year", type=int)
    strategy_search.add_argument("--verified", action="store_true")
    strategy_search.add_argument("--no-memory", action="store_true")

    strategy_consolidate = subparsers.add_parser(
        "consolidate-strategy",
        help="Consolidate paper strategy cards into long-term research memory",
    )
    strategy_consolidate.add_argument("--topic", required=True)
    strategy_consolidate.add_argument("--dimension")
    strategy_consolidate.add_argument("--top-k", type=int, default=8)

    strategy_report = subparsers.add_parser("strategy-report", help="Show one paper strategy card")
    strategy_report.add_argument("--paper-id", required=True)

    submit_experience = subparsers.add_parser(
        "submit-agent-experience",
        help="Submit external agent experience material into HyperVault",
    )
    submit_experience.add_argument("--source", default="hyperagent")
    submit_experience.add_argument("--path", required=True)
    submit_experience.add_argument("--title")

    hyperagent_summary = subparsers.add_parser(
        "call-hyperagent-summary",
        help="Call a configured external HyperAgent runner and store its summary",
    )
    hyperagent_summary.add_argument("--topic", required=True)
    hyperagent_summary.add_argument("--input-path")
    hyperagent_summary.add_argument("--extra-arg", action="append", default=[])
    hyperagent_summary.add_argument("--no-submit", action="store_true")

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

    if args.command == "import-paper":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=True,
        )
        response = await strategy.import_paper(
            PaperImportRequest(
                path=args.path,
                paper_id=args.paper_id,
                title=args.title,
                venue=args.venue,
                year=args.year,
                field=args.field,
            )
        )
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "extract-paper-strategy":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=args.fake_strategy_llm,
        )
        response = await strategy.extract_paper_strategy(
            StrategyExtractionRequest(paper=args.paper)
        )
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "strategy-search":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=True,
        )
        response = await strategy.search_strategy(
            StrategySearchRequest(
                query=args.query,
                top_k=args.top_k,
                dimension=args.dimension,
                paper_id=args.paper_id,
                venue=args.venue,
                year=args.year,
                verified=True if args.verified else None,
                include_memory=not args.no_memory,
            )
        )
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "consolidate-strategy":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=True,
        )
        response = await strategy.consolidate_strategy(
            StrategyConsolidationRequest(
                topic=args.topic,
                dimension=args.dimension,
                top_k=args.top_k,
            )
        )
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "strategy-report":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=True,
        )
        response = await strategy.strategy_report(args.paper_id)
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "submit-agent-experience":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=True,
        )
        response = await strategy.submit_agent_experience(
            AgentExperienceSubmitRequest(
                source=args.source,
                path=args.path,
                title=args.title,
            )
        )
        print_json(response.model_dump(mode="json"))
        return

    if args.command == "call-hyperagent-summary":
        strategy = build_strategy_service(
            settings,
            args.offline_test_embeddings,
            fake_strategy_llm=True,
        )
        response = await strategy.call_hyperagent_summary(
            HyperAgentSummaryRequest(
                topic=args.topic,
                input_path=args.input_path,
                extra_args=args.extra_arg,
                submit_to_vault=not args.no_submit,
            )
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
