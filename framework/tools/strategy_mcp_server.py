from __future__ import annotations

import asyncio

from framework.tools import strategy as strategy_tools

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - optional runtime integration.
    raise SystemExit("Install the optional `mcp` package to run HyperVault MCP tools.") from exc

mcp = FastMCP("hypervault")


@mcp.tool(name="hypervault.import_paper")
async def import_paper(path: str, paper_id: str | None = None) -> dict:
    return await strategy_tools.hypervault_import_paper(path, paper_id)


@mcp.tool(name="hypervault.extract_paper_strategy")
async def extract_paper_strategy(paper: str, fake_llm: bool = False) -> dict:
    return await strategy_tools.hypervault_extract_paper_strategy(paper, fake_llm)


@mcp.tool(name="hypervault.strategy_search")
async def strategy_search(query: str, dimension: str | None = None, top_k: int = 8) -> dict:
    return await strategy_tools.hypervault_strategy_search(query, dimension, top_k)


@mcp.tool(name="hypervault.consolidate_strategy")
async def consolidate_strategy(topic: str, dimension: str | None = None, top_k: int = 8) -> dict:
    return await strategy_tools.hypervault_consolidate_strategy(topic, dimension, top_k)


@mcp.tool(name="hypervault.paper_strategy_report")
async def paper_strategy_report(paper_id: str) -> dict:
    return await strategy_tools.hypervault_paper_strategy_report(paper_id)


@mcp.tool(name="hypervault.submit_agent_experience")
async def submit_agent_experience(source: str, content: str, title: str | None = None) -> dict:
    return await strategy_tools.hypervault_submit_agent_experience(source, content, title)


@mcp.tool(name="hypervault.call_hyperagent_summary")
async def call_hyperagent_summary(topic: str) -> dict:
    return await strategy_tools.hypervault_call_hyperagent_summary(topic)


def main() -> None:
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
