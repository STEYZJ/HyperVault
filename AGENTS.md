# HyperVault Agent Integration

HyperVault is the knowledge owner. External agents, including Codex, Claude Code, and
HyperAgent, should communicate through CLI, API, MCP-compatible wrappers, or Markdown files in the
vault. Do not import HyperAgent Python packages from HyperVault.

## Research Strategy Tools

- Import a paper: `python -m framework.cli import-paper --path <pdf-or-md>`
- Extract a strategy card: `python -m framework.cli --fake-strategy-llm extract-paper-strategy --paper <paper-id-or-path>`
- Search strategy cards: `python -m framework.cli strategy-search --query "baseline selection" --dimension baseline_selection_logic`
- Consolidate long-term memory: `python -m framework.cli consolidate-strategy --topic "baseline selection"`
- Submit agent experience: `python -m framework.cli submit-agent-experience --source hyperagent --path <file>`
- Call external HyperAgent: `python -m framework.cli call-hyperagent-summary --topic <topic>`
- Production checks: `python -m framework.cli secret-scan`, `docker-preflight`, `openai-smoke`

Every generated research lesson must be evidence-backed and phrased as strategy:
`strategy_claim`, `why_it_works`, `evidence_span`, `transferable_template`,
`risk_or_limit`, and `confidence`.

MCP-compatible tool wrappers expose `hypervault.import_paper`,
`hypervault.extract_paper_strategy`, `hypervault.strategy_search`,
`hypervault.consolidate_strategy`, `hypervault.paper_strategy_report`,
`hypervault.submit_agent_experience`, and `hypervault.call_hyperagent_summary`.

Do not print or persist API keys in agent logs. External HyperAgent calls must stay at the
process/API/MCP layer; HyperVault must not import HyperAgent Python modules.

## Git Branch Discipline

Keep `main` stable. Use scoped branches for substantial work, such as `feature/<topic>`,
`fix/<topic>`, `docs/<topic>`, or `ops/<topic>`. Before merging back to `main`, run validation,
update the project worklog, and confirm no secrets or runtime files are staged.
