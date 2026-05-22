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

Every generated research lesson must be evidence-backed and phrased as strategy:
`strategy_claim`, `why_it_works`, `evidence_span`, `transferable_template`,
`risk_or_limit`, and `confidence`.
