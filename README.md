# HyperVault

Production knowledge layer for an Obsidian vault, agent framework, RAG retrieval, and
long-term memory.

## Architecture

HyperVault keeps the Markdown vault independent from the agent framework:

- `knowledge-vault/` is the human-editable Obsidian vault and long-term knowledge store.
- `framework/` contains ingestion, RAG, memory, tools, API, and agent-facing adapters.
- `runtime/` stores rebuildable SQLite state and caches. It is ignored by Git.
- Qdrant stores vectors through Docker Compose. Qdrant storage is ignored by Git.

For offline smoke tests only, `QDRANT_URL=local:runtime/qdrant-local` uses the embedded
qdrant-client local store. Production should use Docker Compose or an external Qdrant
service.

The system uses LlamaIndex's Markdown reader integration, Qdrant, OpenAI embeddings,
FastAPI, Watchdog, SQLite FTS5, and asyncio. It does not use LangChain, Chroma, or Pinecone.

HyperVault also owns an independent `Research Strategy Distillation` layer. HyperAgent,
Codex, and Claude Code are compatibility clients, not internal dependencies. They can submit
experience into HyperVault or call HyperVault retrieval tools through CLI/API/MCP wrappers.

## Quick Start

```bash
scripts/setup_env.sh
cp .env.example .env
# edit .env and set OPENAI_API_KEY
scripts/start_qdrant.sh
scripts/index_once.sh
scripts/verify_collection.sh
```

Production smoke and preflight checks:

```bash
scripts/secret_scan.sh
scripts/docker_preflight.sh
scripts/openai_smoke.sh
```

Run the API:

```bash
scripts/serve_api.sh
```

Search:

```bash
conda run -n HyperVault python -m framework.cli search --query "incremental indexing" --top-k 5
conda run -n HyperVault python -m framework.cli memory-search --query "agent preferences"
```

Research strategy extraction:

```bash
conda run -n HyperVault python -m framework.cli --offline-test-embeddings --fake-strategy-llm \
  extract-paper-strategy --paper example-research-strategy-paper
conda run -n HyperVault python -m framework.cli --offline-test-embeddings \
  strategy-search --query "how to select baselines" --dimension baseline_selection_logic
```

Watch the vault:

```bash
scripts/watch_vault.sh
```

## Configuration

Configuration is loaded with `pydantic-settings` from environment variables and local
`.env`. Keep real secrets in `.env`; it is intentionally ignored by Git.

Important settings:

- `VAULT_PATH=knowledge-vault`
- `RUNTIME_PATH=runtime`
- `QDRANT_URL=http://localhost:6333`
- `QDRANT_COLLECTION=hypervault_chunks`
- `OPENAI_API_KEY=...`
- `OPENAI_BASE_URL=` (optional OpenAI-compatible endpoint)
- `EMBEDDING_MODEL=text-embedding-3-small`
- `EMBEDDING_DIM=1536`
- `OFFLINE_TEST_EMBEDDINGS=false`
- `STRATEGY_LLM_PROVIDER=openai`
- `STRATEGY_LLM_MODEL=gpt-4o-mini`
- `STRATEGY_EXTRACTION_TEMPERATURE=0.2`
- `STRATEGY_LLM_MAX_RETRIES=2`
- `EXPORT_FIGURE_PAGES=false`
- `HYPERAGENT_CLI=/data2/lzj/HyperAgent/HyperAgent` (optional external runner)
- `HYPERAGENT_COMMAND_TEMPLATE=` (optional full external command template)

`OFFLINE_TEST_EMBEDDINGS=true` is only for local smoke tests without OpenAI. Do not use
it for production indexes.

Use `--fake-strategy-llm` or `STRATEGY_LLM_PROVIDER=fake` only for tests and smoke runs.
Real paper strategy extraction requires a configured LLM provider and API key.

`OPENAI_BASE_URL` is passed to both embedding and strategy LLM clients, so an
OpenAI-compatible provider can be used without changing code. `scripts/openai_smoke.sh`
uses a separate `runtime/openai-smoke` local store and only reports model names, base URL
presence, card counts, and search counts. It never prints the API key.

If `scripts/docker_preflight.sh` reports that `/var/run/docker.sock` is not readable and
writable, Docker Qdrant cannot be production-verified from the current user. Fix options
are: have an administrator repair Docker daemon/socket ownership, add the user to the
effective Docker group and re-login, use root/rootless Docker, or point `QDRANT_URL` to an
external Qdrant service.

## Indexing

The indexer is incremental. It stores file hash, file modified time, chunk hash, chunk
metadata, relations, SQLite FTS rows, and embedding cache records. Unchanged files are
skipped. Deleted files are removed from SQLite state and Qdrant.

Chunking is heading-aware, preserves fenced code blocks, keeps Obsidian callouts as
Markdown text, supports overlap, and inherits note metadata into every chunk.

## Retrieval And Memory

Retrieval combines Qdrant semantic search with SQLite FTS5 lexical search. Results are
reranked with metadata filters, recency scoring, and memory boost. Files under
`knowledge-vault/memory/` are treated as long-term agent memory.

Memory consolidation writes a new Markdown note into `knowledge-vault/memory/` with
source backlinks and extracted evidence. Re-run indexing or keep the watcher running to
make new memory notes searchable.

## Research Strategy Distillation

The strategy layer extracts research experience rather than paper summaries. Its target assets are:

- `knowledge-vault/assets/papers/` for source PDFs.
- `knowledge-vault/research/papers/` for Markdown paper text.
- `knowledge-vault/summaries/paper-strategies/` for evidence-backed single-paper strategy cards.
- `knowledge-vault/memory/research-strategy/` for consolidated long-term research patterns.
- `knowledge-vault/research/hyperagent-experience/` for external agent or HyperAgent experience.

Each lesson contains `strategy_claim`, `why_it_works`, `evidence_span`,
`transferable_template`, `risk_or_limit`, and `confidence`. Missing evidence is recorded as
`insufficient_evidence`; unsupported guesses are not promoted into memory.

Research Taste is intentionally stricter than single-paper lessons. A single paper can
surface only a limited signal, so `research_taste` is marked insufficient during
single-paper extraction. Long-term Research Taste memory requires consolidation from at
least two distinct paper strategy cards.

PDF import uses PyMuPDF text layers. A scanned PDF is mostly page images with little or no
extractable text, so HyperVault marks `needs_ocr: true` rather than fabricating evidence.
OCR is not enabled by default because it adds heavier dependencies, is slower, can
introduce recognition errors, and can weaken page/source evidence binding. Figure/table
strategy v1 uses captions and nearby text; when `EXPORT_FIGURE_PAGES=true`, pages with
detected figure/table captions are saved as assets for later optional vision analysis.

CLI commands:

- `python -m framework.cli import-paper --path <pdf-or-md>`
- `python -m framework.cli extract-paper-strategy --paper <path-or-id>`
- `python -m framework.cli strategy-search --query "problem gap" --dimension problem_gap`
- `python -m framework.cli consolidate-strategy --topic "baseline selection"`
- `python -m framework.cli submit-agent-experience --source hyperagent --path <file>`
- `python -m framework.cli call-hyperagent-summary --topic <topic>`
- `python -m framework.cli secret-scan`
- `python -m framework.cli docker-preflight`
- `python -m framework.cli openai-smoke`

HyperAgent compatibility is deliberately process/protocol based. HyperVault never imports
HyperAgent Python packages; configure `HYPERAGENT_CLI` when you want HyperVault to call an
external runner. By default, HyperVault calls `hyperagent research extract --json --no-write`
when an input path is supplied, or `hyperagent research search --json` for a topic-only
lookup. `HYPERAGENT_COMMAND_TEMPLATE` can override this without coupling Python packages.

Dimension filters accept the canonical names and short aliases such as `problem_gap`,
`baseline`, `novelty`, `figure_table`, `reviewer`, `storytelling`, `ablation`, and `taste`.

## API

FastAPI endpoints:

- `GET /health`
- `POST /index/run`
- `GET /index/status`
- `POST /search`
- `POST /memory/search`
- `POST /memory/consolidate`
- `GET /relations`
- `GET /verify/qdrant`
- `POST /papers/import`
- `POST /strategy/extract`
- `POST /strategy/search`
- `POST /strategy/consolidate`
- `GET /strategy/cards`
- `GET /strategy/cards/{paper_id}`
- `POST /agent-experience/submit`
- `POST /integrations/hyperagent/summarize`

## Git Hygiene

Do not commit `.env`, `runtime/`, Qdrant storage, generated caches, or ad hoc runtime logs.
Project worklogs under `logs/worklog/` are intentional documentation. The vault contains sample
notes only; private or sensitive notes should be reviewed before push.

Branch maintenance is part of project maintenance:

- Keep `main` as the stable integration branch.
- Use scoped branches for non-trivial work: `feature/<topic>`, `fix/<topic>`, `docs/<topic>`,
  or `ops/<topic>`.
- Before merging a branch back to `main`, run tests, lint, compile checks, secret scan, and update
  the worklog.
- Delete or archive merged local branches after confirming their commits are integrated.
- Record branch creation, merge, push failures, and cleanup in `logs/worklog/`.

Repository About suggestion:

- Description: `Production Obsidian + Agent Framework + RAG + Long-term Memory knowledge layer`
- Topics: `obsidian`, `rag`, `qdrant`, `llamaindex`, `ai-agent`, `long-term-memory`, `markdown`, `fastapi`
