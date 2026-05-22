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

## Quick Start

```bash
scripts/setup_env.sh
cp .env.example .env
# edit .env and set OPENAI_API_KEY
scripts/start_qdrant.sh
scripts/index_once.sh
scripts/verify_collection.sh
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
- `EMBEDDING_MODEL=text-embedding-3-small`
- `EMBEDDING_DIM=1536`
- `OFFLINE_TEST_EMBEDDINGS=false`

`OFFLINE_TEST_EMBEDDINGS=true` is only for local smoke tests without OpenAI. Do not use
it for production indexes.

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

## Git Hygiene

Do not commit `.env`, `runtime/`, Qdrant storage, logs, or generated caches. The vault
contains sample notes only; private or sensitive notes should be reviewed before push.

Repository About suggestion:

- Description: `Production Obsidian + Agent Framework + RAG + Long-term Memory knowledge layer`
- Topics: `obsidian`, `rag`, `qdrant`, `llamaindex`, `ai-agent`, `long-term-memory`, `markdown`, `fastapi`
