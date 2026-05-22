---
tags:
  - rag
  - qdrant
type: research
priority: medium
---

# RAG Indexing Notes

Incremental indexing depends on file hashes, modified time, chunk hashes, and delete
cleanup. The vector store should not be rebuilt from scratch unless the embedding model
or chunking policy changes.

## Relations

Graph-ready extraction starts with [[Agent Memory Architecture]] and Obsidian wikilinks.

