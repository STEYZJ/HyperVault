from __future__ import annotations

from pathlib import Path

from framework.ingestion.chunking import MarkdownChunker
from framework.ingestion.markdown_loader import MarkdownLoader, parse_frontmatter
from framework.ingestion.relations import extract_wikilink_targets


def test_tolerant_frontmatter_parses_markdown_style_tag_list() -> None:
    raw = """---
tags:
* ai
  type: research
  priority: high
---
# Note
"""
    metadata, body = parse_frontmatter(raw)
    assert metadata["tags"] == ["ai"]
    assert metadata["type"] == "research"
    assert metadata["priority"] == "high"
    assert body.startswith("# Note")


def test_wikilink_targets_support_aliases_and_headings() -> None:
    text = "See [[Agent Memory Architecture|memory]], [[RAG Indexing#Relations]], and [[Plain]]."
    assert extract_wikilink_targets(text) == [
        "Agent Memory Architecture",
        "RAG Indexing",
        "Plain",
    ]


def test_heading_chunker_preserves_fenced_code(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "code.md"
    note.write_text(
        """---
tags: [code]
type: research
---
# Code Note

Before.

```python
def example():
    return "keep together"
```

After.
""",
        encoding="utf-8",
    )
    parsed = MarkdownLoader().load(note, vault)
    chunks = MarkdownChunker(target_chars=80, overlap_chars=10).chunk(parsed)
    joined = "\n\n".join(chunk.text for chunk in chunks)
    assert 'return "keep together"' in joined
    assert any(chunk.heading_path == ["Code Note"] for chunk in chunks)
    assert all(chunk.metadata["type"] == "research" for chunk in chunks)

