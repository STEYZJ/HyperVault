from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from framework.schemas import JsonDict, ParsedMarkdown

logger = logging.getLogger(__name__)

try:  # LlamaIndex is used as the pluggable Markdown reader when installed.
    from llama_index.readers.file import MarkdownReader
except Exception:  # pragma: no cover - import availability depends on env setup.
    MarkdownReader = None  # type: ignore[assignment]

FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<frontmatter>.*?)(?:\n---\s*\n|\n\.\.\.\s*\n)",
    re.DOTALL,
)
HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


class MarkdownLoader:
    """Load Obsidian Markdown while preserving raw frontmatter and body text."""

    def load(self, path: Path, vault_path: Path) -> ParsedMarkdown:
        self._touch_llama_index_reader(path)
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw_text)
        relative_path = path.relative_to(vault_path).as_posix()
        title = infer_title(body, path, metadata)
        is_memory = relative_path.startswith("memory/")
        metadata = normalize_metadata(metadata)
        metadata.setdefault("source_path", relative_path)
        metadata.setdefault("title", title)
        metadata.setdefault("is_memory", is_memory)
        return ParsedMarkdown(
            path=path,
            relative_path=relative_path,
            raw_text=raw_text,
            body=body,
            metadata=metadata,
            title=title,
            is_memory=is_memory,
        )

    def _touch_llama_index_reader(self, path: Path) -> None:
        if MarkdownReader is None:
            return
        try:
            reader = MarkdownReader()
            if hasattr(reader, "load_data"):
                reader.load_data(file=path)
        except Exception as exc:  # pragma: no cover - best effort integration hook.
            logger.debug("LlamaIndex MarkdownReader could not load %s: %s", path, exc)


def parse_frontmatter(raw_text: str) -> tuple[JsonDict, str]:
    match = FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    frontmatter = match.group("frontmatter")
    body = raw_text[match.end() :]
    metadata = parse_yaml_metadata(frontmatter)
    return metadata, body


def parse_yaml_metadata(frontmatter: str) -> JsonDict:
    try:
        parsed = yaml.safe_load(frontmatter) or {}
        if isinstance(parsed, dict):
            return dict(parsed)
    except yaml.YAMLError:
        logger.debug("Falling back to tolerant frontmatter parser")
    return tolerant_frontmatter_parse(frontmatter)


def tolerant_frontmatter_parse(frontmatter: str) -> JsonDict:
    metadata: JsonDict = {}
    current_key: str | None = None
    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ")):
            value = stripped[2:].strip()
            if current_key:
                current_value = metadata.setdefault(current_key, [])
                if not isinstance(current_value, list):
                    current_value = [current_value]
                    metadata[current_key] = current_value
                current_value.append(value)
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if value:
                metadata[key] = coerce_scalar(value)
            else:
                metadata[key] = [] if key == "tags" else None
    return metadata


def normalize_metadata(metadata: JsonDict) -> JsonDict:
    normalized = dict(metadata)
    tags = normalized.get("tags")
    if isinstance(tags, str):
        normalized["tags"] = [tags]
    elif tags is None:
        normalized["tags"] = []
    elif isinstance(tags, list):
        normalized["tags"] = [str(tag) for tag in tags]
    else:
        normalized["tags"] = [str(tags)]
    return normalized


def coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = yaml.safe_load(value)
            if isinstance(parsed, list):
                return parsed
        except yaml.YAMLError:
            return value
    return value.strip("\"'")


def infer_title(body: str, path: Path, metadata: JsonDict) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    match = HEADING_RE.search(body)
    if match:
        return match.group(1).strip()
    return path.stem
