from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from framework.ingestion.hashing import sha256_text
from framework.schemas import ChunkRecord, ParsedMarkdown

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class MarkdownBlock:
    text: str
    heading_path: list[str] = field(default_factory=list)


class MarkdownChunker:
    """Heading-aware chunker that respects fenced code blocks and callout text."""

    def __init__(self, target_chars: int = 1600, overlap_chars: int = 220) -> None:
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(
        self,
        parsed: ParsedMarkdown,
        modified_time: datetime | None = None,
    ) -> list[ChunkRecord]:
        blocks = split_heading_blocks(parsed.body)
        chunks: list[ChunkRecord] = []
        ordinal = 0
        for block in blocks:
            for chunk_text in self._split_block(block.text):
                clean_text = chunk_text.strip()
                if not clean_text:
                    continue
                chunk_hash = sha256_text(clean_text)
                chunk_id = stable_chunk_id(parsed.relative_path, ordinal, chunk_hash)
                metadata = dict(parsed.metadata)
                metadata["heading_path"] = block.heading_path
                metadata["chunk_hash"] = chunk_hash
                metadata["chunk_ordinal"] = ordinal
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        file_path=parsed.relative_path,
                        ordinal=ordinal,
                        chunk_hash=chunk_hash,
                        text=clean_text,
                        heading_path=block.heading_path,
                        metadata=metadata,
                        token_count=approx_token_count(clean_text),
                        is_memory=parsed.is_memory,
                        modified_time=modified_time,
                    )
                )
                ordinal += 1
        return chunks

    def _split_block(self, text: str) -> list[str]:
        if len(text) <= self.target_chars:
            return [text]
        paragraphs = split_paragraphs_keep_code(text)
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if not current:
                current = paragraph
                continue
            if len(current) + len(paragraph) + 2 <= self.target_chars:
                current = f"{current}\n\n{paragraph}"
                continue
            chunks.append(current)
            overlap = tail_overlap(current, self.overlap_chars)
            current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
        if current:
            chunks.append(current)
        return chunks


def split_heading_blocks(markdown_text: str) -> list[MarkdownBlock]:
    blocks: list[MarkdownBlock] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    in_code = False

    for line in markdown_text.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
        heading_match = None if in_code else HEADING_RE.match(line)
        if heading_match:
            if current_lines:
                blocks.append(MarkdownBlock("\n".join(current_lines).strip(), list(current_path)))
                current_lines = []
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = [(lvl, value) for lvl, value in heading_stack if lvl < level]
            heading_stack.append((level, heading))
            current_path = [value for _, value in heading_stack]
        current_lines.append(line)

    if current_lines:
        blocks.append(MarkdownBlock("\n".join(current_lines).strip(), list(current_path)))

    return [block for block in blocks if block.text]


def split_paragraphs_keep_code(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    in_code = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
        if not in_code and not line.strip():
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append("\n".join(current).strip())
    return paragraphs


def tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if len(text) <= overlap_chars else ""
    tail = text[-overlap_chars:]
    first_space = tail.find(" ")
    if first_space > 0:
        return tail[first_space + 1 :].strip()
    return tail.strip()


def stable_chunk_id(file_path: str, ordinal: int, chunk_hash: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hypervault:{file_path}:{ordinal}:{chunk_hash}"))


def approx_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))
