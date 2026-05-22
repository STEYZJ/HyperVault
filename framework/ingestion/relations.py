from __future__ import annotations

import re

from framework.schemas import RelationRecord

WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/-]+)")


def extract_wikilink_targets(markdown_text: str) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for match in WIKILINK_RE.finditer(markdown_text):
        target = match.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


def extract_inline_tags(markdown_text: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for match in TAG_RE.finditer(markdown_text):
        tag = match.group(1).strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def extract_relations(source_path: str, markdown_text: str, metadata: dict) -> list[RelationRecord]:
    relations: list[RelationRecord] = []
    for target in extract_wikilink_targets(markdown_text):
        relations.append(
            RelationRecord(
                source_path=source_path,
                target=target,
                relation_type="wikilink",
                context=None,
            )
        )

    for tag in metadata.get("tags") or []:
        relations.append(
            RelationRecord(
                source_path=source_path,
                target=str(tag),
                relation_type="tag",
                context=None,
            )
        )

    for tag in extract_inline_tags(markdown_text):
        relations.append(
            RelationRecord(
                source_path=source_path,
                target=tag,
                relation_type="tag",
                context=None,
            )
        )

    return relations

