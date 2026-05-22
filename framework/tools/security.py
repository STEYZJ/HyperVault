from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from framework.config import PROJECT_ROOT

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{16,}")),
    (
        "assigned_secret",
        re.compile(
            r"\b(?:OPENAI_API_KEY|GITHUB_TOKEN|QDRANT_API_KEY)\s*=\s*"
            r"(?!(?:$|\.{3}|<[^>]+>|\"\"|''))"
            r"[^\s#`]+"
        ),
    ),
)


class SecretFinding(BaseModel):
    path: str
    line: int
    kind: str
    excerpt: str


class SecretScanResult(BaseModel):
    ok: bool
    scanned_files: int
    findings: list[SecretFinding] = Field(default_factory=list)
    blocked_tracked_paths: list[str] = Field(default_factory=list)


def run_secret_scan(root: Path = PROJECT_ROOT) -> SecretScanResult:
    tracked_files = git_tracked_files(root)
    findings: list[SecretFinding] = []
    blocked_tracked_paths = [
        path for path in tracked_files if path == ".env" or path.startswith("runtime/")
    ]
    for relative_path in tracked_files:
        path = root / relative_path
        if not path.is_file() or is_binary(path):
            continue
        findings.extend(scan_file(path, root))
    return SecretScanResult(
        ok=not findings and not blocked_tracked_paths,
        scanned_files=len(tracked_files),
        findings=findings,
        blocked_tracked_paths=blocked_tracked_paths,
    )


def git_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def scan_file(path: Path, root: Path) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    relative_path = path.relative_to(root).as_posix()
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    SecretFinding(
                        path=relative_path,
                        line=line_number,
                        kind=kind,
                        excerpt=redact_secret(line.strip()),
                    )
                )
    return findings


def is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:1024]
    except OSError:
        return True
    return b"\x00" in chunk


def redact_secret(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}", "sk-***REDACTED***", text)
    text = re.sub(
        r"\b(OPENAI_API_KEY|GITHUB_TOKEN|QDRANT_API_KEY)\s*=\s*[^\s#]+",
        r"\1=***REDACTED***",
        text,
    )
    return text
