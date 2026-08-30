#!/usr/bin/env python3
"""Check that local Markdown link targets exist."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_DIRS = {".git", ".serena", "build"}


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts)
    )


def local_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        # A title after a whitespace-separated path is valid Markdown.
        target = target.split(maxsplit=1)[0]

    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("#") or not parsed.path:
        return None
    return unquote(parsed.path)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    missing: list[tuple[Path, int, str]] = []

    for document in markdown_files(root):
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for match in LINK_RE.finditer(line):
                target = local_target(match.group(1))
                if target is None:
                    continue
                candidate = Path(target) if target.startswith("/") else document.parent / target
                if not candidate.exists():
                    missing.append((document.relative_to(root), line_number, target))

    if missing:
        for document, line_number, target in missing:
            print(f"{document}:{line_number}: missing local target: {target}")
        print(f"Found {len(missing)} missing local Markdown link(s).", file=sys.stderr)
        return 1

    count = len(markdown_files(root))
    print(f"Checked {count} Markdown files: all local link targets exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
