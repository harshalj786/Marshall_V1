"""Marshall must not know the control plane exists.

    pytest tests/ -k forbidden_repo -v

The dependency runs one way: that package may import from here, and nothing here may
import, name, or be documented in terms of it. The agent_hook seam in pipeline_lg.py
is the whole interface, and it is generic — any (node, router) pair.

This is a grep, deliberately. An import check would miss a string, a string check
would miss a comment, and a comment that explains Marshall's code in terms of another
repo is the coupling that survives every refactor. Every .py file under the repo is
scanned, this file included: the tokens are assembled at runtime so the scanner does
not find itself.
"""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: joined at runtime so this file contains neither token as a literal
FORBIDDEN = tuple("_".join(parts) for parts in (("control", "plane"), ("CONTROL", "PLANE")))
_PATTERN = re.compile("|".join(re.escape(t) for t in FORBIDDEN))

#: top-level directories that are not this repo's source
SKIP_DIRS = {".git", "db", "venv", ".venv", "env", "ENV", "__pycache__", ".pytest_cache",
             "node_modules"}


def python_files():
    for path in REPO.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        yield path


def test_forbidden_repo_scanner_has_files_to_scan():
    assert any(python_files()), "the scanner found nothing, so it can prove nothing"


def test_no_python_file_mentions_the_forbidden_repo():
    hits = []
    for path in python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace")
                                      .splitlines(), start=1):
            if _PATTERN.search(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not hits, "Marshall must not mention the control plane:\n" + "\n".join(hits)


def test_forbidden_repo_scanner_would_catch_a_mention(tmp_path, monkeypatch):
    """Prove the negative result means something: a planted mention is found."""
    planted = tmp_path / "planted.py"
    planted.write_text(f"# {FORBIDDEN[0]} mentioned here\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    with pytest.raises(AssertionError, match="planted.py:1"):
        test_no_python_file_mentions_the_forbidden_repo()
