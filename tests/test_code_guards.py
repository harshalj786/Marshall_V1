"""The pure guards in front of the code tools: language, fences, path policy, includes.

    pytest tests/test_code_guards.py -v

No compiler and no model. These are the checks that decide what the code tools are allowed
to look at, and they matter because compiler and parser diagnostics QUOTE source lines: a
checker that opens a file, or compiles an `#include "anything"`, is a file reader. The guest
half of `docs/security.md`'s code-tools exception lives or dies here.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.code_guards import (CodeGuardError, SOURCE_EXTENSIONS, find_forbidden_includes,
                               normalize_language, resolve_source_path, strip_code_fences)


@pytest.mark.parametrize("given,expected", [
    ("Python", "python"), ("py", "python"), ("python3", "python"), ("C", "c"),
    ("c++", "cpp"), ("CPP", "cpp"), ("cxx", "cpp"), ("cc", "cpp"), (" c ", "c"),
])
def test_normalize_language_accepts_aliases(given, expected):
    assert normalize_language(given) == expected


@pytest.mark.parametrize("bad", ["java", "", "   ", "rust"])
def test_normalize_language_rejects_everything_else(bad):
    with pytest.raises(CodeGuardError):
        normalize_language(bad)


def test_source_extensions_cover_exactly_the_three_languages():
    assert set(SOURCE_EXTENSIONS.values()) == {"python", "c", "cpp"}


def test_strip_code_fences_removes_a_fenced_block():
    assert strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    assert strip_code_fences("```\nx = 1\n```\n") == "x = 1"
    # unfenced text is untouched, trailing newline and all
    assert strip_code_fences("print(1)\n") == "print(1)\n"


def test_resolve_source_path_infers_language_from_extension(tmp_path):
    p = tmp_path / "a.cpp"
    p.write_text("int main(){}")
    assert resolve_source_path(str(p)) == (p.resolve(), "cpp")


def test_resolve_source_path_accepts_a_matching_language(tmp_path):
    p = tmp_path / "a.h"
    p.write_text("int x;")
    assert resolve_source_path(str(p), "C") == (p.resolve(), "c")


@pytest.mark.parametrize("name", ["notes.txt", "password.hash", ".env", ".ENV.local", ".env.py"])
def test_resolve_source_path_refuses_non_source_and_env_files(tmp_path, name):
    p = tmp_path / name
    p.write_text("x")
    with pytest.raises(CodeGuardError):
        resolve_source_path(str(p))


def test_resolve_source_path_refuses_a_language_that_contradicts_the_extension(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1")
    with pytest.raises(CodeGuardError):
        resolve_source_path(str(p), "c")


def test_resolve_source_path_refuses_missing_and_empty(tmp_path):
    with pytest.raises(CodeGuardError):
        resolve_source_path(str(tmp_path / "nope.py"))
    with pytest.raises(CodeGuardError):
        resolve_source_path("")


def test_resolve_source_path_refuses_a_directory_with_a_source_suffix(tmp_path):
    d = tmp_path / "pkg.py"
    d.mkdir()
    with pytest.raises(CodeGuardError):
        resolve_source_path(str(d))


def test_find_forbidden_includes_allows_system_headers():
    assert find_forbidden_includes("#include <stdio.h>\n#include <vector>\nint main(){}") == []


@pytest.mark.parametrize("line", [
    '#include "secret.h"',
    "#include <../x.h>",
    "#include </etc/passwd>",
    "#include <C:\\Users\\x\\.env>",
    '# include "a.h"',
    '#embed "x"',
    "#include_next <../a>",
    '#import "a.h"',
    "#include MACRO_NAME",
    '  #  include   "indented.h"',
])
def test_find_forbidden_includes_catches_each_form(line):
    assert find_forbidden_includes("int x;\n" + line + "\nint y;") == [line]


def test_find_forbidden_includes_reports_every_hit_in_order():
    code = '#include "a.h"\n#include <stdio.h>\n#include "b.h"\n'
    assert find_forbidden_includes(code) == ['#include "a.h"', '#include "b.h"']
