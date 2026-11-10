"""The two code tools: `check_code` (compile or syntax-check, never run) and `run_code`.

    pytest tests/test_code_tools.py -v

No model. The gcc/g++ tests skip on a machine without them — skipped means "not verifiable
here", never "passing". Everything a guest can reach is asserted to refuse BEFORE any
compiler runs, because a compiler diagnostic quotes the source line it complains about.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.code_tools as code_tools
from tools.code_tools import (COMPILE_TIMEOUT_SECONDS, SYNTAX_ONLY_NOTE, CheckResult,
                              check_c_family, check_python, make_check_code)
from tools.tools import TOOL_OUTPUT_MAX_CHARS

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None or shutil.which("g++") is None,
                               reason="needs gcc and g++ on PATH")


# ==========================================================================
# check_code
# ==========================================================================

def test_check_python_reports_the_line_of_a_syntax_error():
    r = check_python("def f(:\n    pass\n")
    assert not r.ok
    assert "line 1" in r.report
    assert "def f(:" in r.report


def test_check_python_ok_says_it_is_syntax_only():
    # a runtime bug passes the check: that is the point of the caveat
    r = check_python("x = 1 / 0\n")
    assert r.ok and SYNTAX_ONLY_NOTE in r.report


def test_check_python_never_executes(tmp_path):
    marker = tmp_path / "ran"
    check_python(f"open({str(marker)!r}, 'w').close()\n")
    assert not marker.exists()


@needs_gcc
def test_check_c_reports_the_line_of_an_error(tmp_path):
    src = tmp_path / "pasted.c"
    src.write_text("int main(void) { return 0 }\n")
    r = check_c_family(src, "c", "code")
    assert not r.ok
    assert "code:1" in r.report
    assert "expected" in r.report
    # the temp path is replaced by the display name, so the model never sees it
    assert str(src) not in r.report


@needs_gcc
def test_check_cpp_compiles_cleanly_and_makes_no_executable(tmp_path):
    src = tmp_path / "pasted.cpp"
    src.write_text("#include <vector>\nint main(){ std::vector<int> v; return (int)v.size(); }\n")
    r = check_c_family(src, "cpp", "code")
    assert r.ok and "compiled cleanly" in r.report
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pasted.cpp"]


def test_check_code_tool_is_named_and_documented():
    tool = make_check_code(allow_paths=False)
    assert tool.name == "check_code"
    assert "WITHOUT running it" in tool.description
    assert set(tool.args) == {"code", "language", "path"}


def test_guest_check_code_refuses_paths(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("x = 1\n")
    out = make_check_code(False).invoke({"code": "", "language": "python", "path": str(p)})
    assert out == ("check_code refused: file paths are not available in a guest session; "
                   "paste the code instead")


def test_guest_check_code_refuses_quoted_includes_before_any_compiler_runs(monkeypatch):
    monkeypatch.setattr(code_tools.subprocess, "run", lambda *a, **k: pytest.fail("compiler ran"))
    out = make_check_code(False).invoke({"code": '#include "x.h"\nint main(){}', "language": "c"})
    assert out.startswith("check_code refused: guest code may only #include <system headers>; "
                          'found: #include "x.h"')


def test_admin_check_code_checks_a_file_and_infers_the_language(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def f(:\n")
    assert "python syntax error at line 1" in make_check_code(True).invoke({"path": str(p)})


@pytest.mark.parametrize("name", [".env", "password.hash"])
def test_admin_check_code_refuses_env_and_hash_files(tmp_path, name):
    p = tmp_path / name
    p.write_text("SECRET=1\n")
    out = make_check_code(True).invoke({"path": str(p), "language": "c"})
    assert out.startswith("check_code refused:")
    assert "SECRET" not in out


def test_check_code_strips_markdown_fences():
    out = make_check_code(False).invoke(
        {"code": "```python\nprint(1)\n```", "language": "python"})
    assert "python syntax OK" in out


def test_check_code_without_language_asks_for_it():
    assert make_check_code(False).invoke({"code": "print(1)"}) == (
        "check_code error: say which language the code is: python, c or cpp")


def test_check_code_output_is_capped(monkeypatch):
    monkeypatch.setattr(code_tools, "check_python", lambda *a, **k: CheckResult(False, "e\n" * 20000))
    out = make_check_code(False).invoke({"code": "x", "language": "python"})
    assert "check_code truncated" in out and len(out) < 2 * TOOL_OUTPUT_MAX_CHARS


def test_check_code_reports_a_missing_compiler(monkeypatch):
    monkeypatch.setattr(code_tools, "compiler_for", lambda language: None)
    out = make_check_code(False).invoke({"code": "int main(){}", "language": "c"})
    assert out == "check_code error: no c compiler found: gcc is not on PATH"


def test_check_code_reports_a_compiler_timeout(monkeypatch):
    import subprocess

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="gcc", timeout=COMPILE_TIMEOUT_SECONDS)
    monkeypatch.setattr(code_tools, "compiler_for", lambda language: "gcc")
    monkeypatch.setattr(code_tools.subprocess, "run", slow)
    out = make_check_code(False).invoke({"code": "int main(){}", "language": "c"})
    assert out == f"check_code error: compiler timed out after {COMPILE_TIMEOUT_SECONDS:g}s"
