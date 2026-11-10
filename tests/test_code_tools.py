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


# ==========================================================================
# run_code
# ==========================================================================

import os
import subprocess
import time

from tools.code_tools import RUN_CAPTURE_BYTES, RunRequest, make_run_code, run_program

PY = sys.executable


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_run_program_captures_output_and_exit_code(tmp_path):
    r = run_program([PY, "-c", "import sys; print('hi'); sys.exit(3)"], tmp_path)
    assert (r.exit_code, r.timed_out) == (3, False) and "hi" in r.stdout


def test_run_program_gets_no_stdin(tmp_path):
    r = run_program([PY, "-c", "input()"], tmp_path, timeout=5.0)
    assert r.exit_code != 0 and "EOFError" in r.stderr and r.seconds < 5


def test_run_program_kills_a_looping_program(tmp_path):
    r = run_program([PY, "-c", "while True: pass"], tmp_path, timeout=1.0)
    assert r.timed_out and r.exit_code is None and r.seconds < 4


def test_run_program_kills_grandchildren_on_timeout(tmp_path):
    pidfile = tmp_path / "child.pid"
    parent = ("import subprocess, sys, time; "
              "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              f"open({str(pidfile)!r}, 'w').write(str(c.pid)); time.sleep(60)")
    r = run_program([PY, "-c", parent], tmp_path, timeout=3.0)
    assert r.timed_out
    child = int(pidfile.read_text())
    time.sleep(0.5)
    assert not _pid_alive(child)


def test_run_program_caps_huge_output(tmp_path):
    r = run_program([PY, "-c", "print('x' * 5_000_000)"], tmp_path)
    assert r.exit_code == 0 and len(r.stdout) <= RUN_CAPTURE_BYTES


def test_run_code_declined_executes_nothing(tmp_path):
    marker = tmp_path / "ran"
    out = make_run_code(lambda r: False).invoke(
        {"code": f"open({str(marker)!r}, 'w').close()", "language": "python"})
    assert out == "run_code: the user declined to run it; nothing was executed"
    assert not marker.exists()


def test_run_code_passes_the_request_to_confirm():
    seen = []
    make_run_code(lambda r: seen.append(r) or False).invoke(
        {"code": "```python\nprint(1)\n```", "language": "python"})
    assert seen == [RunRequest(language="python", source="print(1)", origin="pasted")]


def test_run_code_runs_python_when_confirmed():
    out = make_run_code(lambda r: True).invoke({"code": "print(2 + 2)", "language": "python"})
    assert "run_code: exit code 0" in out and "\n4\n" in out


def test_run_code_python_runs_isolated():
    out = make_run_code(lambda r: True).invoke(
        {"code": "import sys; print(sys.flags.isolated)", "language": "python"})
    assert "\n1\n" in out


def test_run_code_reports_a_timeout(monkeypatch):
    monkeypatch.setattr(code_tools, "RUN_TIMEOUT_SECONDS", 1.0)
    out = make_run_code(lambda r: True).invoke({"code": "while True: pass", "language": "python"})
    assert out.startswith("run_code: killed after 1s time limit")


def test_run_code_refuses_env_files_before_asking(tmp_path):
    p = tmp_path / ".env"
    p.write_text("SECRET=1\n")
    asked = []
    out = make_run_code(lambda r: asked.append(r) or True).invoke({"path": str(p)})
    assert out.startswith("run_code refused:") and asked == []


def test_run_code_runs_a_file_by_path(tmp_path):
    p = tmp_path / "hello.py"
    p.write_text("print('from file')\n")
    seen = []
    out = make_run_code(lambda r: seen.append(r) or True).invoke({"path": str(p)})
    assert "from file" in out and seen[0].origin == str(p) and seen[0].language == "python"


@needs_gcc
def test_run_code_compiles_and_runs_c():
    out = make_run_code(lambda r: True).invoke(
        {"code": '#include <stdio.h>\nint main(void){printf("hello\\n");return 7;}',
         "language": "c"})
    assert "run_code: exit code 7" in out and "hello" in out


@needs_gcc
def test_run_code_reports_compile_failure_without_running():
    out = make_run_code(lambda r: True).invoke(
        {"code": "int main(void){ return 0 }", "language": "c"})
    assert out.startswith("run_code: c compile FAILED") and "--- stdout ---" not in out


def test_run_code_tool_is_named_and_documented():
    tool = make_run_code(lambda r: False)
    assert tool.name == "run_code"
    assert "ONLY when the user explicitly asks" in tool.description
