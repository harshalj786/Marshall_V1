"""The code tools: `check_code` (compile or syntax-check, never run) and `run_code`.

Factories, like `make_recall_user_memory`, because what each tool may do depends on the
session it is built for: a guest's `check_code` takes pasted code only, and `run_code` closes
over a confirmer that asks the user OUTSIDE the model. Which session gets which tool is
`AgentGraph`'s decision; this module only builds them.

These two tools are the only ones allowed to start a subprocess or open a file by path —
`docs/security.md` §4 carries the exception and the guards it relies on, and the guards
themselves are in `tools/code_guards.py`.

Neither tool goes through `_run_with_timeout`. That wrapper abandons a thread it cannot
kill, which is right for a hung socket and wrong here: these tools own real subprocesses
with their own time limits, and `run_code` waits on a human answering a prompt.
"""
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool, BaseTool

from tools.code_guards import (CodeGuardError, find_forbidden_includes, normalize_language,
                               resolve_source_path, strip_code_fences)
from tools.tools import _cap_output


#: generous for one translation unit with system headers — `<iostream>` alone costs about a
#: second cold on MinGW — and still short enough that a pathological template cannot hold a
#: voice turn for long
COMPILE_TIMEOUT_SECONDS = 20.0

#: appended to every successful Python check, so Marshall never claims more than was checked.
#: Compiling Python finds syntax errors only; everything else needs it to run.
SYNTAX_ONLY_NOTE = ("syntax only: errors that happen while running, such as a division by zero "
                    "or a missing import, are not detected without running it")

_COMPILERS = {"c": "gcc", "cpp": "g++"}
_PASTED_SUFFIX = {"python": ".py", "c": ".c", "cpp": ".cpp"}


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    report: str


def compiler_for(language: str) -> Optional[str]:
    """The compiler executable for `language`, or None for Python or when it is not on PATH."""
    name = _COMPILERS.get(language)
    return shutil.which(name) if name else None


def check_python(source: str, display_name: str = "code") -> CheckResult:
    """Compile `source` to bytecode and throw it away. Nothing is executed or imported."""
    try:
        compile(source, display_name, "exec")
    except SyntaxError as err:
        line = (err.text or "").rstrip()
        report = f"check_code: python syntax error at line {err.lineno}: {err.msg}"
        return CheckResult(False, report + (f"\n    {line}" if line else ""))
    except ValueError as err:          # e.g. "source code string cannot contain null bytes"
        return CheckResult(False, f"check_code: python syntax error: {err}")
    return CheckResult(True, f"check_code: python syntax OK ({SYNTAX_ONLY_NOTE})")


def check_c_family(source_path: Path, language: str, display_name: str) -> CheckResult:
    """`-fsyntax-only -Wall`: full parse and type check, no object file, no executable."""
    compiler = compiler_for(language)
    name = _COMPILERS[language]
    if compiler is None:
        return CheckResult(False, f"check_code error: no {language} compiler found: "
                                  f"{name} is not on PATH")
    #: a scratch cwd, so nothing the compiler might drop lands in the repo
    with tempfile.TemporaryDirectory(prefix="marshall-check-") as scratch:
        try:
            completed = subprocess.run(
                [compiler, "-fsyntax-only", "-Wall", str(source_path)],
                capture_output=True, text=True, errors="replace",
                timeout=COMPILE_TIMEOUT_SECONDS, cwd=scratch, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return CheckResult(False, f"check_code error: compiler timed out after "
                                      f"{COMPILE_TIMEOUT_SECONDS:g}s")
    #: the temp path means nothing to the user and is not the model's business either
    stderr = (completed.stderr or "").replace(str(source_path), display_name).strip()
    if completed.returncode != 0:
        return CheckResult(False, f"check_code: {language} compile FAILED\n{stderr}")
    report = f"check_code: {language} compiled cleanly ({name} -fsyntax-only -Wall)"
    return CheckResult(True, report + (f"\nwarnings:\n{stderr}" if stderr else ""))


def _check(source: str, language: str, display_name: str) -> CheckResult:
    if language == "python":
        return check_python(source, display_name)
    with tempfile.TemporaryDirectory(prefix="marshall-src-") as workdir:
        src = Path(workdir) / f"pasted{_PASTED_SUFFIX[language]}"
        src.write_text(source, encoding="utf-8")
        return check_c_family(src, language, display_name)


def make_check_code(allow_paths: bool) -> BaseTool:
    """Build `check_code`. `allow_paths=False` is the guest tool: pasted code only."""

    @tool
    def check_code(code: str = "", language: str = "", path: str = "") -> str:
        """Compile or syntax-check a piece of Python, C or C++ code WITHOUT running it.

        Use this whenever the user gives you code and wants to know if it is correct, or asks
        what it does: check it first, then explain. `language` is "python", "c" or "cpp".
        Pass the code exactly as the user gave it in `code`. If the user names a file instead,
        pass its location in `path` and leave `code` empty.

        Returns either a success line or the compiler's own error messages with line numbers.
        A Python check finds syntax errors only; it cannot find errors that happen while
        running."""
        return _cap_output("check_code", _check_code_body(code, language, path, allow_paths))

    return check_code


def _check_code_body(code: str, language: str, path: str, allow_paths: bool) -> str:
    try:
        if (path or "").strip():
            if not allow_paths:
                return ("check_code refused: file paths are not available in a guest session; "
                        "paste the code instead")
            resolved, lang = resolve_source_path(path, language)
            source = resolved.read_text(encoding="utf-8", errors="replace")
            if lang == "python":
                return check_python(source, path.strip()).report
            return check_c_family(resolved, lang, path.strip()).report
        source = strip_code_fences(code)
        if not (language or "").strip():
            return "check_code error: say which language the code is: python, c or cpp"
        lang = normalize_language(language)
        if lang != "python" and not allow_paths:
            hits = find_forbidden_includes(source)
            if hits:
                return ("check_code refused: guest code may only #include <system headers>; "
                        f"found: {hits[0].strip()}")
        return _check(source, lang, "code").report
    except CodeGuardError as err:
        return f"check_code refused: {err}"
