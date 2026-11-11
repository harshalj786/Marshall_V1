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
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from langchain_core.tools import tool, BaseTool

from tools.code_guards import (CodeGuardError, find_forbidden_includes, find_secret_references,
                               is_secret_name, normalize_language, resolve_source_path,
                               strip_code_fences)
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
_GCC_LANGUAGE = {"c": "c", "cpp": "c++"}

#: no caret line under diagnostics. The caret line is gcc RE-READING a source line from
#: disk, from whatever file a `#line` directive named — that is how `#line 1 "secret"`
#: printed the secret. With it off, a diagnostic carries a file name, never file contents.
_CARET_OFF = "-fno-diagnostics-show-caret"

#: BOM-tolerant: Windows editors save "UTF-8 with signature", and a stray U+FEFF would be
#: reported as a syntax error on line 1 of correct code
_SOURCE_ENCODING = "utf-8-sig"


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
                [compiler, "-fsyntax-only", "-Wall", _CARET_OFF, str(source_path)],
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


@lru_cache(maxsize=None)
def _system_include_dirs(compiler: str, language: str) -> tuple:
    """The compiler's own `#include <...>` search list, as `gcc -E -v` reports it."""
    completed = subprocess.run(
        [compiler, "-x", _GCC_LANGUAGE[language], "-E", "-v", "-"],
        input="", capture_output=True, text=True, errors="replace",
        timeout=COMPILE_TIMEOUT_SECONDS)
    dirs, inside = [], False
    for line in (completed.stderr or "").splitlines():
        if line.startswith("#include <...> search starts here"):
            inside = True
        elif line.startswith("End of search list"):
            break
        elif inside and line.strip():
            dirs.append(os.path.normcase(os.path.realpath(line.strip())))
    return tuple(dirs)


def _dependencies(compiler: str, source_path: Path) -> list:
    """Every file the preprocessor opens for `source_path`, via `gcc -M`. Output only parsed.

    Its stdout is a make rule and its stderr is discarded — this pass exists to LOOK at what
    would be read, never to show anything. A failed pass (a missing header is fatal) still
    yields the files read up to that point, and the real check stops at the same place.
    `#embed` files are NOT listed by `-M`; `find_forbidden_includes` and
    `find_secret_references` are what stop those.
    """
    #: a timeout propagates rather than returning [] — "nothing was listed" must never be
    #: read as "nothing was read"; the tool bodies turn it into the compiler-timeout report
    with tempfile.TemporaryDirectory(prefix="marshall-deps-") as scratch:
        completed = subprocess.run(
            [compiler, "-M", str(source_path)], capture_output=True, text=True,
            errors="replace", timeout=COMPILE_TIMEOUT_SECONDS, cwd=scratch,
            stdin=subprocess.DEVNULL)
    rule = (completed.stdout or "").replace("\\\r\n", " ").replace("\\\n", " ")
    _, _, deps = rule.partition(": ")
    #: a make rule escapes spaces in file names as "\ "
    tokens = deps.replace("\\ ", "\0").split()
    return [token.replace("\0", " ") for token in tokens if token != "\\"]


def include_guard(source: str, language: str, guest: bool,
                  source_path: Optional[Path] = None) -> Optional[str]:
    """Why this C/C++ source may not be compiled, or None. Reads, never reports contents.

    Three layers, because compiler diagnostics quote what they read: a normalised text scan
    (comments, splices and digraphs cannot hide a directive), a `gcc -M` pass listing the
    files the preprocessor would really open, and `_CARET_OFF` on every compile. A guest may
    pull in system headers only; in every session `.env*` and the password hash are refused.
    """
    if guest:
        hits = find_forbidden_includes(source)
        if hits:
            return f"guest code may only #include <system headers>; found: {hits[0].strip()}"
    secret = find_secret_references(source)
    if secret:
        return f"{secret[0].strip()} names a secrets file, which is never read"
    compiler = compiler_for(language)
    if compiler is None:
        return None          # nothing can be compiled; the check reports the missing compiler
    with tempfile.TemporaryDirectory(prefix="marshall-guard-") as workdir:
        if source_path is None:
            source_path = Path(workdir) / f"pasted{_PASTED_SUFFIX[language]}"
            source_path.write_text(source, encoding="utf-8")
        own = os.path.normcase(os.path.realpath(source_path))
        system = _system_include_dirs(compiler, language)
        for dep in _dependencies(compiler, source_path):
            real = os.path.normcase(os.path.realpath(dep))
            if real == own:
                continue
            name = os.path.basename(real)
            if is_secret_name(name):
                return f"the code pulls in {name}, a secrets file, which is never read"
            if guest and not any(real.startswith(d + os.sep) for d in system):
                return ("guest code may only #include <system headers>; "
                        f"found a dependency outside them: {name}")
    return None


def _check(source: str, language: str, display_name: str, guest: bool) -> str:
    if language == "python":
        return check_python(source, display_name).report
    refusal = include_guard(source, language, guest)
    if refusal:
        return f"check_code refused: {refusal}"
    with tempfile.TemporaryDirectory(prefix="marshall-src-") as workdir:
        src = Path(workdir) / f"pasted{_PASTED_SUFFIX[language]}"
        src.write_text(source, encoding="utf-8")
        return check_c_family(src, language, display_name).report


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
            source = resolved.read_text(encoding=_SOURCE_ENCODING, errors="replace")
            if lang == "python":
                return check_python(source, path.strip()).report
            refusal = include_guard(source, lang, guest=False, source_path=resolved)
            if refusal:
                return f"check_code refused: {refusal}"
            return check_c_family(resolved, lang, path.strip()).report
        source = strip_code_fences(code)
        if not (language or "").strip():
            return "check_code error: say which language the code is: python, c or cpp"
        lang = normalize_language(language)
        return _check(source, lang, "code", guest=not allow_paths)
    except CodeGuardError as err:
        return f"check_code refused: {err}"
    except subprocess.TimeoutExpired:
        # from the guard's own gcc passes: fail closed, nothing was compiled
        return f"check_code error: compiler timed out after {COMPILE_TIMEOUT_SECONDS:g}s"


# ==========================================================================
# run_code
# ==========================================================================

#: wall-clock budget for one program run. Short on purpose: this is a voice assistant's
#: "try it and tell me", not a job runner, and a program that loops forever has to give the
#: turn back quickly.
RUN_TIMEOUT_SECONDS = 10.0

#: how much of each stream is read back. Output goes to files rather than pipes, so a
#: program printing gigabytes costs disk until it is killed — never the parent's memory —
#: and only this much is ever read. `_cap_output` then bounds what the model sees.
RUN_CAPTURE_BYTES = 65_536


@dataclass(frozen=True)
class RunRequest:
    """What the confirmer is shown before anything runs. `origin` is "pasted" or the path."""
    language: str
    source: str
    origin: str


@dataclass(frozen=True)
class RunResult:
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` AND everything it started. Killing only the parent orphans the rest."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _read_capped(path: Path) -> str:
    #: bytes, then newlines normalised: a Windows program writes "\r\n" in text mode, and
    #: the model should read the same output whichever OS ran the program
    with open(path, "rb") as handle:
        text = handle.read(RUN_CAPTURE_BYTES).decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n")


def run_program(argv: list[str], cwd: Path, timeout: float = RUN_TIMEOUT_SECONDS) -> RunResult:
    """Run `argv` in `cwd` with no stdin, kill the whole tree on timeout, read capped output."""
    cwd = Path(cwd)
    out_path, err_path = cwd / "stdout.txt", cwd / "stderr.txt"
    started = time.perf_counter()
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        #: a new session on POSIX makes the program a process-group leader, which is what
        #: lets `_kill_tree` take its children down with it; Windows uses taskkill /T
        proc = subprocess.Popen(argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=out,
                                stderr=err, start_new_session=(sys.platform != "win32"))
        try:
            exit_code: Optional[int] = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            exit_code, timed_out = None, True
    return RunResult(exit_code=exit_code, stdout=_read_capped(out_path),
                     stderr=_read_capped(err_path), timed_out=timed_out,
                     seconds=time.perf_counter() - started)


def compile_c_family(source_path: Path, language: str, workdir: Path,
                     display_name: str) -> tuple[CheckResult, Optional[Path]]:
    """Build an executable into `workdir`. Returns (result, executable or None)."""
    compiler = compiler_for(language)
    name = _COMPILERS[language]
    if compiler is None:
        return CheckResult(False, f"run_code error: no {language} compiler found: "
                                  f"{name} is not on PATH"), None
    exe = Path(workdir) / ("program.exe" if sys.platform == "win32" else "program")
    try:
        completed = subprocess.run(
            [compiler, "-Wall", _CARET_OFF, "-o", str(exe), str(source_path)],
            capture_output=True, text=True, errors="replace",
            timeout=COMPILE_TIMEOUT_SECONDS, cwd=str(workdir), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return CheckResult(False, f"run_code error: compiler timed out after "
                                  f"{COMPILE_TIMEOUT_SECONDS:g}s"), None
    stderr = (completed.stderr or "").replace(str(source_path), display_name).strip()
    if completed.returncode != 0 or not exe.exists():
        return CheckResult(False, f"run_code: {language} compile FAILED\n{stderr}"), None
    return CheckResult(True, stderr), exe


def _format_run(result: RunResult) -> str:
    if result.timed_out:
        head = (f"run_code: killed after {RUN_TIMEOUT_SECONDS:g}s time limit; "
                f"partial output follows")
    else:
        head = f"run_code: exit code {result.exit_code} in {result.seconds:.1f}s"
    return (f"{head}\n--- stdout ---\n{result.stdout or '(empty)'}"
            f"\n--- stderr ---\n{result.stderr or '(empty)'}")


def make_run_code(confirm: Callable[[RunRequest], bool]) -> BaseTool:
    """Build `run_code`. `confirm` asks the USER, outside the model; False means nothing runs.

    Only ever built for an admin session, so paths are always allowed here — subject to the
    same `.env`/extension policy as `check_code`.
    """

    @tool
    def run_code(code: str = "", language: str = "", path: str = "") -> str:
        """Compile (if C/C++) and RUN a piece of Python, C or C++ code, then return its exit
        code and output. Use ONLY when the user explicitly asks you to run or execute code.

        Before anything runs, the user is asked to confirm outside this conversation; if they
        decline, nothing runs and the result says so. `language` is "python", "c" or "cpp".
        Pass the code exactly as the user gave it in `code`, or a file location in `path`."""
        return _cap_output("run_code", _run_code_body(code, language, path, confirm))

    return run_code


def _run_code_body(code: str, language: str, path: str,
                   confirm: Callable[[RunRequest], bool]) -> str:
    try:
        resolved: Optional[Path] = None
        if (path or "").strip():
            resolved, lang = resolve_source_path(path, language)
            source = resolved.read_text(encoding=_SOURCE_ENCODING, errors="replace")
            origin = path.strip()
        else:
            source = strip_code_fences(code)
            if not (language or "").strip():
                return "run_code error: say which language the code is: python, c or cpp"
            lang = normalize_language(language)
            origin = "pasted"
    except CodeGuardError as err:
        return f"run_code refused: {err}"
    if lang != "python":
        #: before the confirmation, so a refusal costs the user nothing to answer
        try:
            refusal = include_guard(source, lang, guest=False, source_path=resolved)
        except subprocess.TimeoutExpired:
            return f"run_code error: compiler timed out after {COMPILE_TIMEOUT_SECONDS:g}s"
        if refusal:
            return f"run_code refused: {refusal}"

    #: the gate. Everything above only reads; nothing below happens without a yes.
    if not confirm(RunRequest(language=lang, source=source, origin=origin)):
        return "run_code: the user declined to run it; nothing was executed"

    workdir = Path(tempfile.mkdtemp(prefix="marshall-run-"))
    try:
        if resolved is None:
            src = workdir / f"pasted{_PASTED_SUFFIX[lang]}"
            src.write_text(source, encoding="utf-8")
            display = "code"
        else:
            src, display = resolved, origin
        if lang == "python":
            #: -I: isolated — no PYTHON* env vars, no user site, no script dir on sys.path
            argv = [sys.executable, "-I", str(src)]
        else:
            built, exe = compile_c_family(src, lang, workdir, display)
            if exe is None:
                return built.report
            argv = [str(exe)]
        #: read at call time so the limit can be changed (and tested) in one place
        return _format_run(run_program(argv, workdir, timeout=RUN_TIMEOUT_SECONDS))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
