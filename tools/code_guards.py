"""Pure guards for the code tools: which language, which file, which includes.

Nothing here runs a compiler or touches the model. These checks exist because compiler
and parser diagnostics QUOTE source lines — gcc prints the offending line under every
error, and a Python SyntaxError carries its text. So a checker that opens any path it is
handed, or compiles `#include "C:/somewhere/.env"`, reads that file back to whoever asked.
`docs/security.md` §4's code-tools exception rests on the refusals below.

Stdlib only, so `tools/` stays a leaf module.
"""
import re
from pathlib import Path


class CodeGuardError(ValueError):
    """A refusal. The message is shown to the model as the reason, so it is plain English."""


#: the only files either code tool will open. Headers count as source: `.h` is checked as C,
#: `.hpp` as C++. The extension DECIDES the language — see `resolve_source_path`.
SOURCE_EXTENSIONS = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

_LANGUAGE_ALIASES = {
    "python": "python", "py": "python", "python3": "python",
    "c": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp",
}

#: one fenced block, optionally tagged — the shape a model passes through when the user
#: pasted code inside ``` in the chat. Only a WHOLE-string fence is stripped; a fence in the
#: middle of real code is the code's own business.
_FENCE = re.compile(r"^\s*```[\w+#.-]*[ \t]*\r?\n(.*?)\r?\n?```\s*$", re.DOTALL)

#: every preprocessor directive that pulls another file in. `#import` is GCC's Objective-C
#: legacy that still works in C; `#embed` is C23 and pulls raw bytes, which is worse.
_DIRECTIVE = re.compile(r"^\s*#\s*(include_next|include|import|embed)\b\s*(.*)$")
_ANGLE = re.compile(r"^<([^>]*)>")
_DRIVE = re.compile(r"^[A-Za-z]:")


def normalize_language(language: str) -> str:
    """Map a language name the model or user typed to "python", "c" or "cpp"."""
    key = (language or "").strip().lower()
    if key not in _LANGUAGE_ALIASES:
        raise CodeGuardError(
            f"unsupported language {language!r}: only python, c and cpp are supported")
    return _LANGUAGE_ALIASES[key]


def strip_code_fences(code: str) -> str:
    """Remove one markdown fence wrapping the whole input; anything else is returned as is."""
    match = _FENCE.match(code or "")
    return match.group(1) if match else (code or "")


def resolve_source_path(path: str, language: str = "") -> tuple[Path, str]:
    """Check that `path` is a source file a code tool may open. Returns (path, language).

    Refusal order matters only for the message: the `.env` check comes before the suffix
    check so `.env.py` is refused as what it is, not as a Python file.
    """
    if not (path or "").strip():
        raise CodeGuardError("no file path given")
    candidate = Path(path.strip())
    if candidate.name.lower().startswith(".env"):
        raise CodeGuardError(f"{candidate.name} is an environment file and is never read")
    suffix = candidate.suffix.lower()
    if suffix not in SOURCE_EXTENSIONS:
        raise CodeGuardError(f"{candidate.name} is not a Python/C/C++ source file")
    if not candidate.is_file():
        raise CodeGuardError(f"no such file: {path}")
    inferred = SOURCE_EXTENSIONS[suffix]
    #: the language argument can AGREE with the extension, never override it — otherwise
    #: `language="c"` would let any file through as long as it carried a source suffix
    if (language or "").strip() and normalize_language(language) != inferred:
        raise CodeGuardError(
            f"{candidate.name} is a {inferred} file; the language cannot be overridden")
    return candidate.resolve(), inferred


def _forbidden_operand(operand: str) -> bool:
    angle = _ANGLE.match(operand)
    if angle is None:
        # quoted, or a macro that expands to who-knows-what: both can name any file
        return True
    target = angle.group(1).strip()
    return ".." in target or target.startswith(("/", "\\")) or bool(_DRIVE.match(target))


def find_forbidden_includes(code: str) -> list[str]:
    """Every line that would make the compiler open something other than a system header."""
    hits = []
    for line in (code or "").splitlines():
        match = _DIRECTIVE.match(line)
        if match and _forbidden_operand(match.group(2).strip()):
            hits.append(line)
    return hits
