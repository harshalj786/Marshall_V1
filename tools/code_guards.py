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
#: pasted code inside ``` in the chat. A fence around the WHOLE string is stripped; so is
#: exactly one fenced block inside prose ("here is my code: ```...``` is it right?"), which
#: is what llama3.1 passes when it hands over the user's whole message. Two or more blocks
#: are left alone: there is no telling which one was meant.
_FENCE = re.compile(r"^\s*```[\w+#.-]*[ \t]*\r?\n(.*?)\r?\n?```\s*$", re.DOTALL)
_INNER_FENCE = re.compile(r"```[\w+#.-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)

#: the directives that can make the compiler OPEN a file. `#import` is GCC's Objective-C
#: legacy that still works in C; `#embed` (C23, and in g++ 15) pulls raw bytes — g++ prints
#: them back as template arguments. `#line` and the GNU `# 1 "file"` linemarker open nothing
#: themselves, but gcc's caret diagnostics re-read source lines from the file they name.
_FILE_DIRECTIVES = frozenset({"include", "include_next", "import", "embed", "line"})
#: after normalisation: `#`, optional space, then a directive name or a linemarker's number
_DIRECTIVE = re.compile(r"^\s*#\s*([A-Za-z_]\w*|\d+)\b\s*(.*)$")
_ANGLE = re.compile(r"^<([^>]*)>")
_QUOTED_OR_ANGLE = re.compile(r'"([^"]*)"|<([^>]*)>')
_DRIVE = re.compile(r"^[A-Za-z]:")
#: whitespace between a backslash and the newline still splices, in gcc (with a warning)
_SPLICE = re.compile(r"\\[ \t]*\r?\n")
_RAW_PREFIX = re.compile(r"(?:^|[^A-Za-z0-9_])(?:u8|[uUL])?R$")


def is_secret_name(name: str) -> bool:
    """`.env*` and the admin password hash: never read by either code tool, in any session."""
    lowered = name.lower()
    return lowered.startswith(".env") or lowered == "password.hash"


def normalize_language(language: str) -> str:
    """Map a language name the model or user typed to "python", "c" or "cpp"."""
    key = (language or "").strip().lower()
    if key not in _LANGUAGE_ALIASES:
        raise CodeGuardError(
            f"unsupported language {language!r}: only python, c and cpp are supported")
    return _LANGUAGE_ALIASES[key]


def strip_code_fences(code: str) -> str:
    """Return the code inside a markdown fence; input without exactly one fence is unchanged."""
    match = _FENCE.match(code or "")
    if match:
        return match.group(1)
    blocks = _INNER_FENCE.findall(code or "")
    return blocks[0] if len(blocks) == 1 else (code or "")


def resolve_source_path(path: str, language: str = "") -> tuple[Path, str]:
    """Check that `path` is a source file a code tool may open. Returns (path, language).

    Refusal order matters only for the message: the `.env` check comes before the suffix
    check so `.env.py` is refused as what it is, not as a Python file.
    """
    if not (path or "").strip():
        raise CodeGuardError("no file path given")
    candidate = Path(path.strip())
    #: the resolved name too, so a `link.py` pointing at `.env` is refused as what it is
    if is_secret_name(candidate.name) or is_secret_name(candidate.resolve().name):
        raise CodeGuardError(f"{candidate.name} is a secrets file and is never read")
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


def _strip_comments(text: str) -> str:
    """Replace every comment with one space, the way the preprocessor does — string-aware.

    String-awareness is what makes this safe rather than merely tidy: a `/*` inside a string
    must not swallow the real code after it, or a directive hidden behind
    `"/*"` would vanish from the scan while gcc still obeys it. Ordinary strings and
    character literals end at a newline (an unterminated one is a compile error anyway), so
    a mistake can cost at most the rest of one line; raw strings and block comments are the
    only states that span lines, and both are matched exactly. A digit separator (`1'000`)
    is part of a number, not the start of a character literal.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = text.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")
        elif ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
        elif ch == '"' and _RAW_PREFIX.search(text[max(0, i - 3):i]):
            # R"delim( ... )delim" — no escapes, no splices, can span lines
            open_paren = text.find("(", i)
            delim = text[i + 1:open_paren] if open_paren != -1 else ""
            end = text.find(")" + delim + '"', open_paren + 1) if open_paren != -1 else -1
            stop = n if end == -1 else end + len(delim) + 2
            out.append(text[i:stop])
            i = stop
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            stop = min(j + 1, n) if j < n and text[j] == ch else j
            out.append(text[i:stop])
            i = stop
        elif ch.isdigit() and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            # a pp-number: digits, letters, `.`, `'` separators, and e+/e-/p+/p- exponents
            j = i + 1
            while j < n:
                c = text[j]
                if c.isalnum() or c in "._":
                    j += 1
                elif c == "'" and j + 1 < n and text[j + 1].isalnum():
                    j += 2
                elif c in "+-" and text[j - 1] in "eEpP":
                    j += 1
                else:
                    break
            out.append(text[i:j])
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def normalize_preprocessor_text(code: str) -> str:
    """The source as gcc's directive scanner sees it: translation phases 1-3, roughly.

    Trigraphs (off by default in GNU modes; mapped anyway, which can only make the scan
    stricter), then backslash-newline splices, then comments, then the `%:` digraph for `#`.
    """
    text = (code or "").replace("??=", "#").replace("??/", "\\")
    text = _SPLICE.sub("", text)
    text = _strip_comments(text)
    return text.replace("%:", "#")


def _forbidden_operand(operand: str) -> bool:
    angle = _ANGLE.match(operand)
    if angle is None:
        # quoted, or a macro that expands to who-knows-what: both can name any file
        return True
    target = angle.group(1).strip()
    return ".." in target or target.startswith(("/", "\\")) or bool(_DRIVE.match(target))


def find_forbidden_includes(code: str) -> list[str]:
    """Guest rule: every directive that could make the compiler read anything but a system
    header — scanned AFTER normalisation, so comments, splices and digraphs cannot hide one.

    Allowed: `#include <header>` with a relative, `..`-free name. Refused: any other include
    form, `#include_next`, `#import`, `#embed`, `#line` and GNU linemarkers. Returned lines
    are the normalised text, which is what the refusal quotes.
    """
    hits = []
    for line in normalize_preprocessor_text(code).splitlines():
        match = _DIRECTIVE.match(line)
        if not match:
            continue
        name, operand = match.group(1), match.group(2).strip()
        if name.isdigit() or name in _FILE_DIRECTIVES - {"include"}:
            hits.append(line)
        elif name == "include" and _forbidden_operand(operand):
            hits.append(line)
    #: `#embed` has no dependency-pass backstop (gcc -M does not list embedded files), so a
    #: guest's code may not contain the word at all — even somewhere the scan calls harmless
    if "embed" in _SPLICE.sub("", code or "") and not any("embed" in h for h in hits):
        hits.append("embed (the word `embed` is not allowed in guest code)")
    return hits


def find_secret_references(code: str) -> list[str]:
    """Any-session rule: directives that name a secrets file (`.env*`, the password hash)."""
    hits = []
    for line in normalize_preprocessor_text(code).splitlines():
        match = _DIRECTIVE.match(line)
        if not match or not (match.group(1) in _FILE_DIRECTIVES or match.group(1).isdigit()):
            continue
        for quoted, angled in _QUOTED_OR_ANGLE.findall(match.group(2)):
            if is_secret_name(Path((quoted or angled).strip()).name):
                hits.append(line)
                break
    return hits
