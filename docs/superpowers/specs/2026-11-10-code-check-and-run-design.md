# Code Check and Run — Design

Date: 2026-11-10
Status: draft, awaiting approval

## Intent

When someone hands Marshall code, he should be able to tell them whether it is correct and
explain in plain words what it does. In an admin session he can also run it, but only after
asking and getting a yes.

This is the first of two projects. The second, improving Marshall's general coding and
reasoning ability, comes after this one and is out of scope here. This one comes first
because a model that can compile and see real errors is the foundation for the second.

## How code reaches Marshall

- **Text chat**: pasted into the `AgentGraph` CLI loop, or given as a file path.
- **Voice**: given as a file path (reading code aloud through Whisper is not realistic).
  Results are spoken as a short summary, not read out line by line.

Languages: **Python, C, C++** — nothing else.

## Sessions: admin and guest

At startup, in both `pipeline_lg.py`'s agent CLI and `voice_main.py`, before the
conversation begins, Marshall asks for the admin password **in the terminal, with hidden
input** (`getpass`).

- Correct password → **admin session** for the rest of that process.
- Wrong or skipped → **guest session**. *(My call: three attempts, and an empty entry
  skips straight to guest.)*

The password is set once by the owner with a separate command (e.g. a
`scripts/set_admin_password.py`) and stored **hashed** — Argon2id via `argon2-cffi`
(installed, 25.1.0), as `docs/security.md` §2 requires. The hash file lives outside version
control (gitignored). If no password has been set, every session is a guest session.

**The model never sees the password, never sees that it was asked, and is never told the
outcome in a way it could repeat.** The check is plain code that runs before the agent is
built; its only output is which tools the agent gets.

## Tools

| Tool | Guest session | Admin session |
|---|---|---|
| `check_code` | bound | bound |
| `run_code` | **not bound** | bound |
| `recall_user_memory` | **not bound** (see Memory) | bound |

A guest session does not get a refusing `run_code` — it gets no `run_code` at all, and the
system prompt does not mention it (the existing rule: the prompt describes exactly the tools
that are bound, or llama3.1 narrates calls to phantom tools in prose). `AgentGraph`
already has an `extra_tools` seam; the prompt template needs the matching conditional
entries, the way `_RECALL_TOOL_ENTRY` is spliced today.

One `check_code` tool with a `language` argument, not one tool per language: each added tool
has measurably raised spurious tool calls on llama3.1 (4→5 tools took spurious calls on
conversational prompts from 1/4 to 2/4).

### `check_code(code | path, language)`

Compiles or checks without executing anything.

- **C / C++**: `gcc -fsyntax-only -Wall` / `g++ -fsyntax-only -Wall` (gcc/g++ 15.2 MinGW is
  on PATH). Returns diagnostics with line numbers.
- **Python**: parse/compile only (`ast.parse` / `compile`, never `exec`). This catches
  syntax errors only; problems that appear at runtime cannot be found without running. The
  tool's output should say so, so Marshall doesn't claim more than was checked.
- Language is taken from the argument, or from the file extension when a path is given
  (`.py`, `.c`, `.h`, `.cpp`, `.cc`, `.cxx`, `.hpp`).
- After the tool returns, Marshall explains what the code does and, if there were errors,
  what is wrong and why — that part is the model's job, not the tool's.

### `run_code(code | path, language)` — admin only

1. **Confirmation, enforced in code, not by the model.** The tool shows what it is about to
   run and asks for a yes *outside the model*:
   - text: a terminal `[y/N]` prompt;
   - voice: Marshall speaks "I'm about to run this, okay?" and the next utterance is matched
     against yes/no **in code**. Anything other than a clear yes is a no.

   The model can decide to *request* a run; it can never supply the yes. This is the defence
   against code that contains text telling the model to run it.
2. On yes: write to a fresh temp directory, compile if C/C++, execute as a subprocess with
   the temp directory as working directory, no stdin, a time limit, and the whole process
   tree killed on timeout.
3. Return exit code, stdout and stderr, capped by the existing `_cap_output`
   (8,000 characters).

On no: return a plain "the user declined to run it" result; nothing executes.

## Goals

- Pasted or file code in Python/C/C++ gets a correct pass/fail check and an explanation.
- Admin sessions can run code, each run individually confirmed outside the model.
- Works in both text and voice.

## Non-goals

- Containers, VMs or OS-level sandboxing — protection is password + per-run confirmation +
  temp dir + time limit.
- Multi-file projects, build systems, linking against external libraries.
- Interactive programs that read stdin.
- Languages other than Python, C, C++.
- Improving Marshall's general coding/reasoning (project two).
- Elevating to Windows administrator — "admin" here is Marshall's own password.

## Anti-goals (failure even if it "works")

- The password, or any hint of it, appears in the model's context, the message history, the
  checkpointer, the summary file, telemetry or logs.
- Code executes without **both** an admin session **and** a per-run yes given outside the
  model.
- A guest session can cause code to run by any route.
- A guest session can read any file's contents through the checker, or read anything from
  admin conversations through memory or the summary.
- Plain conversation gets noticeably worse: the code tools fire on greetings or ordinary
  questions.

## Constraints

- `pipeline.py` is not touched.
- AgentGraph stays on llama3.1.
- Existing tool conventions apply: `TOOL_TIMEOUT_SECONDS`-style timeouts and `_cap_output`.
- One feature per commit.

## `docs/security.md` exception (owner decided: amend)

`docs/security.md` §4 says tool routines must never call `subprocess`, and that agent file
access is limited to `db/`, `docs/`, `config/` with absolute paths forbidden. The owner
chose to **add an explicit exception** for `check_code` and `run_code` only — in its own
commit, before the feature. Every other tool stays under the existing rules. The exception
should state the guards below, so the policy and the code say the same thing.

## Guests cannot read files through the checker (owner decided)

Compiler and parser errors quote source lines, so a checker that opens files is a file
reader. Therefore:

- **Guest sessions: pasted code only.** `check_code` rejects any path argument.
- **Guest C/C++: only `#include <...>` of system headers.** Any quoted `#include "..."`, and
  any include naming an absolute path or `..`, is rejected before gcc sees the code. *(My
  call: reject quoted includes outright for guests rather than try to allow "safe" ones —
  pasted guest code is a single file, so it has no local headers to include anyway.)*
- **Admin sessions: paths allowed**, except the admin password hash file and `.env*`,
  which are refused by both tools in every session. *(My call: the language argument
  cannot override a path's extension — a path must end in one of the listed source
  extensions, so `check_code(".env", language="c")` is refused.)*
- Python checking never opens files beyond the one given (`ast.parse` does not import), so
  pasted Python needs no extra guard.

## Memory: only admin conversations are kept (owner decided)

- **Admin session**: memory behaves as today — persistent conversation summary, long-term
  user memory, `recall_user_memory`.
- **Guest session**: **nothing is kept.** In-RAM history for that conversation only. No
  summary is loaded or written, the memory gate does not write, and `recall_user_memory`
  is **not bound**, so a guest cannot read the admin's history. A guest also never sees
  an admin's earlier `run_code` calls, which avoids the phantom-tool narration problem.
- **Guest count**: Marshall records how many guest sessions have started — a persisted
  counter, plus a timestamp per guest session *(my call; drop the timestamps if unwanted)*,
  stored outside version control. Nothing a guest says is recorded with it.
  *(My call on surfacing: the admin can ask "how many guests logged in?" — the count is
  given to the model in admin sessions only, e.g. as one line in the admin system prompt.
  Guests are never told it.)* A wrong-password-then-guest start counts as one guest
  session; failed attempts are not counted separately.

## Left to the builder

- Module layout (e.g. `tools/code_tools.py`) and how the session flag is threaded into
  both entry points.
- Exact time limits for compile and run.
- How the voice yes/no capture hooks into `VoiceOrchestrator`.
- Exact wording of the system-prompt entries for the two tools.

## How we'll know it works

- Unit tests: password hash set/verify; guest session binds no `run_code`; `check_code`
  reports a known C error, a known C++ error and a known Python syntax error with line
  numbers; `run_code` executes nothing on a no; timeout kills a looping program; the
  password never appears in agent state; a guest `check_code` refuses a path and a quoted
  `#include`; `.env` and the hash file are refused in admin too; a guest session binds no
  `recall_user_memory`, loads and writes no summary, and increments the guest count.
- Routing A/B, as in earlier tool additions: conversational prompts with and without the
  code tools bound, measuring spurious tool calls. *(My call on the bar: at least 12
  conversational prompts, each run 3 times, since single runs vary even at temperature 0.
  Pass = the code tools are never called on a prompt containing no code or path, and total
  spurious calls are no worse than the current 5-tool baseline on the same set. On a fail,
  the builder stops and reports back rather than choosing a fallback.)*
- A live text session and a live voice session: check a file, explain it, run it as admin
  with a yes, decline a run with a no.
