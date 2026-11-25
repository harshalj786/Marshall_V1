"""Feature 3 prep step 2: num_ctx and thinking reach every model call, at one value.

No model and no Ollama, so this runs in CI. What is pinned:

* `AgentGraph(num_ctx=, reasoning=)` reaches `ChatOllama`, and with neither argument both
  stay None — left out of the request, exactly as before the parameters existed;
* `UserMemory` builds its client (OllamaLLM) with the same num_ctx / reasoning as the
  agent. This is the reload trap: Ollama reloads a model whenever a request's num_ctx
  differs from the loaded one, so a mismatch costs an unload + reload on every gate turn;
* the live entry points send an EXPLICIT 4096 and thinking off — 4096 being what Ollama
  already used, so no VRAM change and no reload versus before;
* the checks always send thinking off, default to the live context, and `--num-ctx`
  reaches the agent, the record, and (via run_evals) every check's command line.

The thinking parameter is langchain's `reasoning`, sent to Ollama as `think`.
"""
import inspect
import json
import sys
from pathlib import Path

import pytest
from langchain_core.runnables import Runnable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import chat_main  # noqa: E402
import memory.long_term_memory as ltm  # noqa: E402
import pipeline_lg  # noqa: E402
import scripts.check_memory as check_memory  # noqa: E402
import scripts.check_persona as check_persona  # noqa: E402
import scripts.check_routing as check_routing  # noqa: E402
import scripts.run_evals as runner  # noqa: E402
import voice_main  # noqa: E402
from admin_session import Session  # noqa: E402
from memory.long_term_memory import UserMemory  # noqa: E402
from pipeline_lg import AgentGraph  # noqa: E402

CHECKS = [check_routing, check_memory, check_persona]
_ids = lambda m: m.__name__.rsplit(".", 1)[-1]  # noqa: E731


# ==========================================================================
# AgentGraph -> ChatOllama
# ==========================================================================

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ctx")
    (root / "doc.txt").write_text("Pipes carry water. Valves stop it.", encoding="utf-8")
    return root


def _agent(workspace, name, **kwargs):
    """A real AgentGraph. Needs no Ollama: building ChatOllama does not connect."""
    return AgentGraph(file_path=str(workspace / "doc.txt"), mode_docs="txt",
                      mode_retreiver="simple", persist_directory_path=str(workspace / "vdb"),
                      user_history_directory_path=str(workspace / name), **kwargs)


def test_num_ctx_and_reasoning_reach_chat_ollama(workspace):
    agent = _agent(workspace, "a", num_ctx=8192, reasoning=False)
    #: summary_llm is the unbound ChatOllama the agent's bound handle is made from
    assert agent.summary_llm.num_ctx == 8192
    assert agent.summary_llm.reasoning is False
    #: and the tool-bound handle carries the same model object
    assert agent.llm.bound.num_ctx == 8192 and agent.llm.bound.reasoning is False


def test_neither_argument_leaves_both_none(workspace):
    """None is left out of the request: a caller passing neither sends what it did before."""
    for name in ("num_ctx", "reasoning"):
        assert inspect.signature(AgentGraph.__init__).parameters[name].default is None
    agent = _agent(workspace, "b")
    assert agent.summary_llm.num_ctx is None and agent.summary_llm.reasoning is None
    assert (agent.num_ctx, agent.reasoning) == (None, None)


def test_chat_ollama_leaves_none_out_of_the_request():
    """What 'left out' means on the wire: ChatOllama drops a None num_ctx from options and
    the ollama client drops a None think, so the request body is the pre-step-2 one."""
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    plain = ChatOllama(model="llama3.1", temperature=0)._chat_params([HumanMessage("hi")])
    assert "num_ctx" not in plain["options"] and plain.get("think") is None
    pinned = ChatOllama(model="llama3.1", temperature=0, num_ctx=4096,
                        reasoning=False)._chat_params([HumanMessage("hi")])
    assert pinned["options"]["num_ctx"] == 4096 and pinned["think"] is False


# ==========================================================================
# UserMemory -> OllamaLLM, the reload trap
# ==========================================================================

class _RecordingLLM(Runnable):
    """Stands in for OllamaLLM; records its constructor kwargs."""
    built: list = []

    def __init__(self, **kwargs):
        _RecordingLLM.built.append(kwargs)

    def invoke(self, input, config=None, **kwargs):
        return "compressed"


@pytest.fixture
def recording_llm(monkeypatch):
    monkeypatch.setattr(ltm, "OllamaLLM", _RecordingLLM)
    _RecordingLLM.built = []
    return _RecordingLLM.built


def test_user_memory_client_gets_the_same_num_ctx_and_reasoning(recording_llm, tmp_path):
    UserMemory(chat_history=[], persist_directory_path=str(tmp_path), model="llama3.1",
               num_ctx=8192, reasoning=False).invoke({})
    assert recording_llm == [{"model": "llama3.1", "temperature": 0,
                              "num_ctx": 8192, "reasoning": False}]


def test_user_memory_defaults_leave_both_out(recording_llm, tmp_path):
    UserMemory(chat_history=[], persist_directory_path=str(tmp_path)).invoke({})
    assert recording_llm[0]["num_ctx"] is None and recording_llm[0]["reasoning"] is None


def test_the_agents_memory_gate_compresses_at_the_agents_context(workspace, recording_llm):
    """End to end through the real funnel: the gate's UserMemory must ask Ollama for the
    SAME context as the agent, or every gate turn reloads the model twice."""
    from langchain_core.messages import HumanMessage
    agent = _agent(workspace, "gate", model="llama3.1", memory_model="llama3.1",
                   num_ctx=8192, reasoning=False)
    memory = agent._build_user_memory([HumanMessage(content="I live in Pune.")], "t-gate")
    memory.invoke({})
    assert recording_llm[-1]["num_ctx"] == agent.summary_llm.num_ctx == 8192
    assert recording_llm[-1]["reasoning"] is agent.summary_llm.reasoning is False


# ==========================================================================
# live entry points: explicit 4096, thinking off
# ==========================================================================

class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_live_constants_are_todays_values():
    """4096 is what Ollama used before this existed (llama3.1 loaded at 5,211 MiB = weights +
    a 4096-token cache), so pinning it is no VRAM change and no reload."""
    assert voice_main.NUM_CTX == chat_main.NUM_CTX == 4096
    assert voice_main.REASONING is False and chat_main.REASONING is False


def test_voice_build_agent_sends_the_live_context(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _Recorder)
    monkeypatch.setattr(voice_main, "_ensure_durable_memory", lambda path: None)
    doc = tmp_path / "d.txt"
    doc.write_text("x")
    agent = voice_main.build_agent(str(doc))
    assert agent.kwargs["num_ctx"] == 4096 and agent.kwargs["reasoning"] is False


def test_chat_build_agent_sends_the_live_context(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _Recorder)
    doc = tmp_path / "d.txt"
    doc.write_text("x")
    agent = chat_main.build_agent(str(doc), Session("guest", 1))
    assert agent.kwargs["num_ctx"] == 4096 and agent.kwargs["reasoning"] is False


# ==========================================================================
# the checks: thinking always off, --num-ctx end to end
# ==========================================================================

def test_checks_never_think():
    assert check_routing.CHECK_REASONING is False


def test_build_check_agent_defaults_to_live_context_and_thinking_off(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _Recorder)
    agent = check_routing.build_check_agent(user_history_path=str(tmp_path / "m"))
    assert agent.kwargs["num_ctx"] == voice_main.NUM_CTX
    assert agent.kwargs["reasoning"] is False


def test_build_check_agent_honours_num_ctx(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _Recorder)
    agent = check_routing.build_check_agent(user_history_path=str(tmp_path / "m"), num_ctx=8192)
    assert agent.kwargs["num_ctx"] == 8192
    assert agent.kwargs["reasoning"] is False, "an override must never turn thinking on"


def test_thinking_off_even_if_live_turns_it_on(monkeypatch, tmp_path):
    """The checks' rule is their own, not a copy of the live setting."""
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _Recorder)
    monkeypatch.setattr(voice_main, "REASONING", True)
    agent = check_routing.build_check_agent(user_history_path=str(tmp_path / "m"))
    assert agent.kwargs["reasoning"] is False


@pytest.mark.parametrize("check", CHECKS, ids=_ids)
def test_num_ctx_flag_default_and_override(check):
    assert check.parse_args([]).num_ctx == voice_main.NUM_CTX
    assert check.parse_args(["--num-ctx", "8192"]).num_ctx == 8192


@pytest.mark.parametrize("check", CHECKS, ids=_ids)
@pytest.mark.parametrize("bad", ["0", "-4096", "lots"])
def test_num_ctx_rejects_nonsense(check, bad):
    with pytest.raises(SystemExit):
        check.parse_args(["--num-ctx", bad])


@pytest.mark.parametrize("check", CHECKS, ids=_ids)
def test_record_carries_num_ctx(check, monkeypatch, tmp_path):
    monkeypatch.setattr(check_routing, "ollama_up", lambda: False)
    out = tmp_path / "record.json"
    assert check.main(["--num-ctx", "8192", "--json", str(out)]) == 2
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["num_ctx"] == 8192 and record["schema_version"] == 4


@pytest.fixture
def captured_builds(monkeypatch):
    calls = []
    monkeypatch.setattr(check_routing, "build_check_agent",
                        lambda **kw: calls.append(kw) or _Recorder(**kw))
    return calls


def test_memory_paths_pass_num_ctx(captured_builds, monkeypatch, tmp_path):
    check_memory.build_agent(str(tmp_path / "m"), "llama3.1", 8192)
    monkeypatch.setattr(check_memory, "CHECK_CARRY_PATH", str(tmp_path / "carry"))
    case = next(c for c in check_memory.CASES if c.told_in_thread)
    check_memory.run_carry_attempt(case, 1, "llama3.1", 8192)
    assert [c["num_ctx"] for c in captured_builds] == [8192, 8192]


def test_persona_builder_passes_num_ctx(captured_builds, monkeypatch, tmp_path):
    monkeypatch.setattr(check_persona, "CHECK_USER_HISTORY_PATH", str(tmp_path / "m"))
    check_persona.build_agent("llama3.1", 8192)
    assert captured_builds[-1]["num_ctx"] == 8192


# ==========================================================================
# run_evals: --num-ctx reaches each check; the header says "@ ctx N"
# ==========================================================================

def test_check_command_adds_num_ctx_only_when_asked():
    plain = runner.check_command("s.py", "r.json")
    assert "--num-ctx" not in plain
    assert runner.check_command("s.py", "r.json", None, 8192) == plain + ["--num-ctx", "8192"]
    assert runner.check_command("s.py", "r.json", "qwen3:8b", 8192) == plain + [
        "--model", "qwen3:8b", "--num-ctx", "8192"]


def test_num_ctx_flag_reaches_every_check_subprocess(tmp_path, monkeypatch, capsys):
    import textwrap
    monkeypatch.setattr(runner, "ollama_vram", lambda: {})
    monkeypatch.setattr(runner, "gpu_used_mib", lambda: None)
    body = textwrap.dedent("""\
        import json, sys
        from pathlib import Path
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        ctx = int(sys.argv[sys.argv.index("--num-ctx") + 1]) if "--num-ctx" in sys.argv else None
        out.write_text(json.dumps({"schema_version": 4, "check": "fake", "model": "m",
                                   "num_ctx": ctx, "error": None,
                                   "cases": [{"id": "c", "verdict": "PASS", "wall_seconds": 0.1}]}))
        """)
    registry = {}
    for name in ("routing", "memory", "persona"):
        script = tmp_path / f"check_{name}.py"
        script.write_text(body, encoding="utf-8")
        registry[name] = str(script)
    log = tmp_path / "runs.jsonl"
    assert runner.main(["--num-ctx", "8192", "--log", str(log)], registry=registry) == 0
    line = json.loads(log.read_text(encoding="utf-8"))
    assert [c["record"]["num_ctx"] for c in line["checks"]] == [8192] * 3
    out = capsys.readouterr().out
    assert "model m @ ctx 8192" in out and "!! ctx" not in out


def _outcome(model, num_ctx):
    from scripts.eval_contract import PASS, CaseResult, CheckRecord
    record = CheckRecord(check="routing", model=model, num_ctx=num_ctx,
                         cases=[CaseResult(id="c", verdict=PASS, wall_seconds=1.0)])
    return runner.CheckOutcome(check="routing", status=runner.ST_PASS, exit_code=0, record=record)


def test_header_reads_model_at_ctx():
    header = runner.format_table([_outcome("llama3.1", 8192)], "abc1234", False, "T", 0,
                                 model="llama3.1", num_ctx=8192).splitlines()[1]
    assert "model llama3.1 @ ctx 8192" in header


def test_a_v3_record_shows_ctx_unknown_rather_than_guessing():
    assert "@ ctx unknown" in runner.format_table([_outcome("llama3.1", None)], None, None, "T", 0)


def test_ctx_mismatch_and_disagreement_are_flagged():
    wrong = runner.format_table([_outcome("llama3.1", 4096)], None, None, "T", 0, num_ctx=8192)
    assert "!! ctx: asked for --num-ctx 8192 but the checks ran 4096" in wrong
    split = runner.format_table([_outcome("llama3.1", 4096), _outcome("llama3.1", 8192)],
                                None, None, "T", 0)
    assert "!! ctx: checks ran different num_ctx values: 4096, 8192" in split
