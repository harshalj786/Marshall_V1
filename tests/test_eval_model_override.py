"""Feature 3 prep step 1: `--model` reaches every check, and the "is it pulled" test is honest.

No model and no Ollama, so this runs in CI. What is pinned:

* each check's `--model` defaults to voice_main.VOICE_MODEL and accepts an override;
* the record a check writes names the model it was told to run, on the exit path that
  needs no Ollama at all (Ollama "down", so the check stops with exit 2 and writes the
  record anyway);
* `build_check_agent(model=X)` gives the agent X for BOTH generation and memory, and the
  memory/persona builders pass the override through;
* the tag rule: a bare name matches any tag, a tagged name only itself. Before this, a
  `granite4:tiny-h` run read as "pulled" when only `granite4:latest` was.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pipeline_lg  # noqa: E402
import scripts.check_memory as check_memory  # noqa: E402
import scripts.check_persona as check_persona  # noqa: E402
import scripts.check_routing as check_routing  # noqa: E402
import voice_main  # noqa: E402

CHECKS = [check_routing, check_memory, check_persona]
OVERRIDE = "qwen3.5:9b"


# ==========================================================================
# the flag
# ==========================================================================

@pytest.mark.parametrize("check", CHECKS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_model_defaults_to_the_voice_model(check):
    assert check.parse_args([]).model == voice_main.VOICE_MODEL


@pytest.mark.parametrize("check", CHECKS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_model_accepts_an_override(check):
    assert check.parse_args(["--model", OVERRIDE]).model == OVERRIDE


# ==========================================================================
# the record names the overridden model
# ==========================================================================

@pytest.mark.parametrize("check", CHECKS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_record_model_equals_the_override(check, monkeypatch, tmp_path):
    #: memory and persona import ollama_up from check_routing at call time, so patching
    #: it there stops all three before anything touches a model or a store
    monkeypatch.setattr(check_routing, "ollama_up", lambda: False)
    out = tmp_path / "record.json"
    assert check.main(["--model", OVERRIDE, "--json", str(out)]) == 2
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["model"] == OVERRIDE
    assert record["error"]                       # it stopped for the stated reason


@pytest.mark.parametrize("check", CHECKS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_record_model_defaults_to_the_voice_model(check, monkeypatch, tmp_path):
    monkeypatch.setattr(check_routing, "ollama_up", lambda: False)
    out = tmp_path / "record.json"
    check.main(["--json", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["model"] == voice_main.VOICE_MODEL


@pytest.mark.parametrize("check", CHECKS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_not_pulled_message_names_the_override(check, monkeypatch, tmp_path):
    monkeypatch.setattr(check_routing, "ollama_up", lambda: True)
    asked = []
    monkeypatch.setattr(check_routing, "model_available", lambda m: asked.append(m) or False)
    out = tmp_path / "record.json"
    assert check.main(["--model", OVERRIDE, "--json", str(out)]) == 2
    assert asked == [OVERRIDE]
    assert json.loads(out.read_text(encoding="utf-8"))["error"] == (
        f"{OVERRIDE} is not pulled: ollama pull {OVERRIDE}")


# ==========================================================================
# the agent gets the override for generation AND memory
# ==========================================================================

class _RecordingAgent:
    """Stands in for AgentGraph: records its kwargs, builds nothing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_check_agent_sets_both_models(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _RecordingAgent)
    agent = check_routing.build_check_agent(user_history_path=str(tmp_path / "mem"),
                                            model=OVERRIDE)
    assert agent.kwargs["model"] == OVERRIDE
    assert agent.kwargs["memory_model"] == OVERRIDE


def test_build_check_agent_without_override_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", _RecordingAgent)
    agent = check_routing.build_check_agent(user_history_path=str(tmp_path / "mem"))
    assert agent.kwargs["model"] == voice_main.VOICE_MODEL
    assert agent.kwargs["memory_model"] == voice_main.VOICE_MODEL


@pytest.fixture
def captured_builds(monkeypatch):
    """Replace check_routing.build_check_agent; memory/persona look it up at call time."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _RecordingAgent(**kwargs)

    monkeypatch.setattr(check_routing, "build_check_agent", fake)
    return calls


def test_memory_builder_passes_the_override(captured_builds, tmp_path):
    check_memory.build_agent(str(tmp_path / "mem"), OVERRIDE)
    assert captured_builds[-1]["model"] == OVERRIDE


def test_memory_carry_over_attempt_passes_the_override(captured_builds, monkeypatch, tmp_path):
    #: the carry-over case builds its OWN agent per attempt, a second path the flag must reach
    monkeypatch.setattr(check_memory, "CHECK_CARRY_PATH", str(tmp_path / "carry"))
    case = next(c for c in check_memory.CASES if c.told_in_thread)
    check_memory.run_carry_attempt(case, 1, OVERRIDE)
    assert captured_builds[0]["model"] == OVERRIDE


def test_persona_builder_passes_the_override(captured_builds, monkeypatch, tmp_path):
    monkeypatch.setattr(check_persona, "CHECK_USER_HISTORY_PATH", str(tmp_path / "mem"))
    check_persona.build_agent(OVERRIDE)
    assert captured_builds[-1]["model"] == OVERRIDE


# ==========================================================================
# the tag rule, against a fake /api/tags reply
# ==========================================================================

#: what Ollama's /api/tags returns, trimmed to the field that is read
FAKE_TAGS = {"models": [{"name": "llama3.1:latest"}, {"name": "granite4:latest"},
                        {"name": "qwen3.5:9b"}]}


@pytest.fixture
def fake_tags(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(FAKE_TAGS).encode("utf-8")

    monkeypatch.setattr(check_routing.urllib.request, "urlopen",
                        lambda url, timeout=None: _Response())


@pytest.mark.parametrize("model, expected", [
    ("llama3.1", True),            # bare name matches any tag: how every caller asks today
    ("llama3.1:latest", True),     # tagged, exact
    ("qwen3.5:9b", True),
    ("qwen3.5", True),             # bare name, some tag pulled
    ("granite4:tiny-h", False),    # THE bug: only granite4:latest is pulled
    ("qwen3.5:27b", False),        # same family, wrong size
    ("llama3.1:8b", False),        # an alias Ollama would resolve, but not what is listed
    ("mistral", False),
])
def test_model_available_tag_rule(fake_tags, model, expected):
    assert check_routing.model_available(model) is expected


def test_model_available_is_false_when_ollama_is_unreachable(monkeypatch):
    def refuse(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(check_routing.urllib.request, "urlopen", refuse)
    assert check_routing.model_available("llama3.1") is False
