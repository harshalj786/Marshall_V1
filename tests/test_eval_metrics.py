"""scripts/eval_metrics.py with no model and no GPU, so this runs in CI.

The absent-reader tests matter most: CI has no Ollama and no nvidia-smi, which is
exactly the path where a careless reader would raise into a check and cost a verdict.
They exercise the REAL failure (a refused port, a missing binary), not a mock of one,
so they prove the same thing on this machine and on the runner.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.eval_metrics as metrics
from scripts.eval_metrics import case_metrics, gpu_used_mib, ollama_vram, turn_timing


def _reply(total_ns=None, load_ns=0, prompt=0, output=0, **extra):
    meta = {"load_duration": load_ns, "prompt_eval_count": prompt, "eval_count": output}
    if total_ns is not None:
        meta["total_duration"] = total_ns
    return AIMessage(content="x", response_metadata=meta, **extra)


# --------------------------------------------------------------------------
# turn_timing
# --------------------------------------------------------------------------

def test_sums_across_the_turns_ai_messages_and_ignores_the_rest():
    turn = [
        HumanMessage(content="what is 47 times 82?"),
        _reply(1_500_000_000, load_ns=90_000_000, prompt=200, output=20,
               tool_calls=[{"name": "calculator", "args": {}, "id": "1"}]),
        ToolMessage(content="3854", tool_call_id="1", name="calculator"),
        _reply(500_000_000, load_ns=10_000_000, prompt=230, output=12),
    ]
    timing = turn_timing(turn)
    assert timing.model_seconds == pytest.approx(2.0)
    assert timing.load_seconds == pytest.approx(0.1)
    assert timing.prompt_tokens == 430
    assert timing.output_tokens == 32


def test_partial_metadata_is_all_null_not_a_partial_sum():
    """A partial sum would undercount and read as a fast model."""
    timing = turn_timing([_reply(1_000_000_000), _reply(total_ns=None)])
    assert timing == metrics.TurnTiming()


@pytest.mark.parametrize("turn", [[], None, [HumanMessage(content="hi")]])
def test_no_ai_message_is_all_null(turn):
    """An ERROR case has no reply to time."""
    assert turn_timing(turn) == metrics.TurnTiming()


def test_garbage_metadata_degrades_to_null():
    assert turn_timing([_reply(total_ns="not a number")]) == metrics.TurnTiming()


# --------------------------------------------------------------------------
# readers that are absent — the CI path
# --------------------------------------------------------------------------

def test_ollama_vram_is_none_when_nothing_listens():
    #: port 9 (discard) is not served on a dev box or a runner: a real refused connection
    assert ollama_vram(url="http://127.0.0.1:9/api/ps", timeout=1.0) is None


def test_gpu_used_mib_is_none_when_nvidia_smi_is_missing(monkeypatch):
    monkeypatch.setattr(metrics, "NVIDIA_SMI", ["no-such-nvidia-smi-binary", "--x"])
    assert gpu_used_mib() is None


def test_gpu_used_mib_is_none_on_timeout_or_failure(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
    monkeypatch.setattr(metrics.subprocess, "run", timeout)
    assert gpu_used_mib() is None

    failed = subprocess.CompletedProcess(args=[], returncode=9, stdout="", stderr="no GPU")
    monkeypatch.setattr(metrics.subprocess, "run", lambda *a, **k: failed)
    assert gpu_used_mib() is None


def test_case_metrics_never_raises_with_every_reader_gone(monkeypatch):
    monkeypatch.setattr(metrics, "OLLAMA_PS_URL", "http://127.0.0.1:9/api/ps")
    monkeypatch.setattr(metrics, "NVIDIA_SMI", ["no-such-nvidia-smi-binary"])
    result = case_metrics([_reply(total_ns=None)])
    assert result == {"model_seconds": None, "load_seconds": None, "prompt_tokens": None,
                      "output_tokens": None, "vram_models": None, "vram_gpu_mib": None,
                      "model_size_mib": None, "model_ctx": None}


# --------------------------------------------------------------------------
# readers that answer
# --------------------------------------------------------------------------

def test_ollama_vram_reports_every_loaded_model_in_mib(monkeypatch):
    """The point of /api/ps: it shows mistral resident beside llama3.1."""
    body = {"models": [{"name": "llama3.1:latest", "size_vram": 5464647680},
                       {"name": "mistral:latest", "size_vram": 5143808000}]}
    monkeypatch.setattr(metrics.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(json.dumps(body).encode()))
    #: 5464647680 / 2**20 = 5211.49, 5143808000 / 2**20 = 4905.52
    assert ollama_vram() == {"llama3.1:latest": 5211, "mistral:latest": 4906}


def test_ollama_vram_empty_dict_means_read_fine_nothing_loaded(monkeypatch):
    monkeypatch.setattr(metrics.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(b'{"models": []}'))
    assert ollama_vram() == {}


def test_gpu_used_mib_parses_the_first_gpu(monkeypatch):
    done = subprocess.CompletedProcess(args=[], returncode=0, stdout="11269\n2048\n")
    monkeypatch.setattr(metrics.subprocess, "run", lambda *a, **k: done)
    assert gpu_used_mib() == 11269


# --------------------------------------------------------------------------
# TurnTimer: live timing by callback (Step 4). Fake chat models, no Ollama.
# --------------------------------------------------------------------------

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from scripts.eval_metrics import TurnTimer, timer_metrics  # noqa: E402

NS = 1_000_000_000


def _fake_model(*replies):
    return FakeMessagesListChatModel(responses=list(replies))


def test_timer_sums_every_response_in_the_turn():
    timer = TurnTimer()
    model = _fake_model(_reply(total_ns=2 * NS, load_ns=NS // 10, prompt=300, output=20),
                        _reply(total_ns=3 * NS, load_ns=NS // 10, prompt=400, output=30))
    model.invoke("first", config={"callbacks": [timer]})
    model.invoke("second", config={"callbacks": [timer]})
    timing = timer.timing()
    assert timer.call_count == 2
    assert timing.model_seconds == pytest.approx(5.0)
    assert timing.load_seconds == pytest.approx(0.2)
    assert (timing.prompt_tokens, timing.output_tokens) == (700, 50)


def test_timer_catches_a_nested_call_that_passed_no_config():
    """The property the memory check relies on: the memory gate and the summariser call
    their model WITHOUT forwarding the config, and the callback still sees them."""
    timer = TurnTimer()
    inner = _fake_model(_reply(total_ns=4 * NS))
    outer = RunnableLambda(lambda _: inner.invoke("nested, no config"))
    outer.invoke(None, config={"callbacks": [timer]})
    assert timer.call_count == 1
    assert timer.timing().model_seconds == pytest.approx(4.0)


def test_timer_is_all_or_null_when_one_response_lacks_timing():
    timer = TurnTimer()
    model = _fake_model(_reply(total_ns=2 * NS), _reply(total_ns=None))
    model.invoke("a", config={"callbacks": [timer]})
    model.invoke("b", config={"callbacks": [timer]})
    assert timer.call_count == 2
    assert timer.timing() == metrics.TurnTiming()


def test_timer_reset_starts_a_new_turn():
    timer = TurnTimer()
    model = _fake_model(_reply(total_ns=9 * NS), _reply(total_ns=1 * NS))
    model.invoke("turn 1", config={"callbacks": [timer]})
    timer.reset()
    model.invoke("turn 2", config={"callbacks": [timer]})
    assert timer.timing().model_seconds == pytest.approx(1.0)


def test_timer_with_nothing_seen_is_null():
    assert TurnTimer().timing() == metrics.TurnTiming()


def test_an_errored_call_nulls_the_turn():
    timer = TurnTimer()
    _fake_model(_reply(total_ns=2 * NS)).invoke("ok", config={"callbacks": [timer]})
    timer.on_llm_error(RuntimeError("model fell over"))
    assert timer.timing() == metrics.TurnTiming()


def test_garbage_response_never_raises_and_nulls():
    timer = TurnTimer()
    timer.on_llm_end(object())          # no .generations at all
    assert timer.call_count == 1
    assert timer.timing() == metrics.TurnTiming()


def test_timer_metrics_with_every_reader_gone(monkeypatch):
    monkeypatch.setattr(metrics, "OLLAMA_PS_URL", "http://127.0.0.1:9/api/ps")
    monkeypatch.setattr(metrics, "NVIDIA_SMI", ["definitely-not-a-real-binary-xyz"])
    assert timer_metrics(None) == {
        "model_seconds": None, "load_seconds": None, "prompt_tokens": None,
        "output_tokens": None, "vram_models": None, "vram_gpu_mib": None,
        "model_size_mib": None, "model_ctx": None}
