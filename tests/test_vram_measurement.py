"""Feature 3 Split B: measure VRAM honestly with voice loaded. No model, no GPU, so CI.

Three prep steps, one file, in build order:

* Step 3 — CPU offload is visible. `/api/ps` reports a model's whole `size` and the part
  in VRAM (`size_vram`); Ollama does not fail when VRAM runs short, it spills layers to the
  CPU, and `size > size_vram` is the only trace. Pinned on fake /api/ps replies.
* Step 6 — `--clean` starts from nothing. `unload_all` waits until /api/ps is EMPTY
  (unloading finishes in the background), refuses a dirty start, and `warm` loads the
  model once at the checks' own num_ctx.
* Step 5 — the ballast and the FIT verdict. The ballast's protocol loop runs on a fake
  voice (no torch); `run_evals --with-voice` runs a fake ballast SCRIPT as a real
  subprocess; `fit_verdict`'s three gates each fail on their own.

The real ballast (Whisper + Kokoro on a GPU) is checked by the documented smoke run, not
here: CI has no GPU.
"""
import io
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.check_memory as check_memory  # noqa: E402
import scripts.eval_metrics as metrics  # noqa: E402
import scripts.ollama_control as oc  # noqa: E402
import scripts.run_evals as runner  # noqa: E402
import scripts.voice_ballast as ballast_mod  # noqa: E402
from scripts.eval_contract import PASS, CaseResult, CheckRecord  # noqa: E402
from scripts.eval_metrics import offload_mib, ollama_models  # noqa: E402
from scripts.run_evals import CheckOutcome, RunInfo, fit_verdict, format_table  # noqa: E402

GiB = 2 ** 30
MiB = 2 ** 20


def _ps(monkeypatch, models):
    body = json.dumps({"models": models}).encode()
    monkeypatch.setattr(metrics.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))


# ==========================================================================
# Step 3 — size vs size_vram
# ==========================================================================

def test_ollama_models_reads_size_vram_and_ctx(monkeypatch):
    #: the 24 Nov 32K measurement: 10,764 MiB of llama3.1, only 10,093 of it on the GPU
    _ps(monkeypatch, [{"name": "llama3.1:latest", "size": 10764 * MiB,
                       "size_vram": 10093 * MiB, "context_length": 32768}])
    assert ollama_models() == {"llama3.1:latest": {"size": 10764, "vram": 10093, "ctx": 32768}}


def test_offload_is_size_minus_vram_and_only_for_spilled_models():
    size = {"llama3.1:latest": 10764, "qwen3:8b": 5800}
    vram = {"llama3.1:latest": 10093, "qwen3:8b": 5800}
    assert offload_mib(size, vram) == {"llama3.1:latest": 671}


def test_offload_skips_what_it_cannot_judge():
    assert offload_mib(None, {"a": 1}) == {}
    assert offload_mib({"a": 100}, None) == {}
    assert offload_mib({"a": None}, {"a": 50}) == {}       # entry without a size
    assert offload_mib({"a": 100}, {"b": 50}) == {}        # models that do not line up


def test_an_entry_without_size_keeps_its_vram_reading(monkeypatch):
    """One model lacking `size` must not cost the whole reading (it used to be all-or-None)."""
    _ps(monkeypatch, [{"name": "old:latest", "size_vram": 4096 * MiB}])
    assert ollama_models() == {"old:latest": {"size": None, "vram": 4096, "ctx": None}}
    assert metrics.ollama_vram() == {"old:latest": 4096}


def test_case_metrics_takes_all_three_model_fields_from_one_read(monkeypatch):
    calls = []

    def fake_models(*a, **k):
        calls.append(1)
        return {"llama3.1:latest": {"size": 5987, "vram": 5987, "ctx": 8192}}

    monkeypatch.setattr(metrics, "ollama_models", fake_models)
    monkeypatch.setattr(metrics, "gpu_used_mib", lambda *a, **k: 7464)
    out = metrics.case_metrics([])
    assert calls == [1], "size, vram and ctx must describe one instant: ONE /api/ps read"
    assert out["vram_models"] == {"llama3.1:latest": 5987}
    assert out["model_size_mib"] == {"llama3.1:latest": 5987}
    assert out["model_ctx"] == {"llama3.1:latest": 8192}


def test_k_of_n_takes_size_and_ctx_from_the_same_attempt_as_vram():
    case = check_memory.CASES[0]
    early = {"vram_models": {"m": 100}, "model_size_mib": {"m": 100}, "model_ctx": {"m": 4096}}
    late = {"vram_models": {"m": 90}, "model_size_mib": {"m": 120}, "model_ctx": {"m": 8192}}
    unread = {"vram_models": None, "model_size_mib": None, "model_ctx": None}
    attempts = [check_memory.Attempt(passed=True, metrics=m) for m in (early, late, unread)]
    result = check_memory.aggregate(case, attempts)
    #: the last attempt that could read /api/ps, all three fields from it — never a mix
    assert (result.vram_models, result.model_size_mib, result.model_ctx) == (
        {"m": 90}, {"m": 120}, {"m": 8192})


def test_a_v4_record_still_reads_with_the_v5_fields_null():
    v4 = {"schema_version": 4, "check": "routing", "model": "llama3.1", "num_ctx": 8192,
          "error": None, "cases": [{"id": "math", "verdict": "PASS", "wall_seconds": 4.0,
                                    "vram_models": {"llama3.1:latest": 5987}}]}
    record = CheckRecord.from_dict(v4)
    assert record.cases[0].model_size_mib is None and record.cases[0].model_ctx is None
    assert record.num_ctx == 8192


def _outcome(check="routing", vram=None, size=None, gpu=None, ctx=None):
    record = CheckRecord(check=check, model="llama3.1", num_ctx=8192, cases=[
        CaseResult(id="c", verdict=PASS, wall_seconds=1.0, vram_models=vram,
                   model_size_mib=size, model_ctx=ctx, vram_gpu_mib=gpu)])
    return CheckOutcome(check=check, status=runner.ST_PASS, exit_code=0, record=record)


def test_table_shows_cpu_spill_loudly():
    spilled = _outcome(vram={"llama3.1:latest": 10093}, size={"llama3.1:latest": 10764})
    table = format_table([spilled], None, None, "T", 0)
    assert "llama3.1 10093 (+671 CPU)" in table
    assert "!! offload: routing: llama3.1 671 MiB on the CPU" in table


def test_table_is_quiet_when_nothing_spilled():
    whole = _outcome(vram={"llama3.1:latest": 5987}, size={"llama3.1:latest": 5987})
    table = format_table([whole], None, None, "T", 0)
    assert "CPU" not in table and "offload" not in table


# ==========================================================================
# Step 6 — --clean: unload, wait for EMPTY, then warm
# ==========================================================================

class _FakeOllama:
    """/api/ps that empties only after `polls_to_empty` reads, like a background unload."""

    def __init__(self, loaded, polls_to_empty=2, unreadable=False):
        self.loaded = dict(loaded)
        self.polls_to_empty = polls_to_empty
        self.unreadable = unreadable
        self.posts, self.reads, self.sleeps = [], 0, 0
        self.now = 0.0

    def read(self):
        self.reads += 1
        if self.unreadable:
            return None
        if self.reads > self.polls_to_empty:
            self.loaded = {}
        return dict(self.loaded)

    def post(self, path, body, timeout=None):
        self.posts.append((path, body))
        return {}

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds

    def clock(self):
        return self.now


def _unload(fake, **kw):
    return oc.unload_all(read=fake.read, post=fake.post, sleep=fake.sleep, clock=fake.clock, **kw)


def test_unload_asks_every_model_to_leave_and_waits_for_empty():
    fake = _FakeOllama({"llama3.1:latest": {}, "mistral:latest": {}}, polls_to_empty=3)
    assert _unload(fake) == {}
    assert fake.posts == [("/api/generate", {"model": "llama3.1:latest", "keep_alive": 0}),
                          ("/api/generate", {"model": "mistral:latest", "keep_alive": 0})]
    #: it did NOT trust the first read after the request: it polled until empty
    assert fake.reads == 4 and fake.sleeps == 2


def test_unload_reports_what_never_left():
    fake = _FakeOllama({"stuck:latest": {"size": 1}}, polls_to_empty=10 ** 9)
    assert _unload(fake, timeout_s=2.0, poll_s=0.5) == {"stuck:latest": {"size": 1}}


def test_unload_with_ollama_down_is_unknown_not_clean():
    assert _unload(_FakeOllama({}, unreadable=True)) is None


def test_unload_with_nothing_loaded_posts_nothing():
    fake = _FakeOllama({}, polls_to_empty=0)
    assert _unload(fake) == {} and fake.posts == []


def test_warm_request_matches_what_the_checks_send():
    body = oc.warm_request("qwen3.5:9b", 8192, "30m")
    assert body["model"] == "qwen3.5:9b"
    assert body["options"]["num_ctx"] == 8192        # a different ctx would be a reload
    assert body["think"] is False                    # never True: 400 on non-thinking models
    assert body["keep_alive"] == "30m"


def test_warm_never_raises():
    def refuse(*a, **k):
        raise OSError("connection refused")
    result = oc.warm("llama3.1", 4096, "30m", post=refuse)
    assert result["error"].startswith("OSError") and result["load_seconds"] is None


def test_warm_reports_ollamas_load_time():
    result = oc.warm("llama3.1", 4096, "30m",
                     post=lambda *a, **k: {"load_duration": 7_660_000_000})
    assert result["error"] is None and result["load_seconds"] == pytest.approx(7.66)


# ==========================================================================
# run_evals --clean / --with-voice orchestration (fake checks, fake ballast)
# ==========================================================================

def _check_script(tmp_path, name):
    script = tmp_path / f"check_{name}.py"
    script.write_text(textwrap.dedent(f"""\
        import json, sys
        from pathlib import Path
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        Path(r"{tmp_path}").joinpath("events.log").open("a").write("check:{name}\\n")
        out.write_text(json.dumps({{"schema_version": 5, "check": "{name}", "model": "m",
            "num_ctx": 8192, "error": None, "cases": [{{"id": "c", "verdict": "PASS",
            "wall_seconds": 0.1, "vram_models": {{"m:latest": 5987}},
            "model_size_mib": {{"m:latest": 5987}}, "vram_gpu_mib": 9000}}]}}))
        """), encoding="utf-8")
    return str(script)


def _ballast_script(tmp_path, ready=True, pass_ms=(500, 600)):
    """A stand-in for voice_ballast.py --serve: same protocol, no torch."""
    script = tmp_path / "fake_ballast.py"
    script.write_text(textwrap.dedent(f"""\
        import json, sys
        from pathlib import Path
        log = Path(r"{tmp_path}").joinpath("events.log")
        def emit(**m):
            print("BALLAST " + json.dumps(m), flush=True)
        print("some library chatter on stdout", flush=True)   # must be ignored
        if not {ready!r}:
            sys.exit(3)
        log.open("a").write("ballast:ready\\n")
        emit(event="ready", stages=[{{"stage": "faster-whisper", "gpu_mib": 3900}}],
             gpu_mib=3912)
        times = list({list(pass_ms)!r})
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "pass":
                ms = times.pop(0)
                log.open("a").write("ballast:pass\\n")
                emit(event="pass", ok=True, round_trip_ms=ms, tts_ms=ms / 2, stt_ms=ms / 2,
                     overlap=1.0, gpu_mib=3950)
            elif cmd == "quit":
                break
        log.open("a").write("ballast:bye\\n")
        emit(event="bye")
        """), encoding="utf-8")
    return [sys.executable, str(script)]


@pytest.fixture
def fake_ollama(monkeypatch, tmp_path):
    """Stub everything run_evals reads or does to Ollama/nvidia-smi; log the order."""
    events = tmp_path / "events.log"

    def note(text):
        with events.open("a") as handle:
            handle.write(text + "\n")

    state = {"leftover": {}, "final": {"m:latest": {"size": 5987, "vram": 5987, "ctx": 8192}}}
    monkeypatch.setattr(runner.ollama_control, "unload_all",
                        lambda: note("unload") or state["leftover"])
    monkeypatch.setattr(runner.ollama_control, "warm",
                        lambda model, ctx, keep: note(f"warm:{model}@{ctx}") or
                        {"seconds": 1.0, "load_seconds": 0.9, "error": None})
    monkeypatch.setattr(runner, "gpu_used_mib", lambda: note("gpu") or 1600)
    monkeypatch.setattr(runner, "gpu_total_mib", lambda: 12282)
    monkeypatch.setattr(runner, "ollama_vram", lambda: {})
    monkeypatch.setattr(runner, "ollama_models", lambda: state["final"])
    state["events"] = lambda: events.read_text().split() if events.exists() else []
    return state


def _main(tmp_path, *flags, ballast_cmd=None, monkeypatch=None):
    registry = {"routing": _check_script(tmp_path, "routing")}
    if ballast_cmd is not None:
        real = runner.VoiceBallast           # captured BEFORE the patch replaces the name
        monkeypatch.setattr(runner, "VoiceBallast",
                            lambda: real(cmd=ballast_cmd, ready_timeout_s=60, pass_timeout_s=60))
    log = tmp_path / "runs.jsonl"
    code = runner.main(["--model", "m", "--num-ctx", "8192", "--log", str(log), *flags],
                       registry=registry)
    return code, json.loads(log.read_text(encoding="utf-8").splitlines()[-1])


def test_plain_run_touches_nothing(tmp_path, fake_ollama):
    code, line = _main(tmp_path)
    assert code == 0
    assert "unload" not in fake_ollama["events"]() and line["clean"] is None


def test_clean_order_is_unload_idle_warm_then_checks(tmp_path, fake_ollama, capsys):
    code, line = _main(tmp_path, "--clean")
    assert code == 0
    events = fake_ollama["events"]()
    assert events.index("unload") < events.index("gpu") < events.index("warm:m@8192") \
        < events.index("check:routing")
    assert line["vram"]["idle_gpu_mib"] == 1600 and line["clean"]["leftover"] == {}
    assert "VRAM     idle 1600" in capsys.readouterr().out


def test_dirty_start_refuses_to_measure(tmp_path, fake_ollama, capsys):
    fake_ollama["leftover"] = {"llama3.1:latest": {"size": 5211}}
    code, line = _main(tmp_path, "--clean")
    assert code == 2
    assert "check:routing" not in fake_ollama["events"](), "no check may run on a dirty start"
    assert "start is not clean" in line["error"]
    assert "!! run: models still loaded" in capsys.readouterr().out


def test_with_voice_loads_ballast_before_the_llm_and_passes_fit(tmp_path, fake_ollama,
                                                              monkeypatch, capsys):
    code, line = _main(tmp_path, "--with-voice", monkeypatch=monkeypatch,
                       ballast_cmd=_ballast_script(tmp_path))
    assert code == 0
    events = fake_ollama["events"]()
    #: voice_main.boot()'s order: voice first, the LLM LAST, then a pass after the checks
    assert events.index("unload") < events.index("ballast:ready") < events.index("warm:m@8192")
    passes = [i for i, e in enumerate(events) if e == "ballast:pass"]
    assert len(passes) == 2, "one round trip before the checks, one after"
    assert passes[0] < events.index("check:routing") < passes[1]
    assert events[-1] == "ballast:bye", "the ballast was told to quit and did"
    assert line["fit"]["passed"] is True and line["settings"]["with_voice"] is True
    assert line["voice"]["pre_pass"]["round_trip_ms"] == 500
    out = capsys.readouterr().out
    assert "FIT      PASS" in out and "+voice 3912" in out


def test_a_ballast_that_never_gets_ready_is_could_not_run(tmp_path, fake_ollama, monkeypatch):
    code, line = _main(tmp_path, "--with-voice", monkeypatch=monkeypatch,
                       ballast_cmd=_ballast_script(tmp_path, ready=False))
    assert code == 2
    assert "voice ballast" in line["error"] and "exited before 'ready'" in line["error"]
    assert "check:routing" not in fake_ollama["events"]()


def test_fit_fail_is_loud_but_leaves_the_exit_code_alone(tmp_path, fake_ollama,
                                                       monkeypatch, capsys):
    #: the final /api/ps shows the model half on the CPU
    fake_ollama["final"] = {"m:latest": {"size": 10764, "vram": 10093, "ctx": 8192}}
    code, line = _main(tmp_path, "--with-voice", monkeypatch=monkeypatch,
                       ballast_cmd=_ballast_script(tmp_path))
    assert code == 0, "a model that does not fit is not a code regression"
    assert line["fit"]["passed"] is False and line["fit"]["offload"] == {"m:latest": 671}
    out = capsys.readouterr().out
    assert "FIT      *** FAIL ***" in out and "[FIT FAIL]" in out and "m 671 MiB" in out


def test_ballast_is_stopped_even_when_the_run_blows_up(tmp_path, fake_ollama, monkeypatch):
    started = []
    real = runner.VoiceBallast

    class Spy(real):
        def stop(self):
            started.append("stopped")
            super().stop()

    monkeypatch.setattr(runner, "VoiceBallast",
                        lambda: Spy(cmd=_ballast_script(tmp_path), ready_timeout_s=60))
    monkeypatch.setattr(runner, "run_check", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        runner.main(["--with-voice", "--log", str(tmp_path / "l.jsonl")],
                    registry={"routing": "x"})
    assert started == ["stopped"]


# ==========================================================================
# Step 5 — the ballast's own loop, on a fake voice
# ==========================================================================

class _FakeVoice:
    def __init__(self, fail_load=False, results=None):
        self.fail_load = fail_load
        #: first = the cold warm-up (slow, like Kokoro's first synthesis), then warm passes
        self.results = list(results or [{"ok": True, "round_trip_ms": 2400.0},
                                        {"ok": True, "round_trip_ms": 400.0}])

    def stages(self):
        def boom():
            raise RuntimeError("no CUDA")
        return [("faster-whisper", boom if self.fail_load else (lambda: "cuda/int8_float16")),
                ("kokoro", lambda: "kokoro")]

    def round_trip(self):
        return dict(self.results.pop(0))


def _messages(text):
    return [m for m in map(ballast_mod.parse_message, text.splitlines()) if m]


def test_serve_warms_up_before_ready_then_pass_quit():
    #: idle, after whisper, after kokoro, after the warm-up pass, at ready, after the pass
    gpu = iter([1600, 2900, 3250, 3670, 3670, 3700])
    out = io.StringIO()
    code = ballast_mod.serve(_FakeVoice(), io.StringIO("pass\nquit\n"), out,
                             gpu=lambda: next(gpu))
    ready, passed, bye = _messages(out.getvalue())
    assert code == 0
    assert [s["stage"] for s in ready["stages"]] == [
        "faster-whisper", "kokoro", "warm-up round trip"]
    assert ready["stages"][0]["delta_mib"] == 1300
    #: the warm-up's memory growth is its own line: what a live session really holds
    assert ready["stages"][2]["delta_mib"] == 420
    assert ready["warmup"]["round_trip_ms"] == 2400.0
    #: the first pass AFTER ready is the warm one, the FIT slowdown reference
    assert passed == {"event": "pass", "ok": True, "round_trip_ms": 400.0, "gpu_mib": 3700}
    assert bye["event"] == "bye"


def test_serve_exits_when_stdin_closes():
    out = io.StringIO()
    assert ballast_mod.serve(_FakeVoice(), io.StringIO(""), out, gpu=lambda: 1) == 0
    assert [m["event"] for m in _messages(out.getvalue())] == ["ready", "bye"]


def test_serve_reports_a_failed_load_and_exits_2():
    out = io.StringIO()
    assert ballast_mod.serve(_FakeVoice(fail_load=True), io.StringIO(""), out,
                             gpu=lambda: 1) == 2
    (message,) = _messages(out.getvalue())
    assert message["event"] == "error" and "no CUDA" in message["error"]


def test_a_crashing_round_trip_is_a_failed_pass_not_a_dead_ballast():
    class Broken(_FakeVoice):
        def round_trip(self):
            raise RuntimeError("CUDA out of memory")
    out = io.StringIO()
    ballast_mod.serve(Broken(), io.StringIO("pass\nquit\n"), out, gpu=lambda: 1)
    ready, passed = _messages(out.getvalue())[:2]
    assert ready["warmup"]["ok"] is False, "a failed warm-up is reported, not fatal"
    assert passed["ok"] is False and "out of memory" in passed["error"]


def test_parse_message_ignores_everything_else():
    assert ballast_mod.parse_message("Defaulting repo_id to hexgrad/Kokoro-82M") is None
    assert ballast_mod.parse_message("BALLAST not json") is None
    assert ballast_mod.parse_message('BALLAST {"no": "event"}') is None
    assert ballast_mod.parse_message('BALLAST {"event": "bye"}') == {"event": "bye"}


# ==========================================================================
# Step 5 — fit_verdict, one gate at a time
# ==========================================================================

def _run(**kw):
    base = dict(clean=True, with_voice=True, total_gpu_mib=12282, voice_gpu_mib=3912,
                pre_pass={"ok": True, "round_trip_ms": 500.0, "gpu_mib": 3950},
                post_pass={"ok": True, "round_trip_ms": 550.0, "gpu_mib": 9100},
                final_models={"m:latest": {"size": 5987, "vram": 5987, "ctx": 8192}})
    base.update(kw)
    return RunInfo(**base)


def _fits():
    return [_outcome(vram={"m:latest": 5987}, size={"m:latest": 5987}, gpu=9000)]


def test_fit_passes_when_all_three_gates_hold():
    verdict = fit_verdict(_fits(), _run())
    assert verdict["passed"] is True and verdict["reasons"] == []
    assert verdict["peak_gpu_mib"] == 9100 and verdict["slowdown"] == pytest.approx(1.1)


def test_fit_fails_on_cpu_offload_seen_in_a_case():
    spilled = [_outcome(vram={"m:latest": 10093}, size={"m:latest": 10764}, gpu=9000)]
    verdict = fit_verdict(spilled, _run())
    assert not verdict["passed"] and verdict["offload"] == {"m:latest": 671}


def test_fit_fails_on_offload_that_began_after_the_last_case():
    late = _run(final_models={"m:latest": {"size": 10764, "vram": 10093, "ctx": 8192}})
    assert fit_verdict(_fits(), late)["offload"] == {"m:latest": 671}


def test_fit_fails_without_headroom():
    tight = _run(post_pass={"ok": True, "round_trip_ms": 550.0, "gpu_mib": 11500})
    verdict = fit_verdict(_fits(), tight)
    assert not verdict["passed"]
    assert "peak 11500 MiB leaves 782 MiB of 12282; needs 1024 MiB headroom" in verdict["reasons"]


def test_fit_fails_when_voice_slows_down():
    slow = _run(post_pass={"ok": True, "round_trip_ms": 900.0, "gpu_mib": 9100})
    verdict = fit_verdict(_fits(), slow)
    assert not verdict["passed"] and verdict["slowdown"] == pytest.approx(1.8)
    assert "1.80x slower" in verdict["reasons"][0]


def test_slowdown_limit_is_adjustable():
    slow = _run(post_pass={"ok": True, "round_trip_ms": 900.0, "gpu_mib": 9100})
    assert fit_verdict(_fits(), slow, slowdown_limit=2.0)["passed"] is True


def test_fit_fails_when_voice_breaks_after_the_checks():
    broken = _run(post_pass={"ok": False, "error": "CUDA out of memory", "gpu_mib": 9100})
    verdict = fit_verdict(_fits(), broken)
    assert not verdict["passed"] and "CUDA out of memory" in verdict["reasons"][0]


def test_missing_evidence_is_a_fail_never_a_pass():
    blind = _run(total_gpu_mib=None, final_models=None)
    no_sizes = [_outcome(vram={"m:latest": 5987}, size=None, gpu=9000)]
    verdict = fit_verdict(no_sizes, blind)
    assert not verdict["passed"]
    assert any("cannot be ruled out" in r for r in verdict["reasons"])
    assert any("headroom is unknown" in r for r in verdict["reasons"])
