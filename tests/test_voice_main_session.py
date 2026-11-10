"""`voice_main.py` with sessions: the password prompt, the late-bound confirmer, the reporter.

    pytest tests/test_voice_main_session.py -v

No audio and no models: `AgentGraph` is a recorder and the durability proof is stubbed.
What is pinned is the wiring — and that a session-less `build_agent` (what `--check` and
scripts/check_routing.py rely on) is unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline_lg
import voice_main
from admin_session import Session


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _doc(tmp_path):
    doc = tmp_path / "d.txt"
    doc.write_text("x")
    return str(doc)


def test_build_agent_passes_session_and_confirmer_through(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", Recorder)
    monkeypatch.setattr(voice_main, "_ensure_durable_memory", lambda path: None)
    confirmer = object()
    agent = voice_main.build_agent(_doc(tmp_path), session=Session("admin", 0),
                                   run_confirm=confirmer)
    assert agent.kwargs["session"] == Session("admin", 0)
    assert agent.kwargs["run_confirm"] is confirmer


def test_build_agent_skips_the_durability_proof_for_guests(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", Recorder)

    def boom(path):
        raise AssertionError("durability proof ran for a guest")
    monkeypatch.setattr(voice_main, "_ensure_durable_memory", boom)
    agent = voice_main.build_agent(_doc(tmp_path), session=Session("guest", 1))
    assert agent.kwargs["session"].role == "guest"


def test_build_agent_without_a_session_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", Recorder)
    seen = []
    monkeypatch.setattr(voice_main, "_ensure_durable_memory", seen.append)
    agent = voice_main.build_agent(_doc(tmp_path))
    assert seen == [voice_main.USER_HISTORY_PATH]
    assert agent.kwargs["session"] is None and agent.kwargs["run_confirm"] is None


def test_reporter_prints_the_confirm_event(capsys):
    report = voice_main._make_reporter()
    report("confirm", {"turn_id": 1, "transcript": "yes", "accepted": True})
    report("state", {"state": "confirming"})
    out = capsys.readouterr().out
    assert "run it? heard 'yes' -> yes" in out
    assert "[?!] confirming" in out


def test_run_voice_binds_the_confirmer_to_the_orchestrator(monkeypatch):
    """The agent is built before the orchestrator, so the confirmer must be bound after."""
    import asyncio
    import voice.orchestrator as orch

    captured = {}

    class FakeMic:
        def start(self):
            return self

    def fake_boot(config, file_path, report=print, session=None, run_confirm=None):
        captured["session"], captured["confirmer"] = session, run_confirm
        return {"agent": object(), "mic": FakeMic(), "vad": object(), "stt": object(),
                "tts": object(), "speaker": object()}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            self.turn_id = 0
            captured["orchestrator"] = self

    async def fake_session(orchestrator):
        captured["bound"] = captured["confirmer"].orchestrator

    monkeypatch.setattr(voice_main, "boot", fake_boot)
    monkeypatch.setattr(orch, "VoiceOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(orch, "run_session", fake_session)

    from voice.config import VoiceConfig
    assert asyncio.run(voice_main.run_voice(VoiceConfig(), "doc.txt",
                                            session=Session("admin", 0))) == 0
    assert captured["session"] == Session("admin", 0)
    assert captured["bound"] is captured["orchestrator"]


def test_main_asks_for_a_password_only_for_a_live_session(monkeypatch, tmp_path):
    asked, ran = [], []
    monkeypatch.setattr(voice_main, "prompt_for_session",
                        lambda: asked.append(1) or Session("guest", 1))
    monkeypatch.setattr(voice_main, "run_check", lambda config, f: ran.append("check") or 0)

    async def fake_run_voice(config, f, session=None):
        ran.append(session)
        return 0
    monkeypatch.setattr(voice_main, "run_voice", fake_run_voice)
    import voice.audio_io as audio_io
    monkeypatch.setattr(audio_io, "resolve_device", lambda *a, **k: None)
    doc = _doc(tmp_path)

    assert voice_main.main(["--file", doc, "--check"]) == 0
    assert asked == [] and ran == ["check"]
    assert voice_main.main(["--file", doc]) == 0
    assert asked == [1] and ran == ["check", Session("guest", 1)]
