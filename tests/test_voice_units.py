"""V1a unit tests: the VAD state machine, telemetry, config, and sentence splitting.

Nothing here loads a model, opens a device, or touches the network. That is the point
of the split in `voice/vad.py`: `UtteranceDetector` consumes (frame, probability) pairs
and knows nothing about silero, so start detection, pre-roll and endpointing can be
driven by hand-written probability sequences and asserted exactly — the same reason
`memory/reconciliation.py` keeps `find_safe_cut_point` a pure function.

    pytest tests/test_voice_units.py -v
    python tests/test_voice_units.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.config import VoiceConfig, VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE
from voice.telemetry import TelemetryWriter, TurnTelemetry, Stopwatch, TURN_FIELDS
from voice.tts import split_sentences
from voice.vad import UtteranceDetector
from voice.audio_io import resample_linear


def frame(value: float = 0.0) -> np.ndarray:
    """One VAD-sized frame, filled with a marker value so provenance is assertable."""
    return np.full(VAD_FRAME_SAMPLES, value, dtype=np.float32)


def feed(detector, probabilities, values=None):
    """Push a probability sequence through. Returns the list of utterances emitted."""
    out = []
    for index, prob in enumerate(probabilities):
        value = values[index] if values is not None else float(index)
        result = detector.process(frame(value), prob)
        if result is not None:
            out.append(result)
    return out


# ==========================================================================
# config
# ==========================================================================

def test_config_defaults():
    config = VoiceConfig()
    assert config.sample_rate == 16_000
    assert config.blocksize == 512
    assert config.channels == 1
    assert config.vad_threshold == 0.5
    assert config.pre_roll_ms == 500
    assert config.silence_ms == 600
    assert config.min_utterance_s == 0.3
    assert config.stt_model == "large-v3-turbo"
    assert config.stt_device == "cuda"
    assert config.stt_compute_type == "int8_float16"
    assert config.stt_cpu_compute_type == "int8"
    assert config.tts_voice == "af_heart"
    assert config.tts_sample_rate == 24_000
    assert config.telemetry_path == "voice_logs/turns.jsonl"
    assert config.thread_id == "voice-main"


def test_config_derived_frame_counts():
    """32 ms per frame at 512/16000, so 500 ms is ~16 frames and 600 ms ~19."""
    config = VoiceConfig()
    assert config.frame_duration_s == pytest.approx(0.032)
    assert config.pre_roll_frames == 16
    assert config.silence_frames == 19
    assert config.max_utterance_frames == 938


def test_config_validate_accepts_defaults():
    VoiceConfig().validate()


@pytest.mark.parametrize("kwargs, expected", [
    ({"sample_rate": 8000}, "16000 Hz"),
    ({"blocksize": 1024}, "512-sample"),
    ({"channels": 2}, "mono"),
    ({"vad_threshold": 0.0}, "vad_threshold"),
    ({"silence_ms": 0}, "silence_ms"),
    ({"max_utterance_s": 0.1}, "max_utterance_s"),
])
def test_config_validate_rejects_bad_values(kwargs, expected):
    """Silero accepts exactly one frame size at one rate; the rest are ours."""
    with pytest.raises(ValueError, match=expected):
        VoiceConfig(**kwargs).validate()


def test_config_is_frozen():
    """A mid-session change would desync the ring buffer from the frames in it."""
    with pytest.raises(Exception):
        VoiceConfig().vad_threshold = 0.9


# ==========================================================================
# VAD state machine — start detection
# ==========================================================================

def test_silence_alone_never_emits():
    detector = UtteranceDetector(VoiceConfig())
    assert feed(detector, [0.0] * 50) == []
    assert not detector.is_speaking


def test_one_loud_frame_starts_an_utterance():
    detector = UtteranceDetector(VoiceConfig())
    feed(detector, [0.9])
    assert detector.is_speaking


def test_probability_exactly_at_threshold_counts_as_speech():
    """`>=`, not `>` — documented here so a later tweak is a deliberate one."""
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    feed(detector, [config.vad_threshold])
    assert detector.is_speaking


def test_just_below_threshold_does_not_start():
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    feed(detector, [config.vad_threshold - 0.01])
    assert not detector.is_speaking


# ==========================================================================
# VAD state machine — pre-roll
# ==========================================================================

def test_pre_roll_frames_are_prepended_to_the_utterance():
    """The whole reason pre-roll exists: silero needs a few frames of a word before
    it crosses the threshold, so the true start is already behind us."""
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    #: 5 silent frames marked 0..4, then speech marked 100, then silence to end it
    values = [0, 1, 2, 3, 4, 100] + [200] * config.silence_frames
    probs = [0.0] * 5 + [0.9] + [0.0] * config.silence_frames
    emitted = feed(detector, probs, values)

    assert len(emitted) == 1
    utterance = emitted[0]
    #: 5 pre-roll + 1 speech + silence_frames of trailing silence
    assert len(utterance) == (6 + config.silence_frames) * VAD_FRAME_SAMPLES
    #: the first sample comes from the frame recorded BEFORE speech was detected
    assert utterance[0] == 0.0
    assert utterance[4 * VAD_FRAME_SAMPLES] == 4.0
    assert utterance[5 * VAD_FRAME_SAMPLES] == 100.0


def test_pre_roll_is_bounded_to_its_configured_length():
    """Long silence must not accumulate — it is a ring buffer, not a list."""
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    silent = 200                                   # far more than pre_roll_frames (16)
    values = list(range(silent)) + [999] + [888] * config.silence_frames
    probs = [0.0] * silent + [0.9] + [0.0] * config.silence_frames
    emitted = feed(detector, probs, values)

    expected_frames = config.pre_roll_frames + 1 + config.silence_frames
    assert len(emitted[0]) == expected_frames * VAD_FRAME_SAMPLES
    #: the oldest kept frame is exactly pre_roll_frames back from the speech frame
    assert emitted[0][0] == float(silent - config.pre_roll_frames)


def test_pre_roll_shorter_than_the_buffer_is_kept_whole():
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    emitted = feed(detector,
                   [0.0, 0.0] + [0.9] + [0.0] * config.silence_frames,
                   [10, 11, 12] + [13] * config.silence_frames)
    assert emitted[0][0] == 10.0


# ==========================================================================
# VAD state machine — endpointing
# ==========================================================================

def test_utterance_ends_after_the_configured_trailing_silence():
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    #: one short of the threshold: still listening
    assert feed(detector, [0.9] + [0.0] * (config.silence_frames - 1)) == []
    assert detector.is_speaking
    #: the next silent frame is the one that endpoints
    assert len(feed(detector, [0.0])) == 1
    assert not detector.is_speaking


def test_a_mid_sentence_pause_does_not_end_the_utterance():
    """People breathe. A cumulative silence count would end on the third comma."""
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    pause = [0.0] * (config.silence_frames - 1)
    emitted = feed(detector, [0.9] + pause + [0.9] + pause + [0.9])
    assert emitted == []
    assert detector.is_speaking


def test_detector_resets_between_utterances():
    """Two utterances in a row must not bleed into each other."""
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    tail = [0.0] * config.silence_frames
    first = feed(detector, [0.9] + tail)
    second = feed(detector, [0.9] + tail)
    assert len(first) == len(second) == 1
    #: the second carries no pre-roll from the first's trailing silence, because
    #: _finish() cleared it
    assert len(second[0]) == VAD_FRAME_SAMPLES * (1 + config.silence_frames)


def test_max_utterance_length_forces_an_endpoint():
    """A stuck-open mic must not grow the buffer without bound.

    Unbroken speech past the cap is chopped into successive bounded utterances rather
    than one growing one — so the detector is legitimately still `is_speaking` at the
    end, having started the next chunk. What matters is that no single emission
    exceeds the cap and that the audio keeps flowing.
    """
    config = VoiceConfig(max_utterance_s=0.5)     # ~15 frames
    detector = UtteranceDetector(config)
    emitted = feed(detector, [0.9] * 100)

    assert len(emitted) >= 1
    cap_samples = config.max_utterance_frames * VAD_FRAME_SAMPLES
    assert all(len(utterance) <= cap_samples for utterance in emitted)
    assert sum(len(u) for u in emitted) <= 100 * VAD_FRAME_SAMPLES, "no frame duplicated"


def test_reset_discards_a_partial_utterance():
    """Called when leaving LISTENING, so half an utterance is not glued to the next."""
    detector = UtteranceDetector(VoiceConfig())
    feed(detector, [0.9, 0.9])
    assert detector.buffered_frames > 0
    detector.reset()
    assert detector.buffered_frames == 0
    assert not detector.is_speaking


# ==========================================================================
# telemetry
# ==========================================================================

def test_turn_row_has_exactly_the_required_fields():
    """§10's list, verbatim: timestamp, turn id, utterance duration, stt_ms, agent_ms,
    tts_first_sentence_ms, speech_end_to_first_audio_ms, transcript length, response
    length, TTS engine."""
    row = TurnTelemetry(turn_id=1, utterance_s=1.5, tts_engine="kokoro").as_row()
    assert list(row) == list(TURN_FIELDS)
    assert set(row) == {
        "timestamp", "turn_id", "utterance_s", "stt_ms", "agent_ms",
        "tts_first_sentence_ms", "speech_end_to_first_audio_ms",
        "transcript_chars", "response_chars", "tts_engine",
    }


def test_unfilled_timings_stay_null_rather_than_zero():
    """A turn that died in STT must not report agent_ms=0 into the baseline."""
    row = TurnTelemetry(turn_id=1, utterance_s=1.0, tts_engine="kokoro").as_row()
    assert row["agent_ms"] is None
    assert row["speech_end_to_first_audio_ms"] is None


def test_timestamp_is_iso_utc():
    row = TurnTelemetry(turn_id=1, utterance_s=1.0, tts_engine="kokoro").as_row()
    from datetime import datetime
    parsed = datetime.fromisoformat(row["timestamp"])
    assert parsed.tzinfo is not None


def test_writer_appends_one_json_line_per_turn(tmp_path):
    path = tmp_path / "nested" / "turns.jsonl"
    writer = TelemetryWriter(str(path))
    for turn in (1, 2, 3):
        telemetry = TurnTelemetry(turn_id=turn, utterance_s=1.0, tts_engine="kokoro")
        telemetry.stt_ms = 123.456
        assert writer.write(telemetry) is True

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(line) for line in lines]
    assert [r["turn_id"] for r in rows] == [1, 2, 3]
    assert rows[0]["stt_ms"] == 123.46, "floats are rounded on the way out"
    assert rows[0]["tts_engine"] == "kokoro"


def test_writer_never_raises_on_a_bad_path():
    """Telemetry may lose data; it may never lose the turn."""
    writer = TelemetryWriter("")                      # Path("") -> unwritable
    assert writer.write(
        TurnTelemetry(turn_id=1, utterance_s=1.0, tts_engine="kokoro")) is False
    assert writer.last_error


def test_stopwatch_records_even_when_the_block_raises():
    """A stage that times out should still report how long it took to fail."""
    watch = Stopwatch()
    with pytest.raises(RuntimeError):
        with watch:
            raise RuntimeError("boom")
    assert watch.ms is not None and watch.ms >= 0


# ==========================================================================
# sentence splitting (Kokoro has a per-call input limit)
# ==========================================================================

def test_split_on_sentence_punctuation():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_ignores_empty_and_whitespace_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_keeps_a_single_unpunctuated_sentence_whole():
    assert split_sentences("no punctuation here") == ["no punctuation here"]


def test_split_enforces_the_length_cap():
    """The load-bearing half: the model may simply not emit sentence punctuation."""
    text = " ".join(["word"] * 500)
    chunks = split_sentences(text, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == text.split(), "no words were lost"


def test_split_cannot_loop_on_an_unbreakable_token():
    """A single token longer than the cap gets a hard cut rather than spinning."""
    chunks = split_sentences("x" * 250, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "x" * 250


def test_split_handles_a_closing_quote_after_the_period():
    assert split_sentences('He said "hi." Then left.') == ['He said "hi."', "Then left."]


# ==========================================================================
# resampling (only used off the live path — check + the round-trip test)
# ==========================================================================

def test_resample_changes_length_by_the_rate_ratio():
    audio = np.linspace(-1, 1, 24_000, dtype=np.float32)
    out = resample_linear(audio, 24_000, 16_000)
    assert len(out) == 16_000
    assert out.dtype == np.float32


def test_resample_is_a_noop_at_equal_rates():
    audio = np.linspace(-1, 1, 100, dtype=np.float32)
    assert np.array_equal(resample_linear(audio, 16_000, 16_000), audio)


def test_resample_preserves_a_constant_signal():
    """Linear interpolation of a constant is that constant — a cheap sanity check
    that the interpolation is not shifting or scaling anything."""
    out = resample_linear(np.full(1000, 0.25, dtype=np.float32), 24_000, 16_000)
    assert np.allclose(out, 0.25)


def test_resample_of_empty_audio_is_empty():
    assert len(resample_linear(np.zeros(0, dtype=np.float32), 24_000, 16_000)) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
