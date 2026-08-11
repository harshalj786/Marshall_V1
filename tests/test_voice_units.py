"""Voice unit tests: the VAD state machines, telemetry, config, and sentence splitting.

Nothing here loads a model, opens a device, or touches the network. That is the point
of the split in `voice/vad.py`: `UtteranceDetector` and `BargeInGuard` consume
(frame, probability) pairs and know nothing about silero, so start detection, pre-roll,
endpointing and the interrupt guard can be driven by hand-written probability sequences
and asserted exactly — the same reason `memory/reconciliation.py` keeps
`find_safe_cut_point` a pure function.

V1b adds three groups: the barge-in guard, the seeding that hands an interruption to the
detector, and the truncation that decides what an interrupted answer left behind.

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
from voice.orchestrator import INTERRUPTION_MARKER, truncate_to_spoken
from voice.telemetry import TelemetryWriter, TurnTelemetry, Stopwatch, TURN_FIELDS
from voice.tts import split_sentences
from voice.vad import BargeInGuard, UtteranceDetector
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


def feed_guard(guard, probabilities, values=None):
    """Push a probability sequence through the guard.

    Returns `(index_that_fired, captured_frames)`, or `(None, None)` if it never fired.
    """
    for index, prob in enumerate(probabilities):
        value = values[index] if values is not None else float(index)
        captured = guard.process(frame(value), prob)
        if captured is not None:
            return index, captured
    return None, None


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


def test_barge_in_defaults():
    """On by default, and harder to trigger than the listening endpointer.

    Both numbers are unmeasured guesses (config.py says so). What is asserted here is
    not that they are right — it is the RELATIONSHIP that makes them defensible: a false
    endpoint costs one bad transcript, a false interrupt cuts Marshall off mid-sentence,
    so the interrupt side must be the harder one to trip.
    """
    config = VoiceConfig()
    assert config.barge_in_mode == "vad"
    assert config.barge_in_enabled is True
    assert config.barge_in_vad_threshold == 0.7
    assert config.barge_in_vad_threshold > config.vad_threshold
    assert config.barge_in_sustained_ms == 300


def test_barge_in_off_reports_itself_disabled():
    assert VoiceConfig(barge_in_mode="off").barge_in_enabled is False


def test_config_derived_frame_counts():
    """32 ms per frame at 512/16000, so 500 ms is ~16 frames and 600 ms ~19."""
    config = VoiceConfig()
    assert config.frame_duration_s == pytest.approx(0.032)
    assert config.pre_roll_frames == 16
    assert config.silence_frames == 19
    assert config.max_utterance_frames == 938
    #: 300 ms of sustained speech is 9 frames — the guard's whole defence against a cough
    assert config.barge_in_sustained_frames == 9


def test_config_validate_accepts_defaults():
    VoiceConfig().validate()


@pytest.mark.parametrize("kwargs, expected", [
    ({"sample_rate": 8000}, "16000 Hz"),
    ({"blocksize": 1024}, "512-sample"),
    ({"channels": 2}, "mono"),
    ({"vad_threshold": 0.0}, "vad_threshold"),
    ({"silence_ms": 0}, "silence_ms"),
    ({"max_utterance_s": 0.1}, "max_utterance_s"),
    ({"barge_in_mode": "wake_word"}, "barge_in_mode"),
    ({"barge_in_vad_threshold": 1.0}, "barge_in_vad_threshold"),
    #: 0 would fire on one frame, which is the transient the guard exists to reject;
    #: "never interrupt" is spelled barge_in_mode="off"
    ({"barge_in_sustained_ms": 0}, "barge_in_sustained_ms"),
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
# barge-in guard (V1b)
#
# The property under test is asymmetry. `UtteranceDetector` starts an utterance on ONE
# frame over 0.5, because clipping the user's first word is the expensive failure there.
# The guard must do the opposite: a cough, a door or a keyboard reaching the mic while
# Marshall is mid-answer must not silence him, so it takes a sustained run over a higher
# threshold — and both numbers come from config, because both were guessed.
# ==========================================================================

def test_a_single_loud_frame_does_not_interrupt():
    """The V1a detector would have started an utterance here. The guard must not fire."""
    guard = BargeInGuard(VoiceConfig())
    fired, captured = feed_guard(guard, [1.0])
    assert fired is None and captured is None


def test_a_transient_below_the_sustained_duration_does_not_interrupt():
    """One frame short of the run, repeatedly — a door, a cough, a chair."""
    config = VoiceConfig()
    guard = BargeInGuard(config)
    burst = [1.0] * (config.barge_in_sustained_frames - 1) + [0.0] * 5
    fired, _ = feed_guard(guard, burst * 6)
    assert fired is None, "a sub-duration transient interrupted Marshall"


def test_sustained_speech_interrupts():
    config = VoiceConfig()
    guard = BargeInGuard(config)
    fired, captured = feed_guard(guard, [0.95] * 50)
    #: fires on the frame that COMPLETES the run, not one later
    assert fired == config.barge_in_sustained_frames - 1
    assert captured, "the guard fired but handed back no audio"


def test_the_run_must_be_unbroken():
    """A single quiet frame resets it. Speech that survives ~300ms is not a transient;
    speech interrupted every few frames is room noise."""
    config = VoiceConfig()
    guard = BargeInGuard(config)
    stutter = ([1.0] * (config.barge_in_sustained_frames - 1) + [0.1]) * 10
    fired, _ = feed_guard(guard, stutter)
    assert fired is None


def test_speech_below_the_barge_in_threshold_never_interrupts():
    """0.6 endpoints an utterance in LISTENING and must NOT interrupt in SPEAKING —
    the one assertion that pins the two thresholds as genuinely separate."""
    config = VoiceConfig()
    assert config.vad_threshold < 0.6 < config.barge_in_vad_threshold
    guard = BargeInGuard(config)
    fired, _ = feed_guard(guard, [0.6] * 200)
    assert fired is None


@pytest.mark.parametrize("sustained_ms, expected_frames", [(300, 9), (1000, 31)])
def test_the_sustained_duration_is_honoured_from_config(sustained_ms, expected_frames):
    """The number is a guess to be corrected from telemetry, so changing it in config
    must be the only thing anyone has to do."""
    config = VoiceConfig(barge_in_sustained_ms=sustained_ms)
    assert config.barge_in_sustained_frames == expected_frames
    guard = BargeInGuard(config)
    assert feed_guard(guard, [1.0] * (expected_frames - 1))[0] is None
    assert feed_guard(BargeInGuard(config), [1.0] * expected_frames)[0] == expected_frames - 1


def test_the_threshold_is_honoured_from_config():
    config = VoiceConfig(barge_in_vad_threshold=0.95)
    guard = BargeInGuard(config)
    assert feed_guard(guard, [0.9] * 100)[0] is None
    assert feed_guard(BargeInGuard(config), [0.96] * 100)[0] is not None


def test_barge_in_mode_off_disables_the_guard_entirely():
    """V1a behaviour, for a user on speakers: nothing interrupts, and no state is kept."""
    guard = BargeInGuard(VoiceConfig(barge_in_mode="off"))
    assert guard.enabled is False
    fired, _ = feed_guard(guard, [1.0] * 500)
    assert fired is None
    assert guard.speech_run == 0, "a disabled guard was still counting"


def test_the_capture_carries_the_pre_roll_and_the_whole_guard_window():
    """The user must not have to repeat themselves.

    What fired the guard is ~300ms of the user already talking, and the pre-roll before
    it is the start of the word that silero needed a few frames to recognise. Both become
    the head of the next utterance, so the interruption IS the next turn's opening.
    """
    config = VoiceConfig()
    guard = BargeInGuard(config)
    quiet = 100                                     # far more than pre_roll_frames
    probs = [0.0] * quiet + [0.95] * config.barge_in_sustained_frames
    values = list(range(quiet)) + [900 + i for i in range(config.barge_in_sustained_frames)]
    fired, captured = feed_guard(guard, probs, values)

    assert fired is not None
    assert len(captured) == config.pre_roll_frames + config.barge_in_sustained_frames
    #: oldest kept frame is exactly pre_roll_frames before the run started
    assert captured[0][0] == float(quiet - config.pre_roll_frames)
    #: and the last is the frame that fired it
    assert captured[-1][0] == float(900 + config.barge_in_sustained_frames - 1)


def test_an_interrupt_that_lands_immediately_still_captures_what_it_heard():
    """No pre-roll available yet (the guard is reset at the top of every turn), so the
    capture is just the run — never empty, which is what would lose the user's words."""
    config = VoiceConfig()
    guard = BargeInGuard(config)
    _fired, captured = feed_guard(guard, [0.95] * config.barge_in_sustained_frames)
    assert len(captured) == config.barge_in_sustained_frames


def test_the_guard_rearms_after_firing():
    """It fires once per run: the orchestrator owns everything after the cut, so a still
    armed run would only race its own interrupt."""
    config = VoiceConfig()
    guard = BargeInGuard(config)
    fired, _ = feed_guard(guard, [0.95] * config.barge_in_sustained_frames)
    assert fired is not None
    assert guard.speech_run == 0
    #: and a fresh sustained run can fire again
    assert feed_guard(guard, [0.95] * config.barge_in_sustained_frames)[0] is not None


# ==========================================================================
# handing an interruption to the endpointer (V1b)
# ==========================================================================

def test_seeded_frames_become_the_head_of_the_next_utterance():
    """The barge-in hand-off: the detector resumes MID-utterance, and endpointing from
    there is completely ordinary."""
    config = VoiceConfig()
    detector = UtteranceDetector(config)
    detector.seed([frame(77.0), frame(78.0), frame(79.0)])
    assert detector.is_speaking
    assert detector.buffered_frames == 3

    emitted = feed(detector, [0.0] * config.silence_frames, [5.0] * config.silence_frames)
    assert len(emitted) == 1
    assert len(emitted[0]) == (3 + config.silence_frames) * VAD_FRAME_SAMPLES
    assert emitted[0][0] == 77.0, "the user's own first word was dropped"


def test_seeding_nothing_leaves_the_detector_idle():
    """A guard that fired with an empty capture must not leave a phantom utterance open."""
    detector = UtteranceDetector(VoiceConfig())
    detector.seed([])
    assert not detector.is_speaking
    assert detector.buffered_frames == 0


def test_seeding_replaces_whatever_was_buffered():
    """Seed is a hand-off, not an append: anything the detector was holding predates the
    interruption and belongs to the turn that just ended."""
    detector = UtteranceDetector(VoiceConfig())
    feed(detector, [0.9, 0.9], [1.0, 2.0])
    detector.seed([frame(50.0)])
    assert detector.buffered_frames == 1


# ==========================================================================
# truncating an interrupted answer (V1b)
#
# H2's summariser and UserMemory both compress from the stored history, so this decides
# what the rest of the system believes Marshall said.
# ==========================================================================

SENTENCES = ["One.", "Two.", "Three.", "Four."]


def test_truncation_keeps_only_the_sentences_that_were_played():
    assert truncate_to_spoken(SENTENCES, 2) == f"One. Two. {INTERRUPTION_MARKER}"


def test_truncation_with_nothing_played_is_the_marker_alone():
    """Cut before the first sentence finished: the honest record is that he said
    nothing the user heard."""
    assert truncate_to_spoken(SENTENCES, 0) == INTERRUPTION_MARKER


def test_truncation_of_a_fully_played_answer_keeps_every_sentence():
    assert truncate_to_spoken(SENTENCES, 4) == (
        f"One. Two. Three. Four. {INTERRUPTION_MARKER}")


def test_truncation_marks_the_interruption_so_history_says_why_it_stops():
    """Without the marker a later turn — and both summarisers — read a truncated answer
    as Marshall's complete one."""
    assert truncate_to_spoken(SENTENCES, 1).endswith(INTERRUPTION_MARKER)


def test_truncation_survives_a_count_it_cannot_honour():
    """Defensive: a negative or over-long count must not raise on the interrupt path."""
    assert truncate_to_spoken(SENTENCES, -1) == INTERRUPTION_MARKER
    assert truncate_to_spoken(SENTENCES, 99).startswith("One. Two. Three. Four.")
    assert truncate_to_spoken([], 3) == INTERRUPTION_MARKER


# ==========================================================================
# telemetry
# ==========================================================================

def test_turn_row_has_exactly_the_required_fields():
    """§10's list, verbatim: timestamp, turn id, utterance duration, stt_ms, agent_ms,
    tts_first_sentence_ms, speech_end_to_first_audio_ms, transcript length, response
    length, TTS engine — plus V1b's four barge-in columns."""
    row = TurnTelemetry(turn_id=1, utterance_s=1.5, tts_engine="kokoro").as_row()
    assert list(row) == list(TURN_FIELDS)
    assert set(row) == {
        "timestamp", "turn_id", "utterance_s", "stt_ms", "agent_ms",
        "tts_first_sentence_ms", "speech_end_to_first_audio_ms",
        "transcript_chars", "response_chars", "tts_engine",
        "was_interrupted", "sentences_played", "sentences_total",
        "time_to_silence_ms",
    }


def test_v1a_fields_stay_the_prefix_of_the_row():
    """V1b appended rather than interleaved, so a log holding both shapes still reads
    column-for-column for the first ten."""
    assert TURN_FIELDS[:10] == (
        "timestamp", "turn_id", "utterance_s", "stt_ms", "agent_ms",
        "tts_first_sentence_ms", "speech_end_to_first_audio_ms",
        "transcript_chars", "response_chars", "tts_engine")


def test_unfilled_timings_stay_null_rather_than_zero():
    """A turn that died in STT must not report agent_ms=0 into the baseline."""
    row = TurnTelemetry(turn_id=1, utterance_s=1.0, tts_engine="kokoro").as_row()
    assert row["agent_ms"] is None
    assert row["speech_end_to_first_audio_ms"] is None


def test_barge_in_fields_default_to_an_uninterrupted_turn():
    """`was_interrupted` is False rather than None — every turn that writes a row has an
    answer to that question. The counts stay None until SPEAKING is actually reached,
    since 0 would read as "spoke nothing" rather than "never got there"."""
    row = TurnTelemetry(turn_id=1, utterance_s=1.0, tts_engine="kokoro").as_row()
    assert row["was_interrupted"] is False
    assert row["sentences_played"] is None
    assert row["sentences_total"] is None
    assert row["time_to_silence_ms"] is None


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


# ==========================================================================
# device resolution
#
# Modelled on a real machine, where the SAME Realtek microphone is enumerated once
# per host API and only some of those can do 16 kHz: the WASAPI view is 48 kHz-native
# and refuses, while the MME and DirectSound views resample and accept. Picking by
# index alone hits that trap roughly half the time, and the index itself moves when a
# Bluetooth controller connects.
# ==========================================================================

#: (name, hostapi, max_in, max_out, rates this device will actually accept)
_FAKE_DEVICES = [
    ("Microsoft Sound Mapper - Input",        "MME",                2, 0, {44100}),
    ("Headset Microphone (Wireless Co",       "MME",                1, 0, {16000, 44100}),
    ("Microphone Array (2- Realtek(R)",       "MME",                2, 0, {16000, 44100}),
    ("Headset Microphone (Wireless Controller)", "Windows DirectSound", 1, 0, {16000}),
    ("Microphone Array (2- Realtek(R) Audio)", "Windows DirectSound", 2, 0, {16000, 44100}),
    ("Headset Microphone (Wireless Controller)", "Windows WASAPI",   1, 0, {16000}),
    ("Microphone Array (2- Realtek(R) Audio)", "Windows WASAPI",     2, 0, {48000}),
    ("Speakers (2- Realtek(R) Audio)",        "Windows DirectSound", 0, 2, {24000, 48000}),
    ("Speakers (2- Realtek(R) Audio)",        "Windows WASAPI",      0, 2, {48000}),
]


@pytest.fixture
def fake_sounddevice(monkeypatch):
    """Install a stand-in `sounddevice`. `resolve_device` imports it inside the
    function, so replacing the module entry is enough and no real audio stack loads."""
    import sys as _sys
    import types

    hostapis = []
    for _n, host, *_ in _FAKE_DEVICES:
        if host not in hostapis:
            hostapis.append(host)

    def _entry(spec):
        name, host, max_in, max_out, _rates = spec
        return {"name": name, "hostapi": hostapis.index(host),
                "max_input_channels": max_in, "max_output_channels": max_out,
                "default_samplerate": 44100.0}

    module = types.ModuleType("sounddevice")
    module.query_devices = lambda index=None: (
        [_entry(s) for s in _FAKE_DEVICES] if index is None else _entry(_FAKE_DEVICES[index]))
    module.query_hostapis = lambda i: {"name": hostapis[i]}

    def _check(want_input):
        def check(device=None, samplerate=None, channels=None, dtype=None):
            name, host, max_in, max_out, rates = _FAKE_DEVICES[device]
            slots = max_in if want_input else max_out
            if slots < (channels or 1) or samplerate not in rates:
                raise ValueError("Invalid sample rate [PaErrorCode -9997]")
        return check

    module.check_input_settings = _check(True)
    module.check_output_settings = _check(False)
    monkeypatch.setitem(_sys.modules, "sounddevice", module)
    return module


def test_resolve_none_means_system_default(fake_sounddevice):
    from voice.audio_io import resolve_device
    assert resolve_device(None, 16_000) is None


def test_resolve_by_name_skips_the_host_api_that_cannot_do_16k(fake_sounddevice):
    """The regression this whole function exists for: index 6 is the WASAPI view of
    the Realtek array and is 48 kHz-only, so it must not be chosen."""
    from voice.audio_io import resolve_device
    assert resolve_device("Realtek", 16_000) == 4        # the DirectSound view


def test_resolve_prefers_the_lowest_latency_usable_host_api(fake_sounddevice):
    """Both the MME (2) and DirectSound (4) views work; DirectSound is the better one.
    WASAPI would win if it were usable, which for this device it is not."""
    from voice.audio_io import resolve_device
    assert resolve_device("Microphone Array", 16_000) == 4


def test_resolve_picks_wasapi_when_it_is_usable(fake_sounddevice):
    """For a natively-16 kHz headset every host API works, so the ranking decides."""
    from voice.audio_io import resolve_device
    assert resolve_device("Wireless Co", 16_000) == 5     # the WASAPI view


def test_resolve_is_case_insensitive(fake_sounddevice):
    from voice.audio_io import resolve_device
    assert resolve_device("realtek", 16_000) == resolve_device("REALTEK", 16_000)


def test_resolve_accepts_an_explicit_index(fake_sounddevice):
    from voice.audio_io import resolve_device
    assert resolve_device(2, 16_000) == 2
    assert resolve_device("2", 16_000) == 2, "an index arriving as a CLI string"


def test_explicit_index_is_never_silently_substituted(fake_sounddevice):
    """Naming one device and getting another is worse than an error."""
    from voice.audio_io import resolve_device, DeviceNotUsableError
    with pytest.raises(DeviceNotUsableError, match="cannot run at 16000 Hz"):
        resolve_device(6, 16_000)                         # the 48 kHz-only WASAPI view


def test_unusable_index_error_names_the_fix(fake_sounddevice):
    from voice.audio_io import resolve_device, DeviceNotUsableError
    with pytest.raises(DeviceNotUsableError) as excinfo:
        resolve_device(6, 16_000)
    message = str(excinfo.value)
    assert "--input-device" in message and "Microphone Array" in message
    assert message.isascii(), "error text is printed to a Windows console"


def test_no_match_names_the_next_step(fake_sounddevice):
    from voice.audio_io import resolve_device, DeviceNotUsableError
    with pytest.raises(DeviceNotUsableError, match="--list-devices"):
        resolve_device("no such microphone", 16_000)


def test_all_matches_unusable_lists_what_was_rejected(fake_sounddevice):
    """A rate nothing supports must report the candidates, not just fail."""
    from voice.audio_io import resolve_device, DeviceNotUsableError
    with pytest.raises(DeviceNotUsableError, match="none can run at 96000"):
        resolve_device("Realtek", 96_000)


def test_input_and_output_are_resolved_separately(fake_sounddevice):
    """A name matching both a mic and a speaker must not cross over."""
    from voice.audio_io import resolve_device
    assert resolve_device("Realtek", 16_000, want_input=True) == 4
    assert resolve_device("Realtek", 24_000, channels=1, want_input=False) == 7


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
