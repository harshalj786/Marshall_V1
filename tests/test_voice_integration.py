"""TTS -> STT round trip against the real models. No microphone, no speakers.

Synthesize a known sentence with Kokoro, resample 24 kHz -> 16 kHz, transcribe it with
faster-whisper, and assert the words survived. This is the one test that exercises both
audio models end to end, and it does it without hardware — so it runs in CI-shaped
environments where `voice_main.py --check` (which plays the audio out loud) cannot.

Everything skips cleanly when the models are unavailable: no GPU, no cuDNN, no network
for the first-run download, or the packages simply not installed. A skip here means
"not verifiable on this machine", never "broken".

    pytest tests/test_voice_integration.py -v
    python tests/test_voice_integration.py

First run downloads ~1.5 GB of model weights and takes minutes. Afterwards it is
seconds, and the models are shared with `--check` and the live session.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.config import VoiceConfig
from voice.audio_io import resample_linear

#: short, phonetically ordinary, and containing nothing Whisper is likely to
#: "correct" into something else. A domain word ("OPVC") would be testing the
#: language model rather than the round trip.
PHRASE = "The quick brown fox jumps over the lazy dog."
#: the bar from the V1a spec. Whisper reliably returns this phrase verbatim, so 80%
#: is slack for casing, punctuation, and a single dropped article — not a target.
MIN_OVERLAP = 0.8


def _tokens(text: str) -> set:
    import re
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def token_overlap(expected: str, actual: str) -> float:
    wanted = _tokens(expected)
    return len(wanted & _tokens(actual)) / len(wanted) if wanted else 0.0


@pytest.fixture(scope="module")
def config():
    return VoiceConfig()


@pytest.fixture(scope="module")
def tts(config):
    """Kokoro, or a skip. Also covers espeak-ng being unavailable."""
    try:
        from voice.tts import build_engine
        return build_engine(config)
    except Exception as exc:
        pytest.skip(f"TTS unavailable ({type(exc).__name__}: {exc})")


@pytest.fixture(scope="module")
def stt(config):
    """faster-whisper, or a skip. `Transcriber` falls back to CPU on its own, so
    reaching the skip means neither device worked — missing package or no weights."""
    try:
        from voice.stt import Transcriber
        return Transcriber(config)
    except Exception as exc:
        pytest.skip(f"STT unavailable ({type(exc).__name__}: {exc})")


# ==========================================================================
# the round trip
# ==========================================================================

def test_tts_stt_round_trip(tts, stt, config):
    """The headline: a sentence spoken by Marshall is understood by Marshall."""
    audio = tts.synthesize(PHRASE)
    assert len(audio) > 0, "TTS produced no audio"

    transcript = stt.transcribe(
        resample_linear(audio, config.tts_sample_rate, config.sample_rate))

    overlap = token_overlap(PHRASE, transcript)
    assert overlap >= MIN_OVERLAP, (
        f"round trip lost too much: {overlap:.0%} token overlap\n"
        f"  spoke: {PHRASE!r}\n  heard: {transcript!r}")


def test_round_trip_survives_sentence_splitting(tts, stt, config):
    """The live path never synthesizes a whole answer at once — it goes sentence by
    sentence, because Kokoro has a per-call input limit. Assert the split path
    round-trips too, since that is the one the user actually hears."""
    text = "Marshall is a local agent. It answers questions about a document."
    pieces = [audio for _i, _s, audio in tts.stream_sentences(text)]
    assert len(pieces) == 2, "the answer should have been split into two sentences"

    joined = np.concatenate(pieces)
    transcript = stt.transcribe(
        resample_linear(joined, config.tts_sample_rate, config.sample_rate))

    overlap = token_overlap(text, transcript)
    assert overlap >= MIN_OVERLAP, (
        f"{overlap:.0%} overlap\n  spoke: {text!r}\n  heard: {transcript!r}")


# ==========================================================================
# properties of the two models the pipeline depends on
# ==========================================================================

def test_tts_output_is_the_configured_rate_and_dtype(tts, config):
    """The speaker opens its stream at `tts_sample_rate`; a mismatch plays at the
    wrong pitch rather than failing, which is the kind of bug that survives a demo."""
    audio = tts.synthesize("Testing.")
    assert audio.dtype == np.float32
    assert np.abs(audio).max() <= 1.0, "audio outside [-1, 1] will clip on playback"
    #: ~0.5-2s of speech for one short word at 24 kHz — a rate mismatch shows up here
    #: as an order-of-magnitude error rather than as a subtle one
    assert 0.2 < len(audio) / config.tts_sample_rate < 5.0


def test_engine_reports_its_name_for_telemetry(tts):
    """`tts_engine` in the turn log is only meaningful if the engine says which it is."""
    assert tts.engine_name and isinstance(tts.engine_name, str)


def test_stt_reports_which_device_it_actually_loaded(stt):
    """A silent CPU fallback must not be mistakable for a slow GPU."""
    assert stt.device in {"cuda", "cpu"}
    assert stt.compute_type
    if stt.fell_back:
        assert stt.device == "cpu"


def test_stt_discards_audio_below_the_minimum_utterance_length(stt, config):
    """Whisper hallucinates confidently on sub-second noise — "Thank you." is its
    favourite — and a hallucinated transcript costs a full agent turn plus a spoken
    answer to nothing."""
    too_short = np.zeros(int(config.sample_rate * config.min_utterance_s * 0.5),
                         dtype=np.float32)
    assert stt.transcribe(too_short) == ""


def test_stt_discards_empty_audio(stt):
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
