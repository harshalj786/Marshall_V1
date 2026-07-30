"""One JSON line per turn, appended to `voice_logs/turns.jsonl`.

`docs/architecture_review.md` §10 closes its list of voice blockers with: *"Nothing
measures latency. There is no instrumentation for time-to-first-token, per-node
duration, or tool latency, so there is no baseline to regress against."* V1a is the
pipe cleaner, so the one thing it must leave behind is the baseline — no timing
means no tuning later, and V1b-V1d are all latency work.

The headline number is `speech_end_to_first_audio_ms`: the wall-clock gap between
the user stopping speaking and hearing anything back. It is the only one the user
experiences directly; `stt_ms`, `agent_ms` and `tts_first_sentence_ms` are there to
say which stage owns it.

JSONL rather than a log line so a later session can just read it:

    import json
    rows = [json.loads(l) for l in open("voice_logs/turns.jsonl", encoding="utf-8")]
"""
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

#: field order in the written line. Fixed so a human tailing the file sees the same
#: shape every turn, and so a reordering shows up as a diff here rather than silently.
TURN_FIELDS = (
    "timestamp",
    "turn_id",
    "utterance_s",
    "stt_ms",
    "agent_ms",
    "tts_first_sentence_ms",
    "speech_end_to_first_audio_ms",
    "transcript_chars",
    "response_chars",
    "tts_engine",
)


@dataclass
class TurnTelemetry:
    """Timings for one utterance, filled in as the turn progresses.

    Every duration is Optional and starts None: a turn that dies in STT should still
    write a row saying so, and `0` would be a lie that averages into the baseline.
    """
    turn_id: int
    utterance_s: float
    tts_engine: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stt_ms: Optional[float] = None
    agent_ms: Optional[float] = None
    tts_first_sentence_ms: Optional[float] = None
    speech_end_to_first_audio_ms: Optional[float] = None
    transcript_chars: Optional[int] = None
    response_chars: Optional[int] = None

    def as_row(self) -> dict:
        data = asdict(self)
        #: rounded on the way out, not on the way in: sub-microsecond precision in a
        #: latency log is noise, but the arithmetic upstream should stay exact
        return {
            key: (round(data[key], 2) if isinstance(data[key], float) else data[key])
            for key in TURN_FIELDS
        }


class TelemetryWriter:
    """Appends turn rows. Never raises — a telemetry failure must not cost a turn.

    Same contract as the memory gate (H4) and the summarizer (H2): the instrumentation
    is allowed to lose data, and is never allowed to lose the user's turn.
    """

    def __init__(self, path: str = "voice_logs/turns.jsonl"):
        self.path = Path(path)
        self.last_error: Optional[str] = None

    def write(self, telemetry: TurnTelemetry) -> bool:
        """True when the row hit the disk. False (and `last_error` set) otherwise."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            #: append mode and one flush per turn — a Ctrl+C mid-session must not cost
            #: the rows already measured, and a turn is far too coarse to buffer
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(telemetry.as_row(), ensure_ascii=False) + "\n")
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False


class Stopwatch:
    """Millisecond timer over a `with` block. `perf_counter`, so no wall-clock jumps."""

    def __init__(self):
        self.ms: Optional[float] = None
        self._started: Optional[float] = None

    def __enter__(self) -> "Stopwatch":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> bool:
        #: recorded even when the block raised, so a failed stage still reports how
        #: long it took to fail — which is the number you want when it is a timeout
        self.ms = (time.perf_counter() - self._started) * 1000
        return False
