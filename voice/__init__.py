"""Marshall's voice layer (V1a — the pipe cleaner).

Open mic -> VAD endpoint -> STT -> the existing synchronous `AgentGraph.invoke` ->
TTS -> playback -> back to listening. No wake word, no barge-in, no streaming; those
are V1b-V1d. `docs/architecture_review.md` §10 is the map this follows.

The dependency arrow points one way: `voice/` imports `pipeline_lg`, and nothing in
the repository imports `voice/`. The agent does not know it is being spoken to, and
V1a required no change to the graph.

    from voice.config import VoiceConfig
    from voice.orchestrator import VoiceOrchestrator
"""
