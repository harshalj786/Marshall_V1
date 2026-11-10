"""The spoken yes/no for `run_code`, matched in code, and the late-bound confirmer.

    pytest tests/test_voice_confirm.py -v

`parse_yes_no` is strict on purpose: anything but a clear yes is a no. A wrong "no" costs
the user one repeat; a wrong "yes" runs code they did not agree to.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.confirm import CONFIRM_SPOKEN, VoiceConfirmer, parse_yes_no


@pytest.mark.parametrize("heard,expected", [
    ("Yes.", True), ("yes please", True), ("yeah go ahead", True), ("okay run it", True),
    ("sure", True), ("Okay.", True),
    ("no", False), ("yes, no wait", False), ("", False), ("not yet", False),
    ("I guess yes", False), ("don't", False), ("yes don't", False), ("   ", False),
])
def test_parse_yes_no_accepts_only_a_clear_yes(heard, expected):
    assert parse_yes_no(heard) is expected


def test_the_question_says_what_is_about_to_happen():
    assert "run" in CONFIRM_SPOKEN.lower() and CONFIRM_SPOKEN.endswith("?")


def test_unbound_confirmer_is_a_no():
    assert VoiceConfirmer()(object()) is False


def test_bound_confirmer_delegates():
    class Orchestrator:
        def confirm_by_voice(self, request):
            self.seen = request
            return True
    confirmer = VoiceConfirmer()
    confirmer.orchestrator = Orchestrator()
    request = object()
    assert confirmer(request) is True
    assert confirmer.orchestrator.seen is request
