"""The spoken yes/no for `run_code`, matched in code, and the late-bound confirmer.

    pytest tests/test_voice_confirm.py -v

`parse_yes_no` is strict on purpose: anything but a clear yes is a no. A wrong "no" costs
the user one repeat; a wrong "yes" runs code they did not agree to.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace

from voice.confirm import VoiceConfirmer, confirm_question, parse_yes_no


@pytest.mark.parametrize("heard,expected", [
    ("Yes.", True), ("yes please", True), ("yeah go ahead", True), ("okay run it", True),
    ("sure", True), ("Okay.", True),
    ("no", False), ("yes, no wait", False), ("", False), ("not yet", False),
    ("I guess yes", False), ("don't", False), ("yes don't", False), ("   ", False),
])
def test_parse_yes_no_accepts_only_a_clear_yes(heard, expected):
    assert parse_yes_no(heard) is expected


def test_the_question_names_a_file_being_run():
    q = confirm_question(SimpleNamespace(language="python", source="x", origin="C:/work/hello.py"))
    assert q == "I'm about to run hello.py. Is that okay?"


@pytest.mark.parametrize("language,spoken", [("python", "Python"), ("c", "C"), ("cpp", "C plus plus")])
def test_the_question_names_the_language_of_pasted_code(language, spoken):
    q = confirm_question(SimpleNamespace(language=language, source="x", origin="pasted"))
    assert q == f"I'm about to run the {spoken} code you gave me. Is that okay?"


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
