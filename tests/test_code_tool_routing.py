"""The code tools must not make llama3.1 worse at ordinary conversation.

    pytest tests/test_code_tool_routing.py -v
    python scripts/check_code_tool_routing.py     # same measurement, readable output

Same contract as tests/test_routing_matrix.py: needs Ollama with llama3.1 pulled, spends a
few real minutes, skips otherwise — skipped means "not verifiable on this machine", never
"passing". Every store is this run's own tmp directory.

The bar is the spec's: over 12 conversational prompts x 3 runs, the code tools fire on
NONE, and total spurious tool calls are no worse than today's session-less agent on the
same set. On a fail the spec says stop and report, not pick a fallback.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import voice_main
from scripts.check_code_tool_routing import CODE_PROMPTS, build_agents, measure, passed


#: same probe as tests/test_routing_matrix.py, restated rather than imported because that
#: is this repo's pattern for it
def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _has_llama31() -> bool:
    from scripts.check_routing import model_available
    return model_available(voice_main.VOICE_MODEL)


needs_llama31 = pytest.mark.skipif(
    not (_ollama_up() and _has_llama31()),
    reason=f"needs Ollama on :11434 with {voice_main.VOICE_MODEL} pulled")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp("code_routing")
    return measure(build_agents(str(root / "vdb"), str(root / "mem"), str(root / "sum")))


@needs_llama31
@pytest.mark.parametrize("config", ["guest", "admin"])
def test_code_tools_never_fire_on_conversation(report, config):
    assert report.code_misfires[config] == 0


@needs_llama31
@pytest.mark.parametrize("config", ["guest", "admin"])
def test_spurious_calls_no_worse_than_baseline(report, config):
    assert report.spurious[config] <= report.spurious["baseline"], report.spurious


@needs_llama31
@pytest.mark.parametrize("name,prompt,expected,admin_only", CODE_PROMPTS,
                         ids=[p[0] for p in CODE_PROMPTS])
def test_code_prompts_reach_the_right_tool(report, name, prompt, expected, admin_only):
    for config in (("admin",) if admin_only else ("guest", "admin")):
        assert expected in report.code_routes[(config, name)], report.code_routes


@needs_llama31
def test_the_run_route_was_declined_outside_the_model(report):
    assert report.confirm_calls == 1


@needs_llama31
def test_admin_keeps_the_r4_routes(report):
    assert all(v == "PASS" for v in report.admin_r4.values()), report.admin_r4


@needs_llama31
def test_the_whole_report_passes(report):
    assert passed(report)
