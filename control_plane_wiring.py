"""Glue between this repo and the standalone `control_plane` package.

This file is the ONLY place the two know about each other. `control_plane` imports
nothing from here and names nothing in here; this module knows both sides, which is
what a wiring layer is for. `pipeline_lg.py` is not modified — the hook goes in
through the `(node, router)` tuple parameter and nothing else.

    from control_plane_wiring import build_marshall_hook, invoke_config
    agent = AgentGraph(..., agent_hook=build_marshall_hook())
    agent.invoke(query, config=invoke_config("my-thread"))

Point `CONTROL_PLANE_PATH` at the package checkout if it is not the sibling
directory next to this repo.
"""
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

#: the package is a separate repository, used from a checkout rather than installed
#: as a wheel, so it has to be findable before it can be imported
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "control_plane"
_PACKAGE_PATH = Path(os.environ.get("CONTROL_PLANE_PATH", _DEFAULT_PATH))
if str(_PACKAGE_PATH) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PATH))

from control_plane import (  # noqa: E402
    DEFAULT_REMEDIES,
    VERIFY_MARKER,
    PolicyConfig,
    build_hook,
    default_payload_from_state,
)

from pipeline_lg import PROFILE_BLOCK_TEMPLATE  # noqa: E402


#: Per-detector remedies, overriding only the entry that needs a tool NAMED. The
#: package deliberately will not name one — a tool name is host-specific and belongs
#: on this side of the boundary — and its generic "if a tool can retrieve that source"
#: is not something an 8B acts on reliably.
#:
#: Only `factuality` is overridden. That detector DOES NOT EXIST YET, so this entry is
#: dormant: it is here so the shape of a retrieval-style remedy stays visible and has
#: an obvious place to plug in. The jailbreak and pattern_leak remedies are taken from
#: the package unchanged, because neither should go anywhere near retrieval — sending
#: an injection attempt to `search_current_doc` was the mistake this replaced.
REMEDIES = {
    **DEFAULT_REMEDIES,
    "factuality": (
        f"{VERIFY_MARKER} Before you answer, check the claim you just made against the "
        "loaded document. Call search_current_doc with a short query about that specific "
        "claim and quote what it returns. If the document does not support the claim, or "
        "the search finds nothing, say so plainly instead of repeating the claim."
    ),
}


#: Supersteps for one turn with the hook installed, worst case:
#:
#:     1                 count_turn
#:   + 2                 both memory-gate nodes, on a turn where both are due
#:   + 2 x (N + R + 1)   agent AND hook, once per generation
#:   + N                 the tool node, once per tool round
#:   = 3N + 2R + 5
#:
#: N is tool rounds; R is revises, which `PolicyConfig.max_revises` caps at 2.
#: Checked against a measured turn: N=3, R=0 and no gate nodes gives 3(3)+3 = 12,
#: and 12 is what the graph actually executed.
#:
#: The limit must EXCEED the step count, not merely equal it — measured directly:
#: that same 12-superstep turn raises GraphRecursionError at recursion_limit=12 and
#: passes at 13. So the requirement is 3N + 2R + 5 + 1.
#:
#: Restoring the pre-hook ceiling of N=11 with the revise budget spent:
#:     3(11) + 2(2) + 5 + 1 = 43
#:
#: On the simplified basis the original "11" came from (2N+2 <= 25, no gate nodes)
#: the answer would be 41. 43 is used because it also covers a turn where history
#: compression and the memory gate both fire — which the old default of 25 never
#: did: unhooked with both gates is 2N+4 supersteps, so N=11 needed 27 and only ever
#: had 25. That ceiling was 11 on paper and 10 on a gate turn.
RECURSION_LIMIT = 43


def payload_from_state(state: dict, thread_id: str):
    """Rebuild the system message as the model actually received it.

    THIS IS THE WHOLE REASON A CUSTOM EXTRACTOR EXISTS. `AgentGraph._agent_node`
    composes the profile block into the system message per call and deliberately
    never writes the result back to state — a written-back copy would compound on
    every gate hit. The consequence is that `state["messages"][0]` is the BARE
    prompt: the composed version the model saw exists only inside that one call and
    is gone by the time any hook runs.

    A detector reading the system slot from unmodified state therefore sees no
    profile block, finds nothing, and reports a clean result — the worst outcome,
    because "clean" and "blind" are indistinguishable in the record. So this
    reassembles the composed text exactly as `_with_profile` does, from the same
    template and the same state field.

    If that composition ever changes, this goes stale silently. The guard is
    `ProfileInjection`, which records `block_found: false` rather than scoring 0
    quietly — though note that a thread whose memory gate has never fired has no
    profile at all, and reports the same thing for an entirely legitimate reason.
    """
    payload = default_payload_from_state(state, thread_id)
    summary = state.get("user_history_summary") or ""
    if not summary:
        return payload
    composed = payload.system_text + PROFILE_BLOCK_TEMPLATE.format(user_history=summary)
    return replace(payload, system_text=composed)


def invoke_config(thread_id: str, **extra) -> dict:
    """Config for a hooked `AgentGraph.invoke`, carrying the raised recursion limit.

    `AgentGraph.invoke` only defaults `recursion_limit` to 25 when a caller supplies
    none, so this raises it without touching that file. Every hooked call site should
    use this; an unhooked one does not need it, since the ceiling only drops when the
    hook adds its superstep per generation.

    The alternative is a one-line change to the default in `pipeline_lg.py`. That
    would be more robust — it cannot be forgotten at a call site — and is the right
    move if the hook ever becomes the normal configuration rather than an opt-in.
    """
    configurable = {"thread_id": thread_id, **extra.pop("configurable", {})}
    return {"configurable": configurable,
            "recursion_limit": extra.pop("recursion_limit", RECURSION_LIMIT),
            **extra}


def build_marshall_hook(shadow: bool = False,
                        config: Optional[PolicyConfig] = None,
                        on_record: Optional[Callable[[dict], None]] = None,
                        **kwargs):
    """Return the `(node, router)` tuple to pass as `AgentGraph(agent_hook=...)`.

    `shadow` defaults to FALSE here, and True in the package. That split is
    deliberate: the package ships safe for anyone who installs it, while this file
    records a decision already taken for this host — the shadow pass ran, the
    thresholds were fitted to its 34 records, and the verdicts are live. Pass
    `shadow=True` to go back to observing without changing either codebase.
    """
    from control_plane import PolicyEngine

    return build_hook(
        shadow=shadow,
        engine=PolicyEngine(config) if config is not None else None,
        on_record=on_record,
        payload_from_state=payload_from_state,
        remedies=REMEDIES,
        **kwargs,
    )
