"""Where can a message list be cut without breaking it?

An `AIMessage` carrying `tool_calls` and the `ToolMessage`s answering it are one
indivisible unit. Ollama rejects a history where the two are separated — a `ToolMessage`
whose call never appears, or an assistant turn whose calls are never answered
(`README.md:267-269`). So any code that shortens a conversation cannot pick an arbitrary
index; it has to pick one that leaves every pair whole.

Two callers need exactly that, in opposite directions:

  * history trimming keeps the SUFFIX (`messages[cut:]`) and summarises the prefix;
  * voice barge-in keeps the PREFIX (`messages[:cut]`) and drops the interrupted tail.

`find_safe_cut_point` therefore validates the *split*, not one side of it: a cut is safe
when nothing emitted before it is answered after it, and nothing before it is left
unanswered. That single condition serves both callers, which is why this lives in its own
module — pure function, no state, no I/O, no graph. Give it a plain list of messages.
"""
from typing import Optional, Sequence

from langchain_core.messages import BaseMessage, ToolMessage


def find_safe_cut_point(messages: Sequence[BaseMessage], target_index: int) -> int:
    """Return the largest index <= `target_index` where `messages` can be split safely.

    The returned index `i` splits the list into `messages[:i]` and `messages[i:]` with
    every tool-call/tool-result pair intact on one side. Walks BACKWARD from the target,
    never forward: both callers are shedding messages, so overshooting the target keeps
    more than asked (harmless) while undershooting would keep less than asked (a silent
    data loss for trimming, and a dropped user turn for barge-in).

    Returns 0 for an empty list, and 0 when no boundary above it is safe — i.e. "the whole
    list is one indivisible run". Callers must treat 0 as "nothing can be cut here".

    `target_index` is clamped into `[0, len(messages)]`, so `len(messages)` is a legal
    request meaning "cut nothing off the end unless the tail is dangling".
    """
    if not messages:
        return 0

    cut = max(0, min(int(target_index), len(messages)))

    # `pending` holds the ids of tool calls emitted so far and not yet answered. Boundary i
    # (the split before messages[i]) is safe exactly when `pending` is empty at that point,
    # so one forward pass finds every safe boundary and we keep the last one at or below
    # the target. Boundary 0 is safe by definition, which is why it is the floor.
    pending: list[Optional[str]] = []
    safe = 0
    for i in range(cut):
        if not pending:
            safe = i
        _consume(messages[i], pending)
    if not pending:
        safe = cut
    return safe


def _consume(message: BaseMessage, pending: list) -> None:
    """Fold one message into the set of outstanding tool calls."""
    if isinstance(message, ToolMessage):
        call_id = getattr(message, "tool_call_id", None) or None
        if call_id in pending:
            pending.remove(call_id)
        elif None in pending:
            # an unidentifiable call answered by an unmatched result: pair them oldest-first
            pending.remove(None)
        # otherwise this result was already orphaned in the input — its call is not in this
        # list at all, so no cut can keep the pair whole and none should be blocked for it
        return

    for call in getattr(message, "tool_calls", None) or []:
        # ids can be missing on some providers; None is still tracked, so an unanswered
        # call keeps the boundary closed rather than being silently ignored
        pending.append(call.get("id") or None)
