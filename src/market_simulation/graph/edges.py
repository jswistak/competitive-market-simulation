"""Conditional edge routing functions for the market graph."""

from typing import Literal

from .state import MarketState


def route_after_announcement(state: MarketState) -> Literal["select_responders", "update_history"]:
    """Route after announcement node.

    If announcement was made, proceed to collect responses.
    Otherwise, update history and continue.
    """
    if state["announcement_made"] and state["announced_price"] is not None:
        return "select_responders"
    return "update_history"


def route_after_response(state: MarketState) -> Literal["record_transaction", "respond", "update_history"]:
    """Route after response node.

    If transaction made, record it.
    If more responders available, try next one.
    Otherwise, update history.
    """
    if state["transaction_made"]:
        return "record_transaction"

    # Check if more responders to query
    responder_idx = state["current_responder_index"]
    total_responders = len(state["potential_responder_ids"])

    if responder_idx < total_responders:
        return "respond"

    return "update_history"


def route_after_update_history(state: MarketState) -> Literal["check_round", "select_announcer"]:
    """Route after history update.

    If iteration complete (transaction made or no announcement), check round.
    Otherwise, try another announcer.
    """
    if state["transaction_made"] or not state["announcement_made"]:
        return "check_round"
    return "select_announcer"


def route_after_check_round(state: MarketState) -> Literal["next_round", "next_iteration"]:
    """Route after round check.

    If round complete, go to next round.
    Otherwise, continue with next iteration.
    """
    if state["round_complete"]:
        return "next_round"
    return "next_iteration"


def route_after_next_round(state: MarketState) -> Literal["select_announcer", "__end__"]:
    """Route after advancing to next round.

    If simulation complete, end.
    Otherwise, start new round.
    """
    if state["simulation_complete"]:
        return "__end__"
    return "select_announcer"
