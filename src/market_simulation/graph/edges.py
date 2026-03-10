"""Conditional edge routing functions for the market graph."""

import logging
from typing import Literal

from .state import MarketState

logger = logging.getLogger(__name__)


def route_after_announcement(state: MarketState) -> Literal["select_responders", "check_iteration"]:
    """Route after announcement node.

    If announcement was made, proceed to collect responses.
    Otherwise, check iteration and continue.
    """
    if state["announcement_made"] and state["announced_price"] is not None:
        logger.debug(
            f"route_after_announcement -> select_responders "
            f"(price={state['announced_price']})"
        )
        return "select_responders"
    logger.debug(
        f"route_after_announcement -> check_iteration "
        f"(announcement_made={state['announcement_made']}, price={state['announced_price']})"
    )
    return "check_iteration"


def route_after_response(state: MarketState) -> Literal["record_transaction", "check_iteration"]:
    """Route after response node.

    If transaction made, record it.
    Otherwise, check iteration.
    """
    if state["transaction_made"]:
        logger.debug("route_after_response -> record_transaction")
        return "record_transaction"

    logger.debug("route_after_response -> check_iteration")
    return "check_iteration"


def route_after_update_history(state: MarketState) -> Literal["check_round", "respond", "select_announcer"]:
    """Route after history update.

    If transaction was made, check if round should continue (advances iteration).
    If no transaction, try another responder or announcer within the same iteration.
    If no announcement could be made (all tried), check round.
    """
    # Transaction made - iteration complete, check round status
    if state["transaction_made"]:
        logger.debug("route_after_update_history -> check_round (transaction made)")
        return "check_round"

    # No announcement made - check if other agents can still announce
    # (announcement_made=False can mean either no eligible agents or a parse failure;
    # in the latter case, other agents should still get a chance)
    if not state["announcement_made"]:
        eligible = [
            aid for aid in state["active_agent_ids"]
            if aid not in state.get("announced_this_iteration", [])
        ]
        if eligible:
            logger.debug(
                f"route_after_update_history -> select_announcer "
                f"(announcement failed but {len(eligible)} eligible agents remain)"
            )
            return "select_announcer"
        logger.debug("route_after_update_history -> check_round (no eligible announcers)")
        return "check_round"

    # Check if more responders to query
    responder_idx = state["current_responder_index"]
    total_responders = len(state["potential_responder_ids"])

    # Try another responder if available
    # (this keeps us in the same iteration)
    if responder_idx < total_responders:
        logger.debug(
            f"route_after_update_history -> respond "
            f"(responder {responder_idx}/{total_responders})"
        )
        return "respond"

    # Announcement was made but rejected - try another announcer
    # (this keeps us in the same iteration)
    logger.debug("route_after_update_history -> select_announcer (all responders exhausted)")
    return "select_announcer"


def route_after_check_round(state: MarketState) -> Literal["next_round", "next_iteration"]:
    """Route after round check.

    If round complete, go to next round.
    Otherwise, continue with next iteration.
    """
    if state["round_complete"]:
        logger.debug("route_after_check_round -> next_round")
        return "next_round"
    logger.debug("route_after_check_round -> next_iteration")
    return "next_iteration"


def route_after_next_round(state: MarketState) -> Literal["select_announcer", "__end__"]:
    """Route after advancing to next round.

    If simulation complete, end.
    Otherwise, start new round.
    """
    if state["simulation_complete"]:
        logger.debug("route_after_next_round -> __end__")
        return "__end__"
    logger.debug("route_after_next_round -> select_announcer")
    return "select_announcer"
