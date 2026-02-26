"""Conditional edge routing for Dutch auction graphs."""

import logging
from typing import Literal

from ...state import DutchAuctionState

logger = logging.getLogger(__name__)


def route_after_solicit_acceptance(
    state: DutchAuctionState,
) -> Literal["solicit_acceptance", "check_dutch_end"]:
    """Route after soliciting acceptance from a bidder.

    If someone accepted or all bidders at this price were queried,
    check the end condition.  Otherwise ask the next bidder.
    """
    if state["accepted"]:
        logger.debug("route_after_solicit_acceptance -> check_dutch_end (accepted)")
        return "check_dutch_end"

    if state["all_queried_at_price"]:
        logger.debug("route_after_solicit_acceptance -> check_dutch_end (all queried)")
        return "check_dutch_end"

    logger.debug(
        f"route_after_solicit_acceptance -> solicit_acceptance "
        f"(bidder {state['current_bidder_index']}/{len(state['bidders'])})"
    )
    return "solicit_acceptance"


def route_after_check_dutch_end(
    state: DutchAuctionState,
) -> Literal["settle", "lower_price"]:
    """Route after checking Dutch end condition.

    - accepted -> settle
    - price at/below floor -> settle (no winner)
    - otherwise -> lower price and ask again
    """
    if state["accepted"]:
        logger.debug("route_after_check_dutch_end -> settle (accepted)")
        return "settle"

    if state["current_price"] - state["dutch_decrement"] < state["dutch_min_price"]:
        logger.debug("route_after_check_dutch_end -> settle (floor reached)")
        return "settle"

    logger.debug("route_after_check_dutch_end -> lower_price")
    return "lower_price"


def route_after_update_history(
    state: DutchAuctionState,
) -> Literal["next_round", "__end__"]:
    """Route after updating history.  More rounds or end."""
    if state["round"] >= state["max_rounds"]:
        logger.debug("route_after_update_history -> __end__")
        return "__end__"
    logger.debug("route_after_update_history -> next_round")
    return "next_round"
