"""Conditional edge routing for English / Open-Outcry auction graphs."""

import logging
from typing import Literal

from ...state import EnglishAuctionState

logger = logging.getLogger(__name__)


def route_after_solicit_bid(
    state: EnglishAuctionState,
) -> Literal["solicit_bid", "check_auction_end"]:
    """Route after soliciting a bid.

    If more active bidders in this cycle, solicit next.
    Otherwise check if auction should end.
    """
    idx = state["current_bidder_index"]
    n_active = len(state["active_bidder_ids"])

    if idx < n_active:
        logger.debug(
            f"route_after_solicit_bid -> solicit_bid "
            f"({idx}/{n_active})"
        )
        return "solicit_bid"

    logger.debug("route_after_solicit_bid -> check_auction_end")
    return "check_auction_end"


def route_after_check_end(
    state: EnglishAuctionState,
) -> Literal["reset_cycle", "settle"]:
    """Route after checking auction end condition.

    If auction ended, settle. Otherwise start a new bidding cycle.
    """
    if state["auction_ended"]:
        logger.debug("route_after_check_end -> settle")
        return "settle"
    logger.debug("route_after_check_end -> reset_cycle")
    return "reset_cycle"


def route_after_update_history(
    state: EnglishAuctionState,
) -> Literal["next_round", "__end__"]:
    """Route after updating history. More rounds or end."""
    if state["round"] >= state["max_rounds"]:
        logger.debug("route_after_update_history -> __end__")
        return "__end__"
    logger.debug("route_after_update_history -> next_round")
    return "next_round"
