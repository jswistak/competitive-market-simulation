"""Conditional edge routing for sealed-bid auction graphs."""

import logging
from typing import Literal

from ...state import SealedBidState

logger = logging.getLogger(__name__)


def route_after_collect_bid(
    state: SealedBidState,
) -> Literal["collect_bid", "determine_winner"]:
    """Route after collecting a bid.

    If all bids collected, determine winner. Otherwise collect next bid.
    """
    if state["all_bids_collected"]:
        logger.debug("route_after_collect_bid -> determine_winner")
        return "determine_winner"
    logger.debug(
        f"route_after_collect_bid -> collect_bid "
        f"(bidder {state['current_bidder_index']}/{len(state['bidders'])})"
    )
    return "collect_bid"


def route_after_update_history(
    state: SealedBidState,
) -> Literal["next_round", "__end__"]:
    """Route after updating history.

    If more rounds to play, advance. Otherwise end.
    """
    if state["round"] >= state["max_rounds"]:
        logger.debug("route_after_update_history -> __end__")
        return "__end__"
    logger.debug("route_after_update_history -> next_round")
    return "next_round"
