"""Conditional edge routing functions for the CDA market graph.

Under the improvement-rule CDA, routing is simple:

- After apply_order: always record history (handled as a normal edge).
- After update_history: check round status.
- After check_round: either advance tick or advance round.
- After next_round: either start the new round or end the simulation.

The old responder-polling routing is gone — crossing is automatic inside
apply_order, so there is no "try another responder" branch.
"""

import logging
from typing import Literal

from .state import MarketState

logger = logging.getLogger(__name__)


def route_after_update_history(state: MarketState) -> Literal["check_round"]:
    """Route after history update.

    Always advance to the round-check. The old CDA mechanism could loop
    back to try another responder / another announcer within an
    iteration; under the tick-based CDA, each tick is one attempt and
    ends here.
    """
    return "check_round"


def route_after_check_round(state: MarketState) -> Literal["next_round", "next_iteration"]:
    """Route after round check: either end the round or advance the tick."""
    if state["round_complete"]:
        logger.debug("route_after_check_round -> next_round")
        return "next_round"
    logger.debug("route_after_check_round -> next_iteration")
    return "next_iteration"


def route_after_next_round(state: MarketState) -> Literal["select_announcer", "__end__"]:
    """Route after advancing to next round. End if simulation is complete."""
    if state["simulation_complete"]:
        logger.debug("route_after_next_round -> __end__")
        return "__end__"
    logger.debug("route_after_next_round -> select_announcer")
    return "select_announcer"
