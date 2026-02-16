"""Dutch (descending) auction graph builder.

Graph topology:
  START -> announce_price -> solicit_acceptance --[next bidder]--> solicit_acceptance
                                               |--[accepted / all queried]--> check_dutch_end
                                                                                |--[accepted]--> settle -> update_history -> ...
                                                                                |--[lower]--> lower_price -> solicit_acceptance
                                                                                |--[floor]--> settle -> update_history -> ...
"""

from typing import Callable

from langgraph.graph import StateGraph, START, END

from ...state import DutchAuctionState
from .nodes import (
    make_announce_price_node,
    make_solicit_acceptance_node,
    make_check_dutch_end_node,
    make_lower_price_node,
    make_settle_dutch_node,
    make_update_dutch_history_node,
    make_next_dutch_round_node,
)
from .edges import (
    route_after_solicit_acceptance,
    route_after_check_dutch_end,
    route_after_update_history,
)
from ....llm.providers.base import LLMProvider
from ....config.schema import AuctionPromptConfig


def build_dutch_graph(
    auction_type: str,
    llm: LLMProvider,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
) -> StateGraph:
    """Build a Dutch (descending) auction LangGraph workflow.

    Args:
        auction_type: "dutch".
        llm: LLM provider for bidder interactions.
        prompts: Auction prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks.

    Returns:
        Compiled StateGraph ready for execution.
    """
    builder = StateGraph(DutchAuctionState)

    # Nodes
    builder.add_node("announce_price", make_announce_price_node())
    builder.add_node("solicit_acceptance", make_solicit_acceptance_node(llm, prompts, callbacks_factory))
    builder.add_node("check_dutch_end", make_check_dutch_end_node())
    builder.add_node("lower_price", make_lower_price_node())
    builder.add_node("settle", make_settle_dutch_node())
    builder.add_node("update_history", make_update_dutch_history_node())
    builder.add_node("next_round", make_next_dutch_round_node())

    # Edges
    builder.add_edge(START, "announce_price")
    builder.add_edge("announce_price", "solicit_acceptance")
    builder.add_conditional_edges("solicit_acceptance", route_after_solicit_acceptance)
    builder.add_conditional_edges("check_dutch_end", route_after_check_dutch_end)
    builder.add_edge("lower_price", "solicit_acceptance")
    builder.add_edge("settle", "update_history")
    builder.add_conditional_edges("update_history", route_after_update_history)
    builder.add_edge("next_round", "announce_price")

    return builder.compile()
