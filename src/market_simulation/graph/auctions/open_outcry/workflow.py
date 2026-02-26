"""First-Price Open-Outcry auction graph builder.

Reuses the English auction graph topology.  The only difference is
the settlement node: open-outcry uses first-price settlement
(make_settle_open_outcry_node) instead of the English settlement.

Graph topology is identical to english/workflow.py.
"""

from typing import Callable

from langgraph.graph import StateGraph, START, END

from ...state import EnglishAuctionState
from ..english.nodes import (
    make_solicit_bid_node,
    make_check_auction_end_node,
    make_reset_cycle_node,
    make_settle_open_outcry_node,
    make_update_english_history_node,
    make_next_english_round_node,
)
from ..english.edges import (
    route_after_solicit_bid,
    route_after_check_end,
    route_after_update_history,
)
from ....llm.providers.base import LLMProvider
from ....config.schema import AuctionPromptConfig


def build_open_outcry_graph(
    auction_type: str,
    llm: LLMProvider,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
) -> StateGraph:
    """Build a first-price open-outcry auction LangGraph workflow.

    Identical to English auction except settlement is explicitly
    first-price (winner pays their own highest bid).

    Args:
        auction_type: "first_price_open_outcry".
        llm: LLM provider for bidder interactions.
        prompts: Auction prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks.

    Returns:
        Compiled StateGraph ready for execution.
    """
    builder = StateGraph(EnglishAuctionState)

    # Nodes — reuse all English nodes except settlement
    builder.add_node("solicit_bid", make_solicit_bid_node(llm, prompts, callbacks_factory))
    builder.add_node("check_auction_end", make_check_auction_end_node())
    builder.add_node("reset_cycle", make_reset_cycle_node())
    builder.add_node("settle", make_settle_open_outcry_node())  # <-- only difference
    builder.add_node("update_history", make_update_english_history_node())
    builder.add_node("next_round", make_next_english_round_node())

    # Edges — identical to English
    builder.add_edge(START, "solicit_bid")
    builder.add_conditional_edges("solicit_bid", route_after_solicit_bid)
    builder.add_conditional_edges("check_auction_end", route_after_check_end)
    builder.add_edge("reset_cycle", "solicit_bid")
    builder.add_edge("settle", "update_history")
    builder.add_conditional_edges("update_history", route_after_update_history)
    builder.add_edge("next_round", "solicit_bid")

    return builder.compile()
