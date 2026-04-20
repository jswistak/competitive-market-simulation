"""Sealed-bid auction graph builder (FPSB, SPSB, All-Pay).

Graph topology:
  START -> collect_bid --[loop]--> collect_bid
                       |--[all collected]--> determine_winner -> update_history
                                                                  |--[more rounds]--> next_round -> collect_bid
                                                                  |--[done]--> END
"""

from typing import Callable

from langgraph.graph import StateGraph, START, END

from ...state import SealedBidState
from .nodes import (
    make_collect_bid_node,
    make_determine_winner_node,
    make_update_sealed_history_node,
    make_next_sealed_round_node,
)
from .edges import route_after_collect_bid, route_after_update_history
from ....llm.providers.base import LLMProvider
from ....llm.response_schemas import get_response_schemas
from ....config.schema import AuctionPromptConfig


def build_sealed_bid_graph(
    auction_type: str,
    llm: LLMProvider,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    include_reasoning: bool = True,
) -> StateGraph:
    """Build a sealed-bid auction LangGraph workflow.

    Works for FPSB, SPSB, and All-Pay — the auction_type in state
    controls only the payment rule in determine_winner.

    Args:
        auction_type: One of "fpsb", "spsb", "all_pay".
        llm: LLM provider for bidder interactions.
        prompts: Auction prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks.
        include_reasoning: Whether to include reasoning field in LLM responses.

    Returns:
        Compiled StateGraph ready for execution.
    """
    schemas = get_response_schemas(include_reasoning)
    builder = StateGraph(SealedBidState)

    # Add nodes
    builder.add_node("collect_bid", make_collect_bid_node(llm, prompts, callbacks_factory, response_schema=schemas.bid))
    builder.add_node("determine_winner", make_determine_winner_node())
    builder.add_node("update_history", make_update_sealed_history_node(prompts))
    builder.add_node("next_round", make_next_sealed_round_node())

    # Edges
    builder.add_edge(START, "collect_bid")
    builder.add_conditional_edges("collect_bid", route_after_collect_bid)
    builder.add_edge("determine_winner", "update_history")
    builder.add_conditional_edges("update_history", route_after_update_history)
    builder.add_edge("next_round", "collect_bid")

    return builder.compile()
