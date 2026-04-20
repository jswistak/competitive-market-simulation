"""English auction graph builder.

Graph topology:
  START -> solicit_bid --[next bidder]--> solicit_bid
                       |--[cycle end]--> check_auction_end
                                            |--[continue]--> reset_cycle -> solicit_bid
                                            |--[done]--> settle -> update_history
                                                                      |--[more rounds]--> next_round -> solicit_bid
                                                                      |--[done]--> END
"""

from typing import Callable

from langgraph.graph import StateGraph, START, END

from ...state import EnglishAuctionState
from .nodes import (
    make_solicit_bid_node,
    make_check_auction_end_node,
    make_reset_cycle_node,
    make_settle_english_node,
    make_update_english_history_node,
    make_next_english_round_node,
)
from .edges import (
    route_after_solicit_bid,
    route_after_check_end,
    route_after_update_history,
)
from ....llm.providers.base import LLMProvider
from ....llm.response_schemas import get_response_schemas
from ....config.schema import AuctionPromptConfig


def build_english_graph(
    auction_type: str,
    llm: LLMProvider,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    include_reasoning: bool = True,
) -> StateGraph:
    """Build an English (ascending) auction LangGraph workflow.

    Args:
        auction_type: "english" (always, but kept for interface consistency).
        llm: LLM provider for bidder interactions.
        prompts: Auction prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks.
        include_reasoning: Whether to include reasoning field in LLM responses.

    Returns:
        Compiled StateGraph ready for execution.
    """
    schemas = get_response_schemas(include_reasoning)
    builder = StateGraph(EnglishAuctionState)

    # Nodes
    builder.add_node("solicit_bid", make_solicit_bid_node(llm, prompts, callbacks_factory, response_schema=schemas.english_bid))
    builder.add_node("check_auction_end", make_check_auction_end_node())
    builder.add_node("reset_cycle", make_reset_cycle_node())
    builder.add_node("settle", make_settle_english_node())
    builder.add_node("update_history", make_update_english_history_node(prompts))
    builder.add_node("next_round", make_next_english_round_node())

    # Edges
    builder.add_edge(START, "solicit_bid")
    builder.add_conditional_edges("solicit_bid", route_after_solicit_bid)
    builder.add_conditional_edges("check_auction_end", route_after_check_end)
    builder.add_edge("reset_cycle", "solicit_bid")
    builder.add_edge("settle", "update_history")
    builder.add_conditional_edges("update_history", route_after_update_history)
    builder.add_edge("next_round", "solicit_bid")

    return builder.compile()
