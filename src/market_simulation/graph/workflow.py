"""LangGraph workflow builder for the continuous double auction.

Implements the Gode & Sunder (1993) improvement-rule CDA. One graph
invocation simulates all rounds of one simulation. Within a round the
loop is tick-based:

    select_announcer -> announce -> apply_order -> update_history -> check_round

Each tick = one randomly-chosen active agent posts an order. The
apply_order node decides crossing / improving / non-improving /
no-announcement; there is no separate response-collection loop.

For mechanisms other than CDA (English / Dutch / sealed-bid /
open-outcry), see ``graph.auctions`` — they have their own subgraphs.
"""

from typing import Callable

import numpy as np
from langgraph.graph import StateGraph, START

from .state import MarketState
from .nodes import (
    make_select_announcer_node,
    make_announce_node,
    make_apply_order_node,
    make_update_history_node,
    make_check_round_node,
    make_next_iteration_node,
    make_next_round_node,
)
from .edges import (
    route_after_update_history,
    route_after_check_round,
    route_after_next_round,
)
from ..llm.providers.base import LLMProvider
from ..llm.response_schemas import get_response_schemas
from ..config.schema import PromptConfig, ZIConfig


def build_market_graph(
    llm: LLMProvider | None,
    prompts: PromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    include_reasoning: bool = True,
    zi_config: ZIConfig | None = None,
    rng: np.random.Generator | None = None,
) -> StateGraph:
    """Build the CDA market-simulation LangGraph workflow.

    Args:
        llm: LLM provider for agent interactions. May be ``None`` when all
            agents use a zero-intelligence strategy.
        prompts: Prompt configuration for agents.
        callbacks_factory: Optional factory for creating tracing callbacks.
        include_reasoning: Whether to include reasoning field in LLM responses.
        zi_config: Hyperparameters for ZI sampling.
        rng: Seeded NumPy ``Generator`` for ZI randomness. A single generator
            is shared across the announce node (the only sampling site
            under the CDA path), so trajectory reproducibility under a seed
            is sensitive to the *order* in which ticks are invoked. Any
            future change that reorders agent selection or inserts new
            sampling calls will produce different (but still
            seed-deterministic) sequences; do not rely on seed-stability
            across such refactors.

    Returns:
        Compiled StateGraph ready for execution.
    """
    schemas = get_response_schemas(include_reasoning)
    builder = StateGraph(MarketState)

    builder.add_node("select_announcer", make_select_announcer_node())
    builder.add_node(
        "announce",
        make_announce_node(
            llm, prompts, callbacks_factory,
            response_schema=schemas.announcement,
            zi_config=zi_config, rng=rng,
        ),
    )
    builder.add_node("apply_order", make_apply_order_node())
    builder.add_node("update_history", make_update_history_node(prompts))
    builder.add_node("check_round", make_check_round_node())
    builder.add_node("next_iteration", make_next_iteration_node())
    builder.add_node("next_round", make_next_round_node())

    builder.add_edge(START, "select_announcer")
    builder.add_edge("select_announcer", "announce")
    builder.add_edge("announce", "apply_order")
    builder.add_edge("apply_order", "update_history")
    builder.add_conditional_edges("update_history", route_after_update_history)
    builder.add_conditional_edges("check_round", route_after_check_round)
    builder.add_edge("next_iteration", "select_announcer")
    builder.add_conditional_edges("next_round", route_after_next_round)

    return builder.compile()
