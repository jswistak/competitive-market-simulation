"""LangGraph workflow builder for market simulation."""

from typing import Callable

import numpy as np
from langgraph.graph import StateGraph, START

from .state import MarketState
from .nodes import (
    make_select_announcer_node,
    make_announce_node,
    make_select_responders_node,
    make_respond_node,
    make_record_transaction_node,
    make_update_history_node,
    make_check_iteration_node,
    make_check_round_node,
    make_next_iteration_node,
    make_next_round_node,
)
from .edges import (
    route_after_announcement,
    route_after_response,
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
    """Build the market simulation LangGraph workflow.

    Args:
        llm: LLM provider for agent interactions. May be ``None`` when all
            agents use a zero-intelligence strategy.
        prompts: Prompt configuration for agents.
        callbacks_factory: Optional factory for creating tracing callbacks.
        include_reasoning: Whether to include reasoning field in LLM responses.
        zi_config: Hyperparameters for ZI sampling.
        rng: Seeded NumPy ``Generator`` for ZI randomness. A single generator
            is shared across all ZI call sites (announce, respond, auction
            nodes), so trajectory reproducibility under a seed is sensitive
            to the *order* in which those sites are invoked. Any future
            change that reorders agent selection or inserts new sampling
            calls will produce different (but still seed-deterministic)
            sequences; do not rely on seed-stability across such refactors.

    Returns:
        Compiled StateGraph ready for execution.
    """
    schemas = get_response_schemas(include_reasoning)
    builder = StateGraph(MarketState)

    # Add nodes
    builder.add_node("select_announcer", make_select_announcer_node())
    builder.add_node(
        "announce",
        make_announce_node(
            llm, prompts, callbacks_factory,
            response_schema=schemas.announcement,
            zi_config=zi_config, rng=rng,
        ),
    )
    builder.add_node("select_responders", make_select_responders_node())
    builder.add_node(
        "respond",
        make_respond_node(
            llm, prompts, callbacks_factory,
            response_schema=schemas.accept_reject,
            zi_config=zi_config, rng=rng,
        ),
    )
    builder.add_node("record_transaction", make_record_transaction_node())
    builder.add_node("check_iteration", make_check_iteration_node())
    builder.add_node("update_history", make_update_history_node())
    builder.add_node("check_round", make_check_round_node())
    builder.add_node("next_iteration", make_next_iteration_node())
    builder.add_node("next_round", make_next_round_node())

    # Add edges
    # Start -> select announcer
    builder.add_edge(START, "select_announcer")

    # Select announcer -> announce
    builder.add_edge("select_announcer", "announce")

    # Announce -> (select_responders | check_iteration)
    builder.add_conditional_edges("announce", route_after_announcement)

    # Select responders -> respond
    builder.add_edge("select_responders", "respond")

    # Respond -> (record_transaction | check_iteration)
    builder.add_conditional_edges("respond", route_after_response)

    # Record transaction -> check_iteration
    builder.add_edge("record_transaction", "check_iteration")

    # Check iteration -> update history
    builder.add_edge("check_iteration", "update_history")

    # Update history -> (check_round | select_announcer | respond)
    builder.add_conditional_edges("update_history", route_after_update_history)

    # Check round -> (next_round | next_iteration)
    builder.add_conditional_edges("check_round", route_after_check_round)

    # Next iteration -> select_announcer
    builder.add_edge("next_iteration", "select_announcer")

    # Next round -> (select_announcer | END)
    builder.add_conditional_edges("next_round", route_after_next_round)

    return builder.compile()
