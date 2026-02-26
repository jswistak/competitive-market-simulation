"""Agent and state factory functions."""

import numpy as np
from typing import Any

from ..config.schema import ExperimentConfig, PersonaConfig
from ..graph.state import MarketState, AgentState


def create_agents(
    config: ExperimentConfig,
    personas: PersonaConfig | None = None,
) -> list[AgentState]:
    """Create buyer and seller agents based on configuration.

    Args:
        config: Experiment configuration with price distributions.
        personas: Optional persona configuration for per-agent customization.

    Returns:
        List of AgentState dictionaries.
    """
    agents: list[AgentState] = []

    if personas is None:
        personas = PersonaConfig()

    # Create buyers
    buyer_prices = np.round(
        np.linspace(
            config.buyers.min,
            config.buyers.max,
            config.buyers.num,
        ),
        2,
    )

    for i, price in enumerate(buyer_prices):
        persona_text = personas.buyers.get(i, personas.buyer_default)
        agent = AgentState(
            id=i,
            type="buyer",
            reservation_price=float(price),
            active=True,
            own_history_prompt="",
            own_history_data=[],
            persona=persona_text,
        )
        agents.append(agent)

    # Create sellers (with offset IDs)
    seller_prices = np.round(
        np.linspace(
            config.sellers.min,
            config.sellers.max,
            config.sellers.num,
        ),
        2,
    )

    id_offset = config.buyers.num
    for i, price in enumerate(seller_prices):
        persona_text = personas.sellers.get(i, personas.seller_default)
        agent = AgentState(
            id=id_offset + i,
            type="seller",
            reservation_price=float(price),
            active=True,
            own_history_prompt="",
            own_history_data=[],
            persona=persona_text,
        )
        agents.append(agent)

    return agents


def create_initial_state(
    config: ExperimentConfig,
    simulation_id: int = 1,
    personas: PersonaConfig | None = None,
) -> MarketState:
    """Create the initial market state for a simulation.

    Args:
        config: Experiment configuration.
        simulation_id: Identifier for this simulation run.
        personas: Optional persona configuration for per-agent persona text.

    Returns:
        Initial MarketState dictionary.
    """
    agents = create_agents(config, personas=personas)
    all_agent_ids = [agent["id"] for agent in agents]

    return MarketState(
        # Experiment context
        round=1,
        iteration=1,
        max_rounds=config.n_rounds,
        max_iterations=config.n_iterations,
        simulation_id=simulation_id,
        # Agent management
        agents=agents,
        active_agent_ids=all_agent_ids,
        potential_responder_ids=[],
        current_responder_index=0,
        # Current turn state
        announcing_agent_id=None,
        announced_price=None,
        announcement_type=None,
        responding_agent_id=None,
        response_accepted=None,
        # History
        market_history_text="",
        iteration_records=[],
        transactions=[],
        # Iteration tracking
        announced_this_iteration=[],
        # Control flow flags
        announcement_made=False,
        transaction_made=False,
        iteration_complete=False,
        round_complete=False,
        simulation_complete=False,
        # Tool usage tracking
        tool_usage_log=[],
        # Error handling
        last_error=None,
        # Diagnostic counters
        parse_failures=0,
        constraint_violations=0,
    )
