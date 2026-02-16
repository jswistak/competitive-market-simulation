"""Agent and state factory functions."""

import numpy as np
from typing import Any

from ..config.schema import ExperimentConfig, AuctionConfig, AuctionType
from ..graph.state import (
    MarketState,
    AgentState,
    BidderState,
    SealedBidState,
    EnglishAuctionState,
    DutchAuctionState,
)


def create_agents(config: ExperimentConfig) -> list[AgentState]:
    """Create buyer and seller agents based on configuration.

    Args:
        config: Experiment configuration with price distributions.

    Returns:
        List of AgentState dictionaries.
    """
    agents: list[AgentState] = []

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
        agent = AgentState(
            id=i,
            type="buyer",
            reservation_price=float(price),
            active=True,
            own_history_prompt="",
            own_history_data=[],
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
        agent = AgentState(
            id=id_offset + i,
            type="seller",
            reservation_price=float(price),
            active=True,
            own_history_prompt="",
            own_history_data=[],
        )
        agents.append(agent)

    return agents


def create_initial_state(
    config: ExperimentConfig,
    simulation_id: int = 1,
) -> MarketState:
    """Create the initial market state for a simulation.

    Args:
        config: Experiment configuration.
        simulation_id: Identifier for this simulation run.

    Returns:
        Initial MarketState dictionary.
    """
    agents = create_agents(config)
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


# ============================================================
# Auction bidder creation
# ============================================================


def create_bidders(config: AuctionConfig) -> list[BidderState]:
    """Create auction bidders with independent private values.

    Args:
        config: Auction configuration with bidder distribution params.

    Returns:
        List of BidderState dictionaries.
    """
    bc = config.bidders

    if bc.distribution == "linspace":
        values = np.round(np.linspace(bc.value_min, bc.value_max, bc.num), 2)
    else:  # uniform
        values = np.round(
            np.random.uniform(bc.value_min, bc.value_max, bc.num), 2
        )

    bidders: list[BidderState] = []
    for i, val in enumerate(values):
        bidders.append(
            BidderState(
                id=i,
                private_value=float(val),
                active=True,
                own_history_prompt="",
                own_history_data=[],
            )
        )
    return bidders


def create_auction_initial_state(
    experiment_config: ExperimentConfig,
    auction_config: AuctionConfig,
    simulation_id: int = 1,
) -> dict:
    """Create the initial state dict for an auction simulation.

    Dispatches to the correct state TypedDict based on auction_type.

    Args:
        experiment_config: Top-level experiment config (carries auction_type).
        auction_config: Auction-specific parameters.
        simulation_id: Identifier for this simulation run.

    Returns:
        Initial state dictionary matching the appropriate auction state type.
    """
    auction_type = experiment_config.auction_type
    bidders = create_bidders(auction_config)

    # Common fields shared by all auction states
    common: dict[str, Any] = {
        "round": 1,
        "max_rounds": auction_config.n_rounds,
        "simulation_id": simulation_id,
        "auction_type": auction_type.value,
        "bidders": bidders,
        "current_bidder_index": 0,
        "auction_results": [],
        "all_bid_records": [],
        "market_history_text": "",
        "tool_usage_log": [],
        "parse_failures": 0,
        "constraint_violations": 0,
    }

    if auction_type in (
        AuctionType.FPSB,
        AuctionType.SPSB,
        AuctionType.ALL_PAY,
    ):
        return SealedBidState(
            **common,
            all_bids_collected=False,
            bids=[],
        )

    if auction_type in (
        AuctionType.ENGLISH,
        AuctionType.FIRST_PRICE_OPEN_OUTCRY,
    ):
        all_ids = [b["id"] for b in bidders]
        return EnglishAuctionState(
            **common,
            active_bidder_ids=all_ids,
            standing_bid=0.0,
            standing_bidder_id=None,
            min_increment=auction_config.min_increment,
            bids_this_cycle=0,
            bid_step=0,
            max_bidding_rounds=auction_config.max_bidding_rounds,
            bids=[],
            auction_ended=False,
        )

    if auction_type == AuctionType.DUTCH:
        return DutchAuctionState(
            **common,
            current_price=auction_config.dutch_start_price,
            dutch_start_price=auction_config.dutch_start_price,
            dutch_decrement=auction_config.dutch_decrement,
            dutch_min_price=auction_config.dutch_min_price,
            accepted=False,
            accepting_bidder_id=None,
            bids=[],
            all_queried_at_price=False,
        )

    raise ValueError(f"Unsupported auction type for state creation: {auction_type}")
