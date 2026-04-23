"""Agent and state factory functions."""

import numpy as np
from typing import Any

from ..config.schema import (
    ExperimentConfig,
    AuctionConfig,
    AuctionType,
    PersonaConfig,
    Strategy,
)
from ..graph.state import (
    MarketState,
    AgentState,
    BidderState,
    SealedBidState,
    EnglishAuctionState,
    DutchAuctionState,
)


def _normalize_strategies(
    strategies: Strategy | list[Strategy],
    num: int,
) -> list[Strategy]:
    """Expand a strategy spec into a per-agent list of length `num`.

    Length mismatch is already caught at config load by
    ``AgentPricesConfig._validate_strategies_length`` /
    ``BiddersConfig._validate_strategies_length``; this helper trusts the
    invariant.
    """
    if isinstance(strategies, list):
        return list(strategies)
    return [strategies] * num


def _build_schedules(
    side_min: float,
    side_max: float,
    num: int,
    units_per_agent: int,
    side: str,  # "buyer" or "seller"
) -> list[list[float]]:
    """Build per-agent marginal value/cost schedules.

    Generates ``num * units_per_agent`` linspace values on ``[min, max]``
    (ascending) and partitions them into contiguous chunks of
    ``units_per_agent`` values assigned by agent ID.

    Convention: higher agent ID means more-aggressive agent on the given
    side. Buyer 0 has the weakest values; buyer N-1 has the strongest.
    Seller 0 has the lowest costs (most aggressive); seller N-1 has the
    highest. This preserves the single-unit convention
    ``reservation[i] = linspace(min, max)[i]`` for both sides and
    generalises it to multi-unit without breaking downstream analysis
    that relies on agent-ID / reservation-price orderings.

    Within each agent's schedule, units are ordered most-aggressive-first
    so advancing ``current_unit_index`` by 1 always steps to the next-best
    (less aggressive) unit:
      - buyer: values descending (first unit = highest value)
      - seller: values ascending (first unit = lowest cost)

    For ``units_per_agent == 1`` this is byte-compatible with the
    pre-multi-unit factory output.
    """
    total = num * units_per_agent
    all_values = np.round(np.linspace(side_min, side_max, total), 2)
    sorted_values = np.sort(all_values)  # ascending

    schedules: list[list[float]] = []
    for i in range(num):
        chunk = sorted_values[i * units_per_agent : (i + 1) * units_per_agent]
        if side == "buyer":
            # Buyer's first (most-aggressive) unit is the highest value
            # in their chunk — reverse so the schedule is descending.
            chunk = chunk[::-1]
        # Seller's first unit is the lowest cost — keep ascending.
        schedules.append([float(v) for v in chunk])
    return schedules


def create_agents(
    config: ExperimentConfig,
    personas: PersonaConfig | None = None,
) -> list[AgentState]:
    """Create buyer and seller agents based on configuration.

    Handles both single-unit (``units_per_agent == 1``) and multi-unit
    (Gode & Sunder 1993 multi-unit markets) traders. Each agent carries
    a ``values`` schedule and a ``current_unit_index`` cursor; the
    initial ``reservation_price`` mirrors ``values[0]``.

    Args:
        config: Experiment configuration with price distributions.
        personas: Optional persona configuration for per-agent customization.

    Returns:
        List of AgentState dictionaries.
    """
    agents: list[AgentState] = []

    if personas is None:
        personas = PersonaConfig()

    # --- Buyers ---
    buyer_schedules = _build_schedules(
        side_min=config.buyers.min,
        side_max=config.buyers.max,
        num=config.buyers.num,
        units_per_agent=config.buyers.units_per_agent,
        side="buyer",
    )
    buyer_strategies = _normalize_strategies(config.buyers.strategies, config.buyers.num)

    for i, schedule in enumerate(buyer_schedules):
        persona_text = personas.buyers.get(i, personas.buyer_default)
        agents.append(
            AgentState(
                id=i,
                type="buyer",
                reservation_price=schedule[0],
                values=schedule,
                current_unit_index=0,
                active=True,
                own_history_prompt="",
                own_history_data=[],
                persona=persona_text,
                strategy=buyer_strategies[i],
            )
        )

    # --- Sellers (with offset IDs) ---
    seller_schedules = _build_schedules(
        side_min=config.sellers.min,
        side_max=config.sellers.max,
        num=config.sellers.num,
        units_per_agent=config.sellers.units_per_agent,
        side="seller",
    )
    seller_strategies = _normalize_strategies(config.sellers.strategies, config.sellers.num)

    id_offset = config.buyers.num
    for i, schedule in enumerate(seller_schedules):
        persona_text = personas.sellers.get(i, personas.seller_default)
        agents.append(
            AgentState(
                id=id_offset + i,
                type="seller",
                reservation_price=schedule[0],
                values=schedule,
                current_unit_index=0,
                active=True,
                own_history_prompt="",
                own_history_data=[],
                persona=persona_text,
                strategy=seller_strategies[i],
            )
        )

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

    # CDA-only factory: max_ticks_per_round is required at config load
    # (SimulationConfig._require_max_ticks_for_cda). Safe to assert here.
    assert config.max_ticks_per_round is not None, (
        "max_ticks_per_round must be set for double_auction configs"
    )

    return MarketState(
        # Experiment context
        round=1,
        iteration=1,
        max_rounds=config.n_rounds,
        # max_iterations holds the tick budget per round under the
        # improvement-rule CDA. One tick = one randomly-chosen active
        # agent attempts to post an order.
        max_iterations=config.max_ticks_per_round,
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
        # Order book — empty at start of simulation.
        standing_bid=None,
        standing_ask=None,
        standing_bid_agent_id=None,
        standing_ask_agent_id=None,
        last_order_outcome=None,
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
        # Chain-of-thought reasoning
        last_announcement_reasoning="",
        last_response_reasoning="",
        # Error handling
        last_error=None,
        # Diagnostic counters
        constraint_violations=0,
        # History display configuration
        history_mode=config.history.mode,
        history_summary_last_n=config.history.summary_last_n_events,
        own_history_mode=config.history.own_history_mode,
    )


# ============================================================
# Auction bidder creation
# ============================================================


def create_bidders(
    config: AuctionConfig,
    personas: PersonaConfig | None = None,
) -> list[BidderState]:
    """Create auction bidders with independent private values.

    Args:
        config: Auction configuration with bidder distribution params.
        personas: Optional persona configuration for per-bidder customization.

    Returns:
        List of BidderState dictionaries.
    """
    bc = config.bidders

    if bc.distribution == "linspace":
        values = np.round(np.linspace(bc.value_min, bc.value_max, bc.num), 2)
    else:  # uniform
        rng = np.random.default_rng(config.random_seed)
        values = np.round(
            rng.uniform(bc.value_min, bc.value_max, bc.num), 2
        )

    if personas is None:
        personas = PersonaConfig()

    bidder_strategies = _normalize_strategies(bc.strategies, bc.num)

    bidders: list[BidderState] = []
    for i, val in enumerate(values):
        persona_text = personas.bidders.get(i, personas.bidder_default)
        bidders.append(
            BidderState(
                id=i,
                private_value=float(val),
                active=True,
                own_history_prompt="",
                own_history_data=[],
                persona=persona_text,
                strategy=bidder_strategies[i],
            )
        )
    return bidders


def create_auction_initial_state(
    experiment_config: ExperimentConfig,
    auction_config: AuctionConfig,
    simulation_id: int = 1,
    personas: PersonaConfig | None = None,
) -> dict:
    """Create the initial state dict for an auction simulation.

    Dispatches to the correct state TypedDict based on auction_type.

    Args:
        experiment_config: Top-level experiment config (carries auction_type).
        auction_config: Auction-specific parameters.
        simulation_id: Identifier for this simulation run.
        personas: Optional persona configuration for per-bidder customization.

    Returns:
        Initial state dictionary matching the appropriate auction state type.
    """
    auction_type = experiment_config.auction_type
    bidders = create_bidders(auction_config, personas=personas)

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
