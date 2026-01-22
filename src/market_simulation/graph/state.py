"""Market simulation state definitions."""

from typing import Annotated, TypedDict
from operator import add


class AgentState(TypedDict):
    """State for a single market agent."""

    id: int
    type: str  # "buyer" or "seller"
    reservation_price: float
    active: bool  # Still in current round
    own_history_prompt: str  # History for prompt rendering
    own_history_data: list[dict]  # Data for CSV export


class Transaction(TypedDict):
    """Record of a completed transaction."""

    round: int
    iteration: int
    price: float
    buyer_id: int
    seller_id: int
    announcement_type: str


class IterationRecord(TypedDict):
    """Record of an iteration's events."""

    round: int
    iteration: int
    price: float | None
    announcement_made: bool
    transaction_made: bool
    announcement_type: str | None
    announcing_agent_id: int | None
    announcing_agent_reservation_price: float | None
    responding_agent_id: int | None
    responding_agent_reservation_price: float | None


class MarketState(TypedDict):
    """Complete state for market simulation graph."""

    # Experiment context
    round: int
    iteration: int
    max_rounds: int
    max_iterations: int
    simulation_id: int

    # Agent management
    agents: list[AgentState]
    active_agent_ids: list[int]  # IDs of agents still in round
    potential_responder_ids: list[int]  # IDs of agents who can respond
    current_responder_index: int  # Index in potential_responder_ids
    announced_this_iteration: list[int]  # IDs of agents who already announced this iteration

    # Current turn state
    announcing_agent_id: int | None
    announced_price: float | None
    announcement_type: str | None  # "buy" or "sell"
    responding_agent_id: int | None
    response_accepted: bool | None

    # History (using add reducer for appending)
    market_history_text: str  # Text history for prompts
    iteration_records: Annotated[list[IterationRecord], add]
    transactions: Annotated[list[Transaction], add]

    # Control flow flags
    announcement_made: bool
    transaction_made: bool
    iteration_complete: bool
    round_complete: bool
    simulation_complete: bool

    # Error handling
    last_error: str | None
