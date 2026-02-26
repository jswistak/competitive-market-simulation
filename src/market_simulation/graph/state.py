"""Market simulation state definitions."""

from typing import Annotated, Literal, NotRequired, TypedDict
from operator import add


class AgentState(TypedDict):
    """State for a single market agent."""

    id: int
    type: str  # "buyer" or "seller"
    reservation_price: float
    active: bool  # Still in current round
    own_history_prompt: str  # History for prompt rendering
    own_history_data: list[dict]  # Data for CSV export
    persona: str  # Per-agent persona text


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

    # Tool usage tracking
    tool_usage_log: Annotated[list[dict], add]

    # Error handling
    last_error: str | None

    # Diagnostic counters
    parse_failures: int
    constraint_violations: int

    # History display configuration
    history_mode: Literal["full", "summary"]
    history_summary_last_n: int
    own_history_mode: Literal["full", "summary"]


# ============================================================
# Auction state definitions (new auction types)
# ============================================================


class BidderState(TypedDict):
    """State for a single auction bidder."""

    id: int
    private_value: float
    active: bool  # Still participating in current round
    own_history_prompt: str  # History for prompt rendering
    own_history_data: list[dict]  # Data for CSV export
    persona: str  # Per-bidder persona text


class BidRecord(TypedDict):
    """A single bid submitted by a bidder."""

    bidder_id: int
    bid_amount: float
    round: int
    bid_step: int  # Which step within the round (cycle for English, 0 for sealed)
    private_value: float


class AuctionResult(TypedDict):
    """Outcome of a single auction round."""

    round: int
    auction_type: str
    winner_id: int | None  # None if no winner (e.g. Dutch hits floor)
    winning_bid: float | None
    payment: float | None
    second_highest_bid: float | None
    all_bids: list[dict]  # [{bidder_id, bid_amount}]
    n_active_bidders: int
    surplus: float | None  # winner's private_value - payment
    total_payments: NotRequired[float | None]  # Sum of ALL bidders' payments (relevant for all-pay auctions where losers also pay)


# --- Sealed-Bid state (FPSB, SPSB, All-Pay) ---


class SealedBidState(TypedDict):
    """State for sealed-bid auction graphs."""

    # Experiment context
    round: int
    max_rounds: int
    simulation_id: int
    auction_type: str  # "fpsb", "spsb", "all_pay"

    # Bidders
    bidders: list[BidderState]
    current_bidder_index: int
    all_bids_collected: bool

    # Current round bids
    bids: list[BidRecord]

    # Accumulated results (add reducer for appending)
    auction_results: Annotated[list[AuctionResult], add]
    all_bid_records: Annotated[list[BidRecord], add]

    # History
    market_history_text: str

    # Tool usage tracking
    tool_usage_log: Annotated[list[dict], add]

    # Diagnostics
    parse_failures: int
    constraint_violations: int


# --- English Auction state ---


class EnglishAuctionState(TypedDict):
    """State for English (ascending) auction graphs."""

    # Experiment context
    round: int
    max_rounds: int
    simulation_id: int
    auction_type: str  # "english"

    # Bidders
    bidders: list[BidderState]
    active_bidder_ids: list[int]
    current_bidder_index: int

    # Auction dynamics
    standing_bid: float  # Current highest bid
    standing_bidder_id: int | None  # Who holds the standing bid
    min_increment: float
    bids_this_cycle: int  # New bids placed in current cycle
    bid_step: int  # Overall step counter within round
    max_bidding_rounds: int  # Safety limit on cycles

    # Accumulated data
    bids: list[BidRecord]  # Bids in current round (reset each round)
    auction_results: Annotated[list[AuctionResult], add]
    all_bid_records: Annotated[list[BidRecord], add]

    # History
    market_history_text: str

    # Control
    auction_ended: bool  # Current round's auction resolved

    # Tool usage tracking
    tool_usage_log: Annotated[list[dict], add]

    # Diagnostics
    parse_failures: int
    constraint_violations: int


# --- Dutch Auction state ---


class DutchAuctionState(TypedDict):
    """State for Dutch (descending) auction graphs."""

    # Experiment context
    round: int
    max_rounds: int
    simulation_id: int
    auction_type: str  # "dutch"

    # Bidders
    bidders: list[BidderState]
    current_bidder_index: int

    # Price mechanism
    current_price: float
    dutch_start_price: float
    dutch_decrement: float
    dutch_min_price: float
    accepted: bool  # Someone accepted at current price
    accepting_bidder_id: int | None

    # Accumulated data
    bids: list[BidRecord]  # Acceptance records for current round
    auction_results: Annotated[list[AuctionResult], add]
    all_bid_records: Annotated[list[BidRecord], add]

    # History
    market_history_text: str

    # Control
    all_queried_at_price: bool  # All bidders asked at current price level

    # Tool usage tracking
    tool_usage_log: Annotated[list[dict], add]

    # Diagnostics
    parse_failures: int
    constraint_violations: int


# --- Open-Outcry state (reuses English topology, different settlement) ---

# Uses EnglishAuctionState with auction_type="first_price_open_outcry"
OpenOutcryState = EnglishAuctionState
