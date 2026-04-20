"""Pydantic configuration schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# --- Auction type enum ---


class AuctionType(str, Enum):
    """Supported auction mechanisms."""

    DOUBLE_AUCTION = "double_auction"
    FPSB = "fpsb"  # First-Price Sealed-Bid
    SPSB = "spsb"  # Second-Price Sealed-Bid (Vickrey)
    ENGLISH = "english"  # English (ascending) auction
    DUTCH = "dutch"  # Dutch (descending) auction
    ALL_PAY = "all_pay"  # All-Pay auction
    FIRST_PRICE_OPEN_OUTCRY = "first_price_open_outcry"


# --- LLM & tool configs ---


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "gemini", "deepseek"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 10
    max_tokens_with_tools: int = 1024
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_backoff_factor: float = 2.0


class ToolConfig(BaseModel):
    """Tool availability configuration for agents."""

    enabled: bool = False
    enable_simple_tools: bool = False
    enable_code_interpreter: bool = False
    e2b_timeout: int = 300
    max_tool_iterations: int = 5


# --- Double auction configs (existing) ---


class AgentPricesConfig(BaseModel):
    """Reservation price distribution for agents."""

    min: float = 0.8
    max: float = 3.2
    num: int = 11


class HistoryConfig(BaseModel):
    """Configuration for how market history is presented in prompts."""

    mode: Literal["full", "summary"] = "full"
    own_history_mode: Literal["full", "summary"] = "full"
    summary_last_n_events: int = 3


# --- Auction-specific configs ---


class BiddersConfig(BaseModel):
    """Bidder private-value distribution for auctions."""

    num: int = 5
    value_min: float = 0.0
    value_max: float = 10.0
    distribution: Literal["linspace", "uniform"] = "linspace"


class AuctionConfig(BaseModel):
    """Mechanism parameters for auctions (non-double-auction)."""

    n_rounds: int = 10
    n_simulations: int = 10
    bidders: BiddersConfig = Field(default_factory=BiddersConfig)

    # English / Open-Outcry
    min_increment: float = 0.5
    max_bidding_rounds: int = 50

    # Dutch
    dutch_start_price: float = 12.0
    dutch_decrement: float = 0.5
    dutch_min_price: float = 0.0

    # Reproducibility
    random_seed: int | None = None  # Seed for random operations (shuffling, sampling)


class AuctionPromptConfig(BaseModel):
    """Prompt templates specific to auction mechanisms."""

    system_template: str = ""
    bid_prompt: str = ""  # Sealed-bid: submit your bid
    english_bid_prompt: str = ""  # English/Open-Outcry: bid or pass
    dutch_accept_prompt: str = ""  # Dutch: accept or reject current price
    value_explanation: str = ""  # Explains profit = value - payment

    # --- Market history entries (shared across auction types) ---
    # Format keys: round, auction_type, winner_id, winning_bid, payment,
    # second_highest_bid (may be None for auctions other than Vickrey).
    market_history_winner_template: str = (
        "Round {round} ({auction_type}): Bidder {winner_id} won at "
        "${winning_bid:.2f}, payment=${payment:.2f}.\n"
    )
    market_history_no_winner_template: str = (
        "Round {round} ({auction_type}): No winner this round.\n"
    )

    # --- Dutch per-bidder history ---
    # Format keys: round, payment, winner_id (for rejected_other_winner).
    dutch_bidder_accepted_template: str = (
        "Round {round}: You accepted at ${payment:.2f}.\n"
    )
    dutch_bidder_rejected_other_winner_template: str = (
        "Round {round}: You did not accept. "
        "Bidder {winner_id} won at ${payment:.2f}.\n"
    )
    dutch_bidder_rejected_no_winner_template: str = (
        "Round {round}: You did not accept. No one accepted.\n"
    )

    # --- English per-bidder history ---
    # Format keys: round, my_bid, payment (when won).
    english_bidder_won_template: str = (
        "Round {round}: Your highest bid was ${my_bid:.2f} and you won. "
        "Payment: ${payment:.2f}.\n"
    )
    english_bidder_lost_template: str = (
        "Round {round}: Your highest bid was ${my_bid:.2f} and you lost.\n"
    )
    english_bidder_no_bid_template: str = "Round {round}: You did not bid.\n"

    # --- Sealed-bid per-bidder history ---
    # Format keys: round, my_bid, payment (when won).
    sealed_bidder_won_template: str = (
        "Round {round}: You bid ${my_bid:.2f} and won. "
        "Payment: ${payment:.2f}.\n"
    )
    sealed_bidder_lost_template: str = (
        "Round {round}: You bid ${my_bid:.2f} and lost.\n"
    )
    # All-pay specific: loser still pays their own bid.
    sealed_bidder_all_pay_loss_template: str = (
        "Round {round}: You bid ${my_bid:.2f} and lost. "
        "You paid your bid of ${my_bid:.2f}.\n"
    )


class ExperimentConfig(BaseModel):
    """Experiment parameters."""

    auction_type: AuctionType = AuctionType.DOUBLE_AUCTION
    include_reasoning: bool = True

    n_rounds: int = 5
    n_iterations: int = 10
    n_simulations: int = 10
    buyers: AgentPricesConfig = Field(default_factory=AgentPricesConfig)
    sellers: AgentPricesConfig = Field(default_factory=AgentPricesConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)

    # Auction-specific config (only used when auction_type != double_auction)
    auction: AuctionConfig | None = None

    # Optional manual override for LangGraph recursion limit
    recursion_limit: int | None = None


class TracingConfig(BaseModel):
    """Langfuse tracing configuration."""

    enabled: bool = True
    llm_call_logging: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


class PromptTemplates(BaseModel):
    """Prompt templates for agent communication."""

    main_template: str = ""

    # Per-agent "own history" entries (injected via {own_history}).
    announcement_history_template: str = (
        "In round {round} at iteration {iteration}, your offer to {announcement_type} "
        "for ${price:.2f} was {outcome}.\n"
    )
    response_history_template: str = (
        "In round {round} at iteration {iteration}, you {outcome} a "
        "{opposite_announcement_type} offer for ${price:.2f}.\n"
    )

    # Shared market-history entries (injected via {market_history}).
    market_history_accepted_template: str = (
        "In round {round} at iteration {iteration}, an announcement to "
        "{announcement_type} for ${price:.2f} was accepted.\n"
    )
    market_history_rejected_template: str = (
        "In round {round} at iteration {iteration}, an announcement to "
        "{announcement_type} for ${price:.2f} was made but no one responded.\n"
    )
    market_history_no_announcement_template: str = (
        "In round {round} at iteration {iteration}, no announcement was made.\n"
    )


class AgentKeywords(BaseModel):
    """Keywords for prompt substitution."""

    role: str
    verb: str
    preference: str
    condition: str


class AgentPromptConfig(BaseModel):
    """Agent-specific prompt configuration."""

    main_keywords: AgentKeywords
    response_prompt: str
    announcement_prompt: str


class PromptConfig(BaseModel):
    """Complete prompt configuration."""

    general: PromptTemplates = Field(default_factory=PromptTemplates)
    tools_preamble: str = ""
    buyer: AgentPromptConfig | None = None
    seller: AgentPromptConfig | None = None
    auction: AuctionPromptConfig | None = None


class PersonaConfig(BaseModel):
    """Per-agent persona/prompt customization."""

    buyer_default: str = ""
    seller_default: str = ""
    buyers: dict[int, str] = Field(default_factory=dict)
    sellers: dict[int, str] = Field(default_factory=dict)
    # Auction bidder personas
    bidder_default: str = ""
    bidders: dict[int, str] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    personas: PersonaConfig = Field(default_factory=PersonaConfig)
