"""Pydantic configuration schemas."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """Base for every config model in this module.

    ``extra='forbid'`` makes Pydantic refuse to load a YAML that
    contains keys we haven't defined here. The point is to catch
    typos and stale fields up front instead of having Pydantic
    silently drop them — which, before this was tightened, allowed
    YAMLs to "configure" things that the runtime never saw (the
    classic example being PR #21's rename leaving the YAML's
    ``market_history_rejected_template`` permanently invisible).
    """

    model_config = ConfigDict(extra="forbid")


Strategy = Literal["llm", "zi_c", "zi_u"]

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


class LLMConfig(_StrictModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "gemini", "deepseek"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 10
    max_tokens_with_tools: int = 1024
    max_retries: int = 5


class ToolConfig(_StrictModel):
    """Tool availability configuration for agents."""

    enabled: bool = False
    enable_simple_tools: bool = False
    enable_code_interpreter: bool = False
    e2b_timeout: int = 300
    max_tool_iterations: int = 5


# --- Double auction configs (existing) ---


class AgentPricesConfig(_StrictModel):
    """Reservation price distribution for agents."""

    min: float = 0.8
    max: float = 3.2
    num: int = 11
    strategies: Strategy | list[Strategy] = "llm"

    @model_validator(mode="after")
    def _validate_strategies_length(self):
        if isinstance(self.strategies, list) and len(self.strategies) != self.num:
            raise ValueError(
                f"strategies list length ({len(self.strategies)}) must equal num ({self.num})"
            )
        return self


class HistoryConfig(_StrictModel):
    """Configuration for how market history is presented in prompts."""

    mode: Literal["full", "summary"] = "full"
    own_history_mode: Literal["full", "summary"] = "full"
    summary_last_n_events: int = 3


# --- Auction-specific configs ---


class BiddersConfig(_StrictModel):
    """Bidder private-value distribution for auctions."""

    num: int = 5
    value_min: float = 0.0
    value_max: float = 10.0
    distribution: Literal["linspace", "uniform"] = "linspace"
    strategies: Strategy | list[Strategy] = "llm"

    @model_validator(mode="after")
    def _validate_strategies_length(self):
        if isinstance(self.strategies, list) and len(self.strategies) != self.num:
            raise ValueError(
                f"strategies list length ({len(self.strategies)}) must equal num ({self.num})"
            )
        return self


class AuctionConfig(_StrictModel):
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

    # Reproducibility.
    # Used for mechanism-level randomness (e.g. Dutch bidder shuffling).
    # Precedence for the ZI RNG: ``ExperimentConfig.random_seed`` wins;
    # this value is the fallback only when experiment seed is unset.
    random_seed: int | None = None  # Seed for random operations (shuffling, sampling)


class AuctionPromptConfig(_StrictModel):
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
        "Round {round}: You did not accept. Bidder {winner_id} won at ${payment:.2f}.\n"
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
        "Round {round}: You bid ${my_bid:.2f} and won. Payment: ${payment:.2f}.\n"
    )
    sealed_bidder_lost_template: str = (
        "Round {round}: You bid ${my_bid:.2f} and lost.\n"
    )
    # All-pay specific: loser still pays their own bid.
    sealed_bidder_all_pay_loss_template: str = (
        "Round {round}: You bid ${my_bid:.2f} and lost. "
        "You paid your bid of ${my_bid:.2f}.\n"
    )


class ExperimentConfig(_StrictModel):
    """Experiment parameters."""

    auction_type: AuctionType = AuctionType.DOUBLE_AUCTION
    include_reasoning: bool = True

    n_rounds: int = 5
    n_simulations: int = 10
    buyers: AgentPricesConfig = Field(default_factory=AgentPricesConfig)
    sellers: AgentPricesConfig = Field(default_factory=AgentPricesConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)

    # Ticks-per-round for the continuous double auction. One tick = one
    # randomly-chosen active agent posts an order into the central order
    # book. Required when auction_type == double_auction; fail-fast at
    # config load (validator on SimulationConfig). No code default:
    # picking the right value depends on market size and
    # expected trading density, so configs must declare it explicitly.
    max_ticks_per_round: int | None = None

    # Auction-specific config (only used when auction_type != double_auction)
    auction: AuctionConfig | None = None

    # Optional manual override for LangGraph recursion limit
    recursion_limit: int | None = None

    # Seed for ZI sampling and any other stochastic double-auction operations.
    # This is the primary source of truth for the ZI RNG; if unset and the
    # run is an auction, the resolver in ``main.py`` falls back to
    # ``AuctionConfig.random_seed`` to stay backward-compatible with older
    # auction configs that only set the auction-level seed.
    random_seed: int | None = None


class TracingConfig(_StrictModel):
    """Langfuse tracing configuration."""

    enabled: bool = True
    llm_call_logging: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


class PromptTemplates(_StrictModel):
    """Prompt templates for agent communication.

    The announcement prompt is split across two messages sent to the
    LLM:

    * ``system_template`` carries the per-agent constants (role, profit
      formula, market rules, reservation price, persona, market size).
      It is identical for the same agent across every tick of the
      simulation, which makes it cacheable by Anthropic / Gemini
      prompt caches.
    * ``user_template`` carries the per-tick state (current standing
      book, market history, own history, round counter, action
      prompt). It changes every tick.

    Either or both may be empty; an empty ``system_template`` simply
    means no SystemMessage is sent (the LLM call uses only the user
    HumanMessage). Tests and minimal configs typically only populate
    ``user_template``.
    """

    system_template: str = ""
    user_template: str = ""

    # Per-agent "own history" entries (injected via {own_history}).
    # The generic announcement_history_template is used for traded
    # ({outcome}=accepted) and posted ({outcome}=posted) outcomes.
    # Non-improving orders use a dedicated template that carries the
    # rejection *reason* — without it the agent only sees the word
    # "rejected" with no signal of why, making the own_history strictly
    # less informative than the market_history broadcast to other
    # agents.
    announcement_history_template: str = (
        "In round {round} at iteration {iteration}, your offer to {announcement_type} "
        "for ${price:.2f} was {outcome}.\n"
    )
    announcement_history_non_improving_template: str = (
        "In round {round} at iteration {iteration}, your offer to "
        "{announcement_type} for ${price:.2f} was rejected because it did "
        "not improve the standing book.\n"
    )
    # Back-annotations for resting orders. The original "posted" line
    # stays in own_history; these are appended later when the order's
    # downstream fate is known. Without them an agent permanently sees
    # only "posted" and cannot tell whether their earlier order traded,
    # was outbid, or expired with the round.
    announcement_history_filled_template: str = (
        "In round {round} at iteration {iteration}, your earlier offer "
        "to {announcement_type} for ${price:.2f} was filled (a "
        "counterparty crossed it).\n"
    )
    announcement_history_outbid_template: str = (
        "In round {round} at iteration {iteration}, your earlier offer "
        "to {announcement_type} for ${price:.2f} was outbid by a better "
        "{announcement_type} order and is no longer on the book.\n"
    )
    announcement_history_expired_template: str = (
        "At the end of round {round}, your offer to {announcement_type} "
        "for ${price:.2f} remained on the book and was never filled.\n"
    )
    # Shared market-history entries (injected via {market_history}).
    # Used by the improvement-rule CDA path. Each renders one of the four
    # possible per-tick outcomes:
    #   accepted        — order crossed the book, trade executed
    #   posted          — order improved the book and is now standing
    #   non_improving   — order failed the improvement rule and was dropped
    #   no_announcement — agent passed (didn't emit a price)
    market_history_accepted_template: str = (
        "In round {round} at iteration {iteration}, an announcement to "
        "{announcement_type} for ${price:.2f} was accepted.\n"
    )
    market_history_posted_template: str = (
        "In round {round} at iteration {iteration}, an announcement to "
        "{announcement_type} for ${price:.2f} was posted as the new best "
        "{announcement_type} but no one crossed it yet.\n"
    )
    market_history_non_improving_template: str = (
        "In round {round} at iteration {iteration}, an announcement to "
        "{announcement_type} for ${price:.2f} was rejected because it did "
        "not improve the standing book.\n"
    )
    market_history_no_announcement_template: str = (
        "In round {round} at iteration {iteration}, no announcement was made.\n"
    )


class AgentKeywords(_StrictModel):
    """Keywords for prompt substitution.

    ``profit_formula`` is optional; templates reference it as
    ``{profit_formula}`` to embed the side-specific definition of
    profit (e.g. "transaction price and reservation price" for a
    seller). Empty string is a safe default for templates that don't
    need it.
    """

    role: str
    verb: str
    preference: str
    condition: str
    profit_formula: str = ""
    order_outcomes: str = ""


class AgentPromptConfig(_StrictModel):
    """Agent-specific prompt configuration."""

    main_keywords: AgentKeywords
    announcement_prompt: str


class PromptConfig(_StrictModel):
    """Complete prompt configuration."""

    general: PromptTemplates = Field(default_factory=PromptTemplates)
    tools_preamble: str = ""
    buyer: AgentPromptConfig | None = None
    seller: AgentPromptConfig | None = None
    auction: AuctionPromptConfig | None = None


class PersonaConfig(_StrictModel):
    """Per-agent persona/prompt customization."""

    buyer_default: str = ""
    seller_default: str = ""
    buyers: dict[int, str] = Field(default_factory=dict)
    sellers: dict[int, str] = Field(default_factory=dict)
    # Auction bidder personas
    bidder_default: str = ""
    bidders: dict[int, str] = Field(default_factory=dict)


class ZIConfig(_StrictModel):
    """Zero-intelligence trader sampling hyperparameters.

    ZI-C (constrained) always respects reservation price / private value and
    needs no bounds — it draws inside the agent's viable range. ZI-U
    (unconstrained) samples from the fixed interval [u_low, u_high] and
    uses Bernoulli gates to decide whether to announce / bid / accept at all.
    """

    u_low: float = 0.0
    u_high: float = 10.0
    # Probabilities used by ZI-U where a node can choose *not* to act at all.
    announce_prob: float = 0.5  # double-auction announce
    accept_prob: float = 0.5  # dutch acceptance (DA respond was removed in PR #18)
    bid_prob: float = 0.5  # english bid-or-pass


class SimulationConfig(_StrictModel):
    """Complete simulation configuration."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    personas: PersonaConfig = Field(default_factory=PersonaConfig)
    zi: ZIConfig = Field(default_factory=ZIConfig)

    @model_validator(mode="after")
    def _validate_zi_c_bounds(self):
        """Ensure ZI-C's non-loss invariant cannot be silently violated.

        Gode & Sunder (1993) define ZI-C buyers as drawing from
        ``[market_floor, v_i]`` and sellers from ``[c_i, market_ceiling]``.
        Our ``zi.u_low`` / ``zi.u_high`` play the role of those market
        bounds. If ``u_low`` exceeds the smallest buyer reservation (or
        ``u_high`` sits below the largest seller reservation), the viable
        range inverts and ``_uniform`` silently returns the out-of-range
        bound — which for buyers means announcing above reservation, the
        exact invariant ZI-C is supposed to preserve. Fail early at config
        load instead.
        """

        def _uses_zi_c(strategies) -> bool:
            if isinstance(strategies, list):
                return "zi_c" in strategies
            return strategies == "zi_c"

        buyers = self.experiment.buyers
        sellers = self.experiment.sellers

        if _uses_zi_c(buyers.strategies) and self.zi.u_low > buyers.min:
            raise ValueError(
                f"ZI-C buyers: zi.u_low ({self.zi.u_low}) exceeds "
                f"experiment.buyers.min ({buyers.min}); a buyer with "
                f"reservation below u_low would announce above its own "
                f"reservation, violating the non-loss constraint."
            )
        if _uses_zi_c(sellers.strategies) and self.zi.u_high < sellers.max:
            raise ValueError(
                f"ZI-C sellers: zi.u_high ({self.zi.u_high}) is below "
                f"experiment.sellers.max ({sellers.max}); a seller with "
                f"reservation above u_high would have an empty viable "
                f"range."
            )
        return self

    @model_validator(mode="after")
    def _require_max_ticks_for_cda(self):
        """max_ticks_per_round is mandatory for the continuous double auction.

        The CDA migration (improvement-rule, automatic crossing, tick-based
        periods) replaced the old iteration/response loop. Period length is
        now set by max_ticks_per_round directly; there is no sensible
        default since the right value depends on market size and desired
        trading density. Fail fast at config load rather than silently
        running with an implicit value.
        """
        if (
            self.experiment.auction_type == AuctionType.DOUBLE_AUCTION
            and self.experiment.max_ticks_per_round is None
        ):
            raise ValueError(
                "experiment.max_ticks_per_round is required for "
                "auction_type=double_auction. Set it explicitly in your "
                "config (e.g. max_ticks_per_round: 50)."
            )
        return self
