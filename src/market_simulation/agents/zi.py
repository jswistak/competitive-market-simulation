"""Zero-Intelligence (ZI) trader decision functions.

Implements Gode & Sunder (1993) style traders as a drop-in alternative to
LLM-driven decisions. Each function takes the agent/bidder, current state,
an ``np.random.Generator`` and a ``ZIConfig``, and returns the same Pydantic
response object the LLM path would return — so the rest of the node logic
(constraint checks, state updates, history writes) stays unchanged.

Two variants are supported per agent:

* ``zi_c`` — constrained / non-loss. Samples only within the agent's viable
  range: buyers never bid above reservation value, sellers never ask below
  cost, and auction bidders never bid above private value. This reproduces
  the surprising Gode-Sunder result of near-100% allocative efficiency.
* ``zi_u`` — unconstrained uniform. Samples from ``[zi.u_low, zi.u_high]``
  ignoring reservation prices, and uses Bernoulli gates (``announce_prob``,
  ``accept_prob``, ``bid_prob``) for nodes where the agent can also opt out.
  Useful as a pure-noise baseline. ZI-U is unconstrained *with respect to
  private value*, not with respect to mechanism legality: e.g. English
  auction bids must still clear the standing improvement rule
  (``bid >= standing + min_increment``).
"""

from __future__ import annotations

import numpy as np

from ..config.schema import ZIConfig
from ..llm.response_schemas import (
    AcceptRejectResponse,
    AcceptRejectResponseWithReasoning,
    AnnouncementResponse,
    AnnouncementResponseWithReasoning,
    BidResponse,
    BidResponseWithReasoning,
    EnglishBidResponse,
    EnglishBidResponseWithReasoning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(schema_cls, payload: dict, include_reasoning: bool, reasoning: str):
    """Instantiate the response schema, conditionally including reasoning."""
    if include_reasoning:
        return schema_cls(**payload, reasoning=reasoning)
    return schema_cls(**payload)


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample uniform in [low, high]. If low >= high, return low (degenerate).

    Draws are rounded to 2 decimal places to match the precision of agent
    ``reservation_price`` / ``private_value`` (``np.round(..., 2)`` in
    ``agents/factory.py``). If that invariant changes — e.g. factories
    emit higher-precision reservations — the rounding here must be
    widened to preserve the ZI-C non-loss constraint.
    """
    if high <= low:
        return round(float(low), 2)
    return round(float(rng.uniform(low, high)), 2)


# ---------------------------------------------------------------------------
# Double-auction: announce
# ---------------------------------------------------------------------------


# Minimum price increment for the improvement rule. Matches the 2-decimal
# rounding in _uniform / factory reservation generation.
PRICE_INCREMENT = 0.01


def decide_announce(
    agent: dict,
    zi_cfg: ZIConfig,
    rng: np.random.Generator,
    include_reasoning: bool = True,
    standing_bid: float | None = None,
    standing_ask: float | None = None,
) -> AnnouncementResponse:
    """Sample a ZI announcement decision for an improvement-rule CDA.

    ZI-C draws in the intersection of the non-loss range and the
    mechanism-legal range:
      - buyer: ``[max(u_low, standing_bid + PRICE_INCREMENT), reservation]``
        — improve on the standing bid without bidding above value. A
        buyer MAY still cross the standing ask (that produces a trade);
        crossing is the apply_order node's responsibility, not the
        sampler's.
      - seller: ``[reservation, min(u_high, standing_ask - PRICE_INCREMENT)]``
        — undercut the standing ask without selling below cost.
    When the range is empty, the agent passes (price=None).

    ZI-U ignores the non-loss range entirely (samples in [u_low, u_high]).
    It does NOT self-filter on the improvement rule either — the
    mechanism discards non-improving ZI-U draws at the apply_order
    stage so "wasted draws" show up as no_announcement ticks, which is
    the paper's behaviour.
    """
    strategy = agent["strategy"]
    reservation = agent["reservation_price"]
    agent_type = agent["type"]

    schema_cls = (
        AnnouncementResponseWithReasoning if include_reasoning else AnnouncementResponse
    )

    if strategy == "zi_c":
        if agent_type == "buyer":
            low = zi_cfg.u_low
            if standing_bid is not None:
                low = max(low, standing_bid + PRICE_INCREMENT)
            high = reservation
        else:
            low = reservation
            high = zi_cfg.u_high
            if standing_ask is not None:
                high = min(high, standing_ask - PRICE_INCREMENT)

        # Round the bounds to the 2-decimal grid that prices live on.
        # Without this, floating-point error in `standing_ask - 0.01`
        # (stored as 1.9999999999999998 when ask=2.01) can falsely
        # empty a range that is genuinely non-empty on the price grid.
        low = round(low, 2)
        high = round(high, 2)

        # Empty improvement range — agent passes.
        if high < low:
            return _mk(
                schema_cls,
                {"price": None},
                include_reasoning,
                f"ZI-C: pass (empty range [{low:.2f}, {high:.2f}] "
                f"under improvement rule; standing_bid={standing_bid}, "
                f"standing_ask={standing_ask})",
            )

        price = _uniform(rng, low, high)
        return _mk(
            schema_cls,
            {"price": price},
            include_reasoning,
            f"ZI-C: uniform draw in [{low:.2f}, {high:.2f}] -> {price:.2f}",
        )

    # zi_u
    if rng.random() >= zi_cfg.announce_prob:
        return _mk(
            schema_cls,
            {"price": None},
            include_reasoning,
            f"ZI-U: skipped announcement (p={zi_cfg.announce_prob})",
        )
    price = _uniform(rng, zi_cfg.u_low, zi_cfg.u_high)
    return _mk(
        schema_cls,
        {"price": price},
        include_reasoning,
        f"ZI-U: uniform draw in [{zi_cfg.u_low:.2f}, {zi_cfg.u_high:.2f}] -> {price:.2f}",
    )


# ---------------------------------------------------------------------------
# Double-auction: respond
# ---------------------------------------------------------------------------


def decide_respond(
    responder: dict,
    announced_price: float,
    zi_cfg: ZIConfig,
    rng: np.random.Generator,
    include_reasoning: bool = True,
) -> AcceptRejectResponse:
    """Sample a ZI accept/reject decision for a double-auction responder."""
    strategy = responder["strategy"]
    schema_cls = (
        AcceptRejectResponseWithReasoning if include_reasoning else AcceptRejectResponse
    )

    if strategy == "zi_c":
        reservation = responder["reservation_price"]
        if responder["type"] == "buyer":
            accept = announced_price <= reservation
        else:
            accept = announced_price >= reservation
        reason = (
            f"ZI-C: {'accept' if accept else 'reject'} "
            f"(price={announced_price:.2f}, reservation={reservation:.2f})"
        )
        return _mk(schema_cls, {"accept": accept}, include_reasoning, reason)

    # zi_u
    accept = bool(rng.random() < zi_cfg.accept_prob)
    return _mk(
        schema_cls,
        {"accept": accept},
        include_reasoning,
        f"ZI-U: Bernoulli({zi_cfg.accept_prob}) -> {'accept' if accept else 'reject'}",
    )


# ---------------------------------------------------------------------------
# Sealed-bid auctions (FPSB / SPSB / All-Pay)
# ---------------------------------------------------------------------------


def decide_sealed_bid(
    bidder: dict,
    zi_cfg: ZIConfig,
    rng: np.random.Generator,
    include_reasoning: bool = True,
) -> BidResponse:
    """Sample a ZI sealed-bid amount."""
    strategy = bidder["strategy"]
    schema_cls = BidResponseWithReasoning if include_reasoning else BidResponse

    if strategy == "zi_c":
        value = bidder["private_value"]
        bid = _uniform(rng, 0.0, value)
        return _mk(
            schema_cls,
            {"bid": bid},
            include_reasoning,
            f"ZI-C: uniform draw in [0.00, {value:.2f}] -> {bid:.2f}",
        )

    # zi_u
    bid = _uniform(rng, zi_cfg.u_low, zi_cfg.u_high)
    return _mk(
        schema_cls,
        {"bid": bid},
        include_reasoning,
        f"ZI-U: uniform draw in [{zi_cfg.u_low:.2f}, {zi_cfg.u_high:.2f}] -> {bid:.2f}",
    )


# ---------------------------------------------------------------------------
# English / open-outcry: bid or pass
# ---------------------------------------------------------------------------


def decide_english(
    bidder: dict,
    standing_bid: float,
    min_increment: float,
    zi_cfg: ZIConfig,
    rng: np.random.Generator,
    include_reasoning: bool = True,
) -> EnglishBidResponse:
    """Sample a ZI English-auction bid-or-pass decision.

    The English auction's improvement rule (``bid >= standing + min_increment``)
    binds both strategies: ZI-U therefore clamps its uniform draw's lower
    bound to ``min_bid`` even though its upper bound still comes from
    ``zi_cfg.u_high``. This is mechanism-legality, not a value-rationality
    constraint — ZI-U still ignores ``private_value``.
    """
    strategy = bidder["strategy"]
    schema_cls = (
        EnglishBidResponseWithReasoning if include_reasoning else EnglishBidResponse
    )
    min_bid = standing_bid + min_increment

    if strategy == "zi_c":
        value = bidder["private_value"]
        if min_bid > value:
            return _mk(
                schema_cls,
                {"action": "pass", "bid": None},
                include_reasoning,
                f"ZI-C: pass (min_bid={min_bid:.2f} > value={value:.2f})",
            )
        bid = _uniform(rng, min_bid, value)
        return _mk(
            schema_cls,
            {"action": "bid", "bid": bid},
            include_reasoning,
            f"ZI-C: uniform draw in [{min_bid:.2f}, {value:.2f}] -> {bid:.2f}",
        )

    # zi_u
    if rng.random() >= zi_cfg.bid_prob:
        return _mk(
            schema_cls,
            {"action": "pass", "bid": None},
            include_reasoning,
            f"ZI-U: pass (Bernoulli {zi_cfg.bid_prob})",
        )
    upper = max(zi_cfg.u_high, min_bid)
    bid = _uniform(rng, min_bid, upper)
    return _mk(
        schema_cls,
        {"action": "bid", "bid": bid},
        include_reasoning,
        f"ZI-U: uniform draw in [{min_bid:.2f}, {upper:.2f}] -> {bid:.2f}",
    )


# ---------------------------------------------------------------------------
# Dutch: accept / reject at current price
# ---------------------------------------------------------------------------


def decide_dutch_accept(
    bidder: dict,
    current_price: float,
    zi_cfg: ZIConfig,
    rng: np.random.Generator,
    include_reasoning: bool = True,
) -> AcceptRejectResponse:
    """Sample a ZI Dutch-auction accept/reject at the current price.

    Dutch auctions are NOT covered by Gode & Sunder (1993), which is
    continuous-double-auction only. This function is a thesis-methodology
    extension of ZI-C to a clock mechanism. In the CDA, ZI randomness
    lives in the bid *level* (a uniform draw in the non-loss range); in a
    Dutch clock auction a bidder only says yes/no at a single price, so
    the analogous randomness must live in the *timing* of acceptance.
    ZI-C here therefore first enforces non-loss (reject iff
    ``current_price > value``) and then gates the accept on
    ``accept_prob`` — without the gate, ZI-C Dutch would degenerate to
    "highest-value bidder always wins at their own value," which is
    indistinguishable from rational play and defeats the purpose of the
    zero-intelligence baseline.
    """
    strategy = bidder["strategy"]
    schema_cls = (
        AcceptRejectResponseWithReasoning if include_reasoning else AcceptRejectResponse
    )

    if strategy == "zi_c":
        value = bidder["private_value"]
        if current_price > value:
            return _mk(
                schema_cls,
                {"accept": False},
                include_reasoning,
                f"ZI-C: reject (price={current_price:.2f} > value={value:.2f})",
            )
        # Price is affordable — gate on accept_prob so ZI-C preserves
        # trader-level randomness in acceptance timing. See docstring.
        accept = bool(rng.random() < zi_cfg.accept_prob)
        return _mk(
            schema_cls,
            {"accept": accept},
            include_reasoning,
            f"ZI-C: price={current_price:.2f} <= value={value:.2f}, "
            f"Bernoulli({zi_cfg.accept_prob}) -> {'accept' if accept else 'reject'}",
        )

    # zi_u
    accept = bool(rng.random() < zi_cfg.accept_prob)
    return _mk(
        schema_cls,
        {"accept": accept},
        include_reasoning,
        f"ZI-U: Bernoulli({zi_cfg.accept_prob}) -> {'accept' if accept else 'reject'}",
    )
