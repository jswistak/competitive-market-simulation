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
  Useful as a pure-noise baseline.
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
    """Sample uniform in [low, high]. If low >= high, return low (degenerate)."""
    if high <= low:
        return round(float(low), 2)
    return round(float(rng.uniform(low, high)), 2)


# ---------------------------------------------------------------------------
# Double-auction: announce
# ---------------------------------------------------------------------------


def decide_announce(
    agent: dict,
    zi_cfg: ZIConfig,
    rng: np.random.Generator,
    include_reasoning: bool = True,
) -> AnnouncementResponse:
    """Sample a ZI announcement decision for a double-auction agent.

    Returns a response with ``price=None`` when the agent chooses not to
    announce (ZI-U only — ZI-C always announces inside its viable range).
    """
    strategy = agent["strategy"]
    reservation = agent["reservation_price"]
    agent_type = agent["type"]

    schema_cls = (
        AnnouncementResponseWithReasoning if include_reasoning else AnnouncementResponse
    )

    if strategy == "zi_c":
        if agent_type == "buyer":
            low, high = zi_cfg.u_low, reservation
        else:
            low, high = reservation, zi_cfg.u_high
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
    """Sample a ZI English-auction bid-or-pass decision."""
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
    """Sample a ZI Dutch-auction accept/reject at the current price."""
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
        # Price is affordable — gate on accept_prob to avoid degenerate
        # "first rational bidder always wins" behaviour that makes the
        # Dutch auction collapse to a race.
        accept = bool(rng.random() < zi_cfg.accept_prob)
        return _mk(
            schema_cls,
            {"accept": accept},
            include_reasoning,
            f"ZI-C: price={current_price:.2f} ≤ value={value:.2f}, "
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
