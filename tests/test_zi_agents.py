"""Tests for Zero-Intelligence (ZI) trader sampling and graph integration."""

import numpy as np
import pytest

from market_simulation.agents import zi as zi_decisions
from market_simulation.agents.factory import (
    _normalize_strategies,
    create_agents,
    create_bidders,
    create_initial_state,
)
from market_simulation.config.schema import (
    AgentPricesConfig,
    AuctionConfig,
    AuctionType,
    BiddersConfig,
    ExperimentConfig,
    PromptConfig,
    ZIConfig,
)
from market_simulation.graph.workflow import build_market_graph


# ---------------------------------------------------------------------------
# Strategy normalisation
# ---------------------------------------------------------------------------


class TestStrategyNormalisation:
    def test_single_string_expands(self):
        assert _normalize_strategies("zi_c", 4) == ["zi_c"] * 4

    def test_list_passthrough(self):
        got = _normalize_strategies(["llm", "zi_c", "zi_u"], 3)
        assert got == ["llm", "zi_c", "zi_u"]

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _normalize_strategies(["llm", "zi_c"], 3)

    def test_schema_validator_rejects_bad_list(self):
        with pytest.raises(Exception):
            AgentPricesConfig(num=3, strategies=["llm", "zi_c"])

    def test_factory_sets_per_agent_strategy(self):
        cfg = ExperimentConfig(
            buyers=AgentPricesConfig(min=1.0, max=2.0, num=3, strategies=["llm", "zi_c", "zi_u"]),
            sellers=AgentPricesConfig(min=1.0, max=2.0, num=2, strategies="zi_c"),
        )
        agents = create_agents(cfg)
        assert [a["strategy"] for a in agents if a["type"] == "buyer"] == ["llm", "zi_c", "zi_u"]
        assert all(a["strategy"] == "zi_c" for a in agents if a["type"] == "seller")

    def test_bidder_factory_sets_strategy(self):
        ac = AuctionConfig(
            bidders=BiddersConfig(num=3, value_min=1.0, value_max=3.0, strategies=["zi_c", "zi_u", "llm"]),
        )
        bidders = create_bidders(ac)
        assert [b["strategy"] for b in bidders] == ["zi_c", "zi_u", "llm"]


# ---------------------------------------------------------------------------
# Sampling correctness
# ---------------------------------------------------------------------------


def _buyer(reservation: float, strategy: str, agent_id: int = 0) -> dict:
    return {
        "id": agent_id,
        "type": "buyer",
        "reservation_price": reservation,
        "active": True,
        "own_history_prompt": "",
        "own_history_data": [],
        "persona": "",
        "strategy": strategy,
    }


def _seller(reservation: float, strategy: str, agent_id: int = 10) -> dict:
    return {
        "id": agent_id,
        "type": "seller",
        "reservation_price": reservation,
        "active": True,
        "own_history_prompt": "",
        "own_history_data": [],
        "persona": "",
        "strategy": strategy,
    }


def _bidder(value: float, strategy: str, bidder_id: int = 0) -> dict:
    return {
        "id": bidder_id,
        "private_value": value,
        "active": True,
        "own_history_prompt": "",
        "own_history_data": [],
        "persona": "",
        "strategy": strategy,
    }


N_DRAWS = 500


class TestZICAnnounce:
    def test_buyer_draw_never_exceeds_reservation(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        agent = _buyer(2.5, "zi_c")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_announce(agent, cfg, rng)
            assert resp.price is not None
            assert 0.0 <= resp.price <= 2.5 + 1e-9

    def test_seller_draw_never_below_reservation(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(u_high=10.0)
        agent = _seller(2.5, "zi_c")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_announce(agent, cfg, rng)
            assert resp.price is not None
            assert 2.5 - 1e-9 <= resp.price <= 10.0 + 1e-9


class TestZIUAnnounce:
    def test_price_in_fixed_interval_when_announcing(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(u_low=1.0, u_high=4.0, announce_prob=1.0)
        agent = _buyer(2.5, "zi_u")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_announce(agent, cfg, rng)
            assert resp.price is not None
            assert 1.0 - 1e-9 <= resp.price <= 4.0 + 1e-9

    def test_announce_prob_zero_never_announces(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(announce_prob=0.0)
        agent = _buyer(2.5, "zi_u")
        for _ in range(50):
            resp = zi_decisions.decide_announce(agent, cfg, rng)
            assert resp.price is None


class TestZICRespond:
    def test_buyer_accepts_iff_price_within_reservation(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        agent = _buyer(2.5, "zi_c")
        assert zi_decisions.decide_respond(agent, 2.0, cfg, rng).accept is True
        assert zi_decisions.decide_respond(agent, 2.5, cfg, rng).accept is True
        assert zi_decisions.decide_respond(agent, 2.6, cfg, rng).accept is False

    def test_seller_accepts_iff_price_at_or_above_reservation(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        agent = _seller(2.5, "zi_c")
        assert zi_decisions.decide_respond(agent, 2.6, cfg, rng).accept is True
        assert zi_decisions.decide_respond(agent, 2.5, cfg, rng).accept is True
        assert zi_decisions.decide_respond(agent, 2.4, cfg, rng).accept is False


class TestZIUResponse:
    def test_accept_prob_one_always_accepts(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(accept_prob=1.0)
        agent = _buyer(2.5, "zi_u")
        assert all(zi_decisions.decide_respond(agent, 99.0, cfg, rng).accept for _ in range(20))

    def test_accept_prob_zero_never_accepts(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(accept_prob=0.0)
        agent = _buyer(2.5, "zi_u")
        assert not any(zi_decisions.decide_respond(agent, 0.01, cfg, rng).accept for _ in range(20))


class TestZISealedBid:
    def test_zi_c_bid_never_exceeds_value(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        bidder = _bidder(5.0, "zi_c")
        for _ in range(N_DRAWS):
            assert 0.0 <= zi_decisions.decide_sealed_bid(bidder, cfg, rng).bid <= 5.0 + 1e-9

    def test_zi_u_bid_within_configured_interval(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(u_low=0.0, u_high=10.0)
        bidder = _bidder(5.0, "zi_u")
        for _ in range(N_DRAWS):
            bid = zi_decisions.decide_sealed_bid(bidder, cfg, rng).bid
            assert 0.0 <= bid <= 10.0 + 1e-9


class TestZIEnglish:
    def test_zi_c_passes_when_min_bid_exceeds_value(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        bidder = _bidder(3.0, "zi_c")
        resp = zi_decisions.decide_english(bidder, standing_bid=3.0, min_increment=0.5, zi_cfg=cfg, rng=rng)
        assert resp.action == "pass"

    def test_zi_c_bid_lies_between_min_bid_and_value(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        bidder = _bidder(5.0, "zi_c")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_english(bidder, standing_bid=1.0, min_increment=0.5, zi_cfg=cfg, rng=rng)
            assert resp.action == "bid"
            assert 1.5 - 1e-9 <= resp.bid <= 5.0 + 1e-9


class TestZIDutch:
    def test_zi_c_rejects_above_value(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        bidder = _bidder(3.0, "zi_c")
        assert zi_decisions.decide_dutch_accept(bidder, 4.0, cfg, rng).accept is False

    def test_zi_c_bernoulli_below_value(self):
        """Should sometimes accept and sometimes reject at an affordable price."""
        rng = np.random.default_rng(42)
        cfg = ZIConfig(accept_prob=0.5)
        bidder = _bidder(10.0, "zi_c")
        accepts = [zi_decisions.decide_dutch_accept(bidder, 1.0, cfg, rng).accept for _ in range(200)]
        # With accept_prob=0.5 and 200 draws, >10 of each outcome is near-certain.
        assert sum(accepts) > 10
        assert sum(accepts) < 190


# ---------------------------------------------------------------------------
# Determinism: same seed -> same outputs
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_reproduces_draws(self):
        cfg = ZIConfig()
        agent = _buyer(2.5, "zi_c")
        r1 = np.random.default_rng(7)
        r2 = np.random.default_rng(7)
        seq1 = [zi_decisions.decide_announce(agent, cfg, r1).price for _ in range(20)]
        seq2 = [zi_decisions.decide_announce(agent, cfg, r2).price for _ in range(20)]
        assert seq1 == seq2


# ---------------------------------------------------------------------------
# End-to-end smoke test: full double auction with all ZI-C agents
# ---------------------------------------------------------------------------


def _smith_experiment_with_strategy(strategy: str) -> ExperimentConfig:
    return ExperimentConfig(
        n_rounds=2,
        n_iterations=6,
        n_simulations=1,
        buyers=AgentPricesConfig(min=0.8, max=3.2, num=5, strategies=strategy),
        sellers=AgentPricesConfig(min=0.8, max=3.2, num=5, strategies=strategy),
        random_seed=42,
    )


class TestEndToEndDoubleAuction:
    def test_zi_c_full_run_produces_transactions(self):
        exp = _smith_experiment_with_strategy("zi_c")
        state = create_initial_state(exp, simulation_id=1)

        rng = np.random.default_rng(exp.random_seed)
        graph = build_market_graph(
            llm=None,
            prompts=PromptConfig(),
            zi_config=ZIConfig(u_low=0.0, u_high=4.0),
            rng=rng,
        )
        # Generous recursion limit — ZI fires fast, no real LLM calls.
        final_state = graph.invoke(state, config={"recursion_limit": 5000})

        assert final_state["round"] > exp.n_rounds or final_state.get("simulation_complete")
        assert len(final_state["transactions"]) > 0
        # ZI-C must never violate the reservation constraint.
        assert final_state["constraint_violations"] == 0
        # Every transaction price must sit between the two reservation prices.
        for t in final_state["transactions"]:
            buyer = next(a for a in final_state["agents"] if a["id"] == t["buyer_id"])
            seller = next(a for a in final_state["agents"] if a["id"] == t["seller_id"])
            assert t["price"] <= buyer["reservation_price"] + 1e-9
            assert t["price"] >= seller["reservation_price"] - 1e-9
