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

    def test_schema_validator_rejects_bad_list(self):
        # Length mismatch is enforced at config load, not in the helper.
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
        """ZI-C Dutch accepts stochastically when price is affordable.

        Gode & Sunder (1993) is CDA-only; Dutch ZI-C is a thesis extension.
        The Bernoulli gate preserves trader-level randomness in acceptance
        timing — without it, ZI-C Dutch collapses to a rational highest-
        value-wins model and loses the "zero intelligence" character.
        """
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
        max_ticks_per_round=40,
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


# ---------------------------------------------------------------------------
# Improvement-rule CDA — ZI-C sampling under the standing book
# ---------------------------------------------------------------------------


class TestZICAnnounceWithImprovementRule:
    def test_buyer_range_bounded_below_by_standing_bid(self):
        """A ZI-C buyer with a standing bid in the book must bid strictly
        above it (or pass if the improvement range is empty)."""
        from market_simulation.agents.zi import PRICE_INCREMENT
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        agent = _buyer(2.5, "zi_c")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_announce(
                agent, cfg, rng, standing_bid=1.50, standing_ask=None,
            )
            # Either posts above standing_bid or passes because of some other
            # tick's deadlock (not expected here since range is non-empty).
            assert resp.price is None or (
                resp.price >= 1.50 + PRICE_INCREMENT - 1e-9
                and resp.price <= 2.5 + 1e-9
            )

    def test_buyer_passes_when_standing_bid_at_reservation(self):
        """No improving bid is possible if standing_bid = reservation."""
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        agent = _buyer(2.5, "zi_c")
        resp = zi_decisions.decide_announce(
            agent, cfg, rng, standing_bid=2.50, standing_ask=None,
        )
        assert resp.price is None

    def test_seller_range_bounded_above_by_standing_ask(self):
        from market_simulation.agents.zi import PRICE_INCREMENT
        rng = np.random.default_rng(0)
        cfg = ZIConfig(u_high=5.0)
        agent = _seller(2.0, "zi_c")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_announce(
                agent, cfg, rng, standing_bid=None, standing_ask=3.50,
            )
            assert resp.price is None or (
                resp.price >= 2.0 - 1e-9
                and resp.price <= 3.50 - PRICE_INCREMENT + 1e-9
            )

    def test_seller_passes_when_standing_ask_below_reservation(self):
        rng = np.random.default_rng(0)
        cfg = ZIConfig(u_high=5.0)
        agent = _seller(2.0, "zi_c")
        # standing_ask undercut to the point where no improving non-loss
        # ask exists.
        resp = zi_decisions.decide_announce(
            agent, cfg, rng, standing_bid=None, standing_ask=1.50,
        )
        assert resp.price is None

    def test_no_standing_book_same_as_unconstrained_zi_c(self):
        """With no standing orders, ZI-C samples in its classic
        [u_low, reservation] buyer range (G&S non-loss only)."""
        rng = np.random.default_rng(0)
        cfg = ZIConfig()
        agent = _buyer(2.5, "zi_c")
        for _ in range(N_DRAWS):
            resp = zi_decisions.decide_announce(
                agent, cfg, rng, standing_bid=None, standing_ask=None,
            )
            assert resp.price is not None
            assert 0.0 <= resp.price <= 2.5 + 1e-9


class TestCDAOrderBookEndToEnd:
    """Integration checks on the improvement-rule CDA graph behaviour."""

    def _run(self, strategy: str):
        exp = _smith_experiment_with_strategy(strategy)
        state = create_initial_state(exp, simulation_id=1)
        rng = np.random.default_rng(exp.random_seed)
        graph = build_market_graph(
            llm=None,
            prompts=PromptConfig(),
            zi_config=ZIConfig(u_low=0.0, u_high=4.0),
            rng=rng,
        )
        return graph.invoke(state, config={"recursion_limit": 5000})

    def test_iteration_records_have_standing_book_columns(self):
        final_state = self._run("zi_c")
        recs = final_state["iteration_records"]
        assert len(recs) > 0
        # Every record carries the new standing_bid/standing_ask keys,
        # even when None.
        for r in recs:
            assert "standing_bid" in r
            assert "standing_ask" in r
            assert "order_outcome" in r

    def test_outcomes_cover_expected_tags(self):
        final_state = self._run("zi_c")
        outcomes = {r["order_outcome"] for r in final_state["iteration_records"]}
        # ZI-C should produce at least trades and posts over two rounds.
        assert "traded" in outcomes
        assert "posted" in outcomes

    def test_crossing_trade_price_equals_standing_price(self):
        """Every trade should execute at the earlier (standing) price
        on its side, which is the core improvement-rule convention.

        Operationally: for each trade, the tick's announced_price must
        be weakly more aggressive than the trade price (buy >= tprice,
        sell <= tprice). Exactly the cross pricing rule.
        """
        final_state = self._run("zi_c")
        records = {(r["round"], r["iteration"]): r for r in final_state["iteration_records"]}
        for tx in final_state["transactions"]:
            rec = records.get((tx["round"], tx["iteration"]))
            assert rec is not None, f"missing iteration record for {tx}"
            ann_price = rec["price"]
            if tx["announcement_type"] == "buy":
                assert ann_price >= tx["price"] - 1e-9
            else:
                assert ann_price <= tx["price"] + 1e-9


class TestConfigValidator:
    def test_cda_requires_max_ticks_per_round(self):
        from market_simulation.config.schema import SimulationConfig, ExperimentConfig
        # Default auction_type is double_auction — must fail without
        # max_ticks_per_round set.
        with pytest.raises(Exception) as exc_info:
            SimulationConfig(experiment=ExperimentConfig())
        assert "max_ticks_per_round" in str(exc_info.value)
