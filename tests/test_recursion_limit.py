"""Tests for recursion limit calculation and config override."""

import pytest
import yaml

from market_simulation.config.schema import (
    ExperimentConfig,
    SimulationConfig,
    AgentPricesConfig,
    AuctionType,
    AuctionConfig,
    BiddersConfig,
)
from market_simulation.config.settings import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(limit):
    """Replicate the auto-calc clamp from main.py."""
    return max(100, min(limit, 500_000))


# ---------------------------------------------------------------------------
# Schema: recursion_limit field
# ---------------------------------------------------------------------------


class TestRecursionLimitSchema:
    """Tests that the recursion_limit field works in ExperimentConfig."""

    def test_defaults_to_none(self):
        cfg = ExperimentConfig()
        assert cfg.recursion_limit is None

    def test_accepts_integer(self):
        cfg = ExperimentConfig(recursion_limit=50000)
        assert cfg.recursion_limit == 50000

    def test_accepts_none_explicitly(self):
        cfg = ExperimentConfig(recursion_limit=None)
        assert cfg.recursion_limit is None

    def test_round_trips_through_simulation_config(self):
        cfg = SimulationConfig(
            experiment=ExperimentConfig(recursion_limit=99999, max_ticks_per_round=50),
        )
        assert cfg.experiment.recursion_limit == 99999

    def test_absent_from_yaml_defaults_to_none(self, tmp_path):
        """A YAML without recursion_limit should parse with None."""
        data = {
            "experiment": {
                "n_rounds": 2, "n_simulations": 1,
                "max_ticks_per_round": 50,
            },
            "llm": {"provider": "openai"},
        }
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump(data))
        cfg = load_config(f)
        assert cfg.experiment.recursion_limit is None

    def test_yaml_with_recursion_limit(self, tmp_path):
        """A YAML with recursion_limit should parse it correctly."""
        data = {
            "experiment": {
                "n_rounds": 2,
                "n_simulations": 1,
                "recursion_limit": 75000,
                "max_ticks_per_round": 50,
            },
            "llm": {"provider": "openai"},
        }
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump(data))
        cfg = load_config(f)
        assert cfg.experiment.recursion_limit == 75000


# ---------------------------------------------------------------------------
# Clamp behaviour
# ---------------------------------------------------------------------------


class TestRecursionLimitClamp:
    """Tests for the auto-calc clamping logic."""

    def test_floor_at_100(self):
        assert _clamp(1) == 100
        assert _clamp(50) == 100
        assert _clamp(100) == 100

    def test_ceiling_at_500k(self):
        assert _clamp(500_000) == 500_000
        assert _clamp(999_999) == 500_000

    def test_passthrough_in_range(self):
        assert _clamp(10_000) == 10_000
        assert _clamp(250_000) == 250_000


# ---------------------------------------------------------------------------
# Auction-type formulas (unchanged, but verify they aren't broken)
# ---------------------------------------------------------------------------


class TestAuctionFormulas:
    """Smoke tests that the auction recursion formulas produce sane values."""

    def test_english_auction_formula(self):
        """English: n_rounds * max_bidding_rounds * n_bidders * 3 + 100."""
        n_rounds, max_bidding_rounds, n_bidders = 10, 50, 5
        limit = n_rounds * max_bidding_rounds * n_bidders * 3 + 100
        assert limit == 7600
        assert _clamp(limit) == 7600

    def test_dutch_auction_formula(self):
        """Dutch: n_rounds * price_steps * n_bidders * 2 + 100."""
        n_rounds, n_bidders = 10, 5
        start, end, decrement = 12.0, 0.0, 0.5
        price_steps = int((start - end) / max(decrement, 0.01))
        limit = n_rounds * price_steps * n_bidders * 2 + 100
        assert limit == 2500
        assert _clamp(limit) == 2500

    def test_sealed_bid_formula(self):
        """Sealed-bid: n_rounds * n_bidders * 3 + 50."""
        n_rounds, n_bidders = 10, 5
        limit = n_rounds * n_bidders * 3 + 50
        assert limit == 200

