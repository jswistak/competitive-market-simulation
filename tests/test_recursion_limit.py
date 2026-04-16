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
# Helper: replicate the double-auction formula from main.py
# ---------------------------------------------------------------------------

def _double_auction_recursion_limit(n_rounds, n_iterations, n_buyers, n_sellers):
    """Replicate the worst-case formula used in main.py for double auction."""
    N = n_buyers + n_sellers
    R = max(n_buyers, n_sellers)
    nodes_per_iteration = N * (3 + 3 * R) + 7
    nodes_per_round = n_iterations * nodes_per_iteration + 2
    return n_rounds * nodes_per_round


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
            experiment=ExperimentConfig(recursion_limit=99999),
        )
        assert cfg.experiment.recursion_limit == 99999

    def test_absent_from_yaml_defaults_to_none(self, tmp_path):
        """A YAML without recursion_limit should parse with None."""
        data = {
            "experiment": {"n_rounds": 2, "n_iterations": 3, "n_simulations": 1},
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
                "n_iterations": 3,
                "n_simulations": 1,
                "recursion_limit": 75000,
            },
            "llm": {"provider": "openai"},
        }
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump(data))
        cfg = load_config(f)
        assert cfg.experiment.recursion_limit == 75000


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


class TestDoubleAuctionFormula:
    """Tests that the recursion limit formula produces sufficient values."""

    def test_small_market(self):
        """3 buyers + 3 sellers (conftest default) should give a reasonable limit."""
        limit = _double_auction_recursion_limit(
            n_rounds=2, n_iterations=3, n_buyers=3, n_sellers=3,
        )
        # N=6, R=3, per_iter = 6*(3+9)+7 = 79, per_round = 3*79+2 = 239
        # total = 2*239 = 478
        assert limit == 478

    def test_smith1_market(self):
        """smith1.yaml: 11 buyers, 11 sellers, 5 rounds, 11 iterations."""
        limit = _double_auction_recursion_limit(
            n_rounds=5, n_iterations=11, n_buyers=11, n_sellers=11,
        )
        # N=22, R=11, per_iter = 22*(3+33)+7 = 799
        # per_round = 11*799+2 = 8791, total = 5*8791 = 43955
        assert limit == 43955
        assert _clamp(limit) == 43955  # within 500k ceiling

    def test_smith6a_market(self):
        """smith6a.yaml: 17 buyers, 12 sellers, 4 rounds, 12 iterations."""
        limit = _double_auction_recursion_limit(
            n_rounds=4, n_iterations=12, n_buyers=17, n_sellers=12,
        )
        # N=29, R=17, per_iter = 29*(3+51)+7 = 1573
        # per_round = 12*1573+2 = 18878, total = 4*18878 = 75512
        assert limit == 75512
        assert _clamp(limit) == 75512

    def test_smith3_market(self):
        """smith3.yaml: 21 buyers, 22 sellers, 4 rounds, 21 iterations."""
        limit = _double_auction_recursion_limit(
            n_rounds=4, n_iterations=21, n_buyers=21, n_sellers=22,
        )
        # N=43, R=22, per_iter = 43*(3+66)+7 = 2974+7 = 2974 -- wait
        # Actually 43*(3+3*22)+7 = 43*69+7 = 2967+7 = 2974
        assert limit > 200_000
        assert _clamp(limit) <= 500_000

    def test_openai_default_market(self):
        """openai.yaml: 11 buyers, 11 sellers, 5 rounds, 10 iterations."""
        limit = _double_auction_recursion_limit(
            n_rounds=5, n_iterations=10, n_buyers=11, n_sellers=11,
        )
        assert limit > 15000  # old ceiling was 15k -- must exceed it

    def test_exceeds_old_15k_ceiling_for_large_markets(self):
        """Any market with 11+ agents per side should exceed the old 15k ceiling."""
        limit = _double_auction_recursion_limit(
            n_rounds=5, n_iterations=10, n_buyers=11, n_sellers=11,
        )
        assert limit > 15000

    def test_single_agent_each_side(self):
        """Edge case: 1 buyer + 1 seller."""
        limit = _double_auction_recursion_limit(
            n_rounds=1, n_iterations=1, n_buyers=1, n_sellers=1,
        )
        # N=2, R=1, per_iter = 2*(3+3)+7 = 19
        # per_round = 1*19+2 = 21, total = 1*21 = 21
        assert limit == 21
        assert _clamp(limit) == 100  # clamped to floor


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


# ---------------------------------------------------------------------------
# Real config files (integration)
# ---------------------------------------------------------------------------


class TestRealConfigRecursionLimits:
    """Verify that real config files produce recursion limits that
    exceed the old 15k ceiling where needed."""

    @pytest.fixture
    def configs_dir(self):
        from pathlib import Path
        return Path(__file__).parent.parent / "configs"

    @pytest.mark.parametrize("config_name,min_expected", [
        ("smith1", 40_000),
        ("smith6a", 70_000),
        ("smith3", 200_000),
    ])
    def test_smith_configs_sufficient(self, configs_dir, config_name, min_expected):
        cfg_file = configs_dir / f"{config_name}.yaml"
        if not cfg_file.exists():
            pytest.skip(f"{config_name}.yaml not found")
        cfg = load_config(cfg_file)
        limit = _double_auction_recursion_limit(
            n_rounds=cfg.experiment.n_rounds,
            n_iterations=cfg.experiment.n_iterations,
            n_buyers=cfg.experiment.buyers.num,
            n_sellers=cfg.experiment.sellers.num,
        )
        assert limit >= min_expected, (
            f"{config_name}: calculated {limit}, expected >= {min_expected}"
        )
        assert _clamp(limit) <= 500_000
