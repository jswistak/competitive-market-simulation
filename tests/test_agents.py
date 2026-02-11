"""Tests for agent creation and initial state factory."""

import pytest
import numpy as np

from market_simulation.agents.factory import create_agents, create_initial_state
from market_simulation.config.schema import ExperimentConfig, AgentPricesConfig


class TestCreateAgents:
    """Tests for create_agents factory function."""

    def test_creates_correct_number_of_agents(self, experiment_config):
        """Should create buyers.num + sellers.num agents total."""
        agents = create_agents(experiment_config)
        expected = experiment_config.buyers.num + experiment_config.sellers.num
        assert len(agents) == expected

    def test_buyers_come_first(self, experiment_config):
        """Buyers should have ids 0..n_buyers-1, sellers offset after."""
        agents = create_agents(experiment_config)
        n_buyers = experiment_config.buyers.num
        for i in range(n_buyers):
            assert agents[i]["type"] == "buyer"
            assert agents[i]["id"] == i
        for i in range(experiment_config.sellers.num):
            assert agents[n_buyers + i]["type"] == "seller"
            assert agents[n_buyers + i]["id"] == n_buyers + i

    def test_reservation_prices_are_linspace(self, experiment_config):
        """Reservation prices should span from min to max as np.linspace."""
        agents = create_agents(experiment_config)
        buyer_prices = [a["reservation_price"] for a in agents if a["type"] == "buyer"]
        expected = np.round(np.linspace(1.0, 2.0, 3), 2).tolist()
        assert buyer_prices == expected

    def test_agents_start_active(self, experiment_config):
        """All agents should start with active=True."""
        agents = create_agents(experiment_config)
        assert all(a["active"] for a in agents)

    def test_agents_start_with_empty_history(self, experiment_config):
        """All agents should start with empty history."""
        agents = create_agents(experiment_config)
        for a in agents:
            assert a["own_history_prompt"] == ""
            assert a["own_history_data"] == []

    def test_single_agent_per_side(self):
        """Should work with just one buyer and one seller."""
        cfg = ExperimentConfig(
            n_rounds=1,
            n_iterations=1,
            n_simulations=1,
            buyers=AgentPricesConfig(min=1.5, max=1.5, num=1),
            sellers=AgentPricesConfig(min=1.5, max=1.5, num=1),
        )
        agents = create_agents(cfg)
        assert len(agents) == 2
        assert agents[0]["type"] == "buyer"
        assert agents[1]["type"] == "seller"
        assert agents[0]["reservation_price"] == 1.5
        assert agents[1]["reservation_price"] == 1.5


class TestCreateInitialState:
    """Tests for create_initial_state factory."""

    def test_state_has_required_keys(self, experiment_config):
        """Initial state should contain all MarketState keys."""
        state = create_initial_state(experiment_config, simulation_id=1)
        required_keys = [
            "round", "iteration", "max_rounds", "max_iterations",
            "simulation_id", "agents", "active_agent_ids",
            "market_history_text", "transactions", "iteration_records",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_initial_round_and_iteration(self, experiment_config):
        """Initial state should start at round 1, iteration 1."""
        state = create_initial_state(experiment_config)
        assert state["round"] == 1
        assert state["iteration"] == 1

    def test_all_agents_in_active_ids(self, experiment_config):
        """All agent ids should be in active_agent_ids initially."""
        state = create_initial_state(experiment_config)
        agent_ids = {a["id"] for a in state["agents"]}
        active_ids = set(state["active_agent_ids"])
        assert agent_ids == active_ids

    def test_max_values_from_config(self, experiment_config):
        """max_rounds and max_iterations should match config."""
        state = create_initial_state(experiment_config)
        assert state["max_rounds"] == experiment_config.n_rounds
        assert state["max_iterations"] == experiment_config.n_iterations

    def test_simulation_id_stored(self, experiment_config):
        """simulation_id should be stored in state."""
        state = create_initial_state(experiment_config, simulation_id=42)
        assert state["simulation_id"] == 42

    def test_control_flags_start_false(self, experiment_config):
        """All control flow flags should be False initially."""
        state = create_initial_state(experiment_config)
        assert state["announcement_made"] is False
        assert state["transaction_made"] is False
        assert state["iteration_complete"] is False
        assert state["round_complete"] is False
        assert state["simulation_complete"] is False

    def test_histories_start_empty(self, experiment_config):
        """Market history and records should be empty initially."""
        state = create_initial_state(experiment_config)
        assert state["market_history_text"] == ""
        assert state["iteration_records"] == []
        assert state["transactions"] == []
