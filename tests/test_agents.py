"""Tests for agent creation and initial state factory."""

import pytest
import numpy as np

from market_simulation.agents.factory import create_agents, create_initial_state
from market_simulation.config.schema import ExperimentConfig, AgentPricesConfig, PersonaConfig


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


class TestPersonaAssignment:
    """Tests for per-agent persona customization."""

    def test_agents_have_empty_persona_by_default(self, experiment_config):
        """Without PersonaConfig, all agents should have empty persona."""
        agents = create_agents(experiment_config)
        assert all(a["persona"] == "" for a in agents)

    def test_role_default_persona_applied(self, experiment_config):
        """Role-level default persona should apply to all agents of that role."""
        personas = PersonaConfig(buyer_default="Aggressive buyer")
        agents = create_agents(experiment_config, personas)
        buyers = [a for a in agents if a["type"] == "buyer"]
        sellers = [a for a in agents if a["type"] == "seller"]
        assert all(b["persona"] == "Aggressive buyer" for b in buyers)
        assert all(s["persona"] == "" for s in sellers)

    def test_individual_persona_overrides_role_default(self, experiment_config):
        """Individual persona should override role-level default."""
        personas = PersonaConfig(
            buyer_default="Default buyer",
            buyers={0: "Special buyer"},
        )
        agents = create_agents(experiment_config, personas)
        buyers = [a for a in agents if a["type"] == "buyer"]
        assert buyers[0]["persona"] == "Special buyer"
        assert buyers[1]["persona"] == "Default buyer"
        assert buyers[2]["persona"] == "Default buyer"

    def test_seller_persona(self, experiment_config):
        """Seller personas should work the same way."""
        personas = PersonaConfig(
            seller_default="Patient seller",
            sellers={1: "Aggressive seller"},
        )
        agents = create_agents(experiment_config, personas)
        sellers = [a for a in agents if a["type"] == "seller"]
        assert sellers[0]["persona"] == "Patient seller"
        assert sellers[1]["persona"] == "Aggressive seller"
        assert sellers[2]["persona"] == "Patient seller"

    def test_personas_passed_through_create_initial_state(self, experiment_config):
        """create_initial_state should forward personas to create_agents."""
        personas = PersonaConfig(buyer_default="Test persona")
        state = create_initial_state(experiment_config, personas=personas)
        buyers = [a for a in state["agents"] if a["type"] == "buyer"]
        assert all(b["persona"] == "Test persona" for b in buyers)


class TestPersonaInRenderedPrompts:
    """Tests that persona text actually appears in rendered prompt output."""

    def test_persona_appears_in_announcement_prompt(self, prompt_config, base_market_state):
        """Persona text assigned to an agent should appear in the rendered announcement prompt."""
        from market_simulation.graph.nodes.announce import _render_announcement_prompt

        # Patch the main_template to include {persona} placeholder
        persona_template = prompt_config.general.main_template.replace(
            "{action_prompt}",
            "{persona} {action_prompt}",
        )
        prompt_config.general.main_template = persona_template

        # Set a distinctive persona on the buyer agent
        persona_text = "You are an aggressive buyer who always pushes for the lowest price."
        agent = base_market_state["agents"][0]  # buyer, id=0
        agent["persona"] = persona_text

        rendered = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config,
            agent_prompts=prompt_config.buyer,
        )

        assert persona_text in rendered

    def test_persona_appears_in_response_prompt(self, prompt_config, base_market_state):
        """Persona text assigned to an agent should appear in the rendered response prompt."""
        from market_simulation.graph.nodes.respond import _render_response_prompt

        # Patch the main_template to include {persona} placeholder
        persona_template = prompt_config.general.main_template.replace(
            "{action_prompt}",
            "{persona} {action_prompt}",
        )
        prompt_config.general.main_template = persona_template

        # Set a distinctive persona on the seller agent and set announced_price for response_prompt
        persona_text = "You are a patient seller who waits for the best offer."
        agent = base_market_state["agents"][3]  # seller, id=3
        agent["persona"] = persona_text
        base_market_state["announced_price"] = 1.50

        rendered = _render_response_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config,
            agent_prompts=prompt_config.seller,
        )

        assert persona_text in rendered

    def test_empty_persona_produces_no_artifact(self, prompt_config, base_market_state):
        """An agent with empty persona should not leave sentinel markers in the rendered prompt."""
        from market_simulation.graph.nodes.announce import _render_announcement_prompt

        # Patch the main_template to include {persona} placeholder
        persona_template = prompt_config.general.main_template.replace(
            "{action_prompt}",
            "{persona} {action_prompt}",
        )
        prompt_config.general.main_template = persona_template

        agent = base_market_state["agents"][0]
        agent["persona"] = ""

        rendered = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config,
            agent_prompts=prompt_config.buyer,
        )

        assert "<<PERSONA>>" not in rendered
        assert "{persona}" not in rendered

    def test_persona_with_curly_braces_not_mangled(self, prompt_config, base_market_state):
        """Persona text containing curly braces should survive rendering intact."""
        from market_simulation.graph.nodes.announce import _render_announcement_prompt

        # Patch the main_template to include {persona} placeholder
        persona_template = prompt_config.general.main_template.replace(
            "{action_prompt}",
            "{persona} {action_prompt}",
        )
        prompt_config.general.main_template = persona_template

        # Curly braces in persona text would break naive str.format()
        persona_text = "You must follow this rule: {always negotiate} and {never give up}."
        agent = base_market_state["agents"][0]
        agent["persona"] = persona_text

        rendered = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config,
            agent_prompts=prompt_config.buyer,
        )

        assert persona_text in rendered
