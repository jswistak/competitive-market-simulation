"""Tests for prompt-related bugs: ambiguous wording and hardcoded participant counts.

These tests verify:
1. Response history wording is unambiguous (no "an offer to buy/sell" phrasing)
2. All config files use {N_BUYERS}/{N_SELLERS} instead of hardcoded counts
3. control.py uses config templates (not hardcoded f-strings) for agent histories
"""

import pytest
from pathlib import Path

from market_simulation.config.schema import PromptConfig, PromptTemplates
from market_simulation.config.settings import load_config
from market_simulation.graph.nodes.control import (
    make_update_history_node,
    _update_agent_histories,
)
from market_simulation.graph.nodes.announce import _render_announcement_prompt
from market_simulation.graph.nodes.respond import _render_response_prompt
from market_simulation.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# All yaml configs that ship with the project (excluding config_used.yaml which is generated)
SHIPPED_CONFIGS = sorted(
    p for p in CONFIGS_DIR.glob("*.yaml") if p.name != "config_used.yaml"
)


def _make_state_for_history_test(
    agents,
    announcing_id,
    responding_id,
    announced_price,
    announcement_type,
    response_accepted,
    transaction_made,
    current_responder_index=1,
    potential_responder_ids=None,
):
    """Create a MarketState dict for testing _update_agent_histories."""
    if potential_responder_ids is None:
        potential_responder_ids = [responding_id]
    return {
        "round": 1,
        "iteration": 5,
        "max_rounds": 5,
        "max_iterations": 10,
        "simulation_id": 1,
        "agents": agents,
        "active_agent_ids": [a["id"] for a in agents if a["active"]],
        "potential_responder_ids": potential_responder_ids,
        "current_responder_index": current_responder_index,
        "announcing_agent_id": announcing_id,
        "announced_price": announced_price,
        "announcement_type": announcement_type,
        "responding_agent_id": responding_id,
        "response_accepted": response_accepted,
        "market_history_text": "",
        "iteration_records": [],
        "transactions": [],
        "announced_this_iteration": [announcing_id],
        "announcement_made": True,
        "transaction_made": transaction_made,
        "iteration_complete": False,
        "round_complete": False,
        "simulation_complete": False,
        "tool_usage_log": [],
        "last_error": None,
        "constraint_violations": 0,
        "history_mode": "full",
        "history_summary_last_n": 3,
        "own_history_mode": "full",
    }


# ===========================================================================
# Bug 2: Response history wording ambiguity
# ===========================================================================


class TestResponseHistoryWording:
    """The phrase 'you rejected an offer to buy' is ambiguous for a seller.

    It could mean:
      (a) someone offered to buy from you and you rejected (intended)
      (b) you were offered a chance to buy and rejected (wrong reading)

    The fix changes 'an offer to buy' -> 'a buy offer' in both
    control.py and config templates.
    """

    def test_seller_rejected_history_no_ambiguous_phrasing(self, sample_agents):
        """Seller's history should not contain 'an offer to buy'."""
        seller = sample_agents[3]  # seller, id=3, reservation_price=1.0
        buyer = sample_agents[0]  # buyer, id=0, reservation_price=2.0

        state = _make_state_for_history_test(
            agents=sample_agents,
            announcing_id=buyer["id"],
            responding_id=seller["id"],
            announced_price=1.53,
            announcement_type="buy",
            response_accepted=False,
            transaction_made=False,
        )

        updated_agents = _update_agent_histories(state)
        seller_updated = next(a for a in updated_agents if a["id"] == seller["id"])

        assert "an offer to buy" not in seller_updated["own_history_prompt"]
        assert "buy offer" in seller_updated["own_history_prompt"]

    def test_buyer_rejected_history_no_ambiguous_phrasing(self, sample_agents):
        """Buyer's history should not contain 'an offer to sell'."""
        buyer = sample_agents[0]  # buyer, id=0
        seller = sample_agents[3]  # seller, id=3

        state = _make_state_for_history_test(
            agents=sample_agents,
            announcing_id=seller["id"],
            responding_id=buyer["id"],
            announced_price=2.50,
            announcement_type="sell",
            response_accepted=False,
            transaction_made=False,
        )

        updated_agents = _update_agent_histories(state)
        buyer_updated = next(a for a in updated_agents if a["id"] == buyer["id"])

        assert "an offer to sell" not in buyer_updated["own_history_prompt"]
        assert "sell offer" in buyer_updated["own_history_prompt"]

    def test_seller_accepted_history_no_ambiguous_phrasing(self, sample_agents):
        """Accepted transactions should also use unambiguous wording."""
        seller = sample_agents[3]
        buyer = sample_agents[0]

        state = _make_state_for_history_test(
            agents=sample_agents,
            announcing_id=buyer["id"],
            responding_id=seller["id"],
            announced_price=1.80,
            announcement_type="buy",
            response_accepted=True,
            transaction_made=True,
        )

        updated_agents = _update_agent_histories(state)
        seller_updated = next(a for a in updated_agents if a["id"] == seller["id"])

        assert "an offer to buy" not in seller_updated["own_history_prompt"]
        assert "buy offer" in seller_updated["own_history_prompt"]


# ===========================================================================
# Bug 2 continued: Config response_history_template
# ===========================================================================


class TestConfigResponseHistoryTemplate:
    """All config files should use unambiguous response history wording."""

    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    def test_response_history_template_unambiguous(self, config_path):
        config = load_config(config_path)
        template = config.prompts.general.response_history_template

        # Should NOT use "an offer to {type}" phrasing
        assert "an offer to" not in template, (
            f"{config_path.name}: response_history_template still uses "
            f"ambiguous 'an offer to' phrasing"
        )


# ===========================================================================
# Bug 4: Hardcoded participant count
# ===========================================================================


class TestDynamicParticipantCount:
    """Config templates should use {N_BUYERS}/{N_SELLERS} not hardcoded numbers."""

    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    def test_main_template_uses_dynamic_participant_count(self, config_path):
        config = load_config(config_path)

        # Auction configs use prompts.auction.system_template instead of main_template
        if config.experiment.auction_type.value != "double_auction":
            pytest.skip("Auction configs use system_template, not main_template")

        # Zero-intelligence configs don't render prompts at all.
        def _only_zi(strategies):
            if isinstance(strategies, list):
                return strategies and all(s != "llm" for s in strategies)
            return strategies != "llm"

        if _only_zi(config.experiment.buyers.strategies) and _only_zi(
            config.experiment.sellers.strategies
        ):
            pytest.skip("Pure ZI configs don't render prompts")

        template = config.prompts.general.main_template

        assert "{N_BUYERS}" in template, (
            f"{config_path.name}: main_template has hardcoded buyer count "
            f"instead of {{N_BUYERS}}"
        )
        assert "{N_SELLERS}" in template, (
            f"{config_path.name}: main_template has hardcoded seller count "
            f"instead of {{N_SELLERS}}"
        )

    def test_rendered_prompt_has_correct_participant_count(
        self, base_market_state, prompt_config
    ):
        """When N_BUYERS/N_SELLERS are in template, rendered prompt should show counts."""
        # Use a template that includes {N_BUYERS} and {N_SELLERS}
        prompt_config_with_counts = PromptConfig(
            general=PromptTemplates(
                main_template=(
                    "There are {N_BUYERS} buyers and {N_SELLERS} sellers. "
                    "You are a {role}. {verb} {preference} {condition}. "
                    "Reservation: {reservation_price}. "
                    "Rounds: {N_ROUNDS}. Iters: {N_ITER}. "
                    "Market: {market_history}. Own: {own_history}. "
                    "Round {round}/{N_ROUNDS} Iter {iteration}/{N_ITER}. "
                    "{action_prompt}"
                ),
            ),
            buyer=prompt_config.buyer,
            seller=prompt_config.seller,
        )

        agent = base_market_state["agents"][0]  # buyer
        prompt = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config_with_counts,
            agent_prompts=prompt_config_with_counts.buyer,
        )

        # base_market_state has 3 buyers and 3 sellers
        assert "There are 3 buyers and 3 sellers" in prompt
