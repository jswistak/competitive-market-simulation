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
from market_simulation.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# Configs that ship under the strict (extra='forbid') schema. Scoped
# to configs/final/ — the canonical home for new configs after the
# user's planned cleanup of legacy YAMLs in configs/*.yaml. Add
# subdirectories or specific files here as the cleanup progresses.
SHIPPED_CONFIGS = sorted(
    p for p in (CONFIGS_DIR / "final").glob("*.yaml")
    if p.name != "config_used.yaml"
)


def _make_state_for_history_test(
    agents,
    announcing_id,
    responding_id,
    announced_price,
    announcement_type,
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
        "counterparty_agent_id": responding_id,
        "market_history_text": "",
        "iteration_records": [],
        "transactions": [],
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


# Note: TestResponseHistoryWording once checked that responders' logged
# history used unambiguous wording ("buy offer" vs "an offer to buy").
# Under the improvement-rule CDA, responders no longer log a new action
# on a cross — the counterparty's standing order was logged when posted,
# so there is no per-tick responder history to render. The
# wording-of-the-config-template concern was previously covered by
# TestConfigResponseHistoryTemplate, deleted alongside the removal of
# PromptTemplates.response_history_template (the field had no live
# reader after PR #18 deleted the responder loop).


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
        _, prompt = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config_with_counts,
            agent_prompts=prompt_config_with_counts.buyer,
        )

        # base_market_state has 3 buyers and 3 sellers
        assert "There are 3 buyers and 3 sellers" in prompt


# ===========================================================================
# Bug 5: Round/iteration leak via hardcoded history f-strings
# ===========================================================================


HISTORY_TEMPLATE_FIELDS = (
    "announcement_history_template",
    "announcement_history_non_improving_template",
    "market_history_accepted_template",
    "market_history_posted_template",
    "market_history_non_improving_template",
    "market_history_no_announcement_template",
)


class TestHistoryTemplatesReachAgents:
    """Regression tests for the round/iteration leak.

    The bug was that control.py used hardcoded f-strings for both market-history
    and own-history entries, so removing {round}/{iteration} from the main_template
    in YAML did not actually hide the round/iteration from the agent — the history
    strings still leaked them. These tests lock in that every template field is
    honoured end-to-end.
    """

    def test_announcement_history_template_from_config_is_used(self, sample_agents):
        """Custom announcement_history_template must appear verbatim in the agent's prompt."""
        sentinel = "CUSTOM-ANNOUNCE-TEMPLATE {announcement_type} {price:.2f} {outcome}\n"
        templates = PromptTemplates(announcement_history_template=sentinel)
        prompts = PromptConfig(general=templates)

        buyer = sample_agents[0]
        seller = sample_agents[3]
        state = _make_state_for_history_test(
            agents=sample_agents,
            announcing_id=buyer["id"],
            responding_id=seller["id"],
            announced_price=1.73,
            announcement_type="buy",
            transaction_made=True,
        )

        updated_agents = _update_agent_histories(state, templates)
        buyer_updated = next(a for a in updated_agents if a["id"] == buyer["id"])

        assert (
            "CUSTOM-ANNOUNCE-TEMPLATE buy 1.73 accepted" in buyer_updated["own_history_prompt"]
        )
        # And the legacy hardcoded prefix must NOT appear.
        assert "In round" not in buyer_updated["own_history_prompt"]
        # Node wrapper must pass templates through as well.
        node_result = make_update_history_node(prompts)(state)
        node_buyer = next(a for a in node_result["agents"] if a["id"] == buyer["id"])
        assert (
            "CUSTOM-ANNOUNCE-TEMPLATE buy 1.73 accepted" in node_buyer["own_history_prompt"]
        )

    # response_history_template was removed from PromptTemplates after
    # PR #18 deleted the responder loop. No CDA path reads it, so the
    # corresponding test was deleted alongside the field.

    def test_market_history_accepted_template_from_config_is_used(self, base_market_state):
        sentinel = "MKT-ACCEPTED {announcement_type} {price:.2f}\n"
        prompts = PromptConfig(
            general=PromptTemplates(market_history_accepted_template=sentinel),
        )

        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 2.25,
            "announcement_type": "sell",
            "announcing_agent_id": 3,
            "counterparty_agent_id": 0,
            "current_responder_index": 1,
            "potential_responder_ids": [0],
        }
        result = make_update_history_node(prompts)(state)

        assert "MKT-ACCEPTED sell 2.25" in result["market_history_text"]
        assert "In round" not in result["market_history_text"]

    def test_market_history_posted_template_from_config_is_used(self, base_market_state):
        sentinel = "MKT-POSTED {announcement_type} {price:.2f}\n"
        prompts = PromptConfig(
            general=PromptTemplates(market_history_posted_template=sentinel),
        )

        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 0.90,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "counterparty_agent_id": 3,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
        }
        result = make_update_history_node(prompts)(state)

        assert "MKT-POSTED buy 0.90" in result["market_history_text"]
        assert "In round" not in result["market_history_text"]

    def test_market_history_no_announcement_template_from_config_is_used(self, base_market_state):
        sentinel = "MKT-SILENT\n"
        prompts = PromptConfig(
            general=PromptTemplates(market_history_no_announcement_template=sentinel),
        )

        state = {
            **base_market_state,
            "announcement_made": False,
            "transaction_made": False,
            "iteration_complete": True,
            "announced_price": None,
            "announcement_type": None,
            "announcing_agent_id": None,
            "counterparty_agent_id": None,
            "current_responder_index": 0,
            "potential_responder_ids": [],
        }
        result = make_update_history_node(prompts)(state)

        assert "MKT-SILENT" in result["market_history_text"]
        assert "In round" not in result["market_history_text"]

    def test_round_iteration_can_be_fully_stripped_via_config(self, base_market_state, sample_agents):
        """End-to-end: with all five templates stripped of {round}/{iteration},
        no agent-visible history string contains the words 'round' or 'iteration'."""
        round_free = PromptTemplates(
            announcement_history_template=(
                "Your offer to {announcement_type} for ${price:.2f} was {outcome}.\n"
            ),
            market_history_accepted_template=(
                "Announcement to {announcement_type} for ${price:.2f} was accepted.\n"
            ),
            market_history_posted_template=(
                "Announcement to {announcement_type} for ${price:.2f} got no takers.\n"
            ),
            market_history_no_announcement_template="No announcement was made.\n",
        )
        prompts = PromptConfig(general=round_free)
        node = make_update_history_node(prompts)

        buyer = sample_agents[0]
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": buyer["id"],
            "counterparty_agent_id": 3,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
        }
        result = node(state)

        assert "round" not in result["market_history_text"].lower()
        assert "iteration" not in result["market_history_text"].lower()
        for agent in result["agents"]:
            assert "round" not in agent["own_history_prompt"].lower()
            assert "iteration" not in agent["own_history_prompt"].lower()


# ===========================================================================
# Bug 5 continued: Config coverage for history templates
# ===========================================================================


class TestAllConfigsHaveHistoryTemplates:
    """Every LLM-driven double-auction config must render non-empty history entries.

    The schema now provides defaults, so a config that omits a template still
    renders something sensible — but we also want to assert every shipped config
    either sets the template explicitly or accepts the schema default. Either way,
    the effective string must be non-empty and contain the expected placeholders
    so history rendering can't silently produce an empty line or crash on a
    missing field.
    """

    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    @pytest.mark.parametrize("field", HISTORY_TEMPLATE_FIELDS)
    def test_history_template_is_non_empty(self, config_path, field):
        config = load_config(config_path)

        # Auctions and pure-ZI configs don't use the double-auction history pipeline.
        if config.experiment.auction_type.value != "double_auction":
            pytest.skip("Auction configs use a different prompt pipeline")

        def _only_zi(pricing):
            # `strategies` was added on a later branch; tolerate its absence.
            strategies = getattr(pricing, "strategies", "llm")
            if isinstance(strategies, list):
                return strategies and all(s != "llm" for s in strategies)
            return strategies != "llm"

        if _only_zi(config.experiment.buyers) and _only_zi(config.experiment.sellers):
            pytest.skip("Pure ZI configs don't render prompts")

        value = getattr(config.prompts.general, field)

        # market_history_no_announcement_template is allowed to be empty
        # by design: when an agent passes, the surrounding lines (which
        # carry their own iteration numbers) let the LLM infer the gap,
        # and the empty template materially reduces token cost on long
        # rounds where many ticks are pass-throughs.
        if field == "market_history_no_announcement_template":
            return

        assert value and value.strip(), (
            f"{config_path.name}: {field} is empty — would silently drop history entries"
        )

    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    def test_history_templates_render_without_keyerror(self, config_path):
        """Every history template must accept the canonical set of format keys."""
        config = load_config(config_path)
        if config.experiment.auction_type.value != "double_auction":
            pytest.skip("Auction configs use a different prompt pipeline")

        templates = config.prompts.general
        canonical = dict(
            round=1,
            iteration=1,
            announcement_type="buy",
            opposite_announcement_type="buy",
            price=1.23,
            outcome="accepted",
        )
        for field in HISTORY_TEMPLATE_FIELDS:
            template = getattr(templates, field)
            try:
                template.format(**canonical)
            except KeyError as exc:
                pytest.fail(
                    f"{config_path.name}: {field} references unsupported placeholder {exc}"
                )


# ===========================================================================
# Bug 6: tools_preamble was declared but unread
# ===========================================================================
#
# The YAML field `prompts.tools_preamble` existed in the schema and was set
# in every tool-augmented config, but nothing on the production path read
# it — researchers worked around the gap by duplicating the same text inside
# main_template. These tests lock in that `{tools_preamble}` is now a real
# placeholder available to main_template.


class TestToolsPreambleWiring:
    def test_tools_preamble_available_in_announcement_prompt(
        self, base_market_state, prompt_config
    ):
        """A {tools_preamble} placeholder in main_template must be filled
        from prompts.tools_preamble."""
        prompt_config_with_preamble = PromptConfig(
            general=PromptTemplates(
                main_template=(
                    "PREAMBLE={tools_preamble} "
                    "You are a {role}. {verb} {preference} {condition}. "
                    "Reservation: {reservation_price}. "
                    "Rounds: {N_ROUNDS}. Iters: {N_ITER}. "
                    "Buyers: {N_BUYERS}. Sellers: {N_SELLERS}. "
                    "Market: {market_history}. Own: {own_history}. "
                    "Round {round}/{N_ROUNDS} Iter {iteration}/{N_ITER}. "
                    "{action_prompt}"
                ),
            ),
            tools_preamble="USE-YOUR-TOOLS-SENTINEL",
            buyer=prompt_config.buyer,
            seller=prompt_config.seller,
        )

        agent = base_market_state["agents"][0]
        _, prompt = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config_with_preamble,
            agent_prompts=prompt_config_with_preamble.buyer,
        )

        assert "PREAMBLE=USE-YOUR-TOOLS-SENTINEL" in prompt

    def test_empty_preamble_renders_empty_string(
        self, base_market_state, prompt_config
    ):
        """Configs that don't set tools_preamble (schema default "") must
        still render main_templates containing {tools_preamble} without error."""
        prompt_config_no_preamble = PromptConfig(
            general=PromptTemplates(
                main_template=(
                    "[{tools_preamble}] "
                    "You are a {role}. {verb} {preference} {condition}. "
                    "Reservation: {reservation_price}. "
                    "Rounds: {N_ROUNDS}. Iters: {N_ITER}. "
                    "Buyers: {N_BUYERS}. Sellers: {N_SELLERS}. "
                    "Market: {market_history}. Own: {own_history}. "
                    "Round {round}/{N_ROUNDS} Iter {iteration}/{N_ITER}. "
                    "{action_prompt}"
                ),
            ),
            # tools_preamble left as default ""
            buyer=prompt_config.buyer,
            seller=prompt_config.seller,
        )

        agent = base_market_state["agents"][0]
        _, prompt = _render_announcement_prompt(
            agent=agent,
            state=base_market_state,
            prompts=prompt_config_no_preamble,
            agent_prompts=prompt_config_no_preamble.buyer,
        )
        assert "[]" in prompt  # empty placeholder rendered cleanly
