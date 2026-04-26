"""Cross-feature interaction tests.

Verifies that features from PRs #2–#5 compose correctly:
  - PR #2: Auction types (FPSB, SPSB, English, Dutch, All-Pay, Open-Outcry)
  - PR #3: Agent personas (PersonaConfig, {persona} placeholder)
  - PR #4: Market history summary (HistoryConfig, full/summary modes)
  - PR #5: Structured output (AnnouncementResponse, AcceptRejectResponse)
"""

import pytest
from unittest.mock import MagicMock

from market_simulation.config.schema import (
    ExperimentConfig,
    AgentPricesConfig,
    AuctionConfig,
    AuctionPromptConfig,
    AuctionType,
    BiddersConfig,
    HistoryConfig,
    PersonaConfig,
    PromptConfig,
    PromptTemplates,
    AgentPromptConfig,
    AgentKeywords,
    SimulationConfig,
)
from market_simulation.agents.factory import (
    create_agents,
    create_initial_state,
    create_bidders,
    create_auction_initial_state,
)
from market_simulation.graph.state import (
    AgentState,
    BidderState,
    MarketState,
    SealedBidState,
    EnglishAuctionState,
    DutchAuctionState,
)
from market_simulation.graph.auctions.base import (
    render_auction_prompt,
)
from market_simulation.graph.nodes.announce import (
    _render_announcement_prompt,
    make_announce_node,
)
from market_simulation.graph.history import (
    build_market_history_for_prompt,
    build_own_history_for_prompt,
)
from market_simulation.llm.response_schemas import (
    AnnouncementResponseWithReasoning,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


MAIN_TEMPLATE = (
    "You are a {role}. Reservation: {reservation_price}. "
    "Rounds: {N_ROUNDS}. Iters: {N_ITER}. "
    "Buyers: {N_BUYERS}. Sellers: {N_SELLERS}. "
    "{persona} "
    "Market: {market_history}. Own: {own_history}. "
    "Round {round}/{N_ROUNDS} Iter {iteration}/{N_ITER}. {action_prompt}"
)


def _make_prompt_config():
    return PromptConfig(
        general=PromptTemplates(
            main_template=MAIN_TEMPLATE,
            announcement_history_template=(
                "Round {round} iter {iteration}: {announcement_type} ${price:.2f} {outcome}.\n"
            ),
            response_history_template=(
                "Round {round} iter {iteration}: {outcome} {opposite_announcement_type} ${price:.2f}.\n"
            ),
        ),
        buyer=AgentPromptConfig(
            main_keywords=AgentKeywords(
                role="buyer", verb="buy", preference="lowest", condition="above"
            ),
            response_prompt="Sell at ${price:.2f}. Buy? yes/no.",
            announcement_prompt="Announce bid price as number.",
        ),
        seller=AgentPromptConfig(
            main_keywords=AgentKeywords(
                role="seller", verb="sell", preference="highest", condition="below"
            ),
            response_prompt="Buy at ${price:.2f}. Sell? yes/no.",
            announcement_prompt="Announce ask price as number.",
        ),
    )


def _make_market_state(
    agents,
    history_mode="full",
    own_history_mode="full",
    summary_last_n=3,
    market_history_text="",
    transactions=None,
    iteration_records=None,
):
    all_ids = [a["id"] for a in agents]
    return MarketState(
        round=1,
        iteration=1,
        max_rounds=2,
        max_iterations=3,
        simulation_id=1,
        agents=agents,
        active_agent_ids=all_ids,
        potential_responder_ids=[],
        current_responder_index=0,
        announcing_agent_id=None,
        announced_price=None,
        announcement_type=None,
        counterparty_agent_id=None,
        market_history_text=market_history_text,
        iteration_records=iteration_records or [],
        transactions=transactions or [],
        announcement_made=False,
        transaction_made=False,
        iteration_complete=False,
        round_complete=False,
        simulation_complete=False,
        tool_usage_log=[],
        last_error=None,
        constraint_violations=0,
        history_mode=history_mode,
        history_summary_last_n=summary_last_n,
        own_history_mode=own_history_mode,
    )


# ===========================================================================
# 1. Persona + Auction rendering
# ===========================================================================


class TestPersonaInAuctionPrompts:
    """Verify persona text is injected into auction prompt rendering."""

    def test_persona_renders_in_auction_template(self):
        bidder = BidderState(
            id=0, private_value=5.0, active=True,
            own_history_prompt="", own_history_data=[],
            persona="You are cautious and risk-averse.",
        )
        template = (
            "Bidder {bidder_id}, value={private_value}. "
            "{persona} "
            "R{round}/{max_rounds}. {action_prompt}"
        )
        state = {"round": 1, "max_rounds": 3, "market_history_text": ""}
        result = render_auction_prompt(
            template, bidder, state, {"action_prompt": "Bid now."}
        )
        assert "cautious and risk-averse" in result
        assert "Bidder 0" in result

    def test_empty_persona_produces_clean_output(self):
        bidder = BidderState(
            id=1, private_value=8.0, active=True,
            own_history_prompt="", own_history_data=[], persona="",
        )
        template = "Bidder {bidder_id}. {persona} {action_prompt}"
        state = {"round": 1, "max_rounds": 2, "market_history_text": ""}
        result = render_auction_prompt(
            template, bidder, state, {"action_prompt": "Go."}
        )
        assert "Bidder 1" in result
        assert "{persona}" not in result

    def test_persona_with_curly_braces_does_not_crash(self):
        bidder = BidderState(
            id=0, private_value=5.0, active=True,
            own_history_prompt="", own_history_data=[],
            persona="Style: {aggressive}, strategy: {contrarian}",
        )
        template = "Bidder {bidder_id}. {persona} {action_prompt}"
        state = {"round": 1, "max_rounds": 2, "market_history_text": ""}
        result = render_auction_prompt(
            template, bidder, state, {"action_prompt": "Go."}
        )
        assert "{aggressive}" in result  # Curly braces preserved literally

    def test_persona_with_template_like_content_not_substituted(self):
        """Persona text like '{round}' should NOT get substituted as round number."""
        bidder = BidderState(
            id=0, private_value=5.0, active=True,
            own_history_prompt="", own_history_data=[],
            persona="In round {round} I should bid more.",
        )
        template = "R{round}. {persona} {action_prompt}"
        state = {"round": 3, "max_rounds": 5, "market_history_text": ""}
        result = render_auction_prompt(
            template, bidder, state, {"action_prompt": "Bid."}
        )
        # The template's {round} → "3", but persona's {round} should remain literal
        assert "R3." in result
        assert "In round {round} I should bid more." in result


class TestPersonaInBidderFactory:
    """Verify create_bidders() integrates PersonaConfig correctly."""

    def test_bidder_default_persona(self):
        config = AuctionConfig(bidders=BiddersConfig(num=3))
        personas = PersonaConfig(bidder_default="I am aggressive.")
        bidders = create_bidders(config, personas=personas)
        assert all(b["persona"] == "I am aggressive." for b in bidders)

    def test_per_bidder_persona_override(self):
        config = AuctionConfig(bidders=BiddersConfig(num=3))
        personas = PersonaConfig(
            bidder_default="Default persona.",
            bidders={1: "Override for bidder 1."},
        )
        bidders = create_bidders(config, personas=personas)
        assert bidders[0]["persona"] == "Default persona."
        assert bidders[1]["persona"] == "Override for bidder 1."
        assert bidders[2]["persona"] == "Default persona."

    def test_no_personas_gives_empty_string(self):
        config = AuctionConfig(bidders=BiddersConfig(num=2))
        bidders = create_bidders(config)
        assert all(b["persona"] == "" for b in bidders)

    def test_auction_initial_state_passes_personas(self):
        exp = ExperimentConfig(
            auction_type=AuctionType.FPSB,
            auction=AuctionConfig(bidders=BiddersConfig(num=3)),
        )
        personas = PersonaConfig(bidder_default="Bidder persona text.")
        state = create_auction_initial_state(
            exp, exp.auction, simulation_id=1, personas=personas,
        )
        for b in state["bidders"]:
            assert b["persona"] == "Bidder persona text."

    def test_buyer_seller_personas_dont_affect_bidders(self):
        """buyer_default/seller_default should NOT leak into auction bidders."""
        config = AuctionConfig(bidders=BiddersConfig(num=2))
        personas = PersonaConfig(
            buyer_default="I am a buyer.",
            seller_default="I am a seller.",
        )
        bidders = create_bidders(config, personas=personas)
        assert all(b["persona"] == "" for b in bidders)


# ===========================================================================
# 2. Persona + History summary in double-auction prompts
# ===========================================================================


class TestPersonaPlusHistorySummary:
    """Verify persona and history summary compose correctly in prompts."""

    def _make_agents_with_personas(self):
        return [
            AgentState(
                id=0, type="buyer", reservation_price=2.0, active=True,
                own_history_prompt="Announced $1.50, accepted.",
                own_history_data=[
                    {"action": "announce", "price": 1.5, "outcome": "accepted"},
                ],
                persona="You are aggressive.",
            ),
            AgentState(
                id=1, type="seller", reservation_price=1.0, active=True,
                own_history_prompt="Sold at $1.50.",
                own_history_data=[
                    {"action": "respond", "price": 1.5, "outcome": "accepted"},
                ],
                persona="You are cautious.",
            ),
        ]

    def test_persona_and_full_history_in_announcement(self):
        agents = self._make_agents_with_personas()
        state = _make_market_state(
            agents,
            history_mode="full",
            market_history_text="Round 1: trade at $1.50",
        )
        prompts = _make_prompt_config()
        result = _render_announcement_prompt(agents[0], state, prompts, prompts.buyer)
        assert "aggressive" in result
        assert "Round 1: trade at $1.50" in result

    def test_persona_and_summary_history_in_announcement(self):
        agents = self._make_agents_with_personas()
        transactions = [{"round": 1, "iteration": 1, "price": 1.5,
                         "buyer_id": 0, "seller_id": 1, "announcement_type": "buy"}]
        records = [{"round": 1, "iteration": 1, "price": 1.5,
                    "announcement_made": True, "transaction_made": True,
                    "announcement_type": "buy", "announcing_agent_id": 0,
                    "announcing_agent_reservation_price": 2.0,
                    "counterparty_agent_id": 1,
                    "counterparty_reservation_price": 1.0}]
        state = _make_market_state(
            agents,
            history_mode="summary",
            own_history_mode="summary",
            market_history_text="Round 1: trade at $1.50",
            transactions=transactions,
            iteration_records=records,
        )
        prompts = _make_prompt_config()
        result = _render_announcement_prompt(agents[0], state, prompts, prompts.buyer)
        assert "aggressive" in result
        # Summary mode should include statistics, not raw text
        assert "Completed transactions: 1" in result
        # Own history in summary mode should include counts
        assert "Total actions: 1" in result

# ===========================================================================
# 3. Persona + Structured output in double-auction
# ===========================================================================


class TestPersonaPlusCoT:
    """Verify persona and structured output compose in the double-auction announce/respond flow."""

    def test_announce_node_with_persona_and_structured_output(self):
        """Full integration: persona agent + structured output in announce node."""
        prompts = _make_prompt_config()
        mock_llm = MagicMock()
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(
            price=1.60, reasoning="My reservation is $2.00 and market is slow."
        )
        mock_llm.last_tool_log = []

        node = make_announce_node(mock_llm, prompts)
        agents = [
            AgentState(
                id=0, type="buyer", reservation_price=2.0, active=True,
                own_history_prompt="", own_history_data=[],
                persona="You are a cautious buyer.",
            ),
        ]
        state = _make_market_state(agents)
        state["announcing_agent_id"] = 0
        result = node(state, {})
        assert result["announced_price"] == 1.60
        assert result["announcement_made"] is True

# ===========================================================================
# 4. Structured output + History summary in double-auction
# ===========================================================================


class TestCoTPlusHistorySummary:
    """Verify structured output and history summary compose correctly."""

    def test_announce_with_structured_output_and_summary_history(self):
        """Announce node with structured output and summary history mode."""
        prompts = _make_prompt_config()
        mock_llm = MagicMock()
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(
            price=1.50, reasoning="Prices are stable."
        )
        mock_llm.last_tool_log = []

        node = make_announce_node(mock_llm, prompts)
        agents = [
            AgentState(
                id=0, type="buyer", reservation_price=2.0, active=True,
                own_history_prompt="", own_history_data=[], persona="",
            ),
        ]
        state = _make_market_state(agents, history_mode="summary")
        state["announcing_agent_id"] = 0
        result = node(state, {})
        assert result["announced_price"] == 1.50
        assert result["announcement_made"] is True


# ===========================================================================
# 5. All features: Persona + Structured output + History + Double-Auction
# ===========================================================================


class TestAllFeaturesDoubleAuction:
    """Verify all features compose correctly in a double-auction scenario."""

    def test_full_announce_with_all_features(self):
        """Announcement with persona, structured output, and summary history all enabled."""
        prompts = _make_prompt_config()
        mock_llm = MagicMock()
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(
            price=1.40,
            reasoning="As a cautious buyer, with only 1 transaction completed at $1.50, I should bid conservatively.",
        )
        mock_llm.last_tool_log = []

        node = make_announce_node(mock_llm, prompts)
        agents = [
            AgentState(
                id=0, type="buyer", reservation_price=2.0, active=True,
                own_history_prompt="Announced $1.50, accepted.",
                own_history_data=[
                    {"action": "announce", "price": 1.5, "outcome": "accepted"},
                ],
                persona="You are cautious and risk-averse.",
            ),
            AgentState(
                id=1, type="seller", reservation_price=1.0, active=True,
                own_history_prompt="", own_history_data=[], persona="",
            ),
        ]
        transactions = [{"round": 1, "iteration": 1, "price": 1.5,
                         "buyer_id": 0, "seller_id": 1, "announcement_type": "buy"}]
        records = [{"round": 1, "iteration": 1, "price": 1.5,
                    "announcement_made": True, "transaction_made": True,
                    "announcement_type": "buy", "announcing_agent_id": 0,
                    "announcing_agent_reservation_price": 2.0,
                    "counterparty_agent_id": 1,
                    "counterparty_reservation_price": 1.0}]
        state = _make_market_state(
            agents,
            history_mode="summary",
            own_history_mode="summary",
            market_history_text="Round 1: trade at $1.50",
            transactions=transactions,
            iteration_records=records,
        )
        state["announcing_agent_id"] = 0
        result = node(state, {})
        assert result["announced_price"] == 1.40
        assert result["announcement_made"] is True

# ===========================================================================
# 6. Config cross-compatibility
# ===========================================================================


class TestConfigCrossCompatibility:
    """Verify SimulationConfig accepts all feature combinations."""

    def test_all_features_enabled_config(self):
        cfg = SimulationConfig(
            experiment=ExperimentConfig(
                max_ticks_per_round=50,
                history=HistoryConfig(mode="summary", own_history_mode="summary"),
            ),
            personas=PersonaConfig(
                buyer_default="Aggressive buyer.",
                seller_default="Cautious seller.",
            ),
        )
        assert cfg.experiment.history.mode == "summary"
        assert cfg.personas.buyer_default == "Aggressive buyer."

    def test_auction_config_with_personas(self):
        cfg = SimulationConfig(
            experiment=ExperimentConfig(
                auction_type=AuctionType.ENGLISH,
                auction=AuctionConfig(
                    n_rounds=5,
                    bidders=BiddersConfig(num=4),
                ),
            ),
            personas=PersonaConfig(
                bidder_default="Rational bidder.",
                bidders={0: "Aggressive bidder."},
            ),
        )
        assert cfg.experiment.auction_type == AuctionType.ENGLISH
        assert cfg.personas.bidder_default == "Rational bidder."

    def test_auction_config_with_personas_strategic(self):
        cfg = SimulationConfig(
            experiment=ExperimentConfig(
                auction_type=AuctionType.FPSB,
                auction=AuctionConfig(),
            ),
            personas=PersonaConfig(bidder_default="Strategic."),
        )
        assert cfg.personas.bidder_default == "Strategic."

    def test_double_auction_with_all_features_config(self):
        cfg = SimulationConfig(
            experiment=ExperimentConfig(
                auction_type=AuctionType.DOUBLE_AUCTION,
                max_ticks_per_round=50,
                history=HistoryConfig(
                    mode="summary",
                    own_history_mode="summary",
                    summary_last_n_events=5,
                ),
            ),
            personas=PersonaConfig(
                buyer_default="Aggressive.",
                seller_default="Conservative.",
                buyers={0: "Wild card."},
            ),
        )
        assert cfg.experiment.history.mode == "summary"
        assert cfg.personas.buyers[0] == "Wild card."


# ===========================================================================
# 7. Factory cross-compatibility
# ===========================================================================


class TestFactoryCrossCompatibility:
    """Verify factory functions handle all feature combinations."""

    def test_create_initial_state_with_all_double_auction_features(self):
        config = ExperimentConfig(
            n_rounds=3,
            max_ticks_per_round=20,
            buyers=AgentPricesConfig(min=1.0, max=2.0, num=2),
            sellers=AgentPricesConfig(min=1.0, max=2.0, num=2),
            history=HistoryConfig(mode="summary", own_history_mode="summary"),
        )
        personas = PersonaConfig(
            buyer_default="Aggressive buyer.",
            seller_default="Passive seller.",
            buyers={0: "Special buyer."},
        )
        state = create_initial_state(config, simulation_id=1, personas=personas)
        # History config applied
        assert state["history_mode"] == "summary"
        assert state["own_history_mode"] == "summary"
        # Personas applied
        assert state["agents"][0]["persona"] == "Special buyer."
        assert state["agents"][1]["persona"] == "Aggressive buyer."
        buyer_count = config.buyers.num
        assert state["agents"][buyer_count]["persona"] == "Passive seller."

    def test_create_auction_state_with_personas(self):
        exp = ExperimentConfig(
            auction_type=AuctionType.SPSB,
            auction=AuctionConfig(bidders=BiddersConfig(num=4)),
        )
        personas = PersonaConfig(
            bidder_default="Rational.",
            bidders={2: "Irrational."},
        )
        state = create_auction_initial_state(
            exp, exp.auction, simulation_id=1, personas=personas,
        )
        assert state["bidders"][0]["persona"] == "Rational."
        assert state["bidders"][2]["persona"] == "Irrational."
        assert state["bidders"][3]["persona"] == "Rational."

    def test_create_auction_state_all_types_accept_personas(self):
        """Verify all auction types can be initialized with personas."""
        auction_types = [
            AuctionType.FPSB,
            AuctionType.SPSB,
            AuctionType.ALL_PAY,
            AuctionType.ENGLISH,
            AuctionType.DUTCH,
            AuctionType.FIRST_PRICE_OPEN_OUTCRY,
        ]
        personas = PersonaConfig(bidder_default="Test persona.")
        for at in auction_types:
            exp = ExperimentConfig(
                auction_type=at,
                auction=AuctionConfig(bidders=BiddersConfig(num=3)),
            )
            state = create_auction_initial_state(
                exp, exp.auction, simulation_id=1, personas=personas,
            )
            assert all(
                b["persona"] == "Test persona." for b in state["bidders"]
            ), f"Persona not set for auction type {at.value}"


# ===========================================================================
# 8. Auction prompt rendering with history and persona
# ===========================================================================


class TestAuctionPromptWithHistoryAndPersona:
    """Verify auction prompts render correctly with history text and persona."""

    def test_prompt_includes_market_history(self):
        bidder = BidderState(
            id=0, private_value=5.0, active=True,
            own_history_prompt="Won round 1 at $4.00.",
            own_history_data=[], persona="Strategic bidder.",
        )
        template = (
            "Bidder {bidder_id}. {persona} "
            "History: {market_history}. Own: {own_history}. {action_prompt}"
        )
        state = {
            "round": 2,
            "max_rounds": 5,
            "market_history_text": "Round 1: Bidder 2 won at $6.00.",
        }
        result = render_auction_prompt(
            template, bidder, state, {"action_prompt": "Bid now."}
        )
        assert "Strategic bidder." in result
        assert "Round 1: Bidder 2 won at $6.00." in result
        assert "Won round 1 at $4.00." in result

    def test_multiple_bidders_get_own_personas(self):
        """Each bidder should see their own persona in rendered prompts."""
        template = "Bidder {bidder_id}. {persona} {action_prompt}"
        state = {"round": 1, "max_rounds": 2, "market_history_text": ""}

        bidders = [
            BidderState(
                id=i, private_value=float(i * 3), active=True,
                own_history_prompt="", own_history_data=[],
                persona=f"Persona for bidder {i}.",
            )
            for i in range(3)
        ]

        for b in bidders:
            result = render_auction_prompt(
                template, b, state, {"action_prompt": "Go."}
            )
            assert f"Persona for bidder {b['id']}." in result


# ===========================================================================
# 9. History summary edge cases with auction-like data
# ===========================================================================


class TestHistorySummaryEdgeCases:
    """History summary functions handle edge cases correctly."""

    def test_summary_with_no_transactions_no_records(self):
        state = _make_market_state(
            [AgentState(
                id=0, type="buyer", reservation_price=2.0, active=True,
                own_history_prompt="", own_history_data=[], persona="",
            )],
            history_mode="summary",
        )
        result = build_market_history_for_prompt(state, mode="summary")
        assert "No market history" in result or "first iteration" in result.lower()

    def test_own_history_summary_with_empty_data(self):
        agent = AgentState(
            id=0, type="buyer", reservation_price=2.0, active=True,
            own_history_prompt="", own_history_data=[], persona="",
        )
        result = build_own_history_for_prompt(agent, mode="summary")
        assert "No actions" in result

    def test_own_history_summary_with_data(self):
        agent = AgentState(
            id=0, type="buyer", reservation_price=2.0, active=True,
            own_history_prompt="Announced $1.50.",
            own_history_data=[
                {"action": "announce", "price": 1.5, "outcome": "accepted"},
                {"action": "respond", "price": 1.8, "outcome": "rejected"},
            ],
            persona="",
        )
        result = build_own_history_for_prompt(agent, mode="summary")
        assert "Total actions: 2" in result
        assert "Successful trades: 1" in result
