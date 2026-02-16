"""Comprehensive tests for all auction types.

Tests cover:
  - Base utilities (extract_bid, extract_yes_no, render_auction_prompt, get_bidder_by_id)
  - Config schema (AuctionType, AuctionConfig, BiddersConfig)
  - Agent factory (create_bidders, create_auction_initial_state)
  - Sealed-bid nodes + edges + full graph (FPSB, SPSB, All-Pay)
  - English auction nodes + edges + full graph
  - Dutch auction nodes + edges + full graph
  - Open-outcry graph (reuses English, different settlement)
"""

import re
import pytest
from unittest.mock import MagicMock

from market_simulation.config.schema import (
    AuctionType,
    AuctionConfig,
    BiddersConfig,
    AuctionPromptConfig,
    ExperimentConfig,
    SimulationConfig,
)
from market_simulation.graph.state import (
    BidderState,
    BidRecord,
    AuctionResult,
    SealedBidState,
    EnglishAuctionState,
    DutchAuctionState,
    OpenOutcryState,
)
from market_simulation.graph.auctions.base import (
    extract_bid,
    extract_yes_no,
    render_auction_prompt,
    get_bidder_by_id,
)
from market_simulation.agents.factory import (
    create_bidders,
    create_auction_initial_state,
)
from market_simulation.graph.auctions.sealed_bid.nodes import (
    make_collect_bid_node,
    make_determine_winner_node,
    make_update_sealed_history_node,
    make_next_sealed_round_node,
)
from market_simulation.graph.auctions.sealed_bid.edges import (
    route_after_collect_bid as sealed_route_collect,
    route_after_update_history as sealed_route_history,
)
from market_simulation.graph.auctions.english.nodes import (
    make_solicit_bid_node,
    make_check_auction_end_node,
    make_reset_cycle_node,
    make_settle_english_node,
    make_settle_open_outcry_node,
    make_update_english_history_node,
    make_next_english_round_node,
)
from market_simulation.graph.auctions.english.edges import (
    route_after_solicit_bid as english_route_solicit,
    route_after_check_end as english_route_check,
    route_after_update_history as english_route_history,
)
from market_simulation.graph.auctions.dutch.nodes import (
    make_announce_price_node,
    make_solicit_acceptance_node,
    make_check_dutch_end_node,
    make_lower_price_node,
    make_settle_dutch_node,
    make_update_dutch_history_node,
    make_next_dutch_round_node,
)
from market_simulation.graph.auctions.dutch.edges import (
    route_after_solicit_acceptance as dutch_route_solicit,
    route_after_check_dutch_end as dutch_route_check,
    route_after_update_history as dutch_route_history,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_bidders():
    """Three bidders with values 0, 5, 10."""
    return [
        BidderState(id=0, private_value=0.0, active=True, own_history_prompt="", own_history_data=[]),
        BidderState(id=1, private_value=5.0, active=True, own_history_prompt="", own_history_data=[]),
        BidderState(id=2, private_value=10.0, active=True, own_history_prompt="", own_history_data=[]),
    ]


@pytest.fixture
def auction_prompts():
    return AuctionPromptConfig(
        system_template=(
            "Bidder {bidder_id}, value={private_value}, R{round}/{max_rounds}. "
            "{value_explanation} {action_prompt}"
        ),
        bid_prompt="Submit your sealed bid.",
        english_bid_prompt="Bid >= ${min_bid} or pass. standing={standing_bid}",
        dutch_accept_prompt="Price is {current_price}. Accept? yes/no.",
        value_explanation="profit=value-payment",
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "5.00"
    llm.last_tool_log = []
    return llm


@pytest.fixture
def auction_config():
    return AuctionConfig(
        n_rounds=2,
        n_simulations=1,
        bidders=BiddersConfig(num=3, value_min=0.0, value_max=10.0),
        min_increment=0.5,
        max_bidding_rounds=50,
        dutch_start_price=12.0,
        dutch_decrement=1.0,
        dutch_min_price=0.0,
    )


def _make_sealed_state(bidders, auction_type="fpsb", round_num=1, max_rounds=2):
    return SealedBidState(
        round=round_num,
        max_rounds=max_rounds,
        simulation_id=1,
        auction_type=auction_type,
        bidders=bidders,
        current_bidder_index=0,
        all_bids_collected=False,
        bids=[],
        auction_results=[],
        all_bid_records=[],
        market_history_text="",
        tool_usage_log=[],
        parse_failures=0,
        constraint_violations=0,
    )


def _make_english_state(bidders, auction_type="english", round_num=1, max_rounds=2):
    return EnglishAuctionState(
        round=round_num,
        max_rounds=max_rounds,
        simulation_id=1,
        auction_type=auction_type,
        bidders=bidders,
        active_bidder_ids=[b["id"] for b in bidders],
        current_bidder_index=0,
        standing_bid=0.0,
        standing_bidder_id=None,
        min_increment=0.5,
        bids_this_cycle=0,
        bid_step=0,
        max_bidding_rounds=50,
        bids=[],
        auction_results=[],
        all_bid_records=[],
        market_history_text="",
        auction_ended=False,
        tool_usage_log=[],
        parse_failures=0,
        constraint_violations=0,
    )


def _make_dutch_state(bidders, round_num=1, max_rounds=2):
    return DutchAuctionState(
        round=round_num,
        max_rounds=max_rounds,
        simulation_id=1,
        auction_type="dutch",
        bidders=bidders,
        current_bidder_index=0,
        current_price=12.0,
        dutch_start_price=12.0,
        dutch_decrement=1.0,
        dutch_min_price=0.0,
        accepted=False,
        accepting_bidder_id=None,
        bids=[],
        auction_results=[],
        all_bid_records=[],
        market_history_text="",
        all_queried_at_price=False,
        tool_usage_log=[],
        parse_failures=0,
        constraint_violations=0,
    )


# ============================================================
# Base utilities
# ============================================================


class TestExtractBid:
    def test_plain_number(self):
        assert extract_bid("5.00") == 5.0

    def test_integer(self):
        assert extract_bid("7") == 7.0

    def test_dollar_prefix(self):
        assert extract_bid("$3.27") == 3.27

    def test_embedded_dollar(self):
        assert extract_bid("I bid $4.50 for this") == 4.50

    def test_bare_decimal(self):
        assert extract_bid("My bid is 2.75 dollars") == 2.75

    def test_comma_separated(self):
        assert extract_bid("1,000.50") == 1000.50

    def test_empty(self):
        assert extract_bid("") is None

    def test_none(self):
        assert extract_bid(None) is None

    def test_no_number(self):
        assert extract_bid("I pass") is None

    def test_whitespace(self):
        assert extract_bid("  3.50  ") == 3.50

    def test_zero(self):
        assert extract_bid("0") == 0.0

    def test_zero_decimal(self):
        assert extract_bid("0.00") == 0.0


class TestExtractYesNo:
    def test_yes(self):
        assert extract_yes_no("yes") is True

    def test_yes_period(self):
        assert extract_yes_no("yes.") is True

    def test_yes_uppercase(self):
        assert extract_yes_no("Yes") is True

    def test_yes_in_sentence(self):
        assert extract_yes_no("I say yes to this") is True

    def test_no(self):
        assert extract_yes_no("no") is False

    def test_no_sentence(self):
        assert extract_yes_no("No, I reject") is False

    def test_yesterday_not_yes(self):
        assert extract_yes_no("yesterday") is False

    def test_empty(self):
        assert extract_yes_no("") is False

    def test_none(self):
        assert extract_yes_no(None) is False

    def test_whitespace_only(self):
        assert extract_yes_no("   ") is False


class TestRenderAuctionPrompt:
    def test_basic_rendering(self, sample_bidders):
        template = "Bidder {bidder_id}, value={private_value}, R{round}/{max_rounds}."
        result = render_auction_prompt(
            template=template,
            bidder=sample_bidders[1],
            state={"round": 3, "max_rounds": 10, "market_history_text": ""},
        )
        assert "Bidder 1" in result
        assert "value=5.0" in result
        assert "R3/10" in result

    def test_extra_vars(self, sample_bidders):
        template = "{bidder_id}: {custom_field}"
        result = render_auction_prompt(
            template=template,
            bidder=sample_bidders[0],
            state={"round": 1, "max_rounds": 1, "market_history_text": ""},
            extra_vars={"custom_field": "hello"},
        )
        assert "hello" in result

    def test_history_included(self, sample_bidders):
        template = "History: {market_history}"
        result = render_auction_prompt(
            template=template,
            bidder=sample_bidders[0],
            state={"round": 1, "max_rounds": 1, "market_history_text": "Round 1: done."},
        )
        assert "Round 1: done." in result


class TestGetBidderById:
    def test_found(self, sample_bidders):
        b = get_bidder_by_id(sample_bidders, 1)
        assert b is not None
        assert b["private_value"] == 5.0

    def test_not_found(self, sample_bidders):
        assert get_bidder_by_id(sample_bidders, 99) is None

    def test_first_bidder(self, sample_bidders):
        b = get_bidder_by_id(sample_bidders, 0)
        assert b["private_value"] == 0.0


# ============================================================
# Config schema
# ============================================================


class TestAuctionType:
    def test_enum_values(self):
        assert AuctionType.FPSB.value == "fpsb"
        assert AuctionType.SPSB.value == "spsb"
        assert AuctionType.ALL_PAY.value == "all_pay"
        assert AuctionType.ENGLISH.value == "english"
        assert AuctionType.DUTCH.value == "dutch"
        assert AuctionType.FIRST_PRICE_OPEN_OUTCRY.value == "first_price_open_outcry"
        assert AuctionType.DOUBLE_AUCTION.value == "double_auction"

    def test_string_comparison(self):
        assert AuctionType.FPSB == "fpsb"
        assert AuctionType.DUTCH == "dutch"

    def test_default_is_double_auction(self):
        ec = ExperimentConfig()
        assert ec.auction_type == AuctionType.DOUBLE_AUCTION


class TestAuctionConfig:
    def test_defaults(self):
        ac = AuctionConfig()
        assert ac.n_rounds == 10
        assert ac.bidders.num == 5
        assert ac.min_increment == 0.5
        assert ac.dutch_start_price == 12.0

    def test_custom(self):
        ac = AuctionConfig(
            n_rounds=5,
            bidders=BiddersConfig(num=10, value_min=1.0, value_max=20.0, distribution="uniform"),
        )
        assert ac.bidders.num == 10
        assert ac.bidders.distribution == "uniform"


class TestBackwardCompatibility:
    def test_simulation_config_no_auction(self):
        sc = SimulationConfig()
        assert sc.experiment.auction_type == AuctionType.DOUBLE_AUCTION
        assert sc.experiment.auction is None
        assert sc.prompts.auction is None

    def test_simulation_config_with_auction(self):
        sc = SimulationConfig(
            experiment=ExperimentConfig(
                auction_type="fpsb",
                auction=AuctionConfig(n_rounds=5),
            ),
            prompts={"auction": AuctionPromptConfig(bid_prompt="Bid!")},
        )
        assert sc.experiment.auction_type == AuctionType.FPSB
        assert sc.experiment.auction.n_rounds == 5
        assert sc.prompts.auction.bid_prompt == "Bid!"


# ============================================================
# Agent factory
# ============================================================


class TestCreateBidders:
    def test_linspace_count(self, auction_config):
        bidders = create_bidders(auction_config)
        assert len(bidders) == 3

    def test_linspace_values(self, auction_config):
        bidders = create_bidders(auction_config)
        assert bidders[0]["private_value"] == 0.0
        assert bidders[1]["private_value"] == 5.0
        assert bidders[2]["private_value"] == 10.0

    def test_ids_sequential(self, auction_config):
        bidders = create_bidders(auction_config)
        assert [b["id"] for b in bidders] == [0, 1, 2]

    def test_all_active(self, auction_config):
        bidders = create_bidders(auction_config)
        assert all(b["active"] for b in bidders)

    def test_empty_histories(self, auction_config):
        bidders = create_bidders(auction_config)
        for b in bidders:
            assert b["own_history_prompt"] == ""
            assert b["own_history_data"] == []

    def test_uniform_distribution(self):
        ac = AuctionConfig(bidders=BiddersConfig(num=100, distribution="uniform", value_min=0, value_max=10))
        bidders = create_bidders(ac)
        assert len(bidders) == 100
        values = [b["private_value"] for b in bidders]
        assert min(values) >= 0.0
        assert max(values) <= 10.0


class TestCreateAuctionInitialState:
    def test_sealed_bid_state(self, auction_config):
        ec = ExperimentConfig(auction_type="fpsb", auction=auction_config)
        state = create_auction_initial_state(ec, auction_config)
        assert state["auction_type"] == "fpsb"
        assert state["round"] == 1
        assert state["max_rounds"] == 2
        assert len(state["bidders"]) == 3
        assert state["all_bids_collected"] is False
        assert state["bids"] == []

    def test_english_state(self, auction_config):
        ec = ExperimentConfig(auction_type="english", auction=auction_config)
        state = create_auction_initial_state(ec, auction_config)
        assert state["auction_type"] == "english"
        assert state["standing_bid"] == 0.0
        assert state["min_increment"] == 0.5
        assert len(state["active_bidder_ids"]) == 3

    def test_dutch_state(self, auction_config):
        ec = ExperimentConfig(auction_type="dutch", auction=auction_config)
        state = create_auction_initial_state(ec, auction_config)
        assert state["auction_type"] == "dutch"
        assert state["current_price"] == 12.0
        assert state["dutch_decrement"] == 1.0
        assert state["accepted"] is False

    def test_open_outcry_state(self, auction_config):
        ec = ExperimentConfig(auction_type="first_price_open_outcry", auction=auction_config)
        state = create_auction_initial_state(ec, auction_config)
        assert state["auction_type"] == "first_price_open_outcry"
        assert "standing_bid" in state  # Uses EnglishAuctionState

    def test_invalid_type_raises(self, auction_config):
        ec = ExperimentConfig(auction_type="double_auction", auction=auction_config)
        with pytest.raises(ValueError, match="Unsupported"):
            create_auction_initial_state(ec, auction_config)

    def test_simulation_id_propagated(self, auction_config):
        ec = ExperimentConfig(auction_type="fpsb", auction=auction_config)
        state = create_auction_initial_state(ec, auction_config, simulation_id=42)
        assert state["simulation_id"] == 42


# ============================================================
# Sealed-bid nodes
# ============================================================


class TestCollectBidNode:
    def test_collects_bid(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "3.00"
        state = _make_sealed_state(sample_bidders)
        node = make_collect_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert len(result["bids"]) == 1
        assert result["bids"][0]["bid_amount"] == 3.0
        assert result["current_bidder_index"] == 1

    def test_all_collected_flag(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "5.00"
        state = _make_sealed_state(sample_bidders)
        state["current_bidder_index"] = 2  # Last bidder
        state["bids"] = [
            BidRecord(bidder_id=0, bid_amount=1.0, round=1, bid_step=0, private_value=0.0),
            BidRecord(bidder_id=1, bid_amount=3.0, round=1, bid_step=0, private_value=5.0),
        ]
        node = make_collect_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["all_bids_collected"] is True

    def test_parse_failure_records_zero(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "I don't want to bid"
        state = _make_sealed_state(sample_bidders)
        node = make_collect_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["bids"][0]["bid_amount"] == 0.0
        assert result["parse_failures"] == 1

    def test_constraint_violation_logged(self, sample_bidders, auction_prompts, mock_llm):
        # Bidder 0 has value 0.0, bidding 5.0 is a violation
        mock_llm.invoke.return_value = "5.00"
        state = _make_sealed_state(sample_bidders)
        node = make_collect_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["constraint_violations"] == 1

    def test_no_violation_within_value(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "4.00"
        state = _make_sealed_state(sample_bidders)
        state["current_bidder_index"] = 1  # Bidder with value 5.0
        node = make_collect_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result.get("constraint_violations", 0) == 0


class TestDetermineWinnerNode:
    def _make_bids(self):
        return [
            BidRecord(bidder_id=0, bid_amount=2.0, round=1, bid_step=0, private_value=0.0),
            BidRecord(bidder_id=1, bid_amount=4.0, round=1, bid_step=0, private_value=5.0),
            BidRecord(bidder_id=2, bid_amount=7.0, round=1, bid_step=0, private_value=10.0),
        ]

    def test_fpsb_winner_pays_own_bid(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "fpsb")
        state["bids"] = self._make_bids()
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["winning_bid"] == 7.0
        assert ar["payment"] == 7.0  # first-price

    def test_spsb_winner_pays_second(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "spsb")
        state["bids"] = self._make_bids()
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["payment"] == 4.0  # second-price
        assert ar["second_highest_bid"] == 4.0

    def test_allpay_winner_pays_own(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "all_pay")
        state["bids"] = self._make_bids()
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["payment"] == 7.0

    def test_surplus_calculation(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "spsb")
        state["bids"] = self._make_bids()
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        # Winner value=10.0, payment=4.0 (second price)
        assert ar["surplus"] == 6.0

    def test_no_bids(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "fpsb")
        state["bids"] = []
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] is None

    def test_single_bidder(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "fpsb")
        state["bids"] = [
            BidRecord(bidder_id=2, bid_amount=5.0, round=1, bid_step=0, private_value=10.0),
        ]
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["second_highest_bid"] is None

    def test_spsb_single_bidder_pays_own(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "spsb")
        state["bids"] = [
            BidRecord(bidder_id=2, bid_amount=5.0, round=1, bid_step=0, private_value=10.0),
        ]
        node = make_determine_winner_node()
        result = node(state)
        ar = result["auction_results"][0]
        # No second bidder, so payment = own bid
        assert ar["payment"] == 5.0

    def test_all_bids_recorded(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "fpsb")
        state["bids"] = self._make_bids()
        node = make_determine_winner_node()
        result = node(state)
        assert len(result["all_bid_records"]) == 3


class TestUpdateSealedHistoryNode:
    def test_updates_market_history(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "fpsb")
        state["bids"] = [
            BidRecord(bidder_id=2, bid_amount=7.0, round=1, bid_step=0, private_value=10.0),
        ]
        state["auction_results"] = [
            AuctionResult(
                round=1, auction_type="fpsb", winner_id=2, winning_bid=7.0,
                payment=7.0, second_highest_bid=None, all_bids=[], n_active_bidders=3, surplus=3.0,
            )
        ]
        node = make_update_sealed_history_node()
        result = node(state)
        assert "Bidder 2 won" in result["market_history_text"]
        assert "$7.00" in result["market_history_text"]

    def test_updates_bidder_histories(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, "fpsb")
        state["bids"] = [
            BidRecord(bidder_id=0, bid_amount=1.0, round=1, bid_step=0, private_value=0.0),
            BidRecord(bidder_id=2, bid_amount=7.0, round=1, bid_step=0, private_value=10.0),
        ]
        state["auction_results"] = [
            AuctionResult(
                round=1, auction_type="fpsb", winner_id=2, winning_bid=7.0,
                payment=7.0, second_highest_bid=1.0, all_bids=[], n_active_bidders=3, surplus=3.0,
            )
        ]
        node = make_update_sealed_history_node()
        result = node(state)
        # Winner's history
        winner = [b for b in result["bidders"] if b["id"] == 2][0]
        assert "won" in winner["own_history_prompt"]
        # Loser's history
        loser = [b for b in result["bidders"] if b["id"] == 0][0]
        assert "lost" in loser["own_history_prompt"]


class TestNextSealedRoundNode:
    def test_advances_round(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, round_num=1, max_rounds=3)
        node = make_next_sealed_round_node()
        result = node(state)
        assert result["round"] == 2
        assert result["bids"] == []
        assert result["current_bidder_index"] == 0
        assert result["all_bids_collected"] is False

    def test_signals_completion(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, round_num=3, max_rounds=3)
        node = make_next_sealed_round_node()
        result = node(state)
        assert result["round"] == 4  # Past max, signals completion


class TestSealedBidEdges:
    def test_route_collect_more(self, sample_bidders):
        state = _make_sealed_state(sample_bidders)
        state["all_bids_collected"] = False
        assert sealed_route_collect(state) == "collect_bid"

    def test_route_collect_done(self, sample_bidders):
        state = _make_sealed_state(sample_bidders)
        state["all_bids_collected"] = True
        assert sealed_route_collect(state) == "determine_winner"

    def test_route_history_more_rounds(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, round_num=1, max_rounds=3)
        assert sealed_route_history(state) == "next_round"

    def test_route_history_end(self, sample_bidders):
        state = _make_sealed_state(sample_bidders, round_num=3, max_rounds=3)
        assert sealed_route_history(state) == "__end__"


# ============================================================
# English auction nodes
# ============================================================


class TestSolicitBidNode:
    def test_valid_bid(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "1.00"
        state = _make_english_state(sample_bidders)
        state["standing_bid"] = 0.0
        node = make_solicit_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["standing_bid"] == 1.0
        assert result["standing_bidder_id"] == 0
        assert result["bids_this_cycle"] == 1
        assert len(result["bids"]) == 1

    def test_pass_drops_bidder(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "pass"
        state = _make_english_state(sample_bidders)
        node = make_solicit_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert 0 not in result["active_bidder_ids"]
        assert len(result["active_bidder_ids"]) == 2

    def test_bid_below_min_treated_as_pass(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "0.20"
        state = _make_english_state(sample_bidders)
        state["standing_bid"] = 5.0  # min_bid = 5.5
        node = make_solicit_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert 0 not in result["active_bidder_ids"]

    def test_constraint_violation_above_value(self, sample_bidders, auction_prompts, mock_llm):
        # Bidder 0 has value 0.0, bid of 1.0 exceeds
        mock_llm.invoke.return_value = "1.00"
        state = _make_english_state(sample_bidders)
        node = make_solicit_bid_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["constraint_violations"] == 1


class TestCheckAuctionEndNode:
    def test_one_bidder_ends(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["active_bidder_ids"] = [2]
        state["bids_this_cycle"] = 0
        node = make_check_auction_end_node()
        result = node(state)
        assert result["auction_ended"] is True

    def test_no_bids_in_cycle_ends(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["bids_this_cycle"] = 0
        state["active_bidder_ids"] = [1, 2]
        node = make_check_auction_end_node()
        result = node(state)
        assert result["auction_ended"] is True

    def test_bids_made_continues(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["bids_this_cycle"] = 2
        node = make_check_auction_end_node()
        result = node(state)
        assert result["auction_ended"] is False

    def test_safety_limit(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["bids_this_cycle"] = 1
        state["bid_step"] = 150  # 50 * 3 = 150
        node = make_check_auction_end_node()
        result = node(state)
        assert result["auction_ended"] is True


class TestSettleEnglishNode:
    def test_winner_payment(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["standing_bid"] = 5.5
        state["standing_bidder_id"] = 2
        state["bids"] = [
            BidRecord(bidder_id=1, bid_amount=5.0, round=1, bid_step=3, private_value=5.0),
            BidRecord(bidder_id=2, bid_amount=5.5, round=1, bid_step=4, private_value=10.0),
        ]
        node = make_settle_english_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["payment"] == 5.5
        assert ar["second_highest_bid"] == 5.0
        assert ar["surplus"] == 4.5

    def test_no_bids(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["standing_bidder_id"] = None
        state["bids"] = []
        node = make_settle_english_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] is None


class TestSettleOpenOutcryNode:
    def test_first_price_payment(self, sample_bidders):
        state = _make_english_state(sample_bidders, auction_type="first_price_open_outcry")
        state["standing_bid"] = 5.5
        state["standing_bidder_id"] = 2
        state["bids"] = [
            BidRecord(bidder_id=1, bid_amount=5.0, round=1, bid_step=3, private_value=5.0),
            BidRecord(bidder_id=2, bid_amount=5.5, round=1, bid_step=4, private_value=10.0),
        ]
        node = make_settle_open_outcry_node()
        result = node(state)
        ar = result["auction_results"][0]
        # First-price: payment = standing bid = winner's own bid
        assert ar["payment"] == 5.5
        assert ar["auction_type"] == "first_price_open_outcry"


class TestEnglishEdges:
    def test_more_bidders_in_cycle(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["current_bidder_index"] = 1
        assert english_route_solicit(state) == "solicit_bid"

    def test_cycle_end(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["current_bidder_index"] = 3  # Past all 3
        assert english_route_solicit(state) == "check_auction_end"

    def test_auction_ended_settles(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["auction_ended"] = True
        assert english_route_check(state) == "settle"

    def test_auction_continues(self, sample_bidders):
        state = _make_english_state(sample_bidders)
        state["auction_ended"] = False
        assert english_route_check(state) == "reset_cycle"

    def test_more_rounds(self, sample_bidders):
        state = _make_english_state(sample_bidders, round_num=1, max_rounds=3)
        assert english_route_history(state) == "next_round"

    def test_end(self, sample_bidders):
        state = _make_english_state(sample_bidders, round_num=3, max_rounds=3)
        assert english_route_history(state) == "__end__"


class TestNextEnglishRoundNode:
    def test_resets_state(self, sample_bidders):
        state = _make_english_state(sample_bidders, round_num=1, max_rounds=3)
        state["standing_bid"] = 5.0
        state["active_bidder_ids"] = [2]
        node = make_next_english_round_node()
        result = node(state)
        assert result["round"] == 2
        assert result["standing_bid"] == 0.0
        assert len(result["active_bidder_ids"]) == 3  # All reactivated
        assert result["bids"] == []
        assert result["auction_ended"] is False


# ============================================================
# Dutch auction nodes
# ============================================================


class TestAnnouncePriceNode:
    def test_sets_start_price(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        node = make_announce_price_node()
        result = node(state)
        assert result["current_price"] == 12.0
        assert result["current_bidder_index"] == 0
        assert result["accepted"] is False
        assert result["bids"] == []


class TestSolicitAcceptanceNode:
    def test_accept(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "yes"
        state = _make_dutch_state(sample_bidders)
        state["current_price"] = 5.0
        state["current_bidder_index"] = 1  # Bidder 1, value=5.0
        node = make_solicit_acceptance_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["accepted"] is True
        assert result["accepting_bidder_id"] == 1
        assert len(result["bids"]) == 1

    def test_reject(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "no"
        state = _make_dutch_state(sample_bidders)
        state["current_bidder_index"] = 0
        node = make_solicit_acceptance_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result.get("accepted") is None or result.get("accepted") is not True
        assert result["current_bidder_index"] == 1

    def test_all_queried(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "no"
        state = _make_dutch_state(sample_bidders)
        state["current_bidder_index"] = 2  # Last bidder
        node = make_solicit_acceptance_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["all_queried_at_price"] is True

    def test_constraint_violation(self, sample_bidders, auction_prompts, mock_llm):
        mock_llm.invoke.return_value = "yes"
        state = _make_dutch_state(sample_bidders)
        state["current_price"] = 12.0
        state["current_bidder_index"] = 0  # Value=0.0
        node = make_solicit_acceptance_node(mock_llm, auction_prompts)
        result = node(state, {})
        assert result["constraint_violations"] == 1


class TestLowerPriceNode:
    def test_decrements(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["current_price"] = 10.0
        node = make_lower_price_node()
        result = node(state)
        assert result["current_price"] == 9.0
        assert result["current_bidder_index"] == 0
        assert result["all_queried_at_price"] is False


class TestSettleDutchNode:
    def test_winner(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["accepted"] = True
        state["accepting_bidder_id"] = 2
        state["current_price"] = 8.0
        state["bids"] = [
            BidRecord(bidder_id=2, bid_amount=8.0, round=1, bid_step=0, private_value=10.0),
        ]
        node = make_settle_dutch_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["payment"] == 8.0
        assert ar["surplus"] == 2.0

    def test_no_winner(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["accepted"] = False
        state["bids"] = []
        node = make_settle_dutch_node()
        result = node(state)
        ar = result["auction_results"][0]
        assert ar["winner_id"] is None
        assert ar["payment"] is None


class TestDutchEdges:
    def test_accepted_goes_to_check(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["accepted"] = True
        assert dutch_route_solicit(state) == "check_dutch_end"

    def test_all_queried_goes_to_check(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["all_queried_at_price"] = True
        assert dutch_route_solicit(state) == "check_dutch_end"

    def test_more_bidders(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["current_bidder_index"] = 1
        assert dutch_route_solicit(state) == "solicit_acceptance"

    def test_accepted_settles(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["accepted"] = True
        assert dutch_route_check(state) == "settle"

    def test_floor_settles(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["current_price"] = 0.5  # decrement=1.0, min=0.0 -> 0.5-1.0 < 0.0
        assert dutch_route_check(state) == "settle"

    def test_lowers_price(self, sample_bidders):
        state = _make_dutch_state(sample_bidders)
        state["current_price"] = 5.0  # 5.0-1.0 = 4.0 >= 0.0
        assert dutch_route_check(state) == "lower_price"

    def test_more_rounds(self, sample_bidders):
        state = _make_dutch_state(sample_bidders, round_num=1, max_rounds=3)
        assert dutch_route_history(state) == "next_round"

    def test_end(self, sample_bidders):
        state = _make_dutch_state(sample_bidders, round_num=3, max_rounds=3)
        assert dutch_route_history(state) == "__end__"


class TestNextDutchRoundNode:
    def test_resets_state(self, sample_bidders):
        state = _make_dutch_state(sample_bidders, round_num=1, max_rounds=3)
        state["accepted"] = True
        state["accepting_bidder_id"] = 2
        node = make_next_dutch_round_node()
        result = node(state)
        assert result["round"] == 2
        assert result["current_price"] == 12.0
        assert result["accepted"] is False
        assert result["accepting_bidder_id"] is None
        assert result["bids"] == []


# ============================================================
# Full graph integration tests
# ============================================================


class _SequentialMockLLM:
    """Mock LLM that returns responses from a list in sequence."""
    def __init__(self, responses):
        self._responses = responses
        self._idx = 0
        self.last_tool_log = []

    def invoke(self, prompt, callbacks=None):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


class TestSealedBidFullGraph:
    def test_fpsb_two_rounds(self, auction_prompts):
        from market_simulation.graph.auctions.sealed_bid import build_sealed_bid_graph

        llm = _SequentialMockLLM(["1.00", "3.00", "6.00"])
        ac = AuctionConfig(n_rounds=2, bidders=BiddersConfig(num=3, value_min=0.0, value_max=10.0))
        ec = ExperimentConfig(auction_type="fpsb", auction=ac)
        state = create_auction_initial_state(ec, ac)
        graph = build_sealed_bid_graph("fpsb", llm, auction_prompts)
        result = graph.invoke(state, {"recursion_limit": 100})

        assert len(result["auction_results"]) == 2
        for ar in result["auction_results"]:
            assert ar["winner_id"] == 2
            assert ar["payment"] == 6.0
        assert len(result["all_bid_records"]) == 6

    def test_spsb_payment_differs(self, auction_prompts):
        from market_simulation.graph.auctions.sealed_bid import build_sealed_bid_graph

        llm = _SequentialMockLLM(["1.00", "4.00", "7.00"])
        ac = AuctionConfig(n_rounds=1, bidders=BiddersConfig(num=3, value_min=0.0, value_max=10.0))
        ec = ExperimentConfig(auction_type="spsb", auction=ac)
        state = create_auction_initial_state(ec, ac)
        graph = build_sealed_bid_graph("spsb", llm, auction_prompts)
        result = graph.invoke(state, {"recursion_limit": 100})

        ar = result["auction_results"][0]
        assert ar["winning_bid"] == 7.0
        assert ar["payment"] == 4.0  # Second-price
        assert ar["payment"] != ar["winning_bid"]


class TestEnglishFullGraph:
    def test_bidders_drop_out_correctly(self, auction_prompts):
        from market_simulation.graph.auctions.english import build_english_graph

        # Bidder 0 (val=0) passes, bidder 1 (val=5) bids then passes,
        # bidder 2 (val=10) wins
        call_count = {"n": 0}

        class SmartLLM:
            def __init__(self):
                self.last_tool_log = []
            def invoke(self, prompt, callbacks=None):
                call_count["n"] += 1
                m = re.search(r"value=(\d+\.?\d*)", prompt)
                pv = float(m.group(1)) if m else 0.0
                m2 = re.search(r"standing=(\d+\.?\d*)", prompt)
                standing = float(m2.group(1)) if m2 else 0.0
                min_bid = standing + 0.5
                if min_bid > pv * 0.9:
                    return "pass"
                return str(min_bid)

        ac = AuctionConfig(
            n_rounds=1,
            bidders=BiddersConfig(num=3, value_min=0.0, value_max=10.0),
            min_increment=0.5,
            max_bidding_rounds=50,
        )
        ec = ExperimentConfig(auction_type="english", auction=ac)
        state = create_auction_initial_state(ec, ac)
        graph = build_english_graph("english", SmartLLM(), auction_prompts)
        result = graph.invoke(state, {"recursion_limit": 500})

        assert len(result["auction_results"]) == 1
        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["payment"] > 0


class TestDutchFullGraph:
    def test_price_descends_to_acceptance(self, auction_prompts):
        from market_simulation.graph.auctions.dutch import build_dutch_graph

        class DutchLLM:
            def __init__(self):
                self.last_tool_log = []
            def invoke(self, prompt, callbacks=None):
                m = re.search(r"value=(\d+\.?\d*)", prompt)
                pv = float(m.group(1)) if m else 0.0
                m2 = re.search(r"Price is (\d+\.?\d*)", prompt)
                cp = float(m2.group(1)) if m2 else 999.0
                return "yes" if cp <= pv else "no"

        ac = AuctionConfig(
            n_rounds=1,
            bidders=BiddersConfig(num=3, value_min=2.0, value_max=8.0),
            dutch_start_price=10.0,
            dutch_decrement=1.0,
            dutch_min_price=0.0,
        )
        ec = ExperimentConfig(auction_type="dutch", auction=ac)
        state = create_auction_initial_state(ec, ac)
        graph = build_dutch_graph("dutch", DutchLLM(), auction_prompts)
        result = graph.invoke(state, {"recursion_limit": 200})

        ar = result["auction_results"][0]
        # Bidder 2 (val=8.0) should accept when price <= 8.0
        assert ar["winner_id"] == 2
        assert ar["payment"] == 8.0
        assert ar["surplus"] == 0.0


class TestOpenOutcryFullGraph:
    def test_open_outcry_uses_first_price(self, auction_prompts):
        from market_simulation.graph.auctions.open_outcry import build_open_outcry_graph

        class OpenOutcryLLM:
            def __init__(self):
                self.last_tool_log = []
            def invoke(self, prompt, callbacks=None):
                m = re.search(r"value=(\d+\.?\d*)", prompt)
                pv = float(m.group(1)) if m else 0.0
                m2 = re.search(r"standing=(\d+\.?\d*)", prompt)
                standing = float(m2.group(1)) if m2 else 0.0
                min_bid = standing + 0.5
                if min_bid > pv * 0.9:
                    return "pass"
                return str(min_bid)

        ac = AuctionConfig(
            n_rounds=1,
            bidders=BiddersConfig(num=3, value_min=0.0, value_max=10.0),
            min_increment=0.5,
            max_bidding_rounds=50,
        )
        ec = ExperimentConfig(auction_type="first_price_open_outcry", auction=ac)
        state = create_auction_initial_state(ec, ac)
        graph = build_open_outcry_graph("first_price_open_outcry", OpenOutcryLLM(), auction_prompts)
        result = graph.invoke(state, {"recursion_limit": 500})

        ar = result["auction_results"][0]
        assert ar["winner_id"] == 2
        assert ar["auction_type"] == "first_price_open_outcry"
        # Payment = standing bid (first-price)
        assert ar["payment"] == ar["winning_bid"]
