"""Tests for market history summary formatting."""

import pytest

from market_simulation.graph.history import (
    build_market_history_for_prompt,
    build_own_history_for_prompt,
)


def _make_state(
    market_history_text="",
    transactions=None,
    iteration_records=None,
    current_round=1,
):
    """Helper to create a minimal state dict for history tests."""
    return {
        "market_history_text": market_history_text,
        "transactions": transactions or [],
        "iteration_records": iteration_records or [],
        "round": current_round,
    }


class TestBuildMarketHistorySummary:
    """Tests for build_market_history_for_prompt."""

    def test_full_mode_returns_raw_text(self):
        state = _make_state(market_history_text="some raw history")
        result = build_market_history_for_prompt(state, mode="full")
        assert result == "some raw history"

    def test_empty_history_returns_first_iteration_message(self):
        state = _make_state()
        result = build_market_history_for_prompt(state, mode="summary")
        assert "first iteration" in result.lower() or "no market history" in result.lower()

    def test_summary_with_transactions(self):
        transactions = [
            {"round": 1, "iteration": 1, "price": 2.0, "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            {"round": 1, "iteration": 2, "price": 1.5, "buyer_id": 1, "seller_id": 4, "announcement_type": "sell"},
        ]
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 2.0, "announcement_type": "buy", "transaction_made": True},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.5, "announcement_type": "sell", "transaction_made": True},
        ]
        state = _make_state(
            market_history_text="line1\nline2\n",
            transactions=transactions,
            iteration_records=records,
        )
        result = build_market_history_for_prompt(state, mode="summary")
        assert "Completed transactions: 2" in result
        assert "Average transaction price: $1.75" in result
        assert "Last transaction price: $1.50" in result

    def test_summary_includes_last_n_events(self):
        history_text = "event1\nevent2\nevent3\nevent4\nevent5\n"
        records = [{"round": 1, "iteration": i, "announcement_made": True, "price": 1.0, "announcement_type": "buy"} for i in range(5)]
        state = _make_state(
            market_history_text=history_text,
            iteration_records=records,
        )
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=2)
        assert "Recent events:" in result
        assert "event4" in result
        assert "event5" in result
        assert "event1" not in result

    def test_summary_bid_ask_spread(self):
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.20, "announcement_type": "buy"},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.80, "announcement_type": "sell"},
        ]
        state = _make_state(iteration_records=records)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Latest bid (buy offer): $1.20" in result
        assert "Latest ask (sell offer): $1.80" in result
        assert "Bid-ask spread: $0.60" in result

    def test_summary_acceptance_rate(self):
        transactions = [
            {"round": 1, "iteration": 1, "price": 1.5, "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
        ]
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.5, "announcement_type": "buy", "transaction_made": True},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 2.0, "announcement_type": "sell", "transaction_made": False},
        ]
        state = _make_state(transactions=transactions, iteration_records=records)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Acceptance rate: 50% (1/2)" in result

    def test_summary_price_trend(self):
        transactions = [
            {"round": 1, "iteration": i, "price": 1.0 + i * 0.2, "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"}
            for i in range(6)
        ]
        records = [
            {"round": 1, "iteration": i, "announcement_made": True, "price": 1.0 + i * 0.2, "announcement_type": "buy"}
            for i in range(6)
        ]
        state = _make_state(transactions=transactions, iteration_records=records)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Price trend: RISING" in result

    def test_summary_current_round_transactions(self):
        transactions = [
            {"round": 1, "iteration": 1, "price": 1.5, "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
            {"round": 2, "iteration": 1, "price": 1.6, "buyer_id": 1, "seller_id": 4, "announcement_type": "buy"},
        ]
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.5, "announcement_type": "buy"},
            {"round": 2, "iteration": 1, "announcement_made": True, "price": 1.6, "announcement_type": "buy"},
        ]
        state = _make_state(transactions=transactions, iteration_records=records, current_round=2)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Transactions in current round (2): 1" in result


class TestBuildOwnHistorySummary:
    """Tests for build_own_history_for_prompt."""

    def test_full_mode_returns_raw_text(self):
        agent = {"own_history_prompt": "raw own history", "own_history_data": []}
        result = build_own_history_for_prompt(agent, mode="full")
        assert result == "raw own history"

    def test_empty_history(self):
        agent = {"own_history_prompt": "", "own_history_data": []}
        result = build_own_history_for_prompt(agent, mode="summary")
        assert "no actions" in result.lower()

    def test_summary_counts_actions(self):
        agent = {
            "own_history_prompt": "",
            "own_history_data": [
                {"action": "announce", "price": 1.5, "outcome": "accepted", "round": 1, "iteration": 1},
                {"action": "respond", "price": 2.0, "outcome": "rejected", "round": 1, "iteration": 2},
                {"action": "announce", "price": 1.8, "outcome": "accepted", "round": 1, "iteration": 3},
            ],
        }
        result = build_own_history_for_prompt(agent, mode="summary")
        assert "Total actions: 3" in result
        assert "2 announcements" in result
        assert "1 responses" in result
        assert "Successful trades: 2" in result

    def test_summary_last_action(self):
        agent = {
            "own_history_prompt": "",
            "own_history_data": [
                {"action": "announce", "price": 1.5, "outcome": "accepted", "round": 1, "iteration": 1},
                {"action": "respond", "price": 2.0, "outcome": "rejected", "round": 1, "iteration": 2},
            ],
        }
        result = build_own_history_for_prompt(agent, mode="summary")
        assert "Last action: respond at $2.00 (rejected)" in result

    def test_summary_average_trade_price(self):
        agent = {
            "own_history_prompt": "",
            "own_history_data": [
                {"action": "announce", "price": 1.0, "outcome": "accepted", "round": 1, "iteration": 1},
                {"action": "announce", "price": 2.0, "outcome": "accepted", "round": 1, "iteration": 2},
            ],
        }
        result = build_own_history_for_prompt(agent, mode="summary")
        assert "Average trade price: $1.50" in result
