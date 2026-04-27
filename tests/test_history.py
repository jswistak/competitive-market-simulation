"""Tests for market history summary formatting."""

import logging

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
        assert "Transactions so far this round: 1" in result


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
        assert "1 response" in result
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


class TestAcceptanceRateDistinctAnnouncements:
    """Verify acceptance rate counts distinct announcements, not records."""

    def test_multiple_responders_per_announcement(self):
        """1 announcement queried to 3 responders (3 rejections then 1 accept)
        should show 100% acceptance rate, not 25%."""
        # All four records share round=1, iteration=1 -> 1 distinct announcement
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.50,
             "announcement_type": "buy", "transaction_made": False},
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.50,
             "announcement_type": "buy", "transaction_made": False},
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.50,
             "announcement_type": "buy", "transaction_made": False},
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.50,
             "announcement_type": "buy", "transaction_made": True},
        ]
        transactions = [
            {"round": 1, "iteration": 1, "price": 1.50,
             "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
        ]
        state = _make_state(
            transactions=transactions,
            iteration_records=records,
        )
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Acceptance rate: 100% (1/1)" in result

    def test_two_announcements_one_accepted(self):
        """2 distinct announcements, 1 accepted -> 50%."""
        records = [
            # First announcement: round=1, iteration=1
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.50,
             "announcement_type": "buy", "transaction_made": True},
            # Second announcement: round=1, iteration=2 (no transaction)
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 2.00,
             "announcement_type": "sell", "transaction_made": False},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 2.00,
             "announcement_type": "sell", "transaction_made": False},
        ]
        transactions = [
            {"round": 1, "iteration": 1, "price": 1.50,
             "buyer_id": 0, "seller_id": 3, "announcement_type": "buy"},
        ]
        state = _make_state(
            transactions=transactions,
            iteration_records=records,
        )
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Acceptance rate: 50% (1/2)" in result


class TestNegativeBidAskSpread:
    """Verify bid-ask spread handles crossed/converged prices."""

    def test_negative_spread_shows_crossed(self):
        """When buy price > sell price, spread should indicate convergence."""
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 2.00,
             "announcement_type": "buy"},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.70,
             "announcement_type": "sell"},
        ]
        state = _make_state(iteration_records=records)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "crossed" in result.lower() or "converged" in result.lower()
        assert "$0.30" in result
        # Must NOT show a negative dollar value
        assert "$-" not in result

    def test_positive_spread_unchanged(self):
        """Normal positive spread should display as before."""
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.20,
             "announcement_type": "buy"},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.80,
             "announcement_type": "sell"},
        ]
        state = _make_state(iteration_records=records)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Bid-ask spread: $0.60" in result
        assert "crossed" not in result.lower()
        assert "converged" not in result.lower()

    def test_zero_spread(self):
        """Zero spread (equal bid/ask) should not show crossed."""
        records = [
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.50,
             "announcement_type": "buy"},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.50,
             "announcement_type": "sell"},
        ]
        state = _make_state(iteration_records=records)
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=0)
        assert "Bid-ask spread: $0.00" in result
        assert "crossed" not in result.lower()


class TestSummaryIntegration:
    """Integration test exercising the full summary render path."""

    def _build_realistic_state(self):
        """Build a state simulating 2 rounds of trading activity."""
        # Round 1: 3 announcements, 2 transactions
        # Round 2: 2 announcements, 1 transaction
        records = [
            # Round 1, iter 1: buy announcement, queried 2 responders, accepted
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.80,
             "announcement_type": "buy", "transaction_made": False},
            {"round": 1, "iteration": 1, "announcement_made": True, "price": 1.80,
             "announcement_type": "buy", "transaction_made": True},
            # Round 1, iter 2: sell announcement, queried 2 responders, accepted
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.60,
             "announcement_type": "sell", "transaction_made": False},
            {"round": 1, "iteration": 2, "announcement_made": True, "price": 1.60,
             "announcement_type": "sell", "transaction_made": True},
            # Round 1, iter 3: buy announcement, queried 1 responder, rejected
            {"round": 1, "iteration": 3, "announcement_made": True, "price": 1.20,
             "announcement_type": "buy", "transaction_made": False},
            # Round 2, iter 1: sell announcement, queried 2 responders, accepted
            {"round": 2, "iteration": 1, "announcement_made": True, "price": 1.70,
             "announcement_type": "sell", "transaction_made": False},
            {"round": 2, "iteration": 1, "announcement_made": True, "price": 1.70,
             "announcement_type": "sell", "transaction_made": True},
            # Round 2, iter 2: buy announcement, queried 1 responder, rejected
            {"round": 2, "iteration": 2, "announcement_made": True, "price": 2.10,
             "announcement_type": "buy", "transaction_made": False},
        ]
        transactions = [
            {"round": 1, "iteration": 1, "price": 1.80, "buyer_id": 0, "seller_id": 3,
             "announcement_type": "buy"},
            {"round": 1, "iteration": 2, "price": 1.60, "buyer_id": 1, "seller_id": 4,
             "announcement_type": "sell"},
            {"round": 2, "iteration": 1, "price": 1.70, "buyer_id": 2, "seller_id": 5,
             "announcement_type": "sell"},
        ]
        raw_history = (
            "R1/I1: Buyer 0 announced buy at $1.80, Seller 3 accepted.\n"
            "R1/I2: Seller 4 announced sell at $1.60, Buyer 1 accepted.\n"
            "R1/I3: Buyer 2 announced buy at $1.20, rejected.\n"
            "R2/I1: Seller 5 announced sell at $1.70, Buyer 2 accepted.\n"
            "R2/I2: Buyer 0 announced buy at $2.10, rejected.\n"
        )
        return _make_state(
            market_history_text=raw_history,
            transactions=transactions,
            iteration_records=records,
            current_round=2,
        )

    def test_summary_contains_expected_statistics(self):
        """Verify that the summary includes all key statistics."""
        state = self._build_realistic_state()
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=2)

        # Transaction count
        assert "Completed transactions: 3" in result
        # Average price: (1.80 + 1.60 + 1.70) / 3 = 1.70
        assert "Average transaction price: $1.70" in result
        # Last transaction price
        assert "Last transaction price: $1.70" in result
        # Acceptance rate: 3 transactions / 5 distinct announcements = 60%
        assert "Acceptance rate: 60% (3/5)" in result
        # Current round transactions
        assert "Transactions so far this round: 1" in result
        # Bid-ask info present
        assert "Latest bid" in result
        assert "Latest ask" in result
        assert "Bid-ask spread" in result

    def test_summary_does_not_contain_full_raw_history(self):
        """Summary mode should NOT include the full raw event log."""
        state = self._build_realistic_state()
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=2)

        # The first events should NOT appear (only last 2 should)
        assert "R1/I1:" not in result
        assert "R1/I2:" not in result
        # But recent events should appear
        assert "Recent events:" in result
        assert "R2/I2:" in result or "R2/I1:" in result

    def test_summary_includes_recent_events(self):
        """Summary should append the last N raw event lines."""
        state = self._build_realistic_state()
        result = build_market_history_for_prompt(state, mode="summary", last_n_events=2)

        assert "Recent events:" in result
        # Last 2 raw lines
        assert "R2/I1:" in result or "rejected" in result

    def test_own_history_summary_integration(self):
        """Verify own-history summary path with mixed actions."""
        agent = {
            "own_history_prompt": "full raw history that should NOT appear in summary",
            "own_history_data": [
                {"action": "announce", "price": 1.80, "outcome": "accepted", "round": 1, "iteration": 1},
                {"action": "respond", "price": 1.60, "outcome": "rejected", "round": 1, "iteration": 2},
                {"action": "announce", "price": 1.70, "outcome": "accepted", "round": 2, "iteration": 1},
                {"action": "respond", "price": 2.10, "outcome": "rejected", "round": 2, "iteration": 2},
            ],
        }
        result = build_own_history_for_prompt(agent, mode="summary")

        assert "Total actions: 4" in result
        assert "2 announcements" in result
        assert "2 responses" in result
        assert "Successful trades: 2" in result
        assert "Average trade price: $1.75" in result
        assert "Last action: respond at $2.10 (rejected)" in result
        # Must NOT contain the full raw text
        assert "full raw history that should NOT appear" not in result

    def test_logging_output(self, caplog):
        """Verify that summary results are logged at DEBUG level."""
        state = self._build_realistic_state()
        agent = {
            "own_history_prompt": "raw",
            "own_history_data": [
                {"action": "announce", "price": 1.0, "outcome": "accepted", "round": 1, "iteration": 1},
            ],
        }

        with caplog.at_level(logging.DEBUG, logger="market_simulation.graph.history"):
            build_market_history_for_prompt(state, mode="summary", last_n_events=0)
            build_own_history_for_prompt(agent, mode="summary")

        log_text = caplog.text
        assert "Market history for prompt (mode=summary)" in log_text
        assert "Own history for prompt (mode=summary)" in log_text
