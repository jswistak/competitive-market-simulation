"""Tests for conditional edge routing functions."""

import pytest

from market_simulation.graph.edges import (
    route_after_announcement,
    route_after_response,
    route_after_update_history,
    route_after_check_round,
    route_after_next_round,
)


class TestRouteAfterAnnouncement:
    """Tests for the announce -> (select_responders | check_iteration) router."""

    def test_routes_to_select_responders_on_announcement(self, base_market_state):
        """When announcement was made with a valid price, route to select_responders."""
        state = {**base_market_state, "announcement_made": True, "announced_price": 1.5}
        assert route_after_announcement(state) == "select_responders"

    def test_routes_to_check_iteration_on_no_announcement(self, base_market_state):
        """When no announcement was made, route to check_iteration."""
        state = {**base_market_state, "announcement_made": False, "announced_price": None}
        assert route_after_announcement(state) == "check_iteration"

    def test_routes_to_check_iteration_when_price_is_none(self, base_market_state):
        """When announcement_made but price is None, route to check_iteration."""
        state = {**base_market_state, "announcement_made": True, "announced_price": None}
        assert route_after_announcement(state) == "check_iteration"


class TestRouteAfterResponse:
    """Tests for the respond -> (record_transaction | check_iteration) router."""

    def test_routes_to_record_transaction_on_acceptance(self, base_market_state):
        """When transaction was made, route to record_transaction."""
        state = {**base_market_state, "transaction_made": True}
        assert route_after_response(state) == "record_transaction"

    def test_routes_to_check_iteration_on_rejection(self, base_market_state):
        """When no transaction, route to check_iteration."""
        state = {**base_market_state, "transaction_made": False}
        assert route_after_response(state) == "check_iteration"


class TestRouteAfterUpdateHistory:
    """Tests for the update_history -> (check_round | respond | select_announcer) router."""

    def test_routes_to_check_round_on_transaction(self, base_market_state):
        """Transaction made means iteration complete -> check_round."""
        state = {**base_market_state, "transaction_made": True, "announcement_made": True}
        assert route_after_update_history(state) == "check_round"

    def test_routes_to_check_round_on_no_announcement(self, base_market_state):
        """No announcement made -> check_round."""
        state = {**base_market_state, "transaction_made": False, "announcement_made": False}
        assert route_after_update_history(state) == "check_round"

    def test_routes_to_respond_when_more_responders(self, base_market_state):
        """Announcement rejected but more responders available -> respond."""
        state = {
            **base_market_state,
            "transaction_made": False,
            "announcement_made": True,
            "potential_responder_ids": [3, 4, 5],
            "current_responder_index": 1,  # still 2 left
        }
        assert route_after_update_history(state) == "respond"

    def test_routes_to_select_announcer_when_all_rejected(self, base_market_state):
        """All responders queried, none accepted -> select_announcer."""
        state = {
            **base_market_state,
            "transaction_made": False,
            "announcement_made": True,
            "potential_responder_ids": [3, 4],
            "current_responder_index": 2,  # exhausted
        }
        assert route_after_update_history(state) == "select_announcer"

    def test_routes_to_select_announcer_with_empty_responder_list(self, base_market_state):
        """Edge case: announcement made, no transaction, empty responder list -> select_announcer."""
        state = {
            **base_market_state,
            "transaction_made": False,
            "announcement_made": True,
            "potential_responder_ids": [],
            "current_responder_index": 0,
        }
        assert route_after_update_history(state) == "select_announcer"


class TestRouteAfterCheckRound:
    """Tests for the check_round -> (next_round | next_iteration) router."""

    def test_routes_to_next_round_when_round_complete(self, base_market_state):
        """When round_complete is True, route to next_round."""
        state = {**base_market_state, "round_complete": True}
        assert route_after_check_round(state) == "next_round"

    def test_routes_to_next_iteration_when_round_continues(self, base_market_state):
        """When round_complete is False, route to next_iteration."""
        state = {**base_market_state, "round_complete": False}
        assert route_after_check_round(state) == "next_iteration"


class TestRouteAfterNextRound:
    """Tests for the next_round -> (select_announcer | __end__) router."""

    def test_routes_to_end_when_simulation_complete(self, base_market_state):
        """When simulation_complete is True, route to __end__."""
        state = {**base_market_state, "simulation_complete": True}
        assert route_after_next_round(state) == "__end__"

    def test_routes_to_select_announcer_when_more_rounds(self, base_market_state):
        """When simulation not complete, route to select_announcer."""
        state = {**base_market_state, "simulation_complete": False}
        assert route_after_next_round(state) == "select_announcer"
