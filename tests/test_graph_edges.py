"""Tests for the improvement-rule CDA edge routers.

The old iteration-based router surface is gone: crossing happens inside
``apply_order`` now, so the fan-out after an announcement no longer
exists. What remains is the post-history fan-out for round completion
and the end-of-simulation branch.
"""

import pytest

from market_simulation.graph.edges import (
    route_after_update_history,
    route_after_check_round,
    route_after_next_round,
)


class TestRouteAfterUpdateHistory:
    def test_always_routes_to_check_round(self, base_market_state):
        # Whatever the prior node did, the next step is always the
        # round check — each tick is one atomic step under the CDA.
        assert route_after_update_history(base_market_state) == "check_round"


class TestRouteAfterCheckRound:
    def test_routes_to_next_round_when_round_complete(self, base_market_state):
        state = {**base_market_state, "round_complete": True}
        assert route_after_check_round(state) == "next_round"

    def test_routes_to_next_iteration_when_round_continues(self, base_market_state):
        state = {**base_market_state, "round_complete": False}
        assert route_after_check_round(state) == "next_iteration"


class TestRouteAfterNextRound:
    def test_routes_to_end_on_simulation_complete(self, base_market_state):
        state = {**base_market_state, "simulation_complete": True}
        assert route_after_next_round(state) == "__end__"

    def test_routes_to_select_announcer_on_new_round(self, base_market_state):
        state = {**base_market_state, "simulation_complete": False}
        assert route_after_next_round(state) == "select_announcer"
