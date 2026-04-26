"""Tests for graph node functions (announce, control flow)."""

import pytest
from unittest.mock import MagicMock, patch

from market_simulation.graph.nodes.announce import (
    make_select_announcer_node,
    make_announce_node,
)
from market_simulation.graph.nodes.control import (
    make_update_history_node,
    make_check_iteration_node,
    make_check_round_node,
    make_next_iteration_node,
    make_next_round_node,
)
from market_simulation.llm.response_schemas import (
    AnnouncementResponse,
    AnnouncementResponseWithReasoning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(callbacks=None):
    """Create a minimal RunnableConfig-like dict."""
    return {"callbacks": callbacks or []}


# ===========================================================================
# TestSelectAnnouncerNode
# ===========================================================================


class TestSelectAnnouncerNode:
    """Tests for select_announcer node."""

    def test_selects_active_agent(self, base_market_state):
        node = make_select_announcer_node()
        result = node(base_market_state)
        assert result["announcing_agent_id"] in base_market_state["active_agent_ids"]

    def test_returns_none_when_no_active_agents(self, base_market_state):
        state = {**base_market_state, "active_agent_ids": []}
        node = make_select_announcer_node()
        result = node(state)
        assert result["announcing_agent_id"] is None
        assert result["announcement_made"] is False

    def test_selected_from_active_ids(self, base_market_state):
        # Only agents 0 and 3 are active
        state = {**base_market_state, "active_agent_ids": [0, 3]}
        node = make_select_announcer_node()
        result = node(state)
        assert result["announcing_agent_id"] in [0, 3]


# ===========================================================================
# TestAnnounceNode
# ===========================================================================


class TestAnnounceNode:
    """Tests for announce node."""

    def test_valid_announcement(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": 0}  # buyer
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert result["announced_price"] == 1.50
        assert result["announcement_type"] == "buy"

    def test_seller_announcement_type(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": 3}  # seller
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_type"] == "sell"

    def test_none_agent_id(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": None}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False
        assert result["announced_price"] is None

    def test_none_price_means_no_announcement(self, base_market_state, mock_llm, prompt_config):
        """Structured output with price=None is a valid response meaning no announcement."""
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=None, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False

    def test_llm_exception(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke_structured.side_effect = RuntimeError("API error")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False
        assert result["announced_price"] is None
        assert "API error" in result["last_error"]

    def test_tool_usage_log_captured(self, base_market_state, mock_llm, prompt_config):
        mock_llm.last_tool_log = [{"tool": "calculator", "input": "1+1"}]
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert len(result["tool_usage_log"]) == 1
        assert result["tool_usage_log"][0]["agent_id"] == 0
        assert result["tool_usage_log"][0]["action"] == "announce"

    def test_buyer_above_reservation_increments_violations(self, base_market_state, mock_llm, prompt_config):
        # Buyer 0 has reservation_price=2.0, announcing $3.00 is a violation
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=3.00, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert result["announced_price"] == 3.00
        assert result["constraint_violations"] == 1

    def test_seller_below_reservation_increments_violations(self, base_market_state, mock_llm, prompt_config):
        # Seller 3 has reservation_price=1.0, announcing $0.50 is a violation
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=0.50, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 3}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert result["announced_price"] == 0.50
        assert result["constraint_violations"] == 1

    def test_no_violation_when_within_bounds(self, base_market_state, mock_llm, prompt_config):
        # Buyer 0 has reservation_price=2.0, announcing $1.50 is fine
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=1.50, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is True
        assert "constraint_violations" not in result

    def test_violation_counter_accumulates(self, base_market_state, mock_llm, prompt_config):
        # Start with 2 existing violations
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(price=3.00, reasoning="")
        state = {**base_market_state, "announcing_agent_id": 0, "constraint_violations": 2}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["constraint_violations"] == 3


# ===========================================================================
# TestUpdateHistoryNode
# ===========================================================================


class TestUpdateHistoryNode:
    """Tests for update_history node."""

    def test_transaction_accepted_history(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "responding_agent_id": 3,
            "response_accepted": True,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
        }
        node = make_update_history_node()
        result = node(state)

        assert "accepted" in result["market_history_text"]
        assert "$1.50" in result["market_history_text"]
        assert len(result["iteration_records"]) == 1

    def test_all_rejected_history(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": False,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "responding_agent_id": 3,
            "response_accepted": False,
            "current_responder_index": 1,
            "potential_responder_ids": [3],  # all queried
        }
        node = make_update_history_node()
        result = node(state)

        assert "no one responded" in result["market_history_text"]

    def test_no_announcement_history(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": False,
            "transaction_made": False,
            "iteration_complete": True,
            "announced_price": None,
            "announcement_type": None,
            "announcing_agent_id": None,
            "responding_agent_id": None,
            "current_responder_index": 0,
            "potential_responder_ids": [],
        }
        node = make_update_history_node()
        result = node(state)

        assert "no announcement was made" in result["market_history_text"]

    def test_announcing_agent_history_updated_on_transaction(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "responding_agent_id": 3,
            "response_accepted": True,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
        }
        node = make_update_history_node()
        result = node(state)

        announcing_agent = next(a for a in result["agents"] if a["id"] == 0)
        assert len(announcing_agent["own_history_data"]) == 1
        assert announcing_agent["own_history_data"][0]["action"] == "announce"
        assert announcing_agent["own_history_data"][0]["outcome"] == "accepted"
        assert "accepted" in announcing_agent["own_history_prompt"]

    def test_iteration_record_fields(self, base_market_state):
        state = {
            **base_market_state,
            "announcement_made": True,
            "transaction_made": True,
            "announced_price": 1.50,
            "announcement_type": "buy",
            "announcing_agent_id": 0,
            "responding_agent_id": 3,
            "response_accepted": True,
            "current_responder_index": 1,
            "potential_responder_ids": [3],
        }
        node = make_update_history_node()
        result = node(state)

        record = result["iteration_records"][0]
        assert record["round"] == 1
        assert record["iteration"] == 1
        assert record["price"] == 1.50
        assert record["announcement_made"] is True
        assert record["transaction_made"] is True
        assert record["announcing_agent_id"] == 0
        assert record["responding_agent_id"] == 3

# ===========================================================================
# TestCheckIterationNode
#
# check_iteration is retained as a no-op stub for any legacy caller —
# the improvement-rule CDA does not wire it into the graph. The node
# simply returns iteration_complete=True.
# ===========================================================================


class TestCheckIterationNode:
    def test_always_returns_iteration_complete(self, base_market_state):
        node = make_check_iteration_node()
        assert node(base_market_state)["iteration_complete"] is True


# ===========================================================================
# TestCheckRoundNode
# ===========================================================================


class TestCheckRoundNode:
    """Tests for check_round node."""

    def test_complete_at_max_iterations(self, base_market_state):
        state = {**base_market_state, "iteration": 3, "max_iterations": 3}
        node = make_check_round_node()
        assert node(state)["round_complete"] is True

    def test_complete_with_too_few_agents(self, base_market_state):
        state = {**base_market_state, "active_agent_ids": [0]}
        node = make_check_round_node()
        assert node(state)["round_complete"] is True

    def test_complete_with_zero_agents(self, base_market_state):
        state = {**base_market_state, "active_agent_ids": []}
        node = make_check_round_node()
        assert node(state)["round_complete"] is True

    def test_not_complete(self, base_market_state):
        state = {**base_market_state, "iteration": 1, "max_iterations": 3, "active_agent_ids": [0, 3]}
        node = make_check_round_node()
        assert node(state)["round_complete"] is False


# ===========================================================================
# TestNextIterationNode
# ===========================================================================


class TestNextIterationNode:
    """Tests for next_iteration node."""

    def test_increments_iteration(self, base_market_state):
        state = {**base_market_state, "iteration": 1}
        node = make_next_iteration_node()
        result = node(state)
        assert result["iteration"] == 2

    def test_resets_flags(self, base_market_state):
        state = {
            **base_market_state,
            "iteration": 1,
            "transaction_made": True,
            "announcement_made": True,
            "announcing_agent_id": 0,
            "announced_price": 1.50,
        }
        node = make_next_iteration_node()
        result = node(state)

        assert result["transaction_made"] is False
        assert result["announcement_made"] is False
        assert result["announcing_agent_id"] is None
        assert result["announced_price"] is None
        assert result["announcement_type"] is None
        assert result["responding_agent_id"] is None
        assert result["response_accepted"] is None


# ===========================================================================
# TestNextRoundNode
# ===========================================================================


class TestNextRoundNode:
    """Tests for next_round node."""

    def test_increments_round(self, base_market_state):
        state = {**base_market_state, "round": 1, "max_rounds": 2}
        node = make_next_round_node()
        result = node(state)
        assert result["round"] == 2

    def test_simulation_complete_at_max_rounds(self, base_market_state):
        state = {**base_market_state, "round": 2, "max_rounds": 2}
        node = make_next_round_node()
        result = node(state)
        assert result["simulation_complete"] is True

    def test_reactivates_all_agents(self, base_market_state):
        # Deactivate some agents
        agents = [
            {**a, "active": False} if a["id"] in (0, 3) else a
            for a in base_market_state["agents"]
        ]
        state = {
            **base_market_state,
            "round": 1,
            "max_rounds": 3,
            "agents": agents,
            "active_agent_ids": [1, 2, 4, 5],
        }
        node = make_next_round_node()
        result = node(state)

        assert all(a["active"] is True for a in result["agents"])
        assert set(result["active_agent_ids"]) == {0, 1, 2, 3, 4, 5}

    def test_resets_iteration_to_one(self, base_market_state):
        state = {**base_market_state, "round": 1, "max_rounds": 3}
        node = make_next_round_node()
        result = node(state)
        assert result["iteration"] == 1

    def test_resets_state_flags(self, base_market_state):
        state = {
            **base_market_state,
            "round": 1,
            "max_rounds": 3,
            "transaction_made": True,
            "announcement_made": True,
        }
        node = make_next_round_node()
        result = node(state)

        assert result["transaction_made"] is False
        assert result["announcement_made"] is False
        assert result["round_complete"] is False

    def test_boundary_new_round_equals_max_rounds_not_complete(self, base_market_state):
        """When round 1 -> 2 and max_rounds=2, simulation is NOT complete (round 2 still runs)."""
        state = {**base_market_state, "round": 1, "max_rounds": 2}
        node = make_next_round_node()
        result = node(state)

        assert result["round"] == 2
        assert result.get("simulation_complete", False) is False
