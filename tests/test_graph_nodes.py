"""Tests for graph node functions (announce, respond, transaction, control flow)."""

import pytest
from unittest.mock import MagicMock, patch

from market_simulation.graph.nodes.announce import (
    make_select_announcer_node,
    make_announce_node,
    _extract_price,
)
from market_simulation.graph.nodes.respond import (
    make_select_responders_node,
    make_respond_node,
    _extract_response,
)
from market_simulation.graph.nodes.transaction import make_record_transaction_node
from market_simulation.graph.nodes.control import (
    make_update_history_node,
    make_check_iteration_node,
    make_check_round_node,
    make_next_iteration_node,
    make_next_round_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(callbacks=None):
    """Create a minimal RunnableConfig-like dict."""
    return {"callbacks": callbacks or []}


# ===========================================================================
# TestExtractPrice
# ===========================================================================


class TestExtractPrice:
    """Tests for _extract_price helper in announce module."""

    def test_plain_number(self):
        assert _extract_price("1.50") == 1.50

    def test_integer(self):
        assert _extract_price("2") == 2.0

    def test_small_decimal(self):
        assert _extract_price("0.99") == 0.99

    def test_dollar_sign(self):
        assert _extract_price("$1.50") == 1.50

    def test_comma_separated(self):
        assert _extract_price("1,500.00") == 1500.00

    def test_whitespace(self):
        assert _extract_price("  1.50  ") == 1.50

    def test_non_numeric_returns_none(self):
        assert _extract_price("hello") is None

    def test_empty_string_returns_none(self):
        assert _extract_price("") is None

    def test_regex_fallback_extracts_last_number(self):
        assert _extract_price("I think $1.50 is fair") == 1.50

    def test_regex_fallback_multiple_numbers(self):
        result = _extract_price("Between 1.0 and 2.0, I choose 1.75")
        assert result == 1.75

    def test_dollar_sign_with_text(self):
        assert _extract_price("My price is $2.50.") == 2.50


# ===========================================================================
# TestExtractResponse
# ===========================================================================


class TestExtractResponse:
    """Tests for _extract_response helper in respond module."""

    def test_yes_lowercase(self):
        assert _extract_response("yes") is True

    def test_yes_capitalized(self):
        assert _extract_response("Yes") is True

    def test_yes_uppercase(self):
        assert _extract_response("YES") is True

    def test_no_lowercase(self):
        assert _extract_response("no") is False

    def test_no_capitalized(self):
        assert _extract_response("No") is False

    def test_yes_in_sentence(self):
        assert _extract_response("Yes, I accept the offer.") is True

    def test_no_yes_in_text(self):
        assert _extract_response("absolutely") is False

    def test_empty_string(self):
        assert _extract_response("") is False

    def test_yesterday_contains_yes(self):
        # "yesterday" contains "yes" — this is the expected behavior
        assert _extract_response("yesterday") is True


# ===========================================================================
# TestSelectAnnouncerNode
# ===========================================================================


class TestSelectAnnouncerNode:
    """Tests for select_announcer node."""

    def test_selects_active_agent(self, base_market_state):
        node = make_select_announcer_node()
        result = node(base_market_state)
        assert result["announcing_agent_id"] in base_market_state["active_agent_ids"]

    def test_no_eligible_agents(self, base_market_state):
        state = {
            **base_market_state,
            "announced_this_iteration": list(base_market_state["active_agent_ids"]),
        }
        node = make_select_announcer_node()
        result = node(state)
        assert result["announcing_agent_id"] is None
        assert result["announcement_made"] is False

    def test_excludes_already_announced(self, base_market_state):
        # Mark all except agent 5 as having announced
        state = {
            **base_market_state,
            "announced_this_iteration": [0, 1, 2, 3, 4],
        }
        node = make_select_announcer_node()
        result = node(state)
        assert result["announcing_agent_id"] == 5

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

    def test_unparsable_response(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke.return_value = "I don't want to announce"
        state = {**base_market_state, "announcing_agent_id": 0}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announcement_made"] is False

    def test_tracks_announced_this_iteration(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": 0, "announced_this_iteration": []}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert 0 in result["announced_this_iteration"]

    def test_appends_to_existing_announced_list(self, base_market_state, mock_llm, prompt_config):
        state = {**base_market_state, "announcing_agent_id": 1, "announced_this_iteration": [0]}
        node = make_announce_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["announced_this_iteration"] == [0, 1]

    def test_llm_exception(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke.side_effect = RuntimeError("API error")
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


# ===========================================================================
# TestSelectRespondersNode
# ===========================================================================


class TestSelectRespondersNode:
    """Tests for select_responders node."""

    def test_buyers_respond_to_sell(self, base_market_state):
        # Seller (id=3) announces sell -> buyers should respond
        state = {
            **base_market_state,
            "announcing_agent_id": 3,
            "announcement_type": "sell",
        }
        node = make_select_responders_node()
        result = node(state)

        # Buyers are ids 0, 1, 2
        assert set(result["potential_responder_ids"]) == {0, 1, 2}
        assert result["current_responder_index"] == 0

    def test_sellers_respond_to_buy(self, base_market_state):
        # Buyer (id=0) announces buy -> sellers should respond
        state = {
            **base_market_state,
            "announcing_agent_id": 0,
            "announcement_type": "buy",
        }
        node = make_select_responders_node()
        result = node(state)

        # Sellers are ids 3, 4, 5
        assert set(result["potential_responder_ids"]) == {3, 4, 5}

    def test_only_active_agents_respond(self, base_market_state):
        # Remove seller 4 and 5 from active
        state = {
            **base_market_state,
            "active_agent_ids": [0, 1, 2, 3],
            "announcing_agent_id": 0,
            "announcement_type": "buy",
        }
        node = make_select_responders_node()
        result = node(state)

        assert result["potential_responder_ids"] == [3]

    def test_no_opposite_type_agents(self, base_market_state):
        # Only buyers active
        state = {
            **base_market_state,
            "active_agent_ids": [0, 1, 2],
            "announcing_agent_id": 0,
            "announcement_type": "buy",
        }
        node = make_select_responders_node()
        result = node(state)

        assert result["potential_responder_ids"] == []

    def test_none_announcement_type(self, base_market_state):
        state = {**base_market_state, "announcement_type": None}
        node = make_select_responders_node()
        result = node(state)

        assert result["potential_responder_ids"] == []
        assert result["current_responder_index"] == 0


# ===========================================================================
# TestRespondNode
# ===========================================================================


class TestRespondNode:
    """Tests for respond node."""

    def test_accept_response(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke.return_value = "Yes"
        state = {
            **base_market_state,
            "potential_responder_ids": [3],
            "current_responder_index": 0,
            "announcing_agent_id": 0,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_respond_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["transaction_made"] is True
        assert result["response_accepted"] is True
        assert result["responding_agent_id"] == 3

    def test_reject_response(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke.return_value = "No"
        state = {
            **base_market_state,
            "potential_responder_ids": [3],
            "current_responder_index": 0,
            "announcing_agent_id": 0,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_respond_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["transaction_made"] is False
        assert result["response_accepted"] is False

    def test_increments_responder_index(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke.return_value = "No"
        state = {
            **base_market_state,
            "potential_responder_ids": [3, 4, 5],
            "current_responder_index": 0,
            "announcing_agent_id": 0,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_respond_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["current_responder_index"] == 1

    def test_index_exceeds_list(self, base_market_state, mock_llm, prompt_config):
        state = {
            **base_market_state,
            "potential_responder_ids": [3],
            "current_responder_index": 1,  # past the end
            "announcing_agent_id": 0,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_respond_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["transaction_made"] is False
        assert result["responding_agent_id"] is None

    def test_llm_exception(self, base_market_state, mock_llm, prompt_config):
        mock_llm.invoke.side_effect = RuntimeError("API error")
        state = {
            **base_market_state,
            "potential_responder_ids": [3],
            "current_responder_index": 0,
            "announcing_agent_id": 0,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_respond_node(mock_llm, prompt_config)
        result = node(state, _make_config())

        assert result["response_accepted"] is False
        assert result["current_responder_index"] == 1


# ===========================================================================
# TestRecordTransactionNode
# ===========================================================================


class TestRecordTransactionNode:
    """Tests for record_transaction node."""

    def test_buy_announcement_transaction(self, base_market_state):
        state = {
            **base_market_state,
            "transaction_made": True,
            "announcing_agent_id": 0,    # buyer announces buy
            "responding_agent_id": 3,    # seller responds
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_record_transaction_node()
        result = node(state)

        assert len(result["transactions"]) == 1
        tx = result["transactions"][0]
        assert tx["buyer_id"] == 0   # announcer is buyer
        assert tx["seller_id"] == 3  # responder is seller
        assert tx["price"] == 1.50

    def test_sell_announcement_transaction(self, base_market_state):
        state = {
            **base_market_state,
            "transaction_made": True,
            "announcing_agent_id": 3,    # seller announces sell
            "responding_agent_id": 0,    # buyer responds
            "announced_price": 2.00,
            "announcement_type": "sell",
        }
        node = make_record_transaction_node()
        result = node(state)

        tx = result["transactions"][0]
        assert tx["buyer_id"] == 0   # responder is buyer
        assert tx["seller_id"] == 3  # announcer is seller

    def test_agents_removed_from_active(self, base_market_state):
        state = {
            **base_market_state,
            "transaction_made": True,
            "announcing_agent_id": 0,
            "responding_agent_id": 3,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_record_transaction_node()
        result = node(state)

        assert 0 not in result["active_agent_ids"]
        assert 3 not in result["active_agent_ids"]
        assert len(result["active_agent_ids"]) == 4  # 6 - 2

    def test_agents_deactivated(self, base_market_state):
        state = {
            **base_market_state,
            "transaction_made": True,
            "announcing_agent_id": 0,
            "responding_agent_id": 3,
            "announced_price": 1.50,
            "announcement_type": "buy",
        }
        node = make_record_transaction_node()
        result = node(state)

        for agent in result["agents"]:
            if agent["id"] in (0, 3):
                assert agent["active"] is False
            else:
                assert agent["active"] is True

    def test_no_transaction(self, base_market_state):
        state = {**base_market_state, "transaction_made": False}
        node = make_record_transaction_node()
        result = node(state)

        assert result == {}


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

    def test_responding_agent_history_updated(self, base_market_state):
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

        responding_agent = next(a for a in result["agents"] if a["id"] == 3)
        assert len(responding_agent["own_history_data"]) == 1
        assert responding_agent["own_history_data"][0]["action"] == "respond"
        assert responding_agent["own_history_data"][0]["outcome"] == "accepted"

    def test_announcer_not_updated_when_more_responders_pending(self, base_market_state):
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
            "potential_responder_ids": [3, 4, 5],  # more responders remain
        }
        node = make_update_history_node()
        result = node(state)

        announcing_agent = next(a for a in result["agents"] if a["id"] == 0)
        # Should NOT be updated yet — more responders are pending
        assert len(announcing_agent["own_history_data"]) == 0

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
# ===========================================================================


class TestCheckIterationNode:
    """Tests for check_iteration node."""

    def test_complete_on_transaction(self, base_market_state):
        state = {**base_market_state, "transaction_made": True, "announcement_made": True}
        node = make_check_iteration_node()
        assert node(state)["iteration_complete"] is True

    def test_complete_on_no_announcement(self, base_market_state):
        state = {**base_market_state, "transaction_made": False, "announcement_made": False}
        node = make_check_iteration_node()
        assert node(state)["iteration_complete"] is True

    def test_complete_when_all_responders_queried(self, base_market_state):
        state = {
            **base_market_state,
            "transaction_made": False,
            "announcement_made": True,
            "potential_responder_ids": [3, 4],
            "current_responder_index": 2,
        }
        node = make_check_iteration_node()
        assert node(state)["iteration_complete"] is True

    def test_not_complete_when_more_responders(self, base_market_state):
        state = {
            **base_market_state,
            "transaction_made": False,
            "announcement_made": True,
            "potential_responder_ids": [3, 4, 5],
            "current_responder_index": 1,
        }
        node = make_check_iteration_node()
        assert node(state)["iteration_complete"] is False


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

    def test_clears_announced_this_iteration(self, base_market_state):
        state = {**base_market_state, "iteration": 1, "announced_this_iteration": [0, 1, 2]}
        node = make_next_iteration_node()
        result = node(state)
        assert result["announced_this_iteration"] == []


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
        assert result["announced_this_iteration"] == []
