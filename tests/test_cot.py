"""Tests for reasoning capture in structured output responses."""

from unittest.mock import MagicMock

from market_simulation.llm.response_schemas import (
    AnnouncementResponseWithReasoning,
)
from market_simulation.graph.nodes.announce import make_announce_node
from market_simulation.graph.nodes.control import (
    make_next_iteration_node,
    make_next_round_node,
    make_update_history_node,
)
from market_simulation.graph.state import IterationRecord


# ===========================================================================
# Unit tests: reasoning field reset in control nodes
# ===========================================================================


class TestReasoningFieldReset:
    """Tests that reasoning fields are reset between iterations/rounds."""

    def test_next_iteration_resets_reasoning(self, base_market_state):
        """next_iteration must reset reasoning fields to empty strings."""
        state = {**base_market_state}
        state["last_announcement_reasoning"] = "old announcement reasoning"
        state["last_response_reasoning"] = "old response reasoning"

        node = make_next_iteration_node()
        result = node(state)

        assert result["last_announcement_reasoning"] == ""
        assert result["last_response_reasoning"] == ""

    def test_next_round_resets_reasoning(self, base_market_state):
        """next_round must reset reasoning fields to empty strings."""
        state = {**base_market_state}
        state["last_announcement_reasoning"] = "old announcement reasoning"
        state["last_response_reasoning"] = "old response reasoning"
        state["round"] = 1
        state["max_rounds"] = 3

        node = make_next_round_node()
        result = node(state)

        assert result["last_announcement_reasoning"] == ""
        assert result["last_response_reasoning"] == ""


# ===========================================================================
# Integration tests: structured output reasoning flows through nodes
# ===========================================================================


def _make_config(callbacks=None):
    """Create a minimal RunnableConfig-like dict."""
    return {"callbacks": callbacks or []}


class TestStructuredOutputReasoning:
    """Integration tests: mock invoke_structured returns Pydantic models with
    reasoning, verify reasoning is captured in state and flows into IterationRecord."""

    def test_announce_captures_reasoning(self, base_market_state, prompt_config):
        """announce node should populate last_announcement_reasoning from structured response."""
        mock_llm = MagicMock()
        mock_llm.invoke_structured.return_value = AnnouncementResponseWithReasoning(
            price=1.75,
            reasoning="My reservation price is $2.00 and the market seems quiet. I should bid conservatively.",
        )
        mock_llm.last_tool_log = []

        node = make_announce_node(mock_llm, prompt_config)

        state = {**base_market_state}
        state["announcing_agent_id"] = 0  # buyer with reservation $2.00

        result = node(state, _make_config())

        assert result["announced_price"] == 1.75
        assert result["announcement_made"] is True
        assert "reservation price" in result["last_announcement_reasoning"]
        assert "conservatively" in result["last_announcement_reasoning"]

    def test_reasoning_recorded_in_iteration_record(self, base_market_state, prompt_config):
        """update_history should store reasoning in IterationRecord."""
        state = {**base_market_state}
        state["round"] = 1
        state["iteration"] = 1
        state["announcing_agent_id"] = 0
        state["announced_price"] = 1.75
        state["announcement_type"] = "buy"
        state["announcement_made"] = True
        state["transaction_made"] = True
        state["responding_agent_id"] = 3
        state["response_accepted"] = True
        state["iteration_complete"] = True
        state["potential_responder_ids"] = [3]
        state["current_responder_index"] = 1
        state["last_announcement_reasoning"] = "I bid low because market is quiet"
        state["last_response_reasoning"] = "Price is above my reservation so I accept"

        node = make_update_history_node()
        result = node(state)

        assert len(result["iteration_records"]) == 1
        record = result["iteration_records"][0]
        assert record["announcement_reasoning"] == "I bid low because market is quiet"
        assert record["response_reasoning"] == "Price is above my reservation so I accept"

