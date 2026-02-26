"""Tests for chain-of-thought reasoning extraction."""

import pytest
from unittest.mock import MagicMock

from market_simulation.graph.nodes.announce import _extract_price, make_announce_node
from market_simulation.graph.nodes.respond import _extract_response, make_respond_node
from market_simulation.graph.nodes.control import (
    make_next_iteration_node,
    make_next_round_node,
    make_update_history_node,
)
from market_simulation.graph.state import IterationRecord


# ===========================================================================
# Unit tests: _extract_price with CoT
# ===========================================================================


class TestExtractPriceWithCoT:
    """Tests for _extract_price with answer_tag support."""

    def test_answer_tag_basic(self):
        response = "I think the price should be high because demand is strong. ANSWER: 2.50"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price == 2.50
        assert "demand is strong" in reasoning

    def test_answer_tag_multiline(self):
        response = "Step 1: My reservation is $2.00\nStep 2: Market is slow\nANSWER: 1.80"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price == 1.80
        assert "Step 1" in reasoning

    def test_no_tag_fallback(self):
        """Without answer_tag, falls back to existing behavior."""
        price, reasoning = _extract_price("1.50", answer_tag=None)
        assert price == 1.50
        assert reasoning == ""

    def test_answer_tag_not_found_in_response(self):
        """If tag not found in response, falls back to existing parsing."""
        price, reasoning = _extract_price("1.50", answer_tag="ANSWER:")
        assert price == 1.50
        assert reasoning == ""

    def test_answer_tag_with_dollar_sign(self):
        response = "My reasoning here. ANSWER: $2.50"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price == 2.50
        assert "reasoning" in reasoning

    def test_case_insensitive_tag(self):
        response = "Some thinking. answer: 3.00"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price == 3.00
        assert "thinking" in reasoning

    def test_plain_number_without_tag(self):
        """Existing behavior preserved when no answer_tag."""
        price, reasoning = _extract_price("2.00")
        assert price == 2.00
        assert reasoning == ""

    def test_empty_response(self):
        price, reasoning = _extract_price("", answer_tag="ANSWER:")
        assert price is None
        assert reasoning == ""

    def test_tag_with_integer(self):
        response = "Let me think... ANSWER: 2"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price == 2.0
        assert "think" in reasoning

    def test_tag_found_but_garbled_answer_returns_none(self):
        """When tag IS found but answer is not a number, return None instead
        of falling back to searching the full response (Issue 1)."""
        response = "My reservation price is $3.00 so I should bid lower. ANSWER: hmm maybe"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price is None
        assert "reservation price" in reasoning

    def test_tag_found_but_empty_answer_returns_none(self):
        """When tag IS found but nothing after it, return None."""
        response = "I think about this carefully. ANSWER:"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price is None
        assert "carefully" in reasoning

    def test_reasoning_price_not_leaked_when_tag_present(self):
        """A price in reasoning must NOT be extracted when the tag is present
        but the answer portion is invalid."""
        response = "The equilibrium is around $2.50 based on history. ANSWER: I'll go with that"
        price, reasoning = _extract_price(response, answer_tag="ANSWER:")
        assert price is None  # Must NOT extract $2.50 from reasoning


# ===========================================================================
# Unit tests: _extract_response with CoT
# ===========================================================================


class TestExtractResponseWithCoT:
    """Tests for _extract_response with answer_tag support."""

    def test_answer_tag_yes(self):
        response = "This is above my reservation price so it's a good deal. ANSWER: yes"
        accepted, reasoning = _extract_response(response, answer_tag="ANSWER:")
        assert accepted is True
        assert "reservation price" in reasoning

    def test_answer_tag_no(self):
        response = "The price is too high for me. ANSWER: no"
        accepted, reasoning = _extract_response(response, answer_tag="ANSWER:")
        assert accepted is False
        assert "too high" in reasoning

    def test_no_tag_fallback(self):
        """Without answer_tag, existing behavior."""
        accepted, reasoning = _extract_response("yes", answer_tag=None)
        assert accepted is True
        assert reasoning == ""

    def test_no_tag_no(self):
        accepted, reasoning = _extract_response("no")
        assert accepted is False
        assert reasoning == ""

    def test_case_insensitive_tag(self):
        response = "Thinking about it... answer: Yes"
        accepted, reasoning = _extract_response(response, answer_tag="ANSWER:")
        assert accepted is True
        assert "Thinking" in reasoning

    def test_empty_response(self):
        accepted, reasoning = _extract_response("", answer_tag="ANSWER:")
        assert accepted is False
        assert reasoning == ""

    def test_tag_not_found_falls_back(self):
        """If tag not found, fall back to existing yes/no detection."""
        accepted, reasoning = _extract_response("yes", answer_tag="ANSWER:")
        assert accepted is True

    def test_reasoning_with_yes_word_doesnt_confuse(self):
        """'yes' in reasoning shouldn't matter if tag splits correctly."""
        response = "Yesterday the price was good, yes it was. ANSWER: no"
        accepted, reasoning = _extract_response(response, answer_tag="ANSWER:")
        assert accepted is False
        assert "Yesterday" in reasoning

    def test_tag_found_but_garbled_answer_returns_false(self):
        """When tag IS found but answer has no yes/no, return False instead
        of falling back to searching the full response (Issue 1)."""
        response = "I think yes this is a good deal, yes indeed. ANSWER: maybe later"
        accepted, reasoning = _extract_response(response, answer_tag="ANSWER:")
        assert accepted is False
        assert "yes indeed" in reasoning

    def test_tag_found_empty_answer_returns_false(self):
        """When tag IS found but nothing after it, return False."""
        response = "Let me think about this yes. ANSWER:"
        accepted, reasoning = _extract_response(response, answer_tag="ANSWER:")
        assert accepted is False


# ===========================================================================
# Unit tests: reasoning field reset in control nodes (Issue 3)
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
# Integration test: full CoT workflow through announce + respond nodes
# ===========================================================================


def _make_config(callbacks=None):
    """Create a minimal RunnableConfig-like dict."""
    return {"callbacks": callbacks or []}


class TestCoTIntegration:
    """Integration tests: mock LLM returns CoT-style responses, verify reasoning
    is captured in state and flows into IterationRecord."""

    def test_announce_captures_reasoning(self, base_market_state, prompt_config):
        """announce node should populate last_announcement_reasoning from CoT response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "My reservation price is $2.00 and the market seems quiet. "
            "I should bid conservatively. ANSWER: 1.75"
        )
        mock_llm.last_tool_log = []

        node = make_announce_node(mock_llm, prompt_config, answer_tag="ANSWER:")

        state = {**base_market_state}
        state["announcing_agent_id"] = 0  # buyer with reservation $2.00

        result = node(state, _make_config())

        assert result["announced_price"] == 1.75
        assert result["announcement_made"] is True
        assert "reservation price" in result["last_announcement_reasoning"]
        assert "conservatively" in result["last_announcement_reasoning"]

    def test_respond_captures_reasoning(self, base_market_state, prompt_config):
        """respond node should populate last_response_reasoning from CoT response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "The offered price of $1.50 is above my reservation price of $1.00. "
            "This is profitable. ANSWER: yes"
        )
        mock_llm.last_tool_log = []

        node = make_respond_node(mock_llm, prompt_config, answer_tag="ANSWER:")

        state = {**base_market_state}
        state["potential_responder_ids"] = [3]  # seller with reservation $1.00
        state["current_responder_index"] = 0
        state["announcement_type"] = "buy"
        state["announcing_agent_id"] = 0
        state["announced_price"] = 1.50

        result = node(state, _make_config())

        assert result["response_accepted"] is True
        assert "profitable" in result["last_response_reasoning"]
        assert "offered price" in result["last_response_reasoning"]

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

    def test_full_announce_respond_reasoning_flow(self, base_market_state, prompt_config):
        """End-to-end: announce with CoT, then respond with CoT, then verify
        both reasoning strings flow into an IterationRecord."""
        # -- Step 1: announce --
        announce_llm = MagicMock()
        announce_llm.invoke.return_value = (
            "Given my reservation of $2.00 I should bid below. ANSWER: 1.60"
        )
        announce_llm.last_tool_log = []

        announce_node = make_announce_node(announce_llm, prompt_config, answer_tag="ANSWER:")

        state = {**base_market_state}
        state["announcing_agent_id"] = 0
        announce_result = announce_node(state, _make_config())

        assert announce_result["announced_price"] == 1.60
        assert "reservation" in announce_result["last_announcement_reasoning"]

        # -- Step 2: respond --
        respond_llm = MagicMock()
        respond_llm.invoke.return_value = (
            "The bid of $1.60 is above my cost of $1.00. Good deal. ANSWER: yes"
        )
        respond_llm.last_tool_log = []

        respond_node = make_respond_node(respond_llm, prompt_config, answer_tag="ANSWER:")

        state["announced_price"] = announce_result["announced_price"]
        state["announcement_type"] = announce_result["announcement_type"]
        state["announcement_made"] = announce_result["announcement_made"]
        state["potential_responder_ids"] = [3]
        state["current_responder_index"] = 0
        state["announcing_agent_id"] = 0

        respond_result = respond_node(state, _make_config())

        assert respond_result["response_accepted"] is True
        assert "Good deal" in respond_result["last_response_reasoning"]

        # -- Step 3: update_history to create IterationRecord --
        state["transaction_made"] = True
        state["responding_agent_id"] = 3
        state["response_accepted"] = True
        state["iteration_complete"] = True
        state["current_responder_index"] = 1
        state["last_announcement_reasoning"] = announce_result["last_announcement_reasoning"]
        state["last_response_reasoning"] = respond_result["last_response_reasoning"]

        history_node = make_update_history_node()
        history_result = history_node(state)

        records = history_result["iteration_records"]
        assert len(records) == 1
        record = records[0]
        assert record["announcement_reasoning"] != ""
        assert record["response_reasoning"] != ""
        assert "reservation" in record["announcement_reasoning"]
        assert "Good deal" in record["response_reasoning"]

    def test_ambiguity_detection_scoped_to_answer_portion(self, base_market_state, prompt_config):
        """When CoT is enabled, ambiguity detection should check only the answer
        portion, not the full response which contains yes/no in reasoning."""
        mock_llm = MagicMock()
        # Reasoning contains "yes" and "no", but the answer portion is garbled
        mock_llm.invoke.return_value = (
            "I'm not sure. On one hand yes the price is good, "
            "on the other hand no it could be better. ANSWER: hmm"
        )
        mock_llm.last_tool_log = []

        node = make_respond_node(mock_llm, prompt_config, answer_tag="ANSWER:")

        state = {**base_market_state}
        state["potential_responder_ids"] = [3]
        state["current_responder_index"] = 0
        state["announcement_type"] = "buy"
        state["announcing_agent_id"] = 0
        state["announced_price"] = 1.50

        result = node(state, _make_config())

        # The answer portion "hmm" has no yes/no, so it should be flagged as ambiguous
        assert "parse_failures" in result
        assert result["parse_failures"] >= 1
