"""Tests for chain-of-thought reasoning extraction."""

import pytest

from market_simulation.graph.nodes.announce import _extract_price
from market_simulation.graph.nodes.respond import _extract_response


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
