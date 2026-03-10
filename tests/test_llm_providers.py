"""Tests for LLM provider base functionality and content normalization."""

import pytest
from unittest.mock import MagicMock, patch

from market_simulation.llm.providers.base import _normalize_content, LLMProvider
from market_simulation.config.schema import LLMConfig


class TestNormalizeContent:
    """Tests for _normalize_content helper that handles Gemini list responses."""

    def test_string_passthrough(self):
        """Plain string content should be returned as-is."""
        assert _normalize_content("1.50") == "1.50"

    def test_string_with_whitespace(self):
        """Strings with whitespace should be preserved."""
        assert _normalize_content("  1.50  ") == "  1.50  "

    def test_empty_string(self):
        """Empty string should be returned as-is."""
        assert _normalize_content("") == ""

    def test_gemini3_text_part(self):
        """Gemini 3 returns list of dicts with 'type' and 'text' keys."""
        content = [{"type": "text", "text": "1.50"}]
        assert _normalize_content(content) == "1.50"

    def test_gemini3_text_part_with_extras(self):
        """Gemini 3 responses include a 'signature' in extras — should be ignored."""
        content = [
            {
                "type": "text",
                "text": "1.0",
                "extras": {"signature": "abc123..."},
            }
        ]
        assert _normalize_content(content) == "1.0"

    def test_gemini3_multiline_response(self):
        """Gemini 3 may return text with leading newline (e.g. '\\nNo.')."""
        content = [{"type": "text", "text": "\nNo."}]
        assert _normalize_content(content) == "\nNo."

    def test_multiple_text_parts(self):
        """Multiple text parts should be concatenated."""
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World"},
        ]
        assert _normalize_content(content) == "Hello World"

    def test_empty_list(self):
        """Empty list (Gemini MAX_TOKENS exhausted) should return empty string."""
        assert _normalize_content([]) == ""

    def test_list_of_strings(self):
        """List of plain strings should be joined."""
        content = ["Hello", " ", "World"]
        assert _normalize_content(content) == "Hello World"

    def test_mixed_list_strings_and_dicts(self):
        """Mixed list of strings and dicts should extract text from both."""
        content = ["prefix ", {"type": "text", "text": "1.50"}]
        assert _normalize_content(content) == "prefix 1.50"

    def test_dict_without_text_key_skipped(self):
        """Dicts without a 'text' key (e.g. image parts) should be skipped."""
        content = [
            {"type": "image", "data": "binary..."},
            {"type": "text", "text": "2.00"},
        ]
        assert _normalize_content(content) == "2.00"

    def test_non_string_non_list_fallback(self):
        """Non-string, non-list types should fall back to str()."""
        assert _normalize_content(42) == "42"
        assert _normalize_content(None) == "None"

    def test_return_type_is_always_str(self):
        """Return value should always be a str regardless of input."""
        cases = [
            "hello",
            "",
            [],
            [{"type": "text", "text": "1.0"}],
            ["a", "b"],
            42,
            None,
        ]
        for case in cases:
            result = _normalize_content(case)
            assert isinstance(result, str), f"Expected str for input {case!r}, got {type(result)}"


class TestLLMProviderInvoke:
    """Tests that LLMProvider.invoke normalizes content for all response types."""

    def _make_provider(self, mock_response_content):
        """Create a concrete LLMProvider with a mocked model returning given content."""
        config = LLMConfig(provider="openai", model="test", max_tokens=10)

        # Create a mock model whose invoke returns an AIMessage-like object
        mock_response = MagicMock()
        mock_response.content = mock_response_content

        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response

        # Create a concrete subclass (can't instantiate ABC directly)
        class TestProvider(LLMProvider):
            def _create_model(self):
                return mock_model

        provider = TestProvider(config)
        return provider

    def test_invoke_with_string_content(self):
        """invoke() should return string content as-is."""
        provider = self._make_provider("1.50")
        assert provider.invoke("test prompt") == "1.50"

    def test_invoke_with_gemini3_list_content(self):
        """invoke() should extract text from Gemini 3 list-of-parts."""
        content = [{"type": "text", "text": "2.50", "extras": {"signature": "..."}}]
        provider = self._make_provider(content)
        assert provider.invoke("test prompt") == "2.50"

    def test_invoke_with_empty_list_content(self):
        """invoke() should return empty string for empty list (MAX_TOKENS hit)."""
        provider = self._make_provider([])
        assert provider.invoke("test prompt") == ""

    def test_invoke_always_returns_str(self):
        """invoke() return type should always be str."""
        for content in ["hello", [], [{"type": "text", "text": "x"}]]:
            provider = self._make_provider(content)
            result = provider.invoke("prompt")
            assert isinstance(result, str)
