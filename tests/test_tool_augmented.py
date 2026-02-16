"""Tests for ToolAugmentedProvider with internal tool-calling loop."""

import pytest
from unittest.mock import MagicMock

from market_simulation.llm.tool_augmented import ToolAugmentedProvider


def _make_provider(tool_registry=None, max_iterations=5):
    """Create a ToolAugmentedProvider with mocked dependencies."""
    base = MagicMock()
    base.config = MagicMock()
    base.config.max_tokens_with_tools = 1024
    base._max_tokens_kwargs.return_value = {"max_tokens": 1024}
    base.model_name = "test-model"
    base.provider_name = "test"

    if tool_registry is None:
        tool_registry = MagicMock()
        tool_registry.has_tools = True
        tool_registry.tools = []
        tool_registry.tool_map = {}

    return ToolAugmentedProvider(base, tool_registry, max_iterations=max_iterations)


class TestToolAugmentedProviderNoTools:
    """Tests for when no tools are available."""

    def test_no_tools_falls_through_to_base_provider(self):
        """When has_tools is False, invoke should delegate to base_provider.invoke."""
        registry = MagicMock()
        registry.has_tools = False
        provider = _make_provider(tool_registry=registry)
        provider.base_provider.invoke.return_value = "1.50"

        result = provider.invoke("test prompt")

        assert result == "1.50"
        provider.base_provider.invoke.assert_called_once_with("test prompt", callbacks=None)

    def test_no_tools_passes_callbacks(self):
        """When has_tools is False, callbacks should be forwarded to base_provider."""
        registry = MagicMock()
        registry.has_tools = False
        provider = _make_provider(tool_registry=registry)
        provider.base_provider.invoke.return_value = "1.50"
        callbacks = [MagicMock()]

        provider.invoke("test", callbacks=callbacks)

        provider.base_provider.invoke.assert_called_once_with("test", callbacks=callbacks)


class TestToolAugmentedProviderWithTools:
    """Tests for the tool-calling loop."""

    def _setup_model(self, provider, responses):
        """Set up a mock model that returns a sequence of responses."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = responses
        mock_model.bind_tools.return_value = mock_model
        provider.base_provider.get_model.return_value = mock_model
        return mock_model

    def _make_tool_response(self, tool_calls, content=""):
        """Create a mock response with tool calls."""
        resp = MagicMock()
        resp.tool_calls = tool_calls
        resp.content = content
        return resp

    def _make_text_response(self, content):
        """Create a mock response with no tool calls (final answer)."""
        resp = MagicMock()
        resp.tool_calls = []
        resp.content = content
        return resp

    def test_direct_text_response(self):
        """When model returns text without tool calls, return immediately."""
        provider = _make_provider()
        self._setup_model(provider, [self._make_text_response("1.50")])

        result = provider.invoke("test")

        assert result == "1.50"
        assert provider.last_tool_log == []

    def test_tool_call_then_text(self):
        """Model calls a tool, gets result, then returns text."""
        provider = _make_provider()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "42"
        provider.tool_registry.tool_map = {"calc": mock_tool}

        responses = [
            self._make_tool_response([{"name": "calc", "args": {"x": 1}, "id": "t1"}]),
            self._make_text_response("The answer is 1.50"),
        ]
        self._setup_model(provider, responses)

        result = provider.invoke("test")

        assert result == "The answer is 1.50"
        assert len(provider.last_tool_log) == 1
        assert provider.last_tool_log[0]["tool_name"] == "calc"
        assert "42" in provider.last_tool_log[0]["tool_result"]

    def test_tool_loop_exhaustion(self):
        """When model always returns tool calls, loop hits max_iterations."""
        provider = _make_provider(max_iterations=2)
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "42"
        provider.tool_registry.tool_map = {"calc": mock_tool}

        # Always return tool calls, never final text
        tool_response = self._make_tool_response(
            [{"name": "calc", "args": {}, "id": "t1"}],
            content="partial",
        )
        self._setup_model(provider, [tool_response, tool_response])

        result = provider.invoke("test")

        # Should have logged 2 tool calls (max_iterations=2)
        assert len(provider.last_tool_log) == 2

    def test_unknown_tool_returns_error_string(self):
        """When model calls a tool not in registry, error string is returned."""
        provider = _make_provider(max_iterations=2)
        provider.tool_registry.tool_map = {}  # No tools registered

        responses = [
            self._make_tool_response([{"name": "nonexistent", "args": {}, "id": "t1"}]),
            self._make_text_response("1.50"),
        ]
        self._setup_model(provider, responses)

        result = provider.invoke("test")

        assert result == "1.50"
        assert len(provider.last_tool_log) == 1
        assert "Error: Unknown tool" in provider.last_tool_log[0]["tool_result"]

    def test_tool_execution_failure_returns_error(self):
        """When a tool raises an exception, error is captured in log and loop continues."""
        provider = _make_provider(max_iterations=2)
        failing_tool = MagicMock()
        failing_tool.invoke.side_effect = ValueError("divide by zero")
        provider.tool_registry.tool_map = {"calc": failing_tool}

        responses = [
            self._make_tool_response([{"name": "calc", "args": {}, "id": "t1"}]),
            self._make_text_response("1.50"),
        ]
        self._setup_model(provider, responses)

        result = provider.invoke("test")

        assert result == "1.50"
        assert "Error executing" in provider.last_tool_log[0]["tool_result"]
        assert "divide by zero" in provider.last_tool_log[0]["tool_result"]

    def test_tool_log_structure(self):
        """Tool log entries should have expected keys."""
        provider = _make_provider()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "result"
        provider.tool_registry.tool_map = {"my_tool": mock_tool}

        responses = [
            self._make_tool_response([{"name": "my_tool", "args": {"a": 1}, "id": "t1"}]),
            self._make_text_response("done"),
        ]
        self._setup_model(provider, responses)

        provider.invoke("test")

        assert len(provider.last_tool_log) == 1
        entry = provider.last_tool_log[0]
        assert "tool_name" in entry
        assert "tool_args" in entry
        assert "tool_result" in entry
        assert "iteration" in entry
        assert entry["tool_name"] == "my_tool"
        assert entry["iteration"] == 1

    def test_last_tool_log_reset_between_invocations(self):
        """last_tool_log should be reset at the start of each invoke()."""
        provider = _make_provider()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "42"
        provider.tool_registry.tool_map = {"calc": mock_tool}

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        provider.base_provider.get_model.return_value = mock_model

        # First call: tool then text
        mock_model.invoke.side_effect = [
            self._make_tool_response([{"name": "calc", "args": {}, "id": "t1"}]),
            self._make_text_response("first"),
        ]
        provider.invoke("test1")
        assert len(provider.last_tool_log) == 1

        # Second call: direct text (no tools)
        mock_model.invoke.side_effect = [self._make_text_response("second")]
        provider.invoke("test2")
        assert len(provider.last_tool_log) == 0


class TestToolAugmentedProviderProperties:
    """Tests for delegated properties."""

    def test_model_name_delegates(self):
        provider = _make_provider()
        provider.base_provider.model_name = "test-model"
        assert provider.model_name == "test-model"

    def test_provider_name_delegates(self):
        provider = _make_provider()
        provider.base_provider.provider_name = "test-provider"
        assert provider.provider_name == "test-provider"

    def test_config_delegates(self):
        provider = _make_provider()
        mock_config = MagicMock()
        provider.base_provider.config = mock_config
        assert provider.config is mock_config
