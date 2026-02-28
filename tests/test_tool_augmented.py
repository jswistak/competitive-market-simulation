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


class TestToolAugmentedProviderStructuredNoTools:
    """Tests for invoke_structured when no tools are available."""

    def test_no_tools_delegates_to_base(self):
        """When has_tools is False, invoke_structured delegates to base_provider."""
        from market_simulation.llm.response_schemas import AnnouncementResponseWithReasoning

        registry = MagicMock()
        registry.has_tools = False
        provider = _make_provider(tool_registry=registry)
        expected = AnnouncementResponseWithReasoning(price=1.5, reasoning="test")
        provider.base_provider.invoke_structured.return_value = expected

        result = provider.invoke_structured("test", AnnouncementResponseWithReasoning)

        assert result is expected
        provider.base_provider.invoke_structured.assert_called_once_with(
            "test", AnnouncementResponseWithReasoning, callbacks=None
        )

    def test_no_tools_passes_callbacks(self):
        """When has_tools is False, callbacks forwarded to base_provider.invoke_structured."""
        from market_simulation.llm.response_schemas import BidResponseWithReasoning

        registry = MagicMock()
        registry.has_tools = False
        provider = _make_provider(tool_registry=registry)
        expected = BidResponseWithReasoning(bid=3.0, reasoning="")
        provider.base_provider.invoke_structured.return_value = expected
        cbs = [MagicMock()]

        provider.invoke_structured("test", BidResponseWithReasoning, callbacks=cbs)

        provider.base_provider.invoke_structured.assert_called_once_with(
            "test", BidResponseWithReasoning, callbacks=cbs
        )


class TestToolAugmentedProviderStructuredWithTools:
    """Tests for invoke_structured with the tool-calling loop."""

    def _make_tool_response(self, tool_calls, content=""):
        resp = MagicMock()
        resp.tool_calls = tool_calls
        resp.content = content
        return resp

    def _make_text_response(self, content):
        resp = MagicMock()
        resp.tool_calls = []
        resp.content = content
        return resp

    def _setup_model(self, provider, tool_responses, structured_result):
        """Set up mock model for tool loop + structured extraction.

        tool_responses: list of responses the tool-bound model returns.
        structured_result: what the structured_model.invoke returns at the end.
        """
        mock_model = MagicMock()
        mock_model.invoke.side_effect = tool_responses
        mock_model.bind_tools.return_value = mock_model

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = structured_result
        # get_model().with_structured_output() returns the structured chain
        base_model = MagicMock()
        base_model.with_structured_output.return_value = mock_structured
        provider.base_provider.get_model.return_value = base_model

        # bind_tools should be called on the base model
        base_model.bind_tools.return_value = mock_model
        provider._model_with_tools = None  # Force re-creation

        return mock_model, mock_structured

    def test_direct_text_triggers_structured_extraction(self):
        """When model returns no tool calls immediately, structured extraction runs."""
        from market_simulation.llm.response_schemas import AcceptRejectResponseWithReasoning

        provider = _make_provider()
        expected = AcceptRejectResponseWithReasoning(accept=True, reasoning="good deal")
        _, mock_structured = self._setup_model(
            provider,
            [self._make_text_response("yes")],
            expected,
        )

        result = provider.invoke_structured("test", AcceptRejectResponseWithReasoning)

        assert result is expected
        assert provider.last_tool_log == []
        mock_structured.invoke.assert_called_once()

    def test_tool_call_then_structured_extraction(self):
        """Model calls a tool, gets result, then structured extraction runs."""
        from market_simulation.llm.response_schemas import BidResponseWithReasoning

        provider = _make_provider()
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "profit=2.5"
        provider.tool_registry.tool_map = {"evaluate_trade": mock_tool}

        expected = BidResponseWithReasoning(bid=3.0, reasoning="calculated")
        _, mock_structured = self._setup_model(
            provider,
            [
                self._make_tool_response([{"name": "evaluate_trade", "args": {"price": 3.0}, "id": "t1"}]),
                self._make_text_response("I'll bid 3.0"),
            ],
            expected,
        )

        result = provider.invoke_structured("test", BidResponseWithReasoning)

        assert result is expected
        assert len(provider.last_tool_log) == 1
        assert provider.last_tool_log[0]["tool_name"] == "evaluate_trade"

    def test_max_iterations_exhausted_still_extracts(self):
        """When tool loop exhausts iterations, structured extraction still attempted."""
        from market_simulation.llm.response_schemas import AnnouncementResponseWithReasoning

        provider = _make_provider(max_iterations=2)
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "stats"
        provider.tool_registry.tool_map = {"compute_market_stats": mock_tool}

        expected = AnnouncementResponseWithReasoning(price=1.5, reasoning="forced")
        tool_resp = self._make_tool_response(
            [{"name": "compute_market_stats", "args": {}, "id": "t1"}]
        )
        _, mock_structured = self._setup_model(
            provider,
            [tool_resp, tool_resp],
            expected,
        )

        result = provider.invoke_structured("test", AnnouncementResponseWithReasoning)

        assert result is expected
        assert len(provider.last_tool_log) == 2
        mock_structured.invoke.assert_called_once()

    def test_tool_log_reset_between_structured_calls(self):
        """last_tool_log is reset at the start of each invoke_structured."""
        from market_simulation.llm.response_schemas import AcceptRejectResponseWithReasoning

        provider = _make_provider()
        expected = AcceptRejectResponseWithReasoning(accept=False, reasoning="")

        # First call with a tool
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "data"
        provider.tool_registry.tool_map = {"calc": mock_tool}

        _, mock_structured = self._setup_model(
            provider,
            [
                self._make_tool_response([{"name": "calc", "args": {}, "id": "t1"}]),
                self._make_text_response("no"),
            ],
            expected,
        )
        provider.invoke_structured("test1", AcceptRejectResponseWithReasoning)
        assert len(provider.last_tool_log) == 1

        # Second call — direct text, no tools
        mock_model_2 = MagicMock()
        mock_model_2.invoke.side_effect = [self._make_text_response("done")]
        provider._model_with_tools = mock_model_2

        provider.invoke_structured("test2", AcceptRejectResponseWithReasoning)
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
