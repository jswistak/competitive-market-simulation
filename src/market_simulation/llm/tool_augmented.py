"""Tool-augmented LLM provider with internal tool-calling loop."""

import logging
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from langchain_core.messages import HumanMessage, ToolMessage

from .providers.base import _normalize_content
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolAugmentedProvider:
    """Wraps an LLMProvider to support tool calling with an internal agent loop.

    From the outside, callers still use invoke(prompt) -> str.
    Internally, the LLM may make tool calls which are executed and fed back
    until a final text response is produced.

    Tool usage is logged to self.last_tool_log after each invoke() call.
    """

    def __init__(
        self,
        base_provider: Any,
        tool_registry: ToolRegistry,
        max_iterations: int = 5,
    ) -> None:
        self.base_provider = base_provider
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._model_with_tools: Any = None
        self.last_tool_log: list[dict] = []

    def _get_model_with_tools(self) -> Any:
        if self._model_with_tools is None:
            model = self.base_provider.get_model()
            self._model_with_tools = model.bind_tools(self.tool_registry.tools)
        return self._model_with_tools

    def invoke(
        self,
        prompt: str,
        callbacks: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the LLM with tool-calling loop.

        Args:
            prompt: The prompt text.
            callbacks: Optional callbacks for tracing.
            metadata: Optional per-call metadata forwarded to LangChain callbacks.

        Returns:
            Final text response from the LLM.
        """
        self.last_tool_log = []

        if not self.tool_registry.has_tools:
            return self.base_provider.invoke(prompt, callbacks=callbacks, metadata=metadata)

        model = self._get_model_with_tools()
        messages: list[Any] = [HumanMessage(content=prompt)]

        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks
        if metadata:
            config["metadata"] = metadata

        logger.info(f"Entering tool-calling loop (max {self.max_iterations} iterations)")

        for iteration in range(self.max_iterations):
            response = model.invoke(
                messages,
                config=config,
                **self.base_provider._max_tokens_kwargs(self.base_provider.config.max_tokens_with_tools),
            )

            # If no tool calls, we have our final answer
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                logger.info(f"Tool loop resolved after {iteration + 1} iteration(s)")
                return _normalize_content(response.content)

            messages.append(response)

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                logger.debug(f"Tool call [{iteration+1}/{self.max_iterations}]: {tool_name}({tool_args})")

                tool = self.tool_registry.tool_map.get(tool_name)
                if tool is None:
                    result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool.invoke(tool_args)
                    except Exception as e:
                        logger.warning(f"Tool {tool_name} failed: {e}")
                        result = f"Error executing {tool_name}: {e}"

                self.last_tool_log.append({
                    "tool_name": tool_name,
                    "tool_args": str(tool_args)[:500],
                    "tool_result": str(result)[:500],
                    "iteration": iteration + 1,
                })

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # Exhausted iterations — extract whatever text we can from the last message
        tool_names_called = [entry["tool_name"] for entry in self.last_tool_log]
        logger.warning(
            f"Tool loop reached max iterations ({self.max_iterations}). "
            f"Tools called: {tool_names_called}"
        )
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            return _normalize_content(last_msg.content)
        return ""

    def invoke_structured(
        self,
        prompt: str,
        schema: type[PydanticBaseModel],
        callbacks: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PydanticBaseModel:
        """Invoke the LLM with tool-calling loop and return structured output.

        When no tools are configured, delegates directly to base provider.
        When tools are enabled, runs the tool loop to accumulate context,
        then uses structured output for the final extraction.

        Args:
            prompt: The prompt text.
            schema: Pydantic model class to parse the response into.
            callbacks: Optional callbacks for tracing.
            metadata: Optional per-call metadata forwarded to LangChain callbacks.

        Returns:
            An instance of the provided schema.
        """
        self.last_tool_log = []

        if not self.tool_registry.has_tools:
            return self.base_provider.invoke_structured(
                prompt, schema, callbacks=callbacks, metadata=metadata,
            )

        model = self._get_model_with_tools()
        messages: list[Any] = [HumanMessage(content=prompt)]

        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks
        if metadata:
            config["metadata"] = metadata

        logger.info(f"Entering structured tool-calling loop (max {self.max_iterations} iterations)")

        for iteration in range(self.max_iterations):
            response = model.invoke(
                messages,
                config=config,
                **self.base_provider._max_tokens_kwargs(self.base_provider.config.max_tokens_with_tools),
            )

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                logger.info(f"Tool loop resolved after {iteration + 1} iteration(s), extracting structured output")
                structured_model = self.base_provider.get_model().with_structured_output(schema)
                return structured_model.invoke(
                    messages, config=config,
                    **self.base_provider._max_tokens_kwargs(self.base_provider.config.max_tokens),
                )

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                logger.debug(f"Tool call [{iteration+1}/{self.max_iterations}]: {tool_name}({tool_args})")

                tool = self.tool_registry.tool_map.get(tool_name)
                if tool is None:
                    result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool.invoke(tool_args)
                    except Exception as e:
                        logger.warning(f"Tool {tool_name} failed: {e}")
                        result = f"Error executing {tool_name}: {e}"

                self.last_tool_log.append({
                    "tool_name": tool_name,
                    "tool_args": str(tool_args)[:500],
                    "tool_result": str(result)[:500],
                    "iteration": iteration + 1,
                })

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        # Exhausted iterations — final structured extraction attempt
        tool_names_called = [entry["tool_name"] for entry in self.last_tool_log]
        logger.warning(
            f"Tool loop reached max iterations ({self.max_iterations}). "
            f"Tools called: {tool_names_called}. Attempting final structured extraction."
        )
        structured_model = self.base_provider.get_model().with_structured_output(schema)
        return structured_model.invoke(
            messages, config=config,
            **self.base_provider._max_tokens_kwargs(self.base_provider.config.max_tokens),
        )

    def get_model(self) -> Any:
        """Get the underlying chat model (without tool binding)."""
        return self.base_provider.get_model()

    @property
    def model_name(self) -> str:
        return self.base_provider.model_name

    @property
    def provider_name(self) -> str:
        return self.base_provider.provider_name

    @property
    def config(self) -> Any:
        return self.base_provider.config
