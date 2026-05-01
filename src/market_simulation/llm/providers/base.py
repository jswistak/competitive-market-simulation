"""Base LLM provider interface."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ...config.schema import LLMConfig

logger = logging.getLogger(__name__)


def _normalize_content(content: Any) -> str:
    """Normalize LLM response content to a plain string.

    Newer Gemini models return content as a list of parts
    instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration.

        Args:
            config: LLM configuration.
        """
        self.config = config
        self._model: BaseChatModel | None = None

    @abstractmethod
    def _create_model(self) -> BaseChatModel:
        """Create the LangChain chat model instance.

        Returns:
            BaseChatModel: LangChain chat model.
        """
        pass

    def get_model(self) -> BaseChatModel:
        """Get or create the chat model.

        Returns:
            BaseChatModel: LangChain chat model instance.
        """
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def _max_tokens_kwargs(self, max_tokens: int) -> dict[str, Any]:
        """Get provider-specific kwargs for max tokens.

        Override in subclasses if the underlying API uses a different parameter name.
        """
        return {"max_tokens": max_tokens}

    def _build_messages(self, prompt: str, system: str | None) -> list[BaseMessage]:
        """Build the message list sent to the underlying chat model.

        With ``system=None`` returns ``[HumanMessage(prompt)]`` —
        identical to the pre-split behaviour. With a non-empty
        ``system`` returns ``[SystemMessage(system), HumanMessage(prompt)]``,
        which is what enables the per-tick / per-agent split and lets
        prompt caches reuse the system prefix across ticks.
        """
        if system:
            return [SystemMessage(content=system), HumanMessage(content=prompt)]
        return [HumanMessage(content=prompt)]

    def invoke(
        self,
        prompt: str,
        callbacks: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> str:
        """Invoke the model with a prompt.

        Args:
            prompt: The prompt text. With the system+user split this is
                the user-facing message; otherwise it is the full prompt.
            callbacks: Optional list of callbacks for tracing. If None, inherits
                from the current LangGraph/LangChain context automatically.
            metadata: Optional per-call metadata forwarded to LangChain callbacks
                via ``config.metadata`` (e.g. agent_id, action, round).
            system: Optional system-message content. When set, prepended
                as a SystemMessage so providers / prompt caches can
                treat it as the constant prefix.

        Returns:
            str: The model's response text.
        """
        model = self.get_model()
        messages = self._build_messages(prompt, system)

        # Build config - if explicit callbacks provided, use them
        # Otherwise, don't set callbacks to allow inheritance from parent context
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks
        if metadata:
            config["metadata"] = metadata

        response = model.invoke(
            messages, config=config, **self._max_tokens_kwargs(self.config.max_tokens)
        )
        content = _normalize_content(response.content)

        if not content.strip():
            metadata = getattr(response, "response_metadata", {}) or {}
            finish_reason = metadata.get("finish_reason", "unknown")
            log_msg = (
                f"Empty response from LLM (finish_reason: {finish_reason}, "
                f"model: {self.config.model}, max_tokens: {self.config.max_tokens})"
            )
            if finish_reason in ("length", "max_tokens"):
                logger.error(
                    f"TRUNCATED: {log_msg} — response was cut off by token limit, "
                    f"consider increasing max_tokens"
                )
            else:
                logger.warning(log_msg)

        return content

    def invoke_structured(
        self,
        prompt: str,
        schema: type[PydanticBaseModel],
        callbacks: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> PydanticBaseModel:
        """Invoke the model and return a structured Pydantic response.

        Args:
            prompt: The prompt text. With the system+user split this is
                the user-facing message.
            schema: Pydantic model class to parse the response into.
            callbacks: Optional list of callbacks for tracing.
            metadata: Optional per-call metadata forwarded to LangChain callbacks
                via ``config.metadata`` (e.g. agent_id, action, round).
            system: Optional system-message content. See :meth:`invoke`.

        Returns:
            An instance of the provided schema.
        """
        model = self.get_model()
        structured_model = model.with_structured_output(schema)
        messages = self._build_messages(prompt, system)

        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks
        if metadata:
            config["metadata"] = metadata

        return structured_model.invoke(
            messages, config=config, **self._max_tokens_kwargs(self.config.max_tokens)
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider
