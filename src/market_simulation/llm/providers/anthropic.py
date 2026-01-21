"""Anthropic LLM provider."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider implementation."""

    def _create_model(self) -> BaseChatModel:
        """Create Anthropic chat model.

        Returns:
            ChatAnthropic: Anthropic chat model instance.
        """
        return ChatAnthropic(
            model=self.config.model,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )
