"""OpenAI LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def _create_model(self) -> BaseChatModel:
        """Create OpenAI chat model.

        Returns:
            ChatOpenAI: OpenAI chat model instance.
        """
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )
