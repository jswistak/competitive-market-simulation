"""DeepSeek LLM provider."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .base import LLMProvider


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider implementation (uses OpenAI-compatible API)."""

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def _create_model(self) -> BaseChatModel:
        """Create DeepSeek chat model via OpenAI-compatible API.

        Returns:
            ChatOpenAI: DeepSeek chat model instance.
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            base_url=self.DEEPSEEK_BASE_URL,
            api_key=api_key,
        )
