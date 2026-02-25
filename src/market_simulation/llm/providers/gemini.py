"""Google Gemini LLM provider."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""

    def _create_model(self) -> BaseChatModel:
        """Create Gemini chat model.

        Returns:
            ChatGoogleGenerativeAI: Gemini chat model instance.
        """
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

    def _max_tokens_kwargs(self, max_tokens: int) -> dict[str, Any]:
        """Gemini uses max_output_tokens instead of max_tokens."""
        return {"max_output_tokens": max_tokens}
