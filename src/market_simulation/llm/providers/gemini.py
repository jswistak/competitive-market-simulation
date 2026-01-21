"""Google Gemini LLM provider."""

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
        )
