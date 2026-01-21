"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ...config.schema import LLMConfig


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

    def invoke(self, prompt: str, callbacks: list[Any] | None = None) -> str:
        """Invoke the model with a prompt.

        Args:
            prompt: The prompt text.
            callbacks: Optional list of callbacks for tracing.

        Returns:
            str: The model's response text.
        """
        model = self.get_model()
        message = HumanMessage(content=prompt)
        config = {"callbacks": callbacks} if callbacks else {}

        response = model.invoke([message], config=config, max_tokens=self.config.max_tokens)
        return response.content

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider
