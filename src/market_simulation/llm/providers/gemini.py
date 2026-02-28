"""Google Gemini LLM provider."""

from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
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
            thinking_level="low",
        )

    def _max_tokens_kwargs(self, max_tokens: int) -> dict[str, Any]:
        """Gemini uses max_output_tokens instead of max_tokens."""
        return {"max_output_tokens": max_tokens}

    def invoke_structured(
        self,
        prompt: str,
        schema: type[PydanticBaseModel],
        callbacks: list[Any] | None = None,
    ) -> PydanticBaseModel:
        """Invoke Gemini with structured output using native JSON schema.

        Uses method="json_schema" which constrains Gemini's generation
        directly via the response_json_schema API parameter, producing
        reliable structured responses.
        """
        model = self.get_model()
        structured_model = model.with_structured_output(schema, method="json_schema")
        message = HumanMessage(content=prompt)

        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks

        return structured_model.invoke(
            [message], config=config, **self._max_tokens_kwargs(self.config.max_tokens)
        )
