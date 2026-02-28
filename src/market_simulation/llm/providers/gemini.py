"""Google Gemini LLM provider."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import LLMProvider, _normalize_content

logger = logging.getLogger(__name__)


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

    def invoke_structured(
        self,
        prompt: str,
        schema: type[PydanticBaseModel],
        callbacks: list[Any] | None = None,
    ) -> PydanticBaseModel:
        """Invoke Gemini with structured output using function calling.

        Gemini's JSON mode can add preamble text that breaks parsing.
        Using method="function_calling" forces the native tool-calling API
        which returns clean structured data. If the model returns text instead
        of a function call, we attempt to extract JSON from the text as a fallback.
        """
        model = self.get_model()
        structured_model = model.with_structured_output(
            schema, method="function_calling", include_raw=True,
        )
        message = HumanMessage(content=prompt)

        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks

        result = structured_model.invoke(
            [message], config=config, **self._max_tokens_kwargs(self.config.max_tokens)
        )

        if result["parsed"] is not None:
            return result["parsed"]

        # Fallback: try to extract JSON from the raw text response
        raw = result["raw"]
        text = _normalize_content(raw.content)
        if text:
            logger.debug(f"Gemini returned text instead of function call, attempting JSON extraction: {text[:200]}")
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return schema.model_validate(data)
                except (json.JSONDecodeError, Exception):
                    pass

        raise ValueError(
            f"Gemini did not return a valid structured response "
            f"(finish_reason: {raw.response_metadata.get('finish_reason', 'unknown')})"
        )
