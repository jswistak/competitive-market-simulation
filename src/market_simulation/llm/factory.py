"""LLM provider factory."""

from ..config.schema import LLMConfig
from .providers.base import LLMProvider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.deepseek import DeepSeekProvider


def create_llm(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: LLM configuration specifying provider and model.

    Returns:
        LLMProvider: Configured LLM provider instance.

    Raises:
        ValueError: If provider is not supported.
    """
    providers: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
    }

    provider_class = providers.get(config.provider)
    if provider_class is None:
        supported = ", ".join(providers.keys())
        raise ValueError(f"Unsupported LLM provider: {config.provider}. Supported: {supported}")

    return provider_class(config)
