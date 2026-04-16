"""LLM provider factory."""

from __future__ import annotations

from ..config.schema import LLMConfig, ToolConfig
from .providers.base import LLMProvider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.deepseek import DeepSeekProvider

from ..tools.sandbox import SandboxManager

from ..tools.registry import ToolRegistry
from .tool_augmented import ToolAugmentedProvider


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
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. Supported: {supported}"
        )

    return provider_class(config)


def create_tool_augmented_llm(
    llm_config: LLMConfig,
    tool_config: ToolConfig,
    sandbox_manager: SandboxManager | None = None,
) -> LLMProvider | ToolAugmentedProvider:
    """Create an LLM provider, optionally wrapped with tool support.

    Args:
        llm_config: LLM configuration.
        tool_config: Tool configuration.
        sandbox_manager: Optional sandbox manager for E2B code interpreter.

    Returns:
        Base LLMProvider when tools disabled, ToolAugmentedProvider when enabled.
    """
    base_provider = create_llm(llm_config)

    if not tool_config.enabled:
        return base_provider

    registry = ToolRegistry(tool_config, sandbox_manager)
    return ToolAugmentedProvider(
        base_provider,
        registry,
        max_iterations=tool_config.max_tool_iterations,
    )
