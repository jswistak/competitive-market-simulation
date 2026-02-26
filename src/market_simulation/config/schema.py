"""Pydantic configuration schemas."""

from pydantic import BaseModel, Field
from typing import Literal


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "gemini", "deepseek"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 10
    max_tokens_with_tools: int = 1024
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_backoff_factor: float = 2.0


class ToolConfig(BaseModel):
    """Tool availability configuration for agents."""

    enabled: bool = False
    enable_simple_tools: bool = True
    enable_code_interpreter: bool = False
    e2b_timeout: int = 300
    max_tool_iterations: int = 5


class AgentPricesConfig(BaseModel):
    """Reservation price distribution for agents."""

    min: float = 0.8
    max: float = 3.2
    num: int = 11


class HistoryConfig(BaseModel):
    """Configuration for how market history is presented in prompts."""

    mode: Literal["full", "summary"] = "full"
    own_history_mode: Literal["full", "summary"] = "full"
    summary_last_n_events: int = 3


class ExperimentConfig(BaseModel):
    """Experiment parameters."""

    n_rounds: int = 5
    n_iterations: int = 10
    n_simulations: int = 10
    buyers: AgentPricesConfig = Field(default_factory=AgentPricesConfig)
    sellers: AgentPricesConfig = Field(default_factory=AgentPricesConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)


class TracingConfig(BaseModel):
    """Langfuse tracing configuration."""

    enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


class PromptTemplates(BaseModel):
    """Prompt templates for agent communication."""

    main_template: str = ""
    announcement_history_template: str = ""
    response_history_template: str = ""


class AgentKeywords(BaseModel):
    """Keywords for prompt substitution."""

    role: str
    verb: str
    preference: str
    condition: str


class AgentPromptConfig(BaseModel):
    """Agent-specific prompt configuration."""

    main_keywords: AgentKeywords
    response_prompt: str
    announcement_prompt: str


class PromptConfig(BaseModel):
    """Complete prompt configuration."""

    general: PromptTemplates = Field(default_factory=PromptTemplates)
    tools_preamble: str = ""
    buyer: AgentPromptConfig | None = None
    seller: AgentPromptConfig | None = None


class PersonaConfig(BaseModel):
    """Per-agent persona/prompt customization."""

    buyer_default: str = ""
    seller_default: str = ""
    buyers: dict[int, str] = Field(default_factory=dict)
    sellers: dict[int, str] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    personas: PersonaConfig = Field(default_factory=PersonaConfig)
