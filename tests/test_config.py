"""Tests for configuration loading and schema validation."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from market_simulation.config.schema import (
    SimulationConfig,
    ExperimentConfig,
    LLMConfig,
    AgentPricesConfig,
    TracingConfig,
    ToolConfig,
    PromptConfig,
    PromptTemplates,
    AgentPromptConfig,
    AgentKeywords,
)
from market_simulation.config.settings import load_config, get_configs_dir


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------


class TestSchemaDefaults:
    """Tests that Pydantic schema defaults are correct."""

    def test_llm_config_defaults(self):
        """LLMConfig should have sensible provider/model defaults."""
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.0
        assert cfg.max_retries == 5

    def test_experiment_config_defaults(self):
        """ExperimentConfig should default to 5 rounds, 10 sims."""
        cfg = ExperimentConfig()
        assert cfg.n_rounds == 5
        assert cfg.n_simulations == 10

    def test_agent_prices_config_defaults(self):
        """AgentPricesConfig should default to 0.8-3.2 with 11 agents."""
        cfg = AgentPricesConfig()
        assert cfg.min == 0.8
        assert cfg.max == 3.2
        assert cfg.num == 11

    def test_tool_config_defaults(self):
        """ToolConfig should be disabled by default."""
        cfg = ToolConfig()
        assert cfg.enabled is False
        assert cfg.enable_simple_tools is True
        assert cfg.enable_code_interpreter is False

    def test_tracing_config_defaults(self):
        """TracingConfig should be enabled by default with no keys."""
        cfg = TracingConfig()
        assert cfg.enabled is True
        assert cfg.langfuse_public_key is None

    def test_simulation_config_constructs_with_defaults(self):
        """SimulationConfig should construct with all defaults (except the
        required CDA tick budget)."""
        cfg = SimulationConfig(experiment=ExperimentConfig(max_ticks_per_round=50))
        assert cfg.llm.provider == "openai"
        assert cfg.experiment.n_rounds == 5


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests that schema validation catches bad inputs."""

    def test_llm_config_rejects_invalid_provider(self):
        """LLMConfig should reject providers not in the literal set."""
        with pytest.raises(Exception):
            LLMConfig(provider="invalid_provider")

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "deepseek"])
    def test_llm_config_accepts_valid_providers(self, provider):
        """LLMConfig should accept all supported provider strings."""
        cfg = LLMConfig(provider=provider)
        assert cfg.provider == provider

    def test_agent_keywords_requires_all_fields(self):
        """AgentKeywords should require role, verb, preference, condition."""
        with pytest.raises(Exception):
            AgentKeywords(role="buyer")  # missing verb, preference, condition

    def test_experiment_config_custom_values(self):
        """ExperimentConfig should accept custom round / sim counts."""
        cfg = ExperimentConfig(n_rounds=3, n_simulations=2)
        assert cfg.n_rounds == 3
        assert cfg.n_simulations == 2


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_config_from_full_path(self, tmp_path):
        """load_config should parse a YAML file given a full path."""
        config_data = {
            "experiment": {
                "n_rounds": 1, "n_simulations": 1,
                "max_ticks_per_round": 10,
            },
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "tracing": {"enabled": False},
        }
        cfg_file = tmp_path / "test_cfg.yaml"
        cfg_file.write_text(yaml.dump(config_data))

        cfg = load_config(cfg_file)
        assert cfg.experiment.n_rounds == 1
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.tracing.enabled is False

    def test_load_config_file_not_found(self):
        """load_config should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_sets_tracing_defaults(self, tmp_path):
        """load_config should inject tracing section when absent in YAML."""
        config_data = {
            "llm": {"provider": "openai"},
            "experiment": {"max_ticks_per_round": 50},
        }
        cfg_file = tmp_path / "no_tracing.yaml"
        cfg_file.write_text(yaml.dump(config_data))

        cfg = load_config(cfg_file)
        assert cfg.tracing is not None

    def test_load_real_test_config(self, configs_dir):
        """load_config should parse the real test.yaml without errors."""
        test_cfg = configs_dir / "test.yaml"
        if not test_cfg.exists():
            pytest.skip("test.yaml not found in configs/")
        cfg = load_config(test_cfg)
        assert cfg.experiment.n_rounds >= 1
        assert cfg.llm.provider in ("openai", "anthropic", "gemini", "deepseek")

    @pytest.mark.parametrize(
        "config_name",
        ["openai", "anthropic", "gemini", "deepseek"],
    )
    def test_load_provider_configs(self, configs_dir, config_name):
        """Each provider config file should parse without errors."""
        cfg_file = configs_dir / f"{config_name}.yaml"
        if not cfg_file.exists():
            pytest.skip(f"{config_name}.yaml not found")
        cfg = load_config(cfg_file)
        assert cfg.llm.provider == config_name


class TestGetConfigsDir:
    """Tests for get_configs_dir utility."""

    def test_configs_dir_is_directory(self):
        """get_configs_dir should return a path that exists."""
        d = get_configs_dir()
        assert d.exists()
        assert d.is_dir()
