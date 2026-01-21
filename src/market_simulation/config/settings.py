"""Configuration loading utilities."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .schema import SimulationConfig


def load_config(config_path: str | Path) -> SimulationConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file or config name (without extension).

    Returns:
        SimulationConfig: Parsed configuration object.
    """
    load_dotenv()

    config_path = Path(config_path)

    # If just a name, look in configs directory
    if not config_path.suffix:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / f"{config_path}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Override tracing config from environment variables if not set
    if "tracing" not in raw_config:
        raw_config["tracing"] = {}

    tracing = raw_config["tracing"]
    if not tracing.get("langfuse_public_key"):
        tracing["langfuse_public_key"] = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not tracing.get("langfuse_secret_key"):
        tracing["langfuse_secret_key"] = os.getenv("LANGFUSE_SECRET_KEY")
    if not tracing.get("langfuse_host"):
        tracing["langfuse_host"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    return SimulationConfig(**raw_config)


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent.parent.parent / "configs"
