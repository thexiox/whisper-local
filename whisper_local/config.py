"""Configuration loading and validation for whisper-local."""

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

VALID_MODELS = ("tiny", "base", "small", "medium", "large-v3")
VALID_DEVICES = ("auto", "cpu", "cuda")
VALID_COMPUTE_TYPES = ("auto", "float16", "int8")
CONFIG_FILE = "config.yaml"


@dataclass
class AppConfig:
    """Application configuration with sane defaults."""

    model_size: str = "large-v3"
    language: str = "es"
    device: str = "auto"
    compute_type: str = "auto"
    hotkey: str = "<ctrl>+<shift>+r"
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    continuous_chunk_seconds: int = 5
    continuous: bool = False
    mic_index: int | None = None
    verbose: bool = False


def _detect_cuda() -> bool:
    """Check if CUDA is available via ctranslate2."""
    try:
        import ctranslate2
        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False


def _resolve_auto(config: AppConfig) -> AppConfig:
    """Resolve 'auto' values for device and compute_type."""
    if config.device == "auto":
        config.device = "cuda" if _detect_cuda() else "cpu"
        logger.debug("Auto-detected device: %s", config.device)

    if config.compute_type == "auto":
        config.compute_type = "float16" if config.device == "cuda" else "int8"
        logger.debug("Auto-selected compute_type: %s", config.compute_type)

    return config


def _validate(config: AppConfig) -> None:
    """Validate configuration values."""
    if config.model_size not in VALID_MODELS:
        raise ValueError(
            f"Invalid model_size '{config.model_size}'. "
            f"Valid options: {', '.join(VALID_MODELS)}"
        )

    if config.device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid device '{config.device}'. "
            f"Valid options: {', '.join(VALID_DEVICES)}"
        )

    if config.compute_type not in VALID_COMPUTE_TYPES:
        raise ValueError(
            f"Invalid compute_type '{config.compute_type}'. "
            f"Valid options: {', '.join(VALID_COMPUTE_TYPES)}"
        )

    if config.sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {config.sample_rate}")

    if config.continuous_chunk_seconds <= 0:
        raise ValueError(
            f"continuous_chunk_seconds must be positive, got {config.continuous_chunk_seconds}"
        )


def load_config(cli_args: dict[str, Any]) -> AppConfig:
    """Load configuration from YAML file and apply CLI overrides.

    Priority: CLI flags > config.yaml > defaults.
    """
    config = AppConfig()

    # Load YAML config if it exists
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        logger.info("Loading config from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        config_fields = {f.name for f in fields(AppConfig)}
        for key, value in yaml_data.items():
            if key in config_fields:
                setattr(config, key, value)
            else:
                logger.warning("Unknown config key '%s' in %s, ignoring", key, CONFIG_FILE)

    # Apply CLI overrides (only non-None values)
    cli_mapping = {
        "model": "model_size",
        "language": "language",
        "cpu": None,  # handled specially
        "continuous": "continuous",
        "mic": "mic_index",
        "verbose": "verbose",
    }

    for cli_key, config_key in cli_mapping.items():
        value = cli_args.get(cli_key)
        if value is None or value is False:
            continue
        if cli_key == "cpu" and value:
            config.device = "cpu"
        elif config_key:
            setattr(config, config_key, value)

    _validate(config)
    config = _resolve_auto(config)

    return config
