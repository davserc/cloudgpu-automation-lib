from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("vast_gpu_manager")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.debug("Loaded environment from %s", _env_path)
except ImportError:
    logger.debug("python-dotenv not installed, using system environment variables")


def get_env(key: str, default: str | None = None) -> str | None:
    """
    Get environment variable with optional default value.

    Args:
        key: Environment variable name.
        default: Default value if variable is not set.

    Returns:
        Environment variable value or default.
    """
    return os.environ.get(key, default)


def get_env_float(key: str, default: float) -> float:
    """
    Get environment variable as float.

    Args:
        key: Environment variable name.
        default: Default value if variable is not set or invalid.

    Returns:
        Float value from environment or default.
    """
    try:
        value = os.environ.get(key)
        return float(value) if value else default
    except ValueError:
        logger.warning("Invalid float value for %s, using default: %s", key, default)
        return default


def get_env_int(key: str, default: int) -> int:
    """
    Get environment variable as integer.

    Args:
        key: Environment variable name.
        default: Default value if variable is not set or invalid.

    Returns:
        Integer value from environment or default.
    """
    try:
        value = os.environ.get(key)
        return int(value) if value else default
    except ValueError:
        logger.warning("Invalid int value for %s, using default: %s", key, default)
        return default
