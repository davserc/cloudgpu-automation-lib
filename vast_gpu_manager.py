"""
Vast.ai GPU Instance Manager.

Facade module: public API remains here, implementation is split across
smaller modules for maintainability.
"""

from __future__ import annotations

import logging

from services.vast.gpu_manager.constants import (
    DEFAULT_DISK_SPACE,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_GPU_NAME,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_RELIABILITY,
    DEFAULT_NUM_GPUS,
    DOCKER_IMAGES,
    MIN_DISK_SPACE,
)
from services.vast.gpu_manager.env import get_env, get_env_float, get_env_int
from services.vast.gpu_manager.errors import APIKeyError, InstanceNotFoundError, VastAIError
from services.vast.gpu_manager.manager import GPUConfig, VastGPUManager, quick_launch
from services.vast.gpu_manager.utils import extract_gpu_model_number

logger = logging.getLogger("vast_gpu_manager")

__all__ = [
    "DEFAULT_GPU_NAME",
    "DEFAULT_NUM_GPUS",
    "DEFAULT_MAX_PRICE",
    "DEFAULT_MIN_RELIABILITY",
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_DISK_SPACE",
    "MIN_DISK_SPACE",
    "DOCKER_IMAGES",
    "get_env",
    "get_env_float",
    "get_env_int",
    "extract_gpu_model_number",
    "GPUConfig",
    "VastGPUManager",
    "VastAIError",
    "APIKeyError",
    "InstanceNotFoundError",
    "quick_launch",
]


if __name__ == "__main__":
    import json

    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        manager = VastGPUManager()

        print("Searching for available GPUs...")
        offers = manager.search_gpus(limit=5)

        if isinstance(offers, list) and offers:
            print(f"\nFound {len(offers)} offers:")
            print(json.dumps(offers[0], indent=2, default=str))
        else:
            print("No offers found matching criteria")

        print("\nYour current instances:")
        instances = manager.list_instances()
        print(json.dumps(instances, indent=2, default=str))

    except APIKeyError as e:
        print(f"Error: {e}")
        print("\nSet your API key in .env file or environment:")
        print("  VASTAI_API_KEY='your_api_key_here'")
