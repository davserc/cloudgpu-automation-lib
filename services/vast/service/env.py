from __future__ import annotations

import logging
import os

logger = logging.getLogger("vast_service")

MIN_FREE_DISK_GB_DEFAULT = 30.0


def _resolve_min_cuda(min_cuda: float | None) -> float | None:
    if min_cuda is not None:
        return min_cuda
    env_min_cuda = os.environ.get("MIN_CUDA")
    if not env_min_cuda:
        return None
    try:
        return float(env_min_cuda)
    except ValueError:
        logger.warning("Invalid MIN_CUDA value %r; ignoring", env_min_cuda)
        return None


def _resolve_min_free_disk_gb(min_free_disk_gb: float | None) -> float:
    if min_free_disk_gb is not None:
        return min_free_disk_gb
    env_val = os.environ.get("MIN_FREE_DISK_GB")
    if not env_val:
        return MIN_FREE_DISK_GB_DEFAULT
    try:
        return float(env_val)
    except ValueError:
        logger.warning("Invalid MIN_FREE_DISK_GB value %r; using default", env_val)
        return MIN_FREE_DISK_GB_DEFAULT
