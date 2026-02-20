from __future__ import annotations

import re


def extract_gpu_model_number(gpu_name: str) -> int | None:
    """
    Extract the numeric model number from a GPU name.

    Args:
        gpu_name: GPU name like "RTX 3090", "GTX 1080 Ti", "A100".

    Returns:
        The extracted model number or None if not found.

    Examples:
        "RTX 3090" -> 3090
        "GTX 1080 Ti" -> 1080
        "A100" -> 100
        "H100" -> 100
        "Tesla V100" -> 100
    """
    numbers = re.findall(r"\d+", gpu_name)
    if numbers:
        # Return the largest number (usually the model number)
        return max(int(n) for n in numbers)
    return None
