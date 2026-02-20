from __future__ import annotations


class VastAIError(Exception):
    """Base exception for Vast.ai operations."""

    pass


class APIKeyError(VastAIError):
    """Raised when API key is missing or invalid."""

    pass


class InstanceNotFoundError(VastAIError):
    """Raised when instance is not found."""

    pass
