from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LaunchResult:
    instance_id: int
    offer_id: int
    raw_response: dict[str, Any]
    exit_code: int | None = None
