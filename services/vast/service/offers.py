from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from services.vast.gpu_manager import VastGPUManager

logger = logging.getLogger("vast_service")


def _format_offer_header() -> str:
    return (
        f"{'ID':<10} "
        f"{'GPU':<16} "
        f"{'#':<2} "
        f"{'VRAM':<5} "
        f"{'$/hr':<7} "
        f"{'DLP':<6} "
        f"{'TF':<5} "
        f"{'DLP/$':<6} "
        f"{'CUDA':<5} "
        f"{'Rel':<4} "
        f"{'Location':<10}"
    )


def _format_offer_line(offer: dict[str, Any]) -> str:
    vram_gb = (offer.get("gpu_ram", 0) or 0) / 1024
    price = offer.get("dph_total", 0) or 0
    dlperf = offer.get("dlperf", 0) or 0
    value = dlperf / price if price > 0 else 0
    location = offer.get("geolocation") or "N/A"
    if len(location) > 10:
        location = location[:8] + ".."
    return (
        f"{offer.get('id', 'N/A'):<10} "
        f"{offer.get('gpu_name', 'N/A'):<16} "
        f"{offer.get('num_gpus', 0):<2} "
        f"{vram_gb:<4.0f}G "
        f"${price:<6.3f} "
        f"{dlperf:<6.1f} "
        f"{offer.get('total_flops', 0):<5.1f} "
        f"{value:<6.0f} "
        f"{offer.get('cuda_max_good', 0):<5.1f} "
        f"{offer.get('reliability', 0) * 100:<4.0f} "
        f"{location:<10}"
    )


def _log_selected_offer(
    offer: dict[str, Any], job_id: str | None, attempt: int, max_attempts: int
) -> None:
    logger.info("Selected offer:")
    logger.info("%s", _format_offer_header())
    logger.info("%s", "-" * 90)
    logger.info("%s", _format_offer_line(offer))
    logger.info(
        "launching offer_id=%s job_id=%s attempt=%s/%s",
        offer.get("id"),
        job_id,
        attempt,
        max_attempts,
    )


def _load_offer_blacklist(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read offer blacklist %s: %s", path, exc)
        return {}


def _save_offer_blacklist(path: str | Path, data: dict[str, Any]) -> None:
    try:
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write offer blacklist %s: %s", path, exc)


def _prune_offer_blacklist(data: dict[str, Any], ttl_sec: int) -> dict[str, Any]:
    now = time.time()
    offers = data.get("offers", {})
    if not isinstance(offers, dict):
        offers = {}
    data["offers"] = {k: v for k, v in offers.items() if isinstance(v, int | float) and v > now}
    data["ttl_sec"] = ttl_sec
    data["updated_at"] = now
    return data


def _add_offer_blacklist(data: dict[str, Any], offer_id: int, ttl_sec: int) -> dict[str, Any]:
    now = time.time()
    offers = data.get("offers", {})
    if not isinstance(offers, dict):
        offers = {}
    offers[str(offer_id)] = now + ttl_sec
    data["offers"] = offers
    data["ttl_sec"] = ttl_sec
    data["updated_at"] = now
    return data


def _is_offer_blacklisted(data: dict[str, Any], offer_id: int) -> bool:
    offers = data.get("offers", {})
    if not isinstance(offers, dict):
        return False
    expiry = offers.get(str(offer_id))
    return isinstance(expiry, int | float) and expiry > time.time()


def find_cheapest_offer(
    manager: VastGPUManager,
    max_price: float | None = None,
    min_cuda: float | None = None,
    max_cuda: float | None = None,
    limit: int = 100,
) -> dict[str, Any] | None:
    """Return the best-value "cheap" offer that satisfies optional constraints."""
    offers = _rank_offers(
        manager, max_price=max_price, min_cuda=min_cuda, max_cuda=max_cuda, limit=limit
    )
    return offers[0] if offers else None


def _rank_offers(
    manager: VastGPUManager,
    max_price: float | None = None,
    min_cuda: float | None = None,
    max_cuda: float | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Return offers ranked by value (DLPerf/$), then VRAM, then price.

    Note: cuda_max_good is the maximum CUDA version supported by the host driver.
    Use min_cuda to enforce compatibility (cuda_max_good >= min_cuda).
    Use max_cuda to impose an upper cap (cuda_max_good <= max_cuda).
    """

    def _is_discouraged_gpu(gpu_name: str) -> bool:
        name = gpu_name.upper()
        if "TESLA P4" in name:
            return True
        if "QUADRO P2000" in name:
            return True
        if "TITAN" in name and "TITAN RTX" not in name:
            return True
        if "GTX" in name:
            # Discard GTX 10-series (older/less efficient for training)
            for model in ("1050", "1060", "1070", "1080", "1090"):
                if f"GTX {model}" in name:
                    return True
        return False

    offers = manager.search_gpus(
        max_price=max_price,
        limit=limit,
        order_by="value",
        order_desc=True,
    )
    if not offers:
        # Fallback: some environments return empty results with value ordering.
        offers = manager.search_gpus(
            max_price=max_price,
            limit=limit,
            order_by="price",
            order_desc=False,
        )
    if not isinstance(offers, list):
        return []

    if max_cuda is not None:
        offers = [
            o
            for o in offers
            if o.get("cuda_max_good") is not None and o["cuda_max_good"] <= max_cuda
        ]
    if min_cuda is not None:
        offers = [
            o
            for o in offers
            if o.get("cuda_max_good") is not None and o["cuda_max_good"] >= min_cuda
        ]

    offers = [
        o
        for o in offers
        if o.get("gpu_name")
        and not _is_discouraged_gpu(o.get("gpu_name", ""))
        and (o.get("dlperf") or 0) > 0
        and (o.get("dph_total") or 0) > 0
    ]

    if not offers:
        return []

    if max_price is None:
        # Define "cheap" as within 2x the cheapest offer.
        min_price = min(o.get("dph_total", float("inf")) for o in offers)
        cheap_cap = min_price * 2.0 if min_price != float("inf") else None
        if cheap_cap:
            cheap_offers = [o for o in offers if o.get("dph_total", 0) <= cheap_cap]
            if cheap_offers:
                offers = cheap_offers

    if not offers:
        return []

    # Rank by value first, then VRAM, then price.
    def _score_key(o: dict[str, Any]) -> tuple[float, float, float]:
        price = o.get("dph_total", 0) or 0
        dlperf = o.get("dlperf", 0) or 0
        vram_gb = (o.get("gpu_ram", 0) or 0) / 1024.0
        value = dlperf / price if price > 0 else 0
        return (value, vram_gb, -price)

    offers.sort(key=_score_key, reverse=True)
    return offers
