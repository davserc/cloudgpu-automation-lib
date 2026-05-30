from __future__ import annotations

import json
import logging
import os
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
        f"{'↓Mbps':<6} "
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
    inet_down = offer.get("inet_down") or 0
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
        f"{inet_down:<6.0f} "
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
    hosts = data.get("hosts", {})
    if not isinstance(hosts, dict):
        hosts = {}
    data["hosts"] = {k: v for k, v in hosts.items() if isinstance(v, int | float) and v > now}
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


def _add_host_blacklist(data: dict[str, Any], host_id: int | str, ttl_sec: int) -> dict[str, Any]:
    now = time.time()
    hosts = data.get("hosts", {})
    if not isinstance(hosts, dict):
        hosts = {}
    hosts[str(host_id)] = now + ttl_sec
    data["hosts"] = hosts
    data["ttl_sec"] = ttl_sec
    data["updated_at"] = now
    return data


def _is_host_blacklisted(data: dict[str, Any], host_id: int | str) -> bool:
    hosts = data.get("hosts", {})
    if not isinstance(hosts, dict):
        return False
    expiry = hosts.get(str(host_id))
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

    _blocked_countries = {
        c.strip().upper()
        for c in os.environ.get("VAST_BLOCKED_COUNTRIES", "CN,RU,IR").split(",")
        if c.strip()
    }

    def _is_blocked_location(geolocation: str) -> bool:
        if not geolocation or not _blocked_countries:
            return False
        # geolocation is like "Washington, US" or ", CN" or "Netherlands"
        parts = [p.strip().upper() for p in geolocation.split(",")]
        country = parts[-1] if parts else ""
        return country in _blocked_countries

    def _is_discouraged_gpu(gpu_name: str) -> bool:
        name = gpu_name.upper()
        # Non-NVIDIA: no CUDA support
        if any(x in name for x in ("RADEON", "RX 6", "RX 7", "RX 8", "INTEL ARC", "ARC A")):
            return True
        # Laptop/embedded GPUs: too weak
        if any(x in name for x in ("MX150", "MX250", "MX350", "MX450", "MX550", "MX570")):
            return True
        # Old Tesla/Quadro compute cards with CUDA 12 support but no good for training
        if "TESLA P4" in name or "TESLA P100" in name:
            return True
        if any(x in name for x in ("QUADRO P2000", "QUADRO P4000", "QUADRO P5000", "QUADRO P6000")):
            return True
        # Non-RTX Titan (K80/M40/etc already filtered by min_cuda=12)
        if "TITAN" in name and "TITAN RTX" not in name:
            return True
        # GTX 10-series: no tensor cores, slow for YOLO
        if "GTX" in name:
            for model in ("1050", "1060", "1070", "1080", "1090"):
                if f"GTX {model}" in name:
                    return True
        # GTX 16-series: no tensor cores (Turing without RT/Tensor)
        if "GTX 16" in name:
            return True
        # RTX 20-series with ≤6GB VRAM (RTX 2060 6GB): too small for imgsz=896
        if "RTX 2060" in name and "SUPER" not in name:
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

    logger.warning(
        "_rank_offers: after search => %d offers (max_price=%s max_cuda=%s min_cuda=%s)",
        len(offers),
        max_price,
        max_cuda,
        min_cuda,
    )
    if offers:
        sample = offers[0]
        logger.warning(
            "_rank_offers: sample offer => id=%s gpu=%s dph_total=%s dlperf=%s cuda=%s",
            sample.get("id"),
            sample.get("gpu_name"),
            sample.get("dph_total"),
            sample.get("dlperf"),
            sample.get("cuda_max_good"),
        )

    if max_cuda is not None:
        before = len(offers)
        offers = [
            o
            for o in offers
            if o.get("cuda_max_good") is not None and o["cuda_max_good"] <= max_cuda
        ]
        logger.warning("_rank_offers: after max_cuda filter => %d (was %d)", len(offers), before)
    if min_cuda is not None:
        before = len(offers)
        offers = [
            o
            for o in offers
            if o.get("cuda_max_good") is not None and o["cuda_max_good"] >= min_cuda
        ]
        logger.warning("_rank_offers: after min_cuda filter => %d (was %d)", len(offers), before)

    _min_compute_cap_raw = os.environ.get("VAST_MIN_COMPUTE_CAP")
    if _min_compute_cap_raw:
        try:
            _min_compute_cap = int(_min_compute_cap_raw)
            before = len(offers)
            offers = [
                o for o in offers
                if (o.get("compute_cap") or 0) >= _min_compute_cap
            ]
            logger.warning(
                "_rank_offers: after compute_cap filter (min=%d) => %d (was %d)",
                _min_compute_cap, len(offers), before,
            )
        except ValueError:
            logger.warning("VAST_MIN_COMPUTE_CAP invalid value=%s, skipping", _min_compute_cap_raw)

    _min_vram_gb = float(os.environ.get("VAST_MIN_VRAM_GB", "10") or "10")
    _min_inet_down_mbps = float(os.environ.get("VAST_MIN_INET_DOWN_MBPS", "0") or "0")

    before = len(offers)
    offers = [
        o
        for o in offers
        if o.get("gpu_name")
        and not _is_discouraged_gpu(o.get("gpu_name", ""))
        and not _is_blocked_location(o.get("geolocation", ""))
        and (o.get("dph_total") or 0) > 0
        and (o.get("gpu_ram") or 0) / 1024 >= _min_vram_gb
    ]
    logger.warning(
        "_rank_offers: after quality filter (min_vram=%.0fGB) => %d (was %d)",
        _min_vram_gb, len(offers), before,
    )

    if _min_inet_down_mbps > 0:
        before = len(offers)
        offers = [o for o in offers if (o.get("inet_down") or 0) >= _min_inet_down_mbps]
        logger.warning(
            "_rank_offers: after inet_down filter (min=%.0f Mbps) => %d (was %d)",
            _min_inet_down_mbps, len(offers), before,
        )

    if not offers:
        return []

    if not offers:
        return []

    # Priority: 1) price (cheapest first), 2) CUDA version (higher first),
    # 3) zone preference (configured via VAST_PREFERRED_ZONES env var).
    #
    # VAST_PREFERRED_ZONES: comma-separated country codes in preference order.
    # Example: "US,CA,PL,DE,NL,GB"  (first = most preferred)
    # Offers whose country code is not in the list get lowest priority.
    _preferred_zones = [
        z.strip().upper()
        for z in os.environ.get("VAST_PREFERRED_ZONES", "US,CA,PL,DE,NL,GB,CZ,FR,ES,IT").split(",")
        if z.strip()
    ]

    def _zone_rank(geolocation: str) -> int:
        if not geolocation:
            return len(_preferred_zones)
        # geolocation format: "Washington, US" / "Quebec, CA" / ", CN"
        parts = [p.strip().upper() for p in geolocation.split(",")]
        country = parts[-1] if parts else ""
        try:
            return _preferred_zones.index(country)
        except ValueError:
            return len(_preferred_zones)

    # Round price to 3 decimal places so near-identical prices ($0.035 vs $0.0351)
    # don't block the CUDA/zone tiebreakers from kicking in.
    def _score_key(o: dict[str, Any]) -> tuple[float, float, int]:
        price = round(o.get("dph_total", 0) or 0, 3)
        cuda = o.get("cuda_max_good", 0) or 0
        zone = _zone_rank(o.get("geolocation", ""))
        return (price, -cuda, zone)  # ascending: cheaper → higher CUDA → preferred zone

    offers.sort(key=_score_key)

    logger.info(
        "_rank_offers: top-3 after priority sort => %s",
        [
            f"{o.get('id')}({o.get('geolocation','?')},${o.get('dph_total',0):.3f},cuda={o.get('cuda_max_good')})"
            for o in offers[:3]
        ],
    )

    return offers
