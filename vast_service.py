"""
Vast.ai service automation helpers.

Use this module from a long-running Python service to automate:
search -> launch -> download (GCS) -> run -> download -> destroy.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vast_gpu_manager import DOCKER_IMAGES, VastGPUManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaunchResult:
    instance_id: int
    offer_id: int
    raw_response: dict[str, Any]
    exit_code: int | None = None


def _resolve_image(image: str) -> str:
    return DOCKER_IMAGES.get(image, image)


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


def _get_ssh_info(manager: VastGPUManager, instance_id: int) -> tuple[str, int] | None:
    instance = manager.get_instance(instance_id)
    if instance:
        host = instance.get("ssh_host")
        port = instance.get("ssh_port")
        if host and port:
            return (host, int(port))
    return None


def wait_for_ssh(
    manager: VastGPUManager,
    instance_id: int,
    timeout_sec: int = 300,
    poll_interval_sec: int = 5,
    job_id: str | None = None,
) -> tuple[str, int]:
    """Wait until SSH is available for an instance or raise TimeoutError."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        ssh_info = _get_ssh_info(manager, instance_id)
        if ssh_info:
            host, port = ssh_info
            try:
                import socket

                with socket.create_connection((host, port), timeout=5):
                    return ssh_info
            except OSError:
                pass
        time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"SSH not available for instance {instance_id} after {timeout_sec}s job_id={job_id}"
    )


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


def launch_offer(
    manager: VastGPUManager,
    offer_id: int,
    image: str,
    ports: str | None = None,
    onstart_cmd: str | None = None,
    env_vars: dict[str, str] | None = None,
    disk_space: float | None = None,
    fallback_timeout_sec: int = 60,
    fallback_poll_sec: int = 5,
    job_id: str | None = None,
) -> LaunchResult:
    image = _resolve_image(image)
    before_ids: set[int] = set()
    existing = manager.list_instances()
    if isinstance(existing, list):
        before_ids = {inst.get("id") for inst in existing if inst.get("id")}

    response = manager.launch_by_offer_id(
        offer_id=offer_id,
        image=image,
        disk_space=disk_space,
        ports=ports,
        onstart_cmd=onstart_cmd,
        env_vars=env_vars,
        ssh=True,
    )

    instance_id: int | None = None
    if isinstance(response, dict):
        raw_id = response.get("new_contract") or response.get("instance_id")
        if raw_id:
            instance_id = int(raw_id)

    if instance_id is None:
        # SDK may return None/empty response even if instance was created.
        logger.warning(
            "launch_offer missing instance_id offer_id=%s job_id=%s response=%s",
            offer_id,
            job_id,
            response,
        )
        deadline = time.time() + fallback_timeout_sec
        while time.time() < deadline:
            instances = manager.list_instances()
            if isinstance(instances, list):
                candidates = [
                    inst
                    for inst in instances
                    if inst.get("id") and inst.get("id") not in before_ids
                ]
                if candidates:
                    matched = [c for c in candidates if c.get("offer_id") == offer_id]
                    if matched:
                        instance_id = int(matched[0]["id"])
                        break

                    def _parse_instance_time(inst: dict[str, Any]) -> float:
                        for key in ("created_at", "start_date", "start_time"):
                            val = inst.get(key)
                            if isinstance(val, int | float):
                                return float(val)
                            if isinstance(val, str):
                                try:
                                    return datetime.fromisoformat(
                                        val.replace("Z", "+00:00")
                                    ).timestamp()
                                except ValueError:
                                    continue
                        return 0.0

                    now = time.time()
                    candidates = [
                        c
                        for c in candidates
                        if (ts := _parse_instance_time(c)) == 0.0
                        or ts >= now - fallback_timeout_sec
                    ]
                    if not candidates:
                        time.sleep(fallback_poll_sec)
                        continue

                    candidates.sort(key=_parse_instance_time, reverse=True)
                    if len(candidates) > 1 and _parse_instance_time(candidates[0]) == 0.0:
                        logger.warning(
                            "launch_offer ambiguous fallback (no timestamps) offer_id=%s job_id=%s candidates=%s",
                            offer_id,
                            job_id,
                            [c.get("id") for c in candidates],
                        )
                    instance_id = int(candidates[0]["id"])
                    logger.warning(
                        "launch_offer fallback selected instance_id=%s offer_id=%s job_id=%s",
                        instance_id,
                        offer_id,
                        job_id,
                    )
                    break
            time.sleep(fallback_poll_sec)

    if instance_id is None:
        raise RuntimeError(f"Failed to launch offer {offer_id}: {response}")

    return LaunchResult(instance_id=int(instance_id), offer_id=offer_id, raw_response=response)


def download(
    manager: VastGPUManager,
    instance_id: int,
    src: str,
    dst: str | Path = "./",
    scp_bin: str = "scp",
    job_id: str | None = None,
) -> None:
    host, port = wait_for_ssh(manager, instance_id, job_id=job_id)
    dst_path = str(dst)
    scp_cmd = [
        scp_bin,
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-r",
        f"root@{host}:{src}",
        dst_path,
    ]
    logger.info(
        "download instance_id=%s job_id=%s host=%s src=%s dst=%s",
        instance_id,
        job_id,
        host,
        src,
        dst_path,
    )
    subprocess.run(scp_cmd, check=True)


def run(
    manager: VastGPUManager,
    instance_id: int,
    cmd: str,
    ssh_bin: str = "ssh",
    job_id: str | None = None,
) -> int:
    host, port = wait_for_ssh(manager, instance_id, job_id=job_id)
    ssh_cmd = [
        ssh_bin,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{host}",
        cmd,
    ]
    logger.info(
        "run instance_id=%s job_id=%s host=%s port=%s cmd=%s",
        instance_id,
        job_id,
        host,
        port,
        cmd,
    )
    completed = subprocess.run(ssh_cmd)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, ssh_cmd)
    return completed.returncode


def run_with_retries(
    manager: VastGPUManager,
    instance_id: int,
    cmd: str,
    retries: int = 5,
    backoff_sec: float = 5.0,
    ssh_bin: str = "ssh",
    job_id: str | None = None,
) -> int:
    """Run a command with retries to handle transient SSH errors."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return run(manager, instance_id, cmd, ssh_bin=ssh_bin, job_id=job_id)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
            else:
                break
    if last_error:
        raise last_error
    return 1


def run_and_capture(
    manager: VastGPUManager,
    instance_id: int,
    cmd: str,
    log_path: str | Path,
    ssh_bin: str = "ssh",
    job_id: str | None = None,
) -> int:
    """Run a command on the instance and capture stdout/stderr to a local log file."""
    host, port = wait_for_ssh(manager, instance_id, job_id=job_id)
    ssh_cmd = [
        ssh_bin,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{host}",
        cmd,
    ]
    log_file_path = Path(log_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "run_capture instance_id=%s job_id=%s host=%s port=%s cmd=%s",
        instance_id,
        job_id,
        host,
        port,
        cmd,
    )
    with log_file_path.open("wb") as log_file:
        completed = subprocess.run(ssh_cmd, stdout=log_file, stderr=log_file)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, ssh_cmd)
    return completed.returncode


def destroy_with_retries(
    manager: VastGPUManager,
    instance_id: int,
    retries: int = 3,
    backoff_sec: float = 5.0,
    verify: bool = True,
    verify_timeout_sec: int = 60,
    verify_poll_interval_sec: int = 5,
    job_id: str | None = None,
) -> None:
    """Destroy an instance with retries in case the API call fails."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "destroy instance_id=%s job_id=%s attempt=%s/%s",
                instance_id,
                job_id,
                attempt,
                retries,
            )
            manager.destroy_instance(instance_id)
            if verify:
                deadline = time.time() + verify_timeout_sec
                while time.time() < deadline:
                    instances = manager.list_instances()
                    if not isinstance(instances, list):
                        break
                    if not any(inst.get("id") == instance_id for inst in instances):
                        return
                    time.sleep(verify_poll_interval_sec)
                logger.warning(
                    "Instance %s still present after destroy attempt %s job_id=%s",
                    instance_id,
                    attempt,
                    job_id,
                )
                if attempt < retries:
                    time.sleep(backoff_sec * attempt)
                continue
            return
        except Exception as exc:  # noqa: BLE001 - surface API failures
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
            else:
                logger.error(
                    "Failed to destroy instance %s after %s attempts", instance_id, retries
                )
    if last_error:
        raise last_error


def train_with_cheapest_instance(
    api_key: str | None,
    *,
    job_id: str | None = None,
    image: str,
    ports: str | None,
    dataset_dst: str,
    run_cmd: str,
    artifact_src: str,
    artifact_dst: str | Path = "./",
    log_path: str | Path | None = None,
    raise_on_nonzero: bool = True,
    gcp_sa_b64: str | None = None,
    dataset_gs_uri: str | None = None,
    dataset_archive_name: str | None = None,
    extract_cmd: str | None = None,
    install_gsutil: bool = False,
    train_dataset_url: str | None = None,
    train_env: dict[str, str] | None = None,
    ssh_timeout_sec: int = 120,
    ssh_poll_interval_sec: int = 10,
    cmd_retries: int = 10,
    cmd_backoff_sec: float = 10.0,
    max_price: float | None = 0.04,
    min_cuda: float | None = None,
    max_cuda: float | None = 12.9,
    max_launch_attempts: int = 5,
    launch_retry_backoff_sec: float = 5.0,
    destroy_retries: int = 3,
    destroy_backoff_sec: float = 5.0,
    offer_blacklist_path: str | Path = ".vast_offer_blacklist.json",
    offer_blacklist_ttl_sec: int = 3600,
) -> LaunchResult:
    """
    End-to-end workflow:
    1) find cheapest offer (optional max_cuda)
    2) launch instance
    3) download dataset (GCS)
    4) run command
    5) download artifact
    6) destroy instance (always, with retries)

    If raise_on_nonzero is False, the function will return with exit_code set
    even when the remote command fails (non-zero).
    """
    manager = VastGPUManager(api_key=api_key)
    tried_offer_ids: set[int] = set()
    stale_instance_ids: list[int] = []

    if min_cuda is None:
        env_min_cuda = os.environ.get("MIN_CUDA")
        if env_min_cuda:
            try:
                min_cuda = float(env_min_cuda)
            except ValueError:
                logger.warning("Invalid MIN_CUDA value %r; ignoring", env_min_cuda)
                min_cuda = None

    if gcp_sa_b64 is None:
        gcp_sa_b64 = os.environ.get("GCP_SA_B64")

    env_vars = None
    onstart_cmd = None

    if gcp_sa_b64:
        env_vars = {"GCP_SA_B64": gcp_sa_b64}
        onstart_parts = []
        if install_gsutil:
            onstart_parts.append("apt-get update && apt-get install -y google-cloud-cli")
        onstart_parts.append("printf %s \"$GCP_SA_B64\" | tr -d '\\r' | base64 -d > /root/gcp.json")
        onstart_parts.append("chmod 600 /root/gcp.json")
        onstart_cmd = " && ".join(onstart_parts)

    launch: LaunchResult | None = None
    last_boot_error: Exception | None = None
    attempts = 0
    blacklist = _load_offer_blacklist(offer_blacklist_path)
    blacklist = _prune_offer_blacklist(blacklist, offer_blacklist_ttl_sec)
    while attempts < max_launch_attempts:
        offers = _rank_offers(manager, max_price=max_price, min_cuda=min_cuda, max_cuda=max_cuda)
        if not offers:
            break
        next_offer = None
        for offer in offers:
            offer_id = int(offer.get("id", 0) or 0)
            if offer_id and offer_id not in tried_offer_ids:
                if _is_offer_blacklisted(blacklist, offer_id):
                    continue
                next_offer = offer
                tried_offer_ids.add(offer_id)
                break
        if next_offer is None:
            break

        attempts += 1
        logger.info("Selected offer:")
        logger.info("%s", _format_offer_header())
        logger.info("%s", "-" * 90)
        logger.info("%s", _format_offer_line(next_offer))
        logger.info(
            "launching offer_id=%s job_id=%s attempt=%s/%s",
            next_offer.get("id"),
            job_id,
            attempts,
            max_launch_attempts,
        )
        try:
            launch = launch_offer(
                manager,
                offer_id=int(next_offer["id"]),
                image=image,
                ports=ports,
                env_vars=env_vars,
                onstart_cmd=onstart_cmd,
                job_id=job_id,
            )
        except Exception as exc:  # noqa: BLE001
            last_boot_error = exc
            blacklist = _add_offer_blacklist(
                blacklist, int(next_offer["id"]), offer_blacklist_ttl_sec
            )
            _save_offer_blacklist(offer_blacklist_path, blacklist)
            launch = None
            time.sleep(launch_retry_backoff_sec)
            continue
        try:
            wait_for_ssh(
                manager,
                launch.instance_id,
                timeout_sec=ssh_timeout_sec,
                poll_interval_sec=ssh_poll_interval_sec,
                job_id=job_id,
            )
            last_boot_error = None
            break
        except TimeoutError as exc:
            last_boot_error = exc
            blacklist = _add_offer_blacklist(
                blacklist, int(next_offer["id"]), offer_blacklist_ttl_sec
            )
            _save_offer_blacklist(offer_blacklist_path, blacklist)
            # Instance didn't come up in time; destroy and try next offer.
            try:
                destroy_with_retries(
                    manager,
                    launch.instance_id,
                    retries=destroy_retries,
                    backoff_sec=destroy_backoff_sec,
                    job_id=job_id,
                )
            except Exception as destroy_exc:  # noqa: BLE001 - log and keep retrying
                logger.warning(
                    "Failed to destroy instance %s after boot timeout: %s",
                    launch.instance_id,
                    destroy_exc,
                )
                stale_instance_ids.append(launch.instance_id)
            launch = None
            time.sleep(launch_retry_backoff_sec)

    if launch is None:
        for instance_id in stale_instance_ids:
            try:
                destroy_with_retries(
                    manager,
                    instance_id,
                    retries=destroy_retries,
                    backoff_sec=destroy_backoff_sec,
                    job_id=job_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Final cleanup failed for instance %s: %s", instance_id, exc)
        raise RuntimeError(
            f"No instance booted within {ssh_timeout_sec}s after {attempts} attempts"
        ) from last_boot_error

    if train_dataset_url is None and dataset_gs_uri:
        train_dataset_url = dataset_gs_uri

    if not train_dataset_url:
        raise ValueError("train_dataset_url is required (e.g. gs://bucket/path)")

    env_vars_cmd: dict[str, str] = {"TRAIN_DATASET_URL": train_dataset_url}
    if gcp_sa_b64:
        env_vars_cmd["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/gcp.json"
    if train_env:
        env_vars_cmd.update(train_env)
    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_vars_cmd.items())
    run_cmd_with_dataset = f"{env_prefix} {run_cmd}"
    logger.info("train_command job_id=%s cmd=%s", job_id, run_cmd_with_dataset)

    exit_code: int | None = None
    try:
        if dataset_gs_uri:
            if not gcp_sa_b64:
                raise ValueError("gcp_sa_b64 is required when dataset_gs_uri is provided")
            archive_name = dataset_archive_name or dataset_gs_uri.rstrip("/").split("/")[-1]
            archive_path = f"{dataset_dst.rstrip('/')}/{archive_name}"
            cmds: list[str] = [f"mkdir -p {dataset_dst}"]
            if gcp_sa_b64:
                cmds.append("test -s /root/gcp.json")
                cmds.append(
                    "GOOGLE_APPLICATION_CREDENTIALS=/root/gcp.json "
                    "gcloud auth activate-service-account --key-file /root/gcp.json "
                    "2>/dev/null || true"
                )
                cmds.append(
                    "GOOGLE_APPLICATION_CREDENTIALS=/root/gcp.json "
                    f"gsutil -m cp {dataset_gs_uri} {archive_path}"
                )
            else:
                cmds.append(f"gsutil -m cp {dataset_gs_uri} {archive_path}")
            if extract_cmd:
                cmds.append(extract_cmd.format(archive=archive_path, dst=dataset_dst))
            elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
                cmds.append(f"tar -xzf {archive_path} -C {dataset_dst}")
            logger.info("dataset_prepare job_id=%s cmd=%s", job_id, " && ".join(cmds))
            run_with_retries(
                manager,
                launch.instance_id,
                " && ".join(cmds),
                retries=cmd_retries,
                backoff_sec=cmd_backoff_sec,
                job_id=job_id,
            )
            if dataset_dst:
                verify_cmd = (
                    f"ls -la {dataset_dst} && "
                    "find /work/datasets -maxdepth 3 -type f -name '*.yaml' | head -n 20"
                )
                logger.info("dataset_verify job_id=%s cmd=%s", job_id, verify_cmd)
                run_with_retries(
                    manager,
                    launch.instance_id,
                    verify_cmd,
                    retries=3,
                    backoff_sec=5.0,
                    job_id=job_id,
                )
        verify_yolo_cmd = "command -v yolo && yolo --version"
        logger.info("yolo_verify job_id=%s cmd=%s", job_id, verify_yolo_cmd)
        run_with_retries(
            manager,
            launch.instance_id,
            verify_yolo_cmd,
            retries=2,
            backoff_sec=5.0,
            job_id=job_id,
        )
        if "yolo11s.pt" in run_cmd:
            weights_cmd = "ls -la yolo11s.pt || true"
            logger.info("weights_verify job_id=%s cmd=%s", job_id, weights_cmd)
            run_with_retries(
                manager,
                launch.instance_id,
                weights_cmd,
                retries=2,
                backoff_sec=5.0,
                job_id=job_id,
            )
        try:
            if log_path is None:
                exit_code = run_with_retries(
                    manager,
                    launch.instance_id,
                    run_cmd_with_dataset,
                    retries=cmd_retries,
                    backoff_sec=cmd_backoff_sec,
                    job_id=job_id,
                )
            else:
                exit_code = run_and_capture(
                    manager,
                    launch.instance_id,
                    run_cmd_with_dataset,
                    log_path,
                    job_id=job_id,
                )
        except subprocess.CalledProcessError as exc:
            exit_code = exc.returncode
            if raise_on_nonzero:
                raise
        download(manager, launch.instance_id, artifact_src, artifact_dst, job_id=job_id)
    finally:
        # Always destroy to avoid accidental billing.
        destroy_with_retries(
            manager,
            launch.instance_id,
            retries=destroy_retries,
            backoff_sec=destroy_backoff_sec,
            job_id=job_id,
        )

    return LaunchResult(
        instance_id=launch.instance_id,
        offer_id=launch.offer_id,
        raw_response=launch.raw_response,
        exit_code=exit_code,
    )
