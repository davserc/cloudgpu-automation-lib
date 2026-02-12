"""
Vast.ai service automation helpers.

Use this module from a long-running Python service to automate:
search -> launch -> download (GCS) -> run -> download -> destroy.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
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
    raise TimeoutError(f"SSH not available for instance {instance_id} after {timeout_sec}s")


def find_cheapest_offer(
    manager: VastGPUManager,
    max_price: float | None = None,
    max_cuda: float | None = None,
    limit: int = 100,
) -> dict[str, Any] | None:
    """Return the best-value "cheap" offer that satisfies optional constraints."""
    offers = _rank_offers(manager, max_price=max_price, max_cuda=max_cuda, limit=limit)
    return offers[0] if offers else None


def _rank_offers(
    manager: VastGPUManager,
    max_price: float | None = None,
    max_cuda: float | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return offers ranked by value (DLPerf/$), then VRAM, then price."""
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
) -> LaunchResult:
    image = _resolve_image(image)
    response = manager.launch_by_offer_id(
        offer_id=offer_id,
        image=image,
        disk_space=disk_space,
        ports=ports,
        onstart_cmd=onstart_cmd,
        env_vars=env_vars,
        ssh=True,
    )

    instance_id = response.get("new_contract") or response.get("instance_id")
    if not instance_id:
        raise RuntimeError(f"Failed to launch offer {offer_id}: {response}")

    return LaunchResult(instance_id=int(instance_id), offer_id=offer_id, raw_response=response)


def download(
    manager: VastGPUManager,
    instance_id: int,
    src: str,
    dst: str | Path = "./",
    scp_bin: str = "scp",
) -> None:
    host, port = wait_for_ssh(manager, instance_id)
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
    logger.info("Downloading %s:%s -> %s", host, src, dst_path)
    subprocess.run(scp_cmd, check=True)


def run(
    manager: VastGPUManager,
    instance_id: int,
    cmd: str,
    ssh_bin: str = "ssh",
) -> int:
    host, port = wait_for_ssh(manager, instance_id)
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
    logger.info("Running on %s:%s -> %s", host, port, cmd)
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
) -> int:
    """Run a command with retries to handle transient SSH errors."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return run(manager, instance_id, cmd, ssh_bin=ssh_bin)
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
) -> int:
    """Run a command on the instance and capture stdout/stderr to a local log file."""
    host, port = wait_for_ssh(manager, instance_id)
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
    logger.info("Running (capturing logs) on %s:%s -> %s", host, port, cmd)
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
) -> None:
    """Destroy an instance with retries in case the API call fails."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Destroying instance %s (attempt %s/%s)", instance_id, attempt, retries)
            manager.destroy_instance(instance_id)
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
    max_cuda: float | None = 12.9,
    max_launch_attempts: int = 3,
    destroy_retries: int = 3,
    destroy_backoff_sec: float = 5.0,
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
    offers = _rank_offers(manager, max_price=max_price, max_cuda=max_cuda)
    if not offers:
        raise RuntimeError("No offers found that match the constraints")

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
    for offer in offers:
        if attempts >= max_launch_attempts:
            break
        attempts += 1
        launch = launch_offer(
            manager,
            offer_id=int(offer["id"]),
            image=image,
            ports=ports,
            env_vars=env_vars,
            onstart_cmd=onstart_cmd,
        )
        try:
            wait_for_ssh(
                manager,
                launch.instance_id,
                timeout_sec=ssh_timeout_sec,
                poll_interval_sec=ssh_poll_interval_sec,
            )
            last_boot_error = None
            break
        except TimeoutError as exc:
            last_boot_error = exc
            # Instance didn't come up in time; destroy and try next offer.
            destroy_with_retries(
                manager,
                launch.instance_id,
                retries=destroy_retries,
                backoff_sec=destroy_backoff_sec,
            )
            launch = None

    if launch is None:
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
            run_with_retries(
                manager,
                launch.instance_id,
                " && ".join(cmds),
                retries=cmd_retries,
                backoff_sec=cmd_backoff_sec,
            )
        try:
            if log_path is None:
                exit_code = run_with_retries(
                    manager,
                    launch.instance_id,
                    run_cmd_with_dataset,
                    retries=cmd_retries,
                    backoff_sec=cmd_backoff_sec,
                )
            else:
                exit_code = run_and_capture(
                    manager, launch.instance_id, run_cmd_with_dataset, log_path
                )
        except subprocess.CalledProcessError as exc:
            exit_code = exc.returncode
            if raise_on_nonzero:
                raise
        download(manager, launch.instance_id, artifact_src, artifact_dst)
    finally:
        # Always destroy to avoid accidental billing.
        destroy_with_retries(
            manager,
            launch.instance_id,
            retries=destroy_retries,
            backoff_sec=destroy_backoff_sec,
        )

    return LaunchResult(
        instance_id=launch.instance_id,
        offer_id=launch.offer_id,
        raw_response=launch.raw_response,
        exit_code=exit_code,
    )
