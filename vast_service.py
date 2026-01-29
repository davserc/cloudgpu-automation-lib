"""
Vast.ai service automation helpers.

Use this module from a long-running Python service to automate:
search -> launch -> upload -> run -> download -> stop/destroy.
"""

from __future__ import annotations

import logging
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
            return ssh_info
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"SSH not available for instance {instance_id} after {timeout_sec}s")


def find_cheapest_offer(
    manager: VastGPUManager,
    max_price: float | None = None,
    max_cuda: float | None = None,
    limit: int = 100,
) -> dict[str, Any] | None:
    """Return the cheapest offer that satisfies the optional max_cuda constraint."""
    offers = manager.search_gpus(
        max_price=max_price,
        limit=limit,
        order_by="price",
        order_desc=False,
    )
    if not isinstance(offers, list):
        return None

    if max_cuda is not None:
        offers = [
            o
            for o in offers
            if o.get("cuda_max_good") is not None and o["cuda_max_good"] <= max_cuda
        ]

    if not offers:
        return None

    offers.sort(key=lambda o: o.get("dph_total", float("inf")))
    return offers[0]


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


def upload(
    manager: VastGPUManager,
    instance_id: int,
    src: str | Path,
    dst: str = "/root/",
    scp_bin: str = "scp",
) -> None:
    host, port = wait_for_ssh(manager, instance_id)
    src_path = str(src)
    scp_cmd = [
        scp_bin,
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-r",
        src_path,
        f"root@{host}:{dst}",
    ]
    logger.info("Uploading %s -> %s:%s", src_path, host, dst)
    subprocess.run(scp_cmd, check=True)


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
    dataset_src: str | Path,
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
    max_price: float | None = None,
    max_cuda: float | None = 12.9,
    destroy_retries: int = 3,
    destroy_backoff_sec: float = 5.0,
) -> LaunchResult:
    """
    End-to-end workflow:
    1) find cheapest offer (optional max_cuda)
    2) launch instance
    3) upload dataset
    4) run command
    5) download artifact
    6) destroy instance (always, with retries)

    If raise_on_nonzero is False, the function will return with exit_code set
    even when the remote command fails (non-zero).
    """
    manager = VastGPUManager(api_key=api_key)
    offer = find_cheapest_offer(manager, max_price=max_price, max_cuda=max_cuda)
    if not offer:
        raise RuntimeError("No offers found that match the constraints")

    env_vars = None
    onstart_cmd = None

    if gcp_sa_b64:
        env_vars = {"GCP_SA_B64": gcp_sa_b64}
        onstart_parts = []
        if install_gsutil:
            onstart_parts.append("apt-get update && apt-get install -y google-cloud-cli")
        onstart_parts.append("echo $GCP_SA_B64 | base64 -d > /root/gcp.json")
        onstart_parts.append("export GOOGLE_APPLICATION_CREDENTIALS=/root/gcp.json")
        onstart_cmd = " && ".join(onstart_parts)

    launch = launch_offer(
        manager,
        offer_id=int(offer["id"]),
        image=image,
        ports=ports,
        env_vars=env_vars,
        onstart_cmd=onstart_cmd,
    )

    exit_code: int | None = None
    try:
        if dataset_gs_uri:
            if not gcp_sa_b64:
                raise ValueError("gcp_sa_b64 is required when dataset_gs_uri is provided")
            archive_name = dataset_archive_name or dataset_gs_uri.rstrip("/").split("/")[-1]
            archive_path = f"{dataset_dst.rstrip('/')}/{archive_name}"
            cmds: list[str] = [f"mkdir -p {dataset_dst}"]
            cmds.append(f"gsutil -m cp {dataset_gs_uri} {archive_path}")
            if extract_cmd:
                cmds.append(extract_cmd.format(archive=archive_path, dst=dataset_dst))
            elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
                cmds.append(f"tar -xzf {archive_path} -C {dataset_dst}")
            run(manager, launch.instance_id, " && ".join(cmds))
        else:
            upload(manager, launch.instance_id, dataset_src, dataset_dst)
        try:
            if log_path is None:
                exit_code = run(manager, launch.instance_id, run_cmd)
            else:
                exit_code = run_and_capture(manager, launch.instance_id, run_cmd, log_path)
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
