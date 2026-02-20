from __future__ import annotations

import logging
import shlex
import subprocess
import time
from pathlib import Path

from services.vast.gpu_manager import VastGPUManager

logger = logging.getLogger("vast_service")


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


def run_and_get_output(
    manager: VastGPUManager,
    instance_id: int,
    cmd: str,
    ssh_bin: str = "ssh",
    job_id: str | None = None,
) -> str:
    """Run a command on the instance and return stdout as text."""
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
        "run_output instance_id=%s job_id=%s host=%s port=%s cmd=%s",
        instance_id,
        job_id,
        host,
        port,
        cmd,
    )
    completed = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, ssh_cmd, completed.stdout, completed.stderr
        )
    return completed.stdout


def _ensure_min_free_space(
    manager: VastGPUManager,
    instance_id: int,
    min_free_gb: float,
    path: str,
    job_id: str | None = None,
) -> bool:
    path_q = shlex.quote(path)
    cmd = f"df -Pk {path_q} | tail -n 1"
    out = run_and_get_output(manager, instance_id, cmd, job_id=job_id).strip()
    if not out:
        logger.warning("disk_check empty output path=%s job_id=%s", path, job_id)
        return False
    parts = out.split()
    if len(parts) < 4:
        logger.warning("disk_check parse failed output=%s job_id=%s", out, job_id)
        return False
    try:
        available_kb = float(parts[3])
    except ValueError:
        logger.warning("disk_check invalid available field output=%s job_id=%s", out, job_id)
        return False
    available_gb = available_kb / 1024.0 / 1024.0
    logger.info(
        "disk_check path=%s available_gb=%.2f min_required_gb=%.2f job_id=%s",
        path,
        available_gb,
        min_free_gb,
        job_id,
    )
    return available_gb >= min_free_gb
