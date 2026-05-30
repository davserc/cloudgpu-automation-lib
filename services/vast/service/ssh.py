from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

from services.vast.gpu_manager import VastGPUManager

logger = logging.getLogger("vast_service")

SSH_CONNECT_TIMEOUT_SEC = 10
SSH_SERVER_ALIVE_INTERVAL_SEC = 30
SSH_SERVER_ALIVE_COUNT_MAX = 5


def ssh_user() -> str:
    return os.getenv("VAST_SSH_USER", "vast")


def known_hosts_file() -> str:
    return os.getenv("VAST_SSH_KNOWN_HOSTS_FILE", "/root/.ssh/known_hosts")


def ssh_base_args() -> list[str]:
    return [
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts_file()}",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SEC}",
        "-o",
        f"ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL_SEC}",
        "-o",
        f"ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
        "-o",
        "TCPKeepAlive=yes",
    ]


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
    _last_status_msg: str = ""
    _tcp_up_logged: bool = False
    while time.time() < deadline:
        instance = manager.get_instance(instance_id)
        actual_status = ""
        if instance:
            status_msg = instance.get("status_msg") or ""
            actual_status = instance.get("actual_status") or instance.get("status") or ""
            if status_msg != _last_status_msg:
                logger.info(
                    "wait_for_ssh: status_change instance_id=%s actual_status=%s status_msg=%s job_id=%s",
                    instance_id,
                    actual_status,
                    status_msg[:200],
                    job_id,
                )
                _last_status_msg = status_msg
            if (
                "Error response from daemon" in status_msg
                or "failed to create" in status_msg
                or "invalid hostPort" in status_msg
            ):
                logger.error(
                    "instance_id=%s job_id=%s container_error status_msg=%s",
                    instance_id,
                    job_id,
                    status_msg[:300],
                )
                raise TimeoutError(
                    f"Instance {instance_id} container failed to start: {status_msg[:120]} job_id={job_id}"
                )
            if actual_status in ("stopped", "offline", "exited", "deleted"):
                logger.error(
                    "instance_id=%s job_id=%s dead_status actual_status=%s status_msg=%s",
                    instance_id,
                    job_id,
                    actual_status,
                    status_msg[:200],
                )
                raise TimeoutError(
                    f"Instance {instance_id} is {actual_status} — container exited job_id={job_id}"
                )

        ssh_info = _get_ssh_info(manager, instance_id)
        if ssh_info:
            host, port = ssh_info
            try:
                import socket

                with socket.create_connection((host, port), timeout=5):
                    pass
            except OSError:
                time.sleep(poll_interval_sec)
                continue

            if not _tcp_up_logged:
                logger.info(
                    "wait_for_ssh: tcp_up instance_id=%s host=%s port=%s job_id=%s",
                    instance_id,
                    host,
                    port,
                    job_id,
                )
                _tcp_up_logged = True

            # TCP is open — verify the SSH daemon inside the container is ready.
            try:
                result = subprocess.run(
                    ["ssh", "-p", str(port), *ssh_base_args(), f"{ssh_user()}@{host}", "true"],
                    capture_output=True,
                    timeout=12,
                )
            except subprocess.TimeoutExpired:
                logger.debug(
                    "wait_for_ssh: SSH 'true' timed out instance_id=%s job_id=%s — retrying",
                    instance_id,
                    job_id,
                )
                time.sleep(poll_interval_sec)
                continue
            if result.returncode == 0:
                logger.info(
                    "wait_for_ssh: SSH ready instance_id=%s host=%s port=%s job_id=%s",
                    instance_id,
                    host,
                    port,
                    job_id,
                )
                return ssh_info
            # Exit 255 = gateway accepted TCP but container SSH not ready yet.
            stderr_hint = (result.stderr or b"").decode(errors="replace").strip()[:120]
            logger.warning(
                "wait_for_ssh: SSH auth failed rc=%s instance_id=%s job_id=%s stderr=%s — retrying",
                result.returncode,
                instance_id,
                job_id,
                stderr_hint,
            )
        time.sleep(poll_interval_sec)
    raise TimeoutError(
        f"SSH not available for instance {instance_id} after {timeout_sec}s job_id={job_id}"
    )


def download(
    manager: VastGPUManager,
    instance_id: int,
    src: str | list[str],
    dst: str | Path = "./",
    scp_bin: str = "scp",
    job_id: str | None = None,
) -> None:
    """Download one file/dir from the remote instance.

    If *src* is a list, each path is tried in order and the first successful
    one wins.  This lets callers express fallbacks like ``[best.pt, last.pt]``
    without knowing in advance which file the training produced.
    """
    host, port = wait_for_ssh(manager, instance_id, job_id=job_id)
    dst_path = str(dst)
    sources = [src] if isinstance(src, str) else src

    last_exc: Exception | None = None
    for s in sources:
        scp_cmd = [
            scp_bin,
            "-P",
            str(port),
            *ssh_base_args(),
            "-r",
            f"{ssh_user()}@{host}:{s}",
            dst_path,
        ]
        logger.info(
            "download instance_id=%s job_id=%s host=%s src=%s dst=%s",
            instance_id,
            job_id,
            host,
            s,
            dst_path,
        )
        try:
            subprocess.run(scp_cmd, check=True)
            return  # success — stop trying
        except subprocess.CalledProcessError as exc:
            logger.warning("download failed for src=%s (will try next fallback): %s", s, exc)
            last_exc = exc

    raise last_exc  # type: ignore[misc]  # all candidates exhausted


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
        *ssh_base_args(),
        f"{ssh_user()}@{host}",
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
        *ssh_base_args(),
        f"{ssh_user()}@{host}",
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
        *ssh_base_args(),
        f"{ssh_user()}@{host}",
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
    cmd = f"mkdir -p {path_q} && df -Pk {path_q} | tail -n 1"
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
