from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from services.vast.gpu_manager import DOCKER_IMAGES, VastGPUManager
from services.vast.service.dataset import (
    _build_dataset_cmds,
    _build_onstart_cmd,
    _build_train_command,
)
from services.vast.service.env import _resolve_min_cuda, _resolve_min_free_disk_gb
from services.vast.service.offers import (
    _add_offer_blacklist,
    _is_offer_blacklisted,
    _load_offer_blacklist,
    _log_selected_offer,
    _prune_offer_blacklist,
    _rank_offers,
    _save_offer_blacklist,
)
from services.vast.service.ssh import (
    _ensure_min_free_space,
    download,
    run_and_capture,
    run_and_get_output,
    run_with_retries,
    ssh_base_args,
    wait_for_ssh,
)
from services.vast.service.types import LaunchResult

logger = logging.getLogger("vast_service")


def _resolve_image(image: str) -> str:
    return DOCKER_IMAGES.get(image, image)


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


def _ensure_remote_gcp_json(
    manager: VastGPUManager,
    instance_id: int,
    gcp_sa_b64: str,
    job_id: str | None = None,
) -> None:
    """Ensure /root/gcp.json exists on the instance; fallback to SCP if missing."""
    try:
        decoded = base64.b64decode(gcp_sa_b64, validate=True)
    except Exception as exc:  # noqa: BLE001 - surface invalid secrets early
        logger.warning("Invalid GCP_SA_B64; cannot create /root/gcp.json (%s)", exc)
        return

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(decoded)
            temp_path = tmp.name
        os.chmod(temp_path, 0o600)

        try:
            out = run_and_get_output(
                manager,
                instance_id,
                "test -s /root/gcp.json && echo present || echo missing",
                job_id=job_id,
            ).strip()
            if "present" in out:
                return
        except subprocess.CalledProcessError:
            pass

        host, port = wait_for_ssh(manager, instance_id, job_id=job_id)
        scp_cmd = [
            "scp",
            "-P",
            str(port),
            *ssh_base_args(),
            temp_path,
            f"root@{host}:/root/gcp.json",
        ]
        logger.info(
            "gcp.json missing on instance %s; uploading via scp (host=%s port=%s)",
            instance_id,
            host,
            port,
        )
        try:
            result = subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.info("gcp.json scp stdout:\n%s", result.stdout)
            if result.stderr:
                logger.warning("gcp.json scp stderr:\n%s", result.stderr)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to upload /root/gcp.json via scp for instance {instance_id}"
            ) from exc

        try:
            out = run_and_get_output(
                manager,
                instance_id,
                "test -s /root/gcp.json && echo present || echo missing",
                job_id=job_id,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to verify /root/gcp.json after scp for instance {instance_id}"
            ) from exc
        if "present" not in out:
            raise RuntimeError(f"/root/gcp.json still missing after scp for instance {instance_id}")
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


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
    min_free_disk_gb: float | None = None,
    free_disk_path: str | None = None,
    disk_space_gb: float | None = None,
    ensure_yolo_weights: bool = True,
    yolo_weights_name: str = "yolo11s.pt",
    ensure_ultralytics: bool = True,
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

    min_cuda = _resolve_min_cuda(min_cuda)
    min_free_disk_gb = _resolve_min_free_disk_gb(min_free_disk_gb)
    free_disk_path = free_disk_path or (dataset_dst if dataset_dst else "/")

    if gcp_sa_b64 is None:
        gcp_sa_b64 = os.environ.get("GCP_SA_B64")

    env_vars, onstart_cmd = _build_onstart_cmd(gcp_sa_b64, install_gsutil)

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
        _log_selected_offer(next_offer, job_id, attempts, max_launch_attempts)
        try:
            launch = launch_offer(
                manager,
                offer_id=int(next_offer["id"]),
                image=image,
                ports=ports,
                env_vars=env_vars,
                onstart_cmd=onstart_cmd,
                disk_space=disk_space_gb,
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
            if gcp_sa_b64:
                _ensure_remote_gcp_json(manager, launch.instance_id, gcp_sa_b64, job_id=job_id)
            if min_free_disk_gb > 0:
                if not _ensure_min_free_space(
                    manager,
                    launch.instance_id,
                    min_free_disk_gb,
                    free_disk_path,
                    job_id=job_id,
                ):
                    logger.warning(
                        "Insufficient free disk on instance %s (min %.1f GB). Retrying.",
                        launch.instance_id,
                        min_free_disk_gb,
                    )
                    blacklist = _add_offer_blacklist(
                        blacklist, int(next_offer["id"]), offer_blacklist_ttl_sec
                    )
                    _save_offer_blacklist(offer_blacklist_path, blacklist)
                    try:
                        destroy_with_retries(
                            manager,
                            launch.instance_id,
                            retries=destroy_retries,
                            backoff_sec=destroy_backoff_sec,
                            job_id=job_id,
                        )
                    except Exception as destroy_exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to destroy instance %s after disk check: %s",
                            launch.instance_id,
                            destroy_exc,
                        )
                        stale_instance_ids.append(launch.instance_id)
                    launch = None
                    time.sleep(launch_retry_backoff_sec)
                    continue
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

    run_cmd_with_dataset = _build_train_command(train_dataset_url, gcp_sa_b64, train_env, run_cmd)
    logger.info("train_command job_id=%s cmd=%s", job_id, run_cmd_with_dataset)

    exit_code: int | None = None
    try:
        if dataset_gs_uri:
            if not gcp_sa_b64:
                raise ValueError("gcp_sa_b64 is required when dataset_gs_uri is provided")
            cmd_str, _ = _build_dataset_cmds(
                dataset_gs_uri,
                dataset_dst,
                dataset_archive_name,
                extract_cmd,
                gcp_sa_b64,
            )
            logger.info("dataset_prepare job_id=%s cmd=%s", job_id, cmd_str)
            run_with_retries(
                manager,
                launch.instance_id,
                cmd_str,
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
        if ensure_ultralytics and yolo_weights_name in run_cmd:
            install_ultralytics_cmd = "python3 -m pip install -U ultralytics"
            logger.info("ultralytics_install job_id=%s cmd=%s", job_id, install_ultralytics_cmd)
            run_with_retries(
                manager,
                launch.instance_id,
                install_ultralytics_cmd,
                retries=2,
                backoff_sec=10.0,
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
        if yolo_weights_name in run_cmd:
            weights_cmd = f"ls -la {yolo_weights_name} || true"
            logger.info("weights_verify job_id=%s cmd=%s", job_id, weights_cmd)
            run_with_retries(
                manager,
                launch.instance_id,
                weights_cmd,
                retries=2,
                backoff_sec=5.0,
                job_id=job_id,
            )
            if ensure_yolo_weights:
                ensure_weights_cmd = (
                    f"test -s {yolo_weights_name} || "
                    'python3 -c "from ultralytics.utils.downloads import '
                    "attempt_download_asset; attempt_download_asset("
                    f"'{yolo_weights_name}')\""
                )
                logger.info("weights_download job_id=%s cmd=%s", job_id, ensure_weights_cmd)
                run_with_retries(
                    manager,
                    launch.instance_id,
                    ensure_weights_cmd,
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
