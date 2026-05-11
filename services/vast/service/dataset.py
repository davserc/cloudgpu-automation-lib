from __future__ import annotations

import base64
import json
import logging
import shlex
from urllib import parse as urllib_parse
from urllib import request as urllib_request

logger = logging.getLogger("vast_service")


def _get_user_access_token(gcp_sa_b64: str) -> str | None:
    """Exchange refresh_token for a short-lived access token (authorized_user only)."""
    try:
        creds = json.loads(base64.b64decode(gcp_sa_b64))
        if creds.get("type") != "authorized_user":
            return None
        data = urllib_parse.urlencode({
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "grant_type": "refresh_token",
        }).encode()
        req = urllib_request.Request("https://oauth2.googleapis.com/token", data=data)
        with urllib_request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["access_token"]
    except Exception as exc:
        logger.warning("Failed to get user access token: %s", exc)
        return None


def _build_onstart_cmd(
    gcp_sa_b64: str | None, install_gsutil: bool
) -> tuple[dict[str, str] | None, str | None]:
    if not gcp_sa_b64:
        return None, None
    env_vars = {"GCP_SA_B64": gcp_sa_b64}
    onstart_parts = []
    if install_gsutil:
        onstart_parts.append("apt-get update && apt-get install -y google-cloud-cli")
    onstart_parts.append("printf %s \"$GCP_SA_B64\" | tr -d '\\r' | base64 -d > /root/gcp.json")
    onstart_parts.append("chmod 600 /root/gcp.json")
    onstart_parts.append(
        "test -s /root/gcp.json && echo 'gcp.json: present' || echo 'gcp.json: missing'"
    )
    onstart_parts.append("gcloud auth list || true")
    onstart_parts.append("gsutil ls gs://unlu-genai-serranodavid-computer_vision_yolo || true")
    return env_vars, " && ".join(onstart_parts)


def _build_train_command(
    train_dataset_url: str,
    gcp_sa_b64: str | None,
    train_env: dict[str, str] | None,
    run_cmd: str,
) -> str:
    env_vars_cmd: dict[str, str] = {"TRAIN_DATASET_URL": train_dataset_url}
    if gcp_sa_b64:
        env_vars_cmd["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/gcp.json"
    if train_env:
        env_vars_cmd.update(train_env)
    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env_vars_cmd.items())
    return f"{env_prefix} {run_cmd}"


def _build_dataset_cmds(
    dataset_gs_uri: str,
    dataset_dst: str,
    dataset_archive_name: str | None,
    extract_cmd: str | None,
    gcp_sa_b64: str | None,
    access_token: str | None = None,
) -> tuple[str, str]:
    archive_name = dataset_archive_name or dataset_gs_uri.rstrip("/").split("/")[-1]
    archive_path = f"{dataset_dst.rstrip('/')}/{archive_name}"
    cmds: list[str] = [f"mkdir -p {dataset_dst}"]

    if access_token:
        # Use curl with Bearer token — works with authorized_user credentials without
        # needing gcloud account setup on the remote instance.
        # gs://bucket/path → https://storage.googleapis.com/storage/v1/b/bucket/o/encoded-path?alt=media
        parts = dataset_gs_uri[len("gs://"):].split("/", 1)
        bucket = parts[0]
        obj = urllib_parse.quote(parts[1], safe="") if len(parts) > 1 else ""
        https_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{obj}?alt=media"
        cmds.append(
            f"curl -fL -H {shlex.quote('Authorization: Bearer ' + access_token)} "
            f"-o {shlex.quote(archive_path)} {shlex.quote(https_url)}"
        )
    elif gcp_sa_b64:
        cmds.append("test -s /root/gcp.json")
        cmds.append(
            "gcloud auth activate-service-account --key-file /root/gcp.json 2>/dev/null "
            "|| gcloud auth login --cred-file /root/gcp.json --quiet 2>/dev/null "
            "|| true"
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
        cmds.append(f"tar -xzf {shlex.quote(archive_path)} -C {shlex.quote(dataset_dst)}")
    elif archive_name.endswith(".zip"):
        cmds.append(
            f"python3 -c \"import zipfile; zipfile.ZipFile({repr(archive_path)}).extractall({repr(dataset_dst)})\""
        )

    return " && ".join(cmds), archive_path
