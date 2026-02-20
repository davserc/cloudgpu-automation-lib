from __future__ import annotations

import logging
import shlex

logger = logging.getLogger("vast_service")


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
) -> tuple[str, str]:
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
    return " && ".join(cmds), archive_path
