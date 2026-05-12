from unittest.mock import MagicMock, patch

from services.vast.service.dataset import (
    _build_dataset_cmds,
    _build_onstart_cmd,
    _build_train_command,
    _get_user_access_token,
)


class TestBuildDatasetCmds:
    def test_with_access_token_uses_curl(self):
        cmd, path = _build_dataset_cmds(
            "gs://my-bucket/data.zip",
            "/work/datasets",
            None,
            None,
            None,
            access_token="tok123",
        )
        assert "curl" in cmd
        assert "Authorization: Bearer tok123" in cmd
        assert "/work/datasets/data.zip" in path

    def test_with_gcp_sa_b64_uses_gsutil(self):
        cmd, path = _build_dataset_cmds(
            "gs://my-bucket/data.zip",
            "/work/datasets",
            None,
            None,
            gcp_sa_b64="dGVzdA==",
        )
        assert "gsutil" in cmd
        assert "gcloud auth" in cmd

    def test_zip_extraction_with_token(self):
        cmd, _ = _build_dataset_cmds(
            "gs://bucket/data.zip",
            "/work/datasets",
            None,
            None,
            None,
            access_token="tok",
        )
        assert "zipfile" in cmd

    def test_tgz_extraction_with_gsutil(self):
        cmd, _ = _build_dataset_cmds(
            "gs://bucket/data.tar.gz",
            "/work/datasets",
            None,
            None,
            gcp_sa_b64="dGVzdA==",
        )
        assert "tar" in cmd

    def test_custom_extract_cmd(self):
        cmd, _ = _build_dataset_cmds(
            "gs://bucket/data.zip",
            "/dst",
            None,
            "unzip {archive} -d {dst}",
            None,
            access_token="tok",
        )
        assert "unzip" in cmd

    def test_custom_archive_name(self):
        _, path = _build_dataset_cmds(
            "gs://bucket/file.zip",
            "/dst",
            "custom_name.zip",
            None,
            None,
            access_token="tok",
        )
        assert "custom_name.zip" in path

    def test_no_credentials_uses_plain_gsutil(self):
        cmd, _ = _build_dataset_cmds(
            "gs://bucket/data.zip",
            "/work",
            None,
            None,
            None,
        )
        assert "gsutil" in cmd


class TestBuildTrainCommand:
    def test_includes_dataset_url(self):
        cmd = _build_train_command("gs://bucket/data.zip", None, None, "bash run.sh")
        assert "TRAIN_DATASET_URL=" in cmd
        assert "bash run.sh" in cmd

    def test_includes_gcp_credentials_when_sa_b64(self):
        cmd = _build_train_command("gs://b/d.zip", "abc123", None, "bash run.sh")
        assert "GOOGLE_APPLICATION_CREDENTIALS=/root/gcp.json" in cmd

    def test_includes_train_env(self):
        cmd = _build_train_command("url", None, {"MY_VAR": "42"}, "bash run.sh")
        assert "MY_VAR=" in cmd


class TestBuildOnstartCmd:
    def test_returns_none_when_no_credentials(self):
        env_vars, cmd = _build_onstart_cmd(None, False)
        assert env_vars is None
        assert cmd is None

    def test_returns_env_and_cmd_with_credentials(self):
        env_vars, cmd = _build_onstart_cmd("dGVzdA==", False)
        assert env_vars is not None
        assert "GCP_SA_B64" in env_vars
        assert cmd is not None

    def test_includes_gsutil_install_when_requested(self):
        _, cmd = _build_onstart_cmd("dGVzdA==", install_gsutil=True)
        assert "apt-get" in cmd


class TestGetUserAccessToken:
    def test_returns_none_for_service_account_type(self):
        import base64
        import json

        creds = {"type": "service_account", "private_key": "..."}
        b64 = base64.b64encode(json.dumps(creds).encode()).decode()
        assert _get_user_access_token(b64) is None

    def test_returns_none_for_invalid_base64(self):
        assert _get_user_access_token("not-valid-base64!!!") is None

    def test_exchanges_token_for_authorized_user(self):
        import base64
        import json

        creds = {
            "type": "authorized_user",
            "client_id": "client-id",
            "client_secret": "secret",
            "refresh_token": "refresh-tok",
        }
        b64 = base64.b64encode(json.dumps(creds).encode()).decode()
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"access_token": "ya29.fake"}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch(
            "services.vast.service.dataset.urllib_request.urlopen", return_value=mock_response
        ):
            token = _get_user_access_token(b64)
        assert token == "ya29.fake"
