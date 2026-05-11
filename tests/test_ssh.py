import pytest
import subprocess
from unittest.mock import MagicMock, patch, call
from services.vast.service.ssh import ssh_base_args, ssh_user, known_hosts_file, wait_for_ssh, download, run, run_and_capture, run_with_retries


def test_ssh_base_args_contains_strict_checking():
    args = ssh_base_args()
    assert "StrictHostKeyChecking=accept-new" in args


def test_ssh_user_default(monkeypatch):
    monkeypatch.delenv("VAST_SSH_USER", raising=False)
    assert ssh_user() == "vast"


def test_ssh_user_from_env(monkeypatch):
    monkeypatch.setenv("VAST_SSH_USER", "root")
    assert ssh_user() == "root"


def test_known_hosts_file_default(monkeypatch):
    monkeypatch.delenv("VAST_SSH_KNOWN_HOSTS_FILE", raising=False)
    assert "known_hosts" in known_hosts_file()


class TestWaitForSsh:
    def _make_manager(self, ssh_host="ssh1.vast.ai", ssh_port=12345, status_msg=""):
        mgr = MagicMock()
        mgr.get_instance.return_value = {
            "id": 1,
            "ssh_host": ssh_host,
            "ssh_port": ssh_port,
            "status_msg": status_msg,
        }
        return mgr

    def test_returns_ssh_info_when_ready(self):
        mgr = self._make_manager()
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0)
            result = wait_for_ssh(mgr, 1, timeout_sec=5, poll_interval_sec=1)
        assert result == ("ssh1.vast.ai", 12345)

    def test_raises_timeout_when_ssh_not_available(self):
        mgr = self._make_manager()
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run, \
             patch("time.sleep"):
            mock_conn.side_effect = OSError("refused")
            with pytest.raises(TimeoutError):
                wait_for_ssh(mgr, 1, timeout_sec=0, poll_interval_sec=1)

    def test_raises_timeout_on_container_error(self):
        mgr = self._make_manager(status_msg="Error response from daemon: failed to create task")
        with patch("time.sleep"):
            with pytest.raises(TimeoutError, match="container failed to start"):
                wait_for_ssh(mgr, 1, timeout_sec=10, poll_interval_sec=1)

    def test_retries_when_ssh_returns_255(self):
        mgr = self._make_manager()
        call_count = {"n": 0}

        def fake_run(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                return MagicMock(returncode=255)
            return MagicMock(returncode=0)

        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep"):
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            result = wait_for_ssh(mgr, 1, timeout_sec=30, poll_interval_sec=0)
        assert result is not None
        assert call_count["n"] == 2


class TestDownload:
    def _make_manager(self):
        mgr = MagicMock()
        mgr.get_instance.return_value = {
            "id": 1, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345, "status_msg": ""
        }
        return mgr

    def test_download_runs_scp(self):
        mgr = self._make_manager()
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0)
            download(mgr, 1, "/remote/file.pt", "/local/")
        scp_call = mock_run.call_args_list[-1]
        cmd = scp_call[0][0]
        assert "scp" in cmd[0]
        assert "ssh1.vast.ai" in " ".join(str(c) for c in cmd)

    def test_download_raises_on_scp_failure(self):
        mgr = self._make_manager()
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_run.side_effect = [
                MagicMock(returncode=0),  # ssh test in wait_for_ssh
                subprocess.CalledProcessError(1, ["scp"]),  # actual scp
            ]
            with pytest.raises(subprocess.CalledProcessError):
                download(mgr, 1, "/remote/missing.pt", "/local/")


class TestRunAndCapture:
    def _make_manager(self):
        mgr = MagicMock()
        mgr.get_instance.return_value = {
            "id": 1, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345, "status_msg": ""
        }
        return mgr

    def test_run_and_capture_writes_to_log(self, tmp_path):
        mgr = self._make_manager()
        log_path = tmp_path / "out.log"
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0)
            rc = run_and_capture(mgr, 1, "echo hello", log_path)
        assert rc == 0

    def test_run_and_capture_raises_on_nonzero(self, tmp_path):
        mgr = self._make_manager()
        log_path = tmp_path / "out.log"
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_run.side_effect = [
                MagicMock(returncode=0),   # ssh probe
                MagicMock(returncode=1),   # actual command
            ]
            with pytest.raises(subprocess.CalledProcessError):
                run_and_capture(mgr, 1, "false", log_path)


class TestRun:
    def _make_manager(self):
        mgr = MagicMock()
        mgr.get_instance.return_value = {
            "id": 1, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345, "status_msg": ""
        }
        return mgr

    def test_run_returns_zero_on_success(self):
        mgr = self._make_manager()
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0)
            rc = run(mgr, 1, "echo hello")
        assert rc == 0

    def test_run_raises_on_nonzero(self):
        mgr = self._make_manager()
        with patch("socket.create_connection") as mock_conn, \
             patch("subprocess.run") as mock_run:
            mock_conn.return_value.__enter__ = lambda s: s
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            # First call: SSH probe returns 0 so wait_for_ssh succeeds.
            # Second call: the actual command returns 1 → CalledProcessError.
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1),
            ]
            with pytest.raises(subprocess.CalledProcessError):
                run(mgr, 1, "false")
