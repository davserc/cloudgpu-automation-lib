from unittest.mock import MagicMock, patch

import pytest

from services.vast.gpu_manager.constants import (
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_GPU_NAME,
    DEFAULT_MAX_PRICE,
    DEFAULT_MIN_RELIABILITY,
    DEFAULT_NUM_GPUS,
    MIN_DISK_SPACE,
)
from services.vast.gpu_manager.errors import APIKeyError, InstanceNotFoundError, VastAIError
from services.vast.gpu_manager.manager import GPUConfig, VastGPUManager, quick_launch


class TestGPUConfig:
    def test_default_values(self, monkeypatch):
        for var in (
            "DEFAULT_GPU_NAME",
            "DEFAULT_NUM_GPUS",
            "DEFAULT_MAX_PRICE",
            "DEFAULT_MIN_RELIABILITY",
            "DEFAULT_DOCKER_IMAGE",
            "DEFAULT_DISK_SPACE",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = GPUConfig()
        assert cfg.gpu_name == DEFAULT_GPU_NAME
        assert cfg.num_gpus == DEFAULT_NUM_GPUS
        assert cfg.max_price == pytest.approx(DEFAULT_MAX_PRICE)
        assert cfg.min_reliability == pytest.approx(DEFAULT_MIN_RELIABILITY)
        assert cfg.docker_image == DEFAULT_DOCKER_IMAGE
        assert cfg.disk_space >= MIN_DISK_SPACE

    def test_disk_space_min_enforced(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_DISK_SPACE", "1.0")
        cfg = GPUConfig()
        assert cfg.disk_space >= MIN_DISK_SPACE


class TestVastGPUManager:
    def _manager(self, api_key="test-key"):
        return VastGPUManager(api_key=api_key)

    def test_raises_api_key_error_without_key(self, monkeypatch):
        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        with pytest.raises(APIKeyError):
            VastGPUManager(api_key=None)

    def test_strips_whitespace_from_key(self):
        mgr = self._manager(api_key="  my-key  ")
        assert mgr.api_key == "my-key"

    def test_search_gpus_returns_list(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.search_offers.return_value = [
            {
                "id": 1,
                "gpu_name": "RTX 3080",
                "dph_total": 0.03,
                "dlperf": 27.0,
                "gpu_ram": 10240,
                "total_flops": 29.0,
                "reliability": 0.98,
            }
        ]
        mgr._sdk = mock_sdk
        results = mgr.search_gpus(max_price=0.05, limit=10)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_search_gpus_filters_by_gpu_name(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.search_offers.return_value = [
            {
                "id": 1,
                "gpu_name": "RTX 3080",
                "dph_total": 0.03,
                "dlperf": 27.0,
                "gpu_ram": 10240,
                "total_flops": 29.0,
                "reliability": 0.98,
            },
            {
                "id": 2,
                "gpu_name": "Tesla V100",
                "dph_total": 0.02,
                "dlperf": 25.0,
                "gpu_ram": 32768,
                "total_flops": 12.5,
                "reliability": 0.99,
            },
        ]
        mgr._sdk = mock_sdk
        results = mgr.search_gpus(gpu_name="V100")
        ids = [r["id"] for r in results]
        assert 2 in ids
        assert 1 not in ids

    def test_list_instances_returns_result(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_instances.return_value = []
        mgr._sdk = mock_sdk
        assert mgr.list_instances() == []

    def test_get_instance_finds_by_id(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_instances.return_value = [{"id": 123, "gpu_name": "V100"}]
        mgr._sdk = mock_sdk
        inst = mgr.get_instance(123)
        assert inst is not None
        assert inst["id"] == 123

    def test_get_instance_returns_none_when_not_found(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_instances.return_value = [{"id": 999}]
        mgr._sdk = mock_sdk
        assert mgr.get_instance(1) is None

    def test_destroy_instance_calls_sdk(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.destroy_instance.return_value = {"success": True}
        mgr._sdk = mock_sdk
        mgr.destroy_instance(42)
        mock_sdk.destroy_instance.assert_called_once_with(id=42)

    def test_get_balance_returns_dict(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_user.return_value = {
            "credit": 10.0,
            "total_spend": -5.0,
            "username": "user",
            "email": "u@test.com",
        }
        mgr._sdk = mock_sdk
        bal = mgr.get_balance()
        assert bal["credit"] == 10.0
        assert bal["total_spend"] == 5.0

    def test_get_ssh_command_returns_string(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_instances.return_value = [
            {"id": 1, "ssh_host": "ssh1.vast.ai", "ssh_port": 12345}
        ]
        mgr._sdk = mock_sdk
        cmd = mgr.get_ssh_command(1)
        assert "12345" in cmd
        assert "ssh1.vast.ai" in cmd

    def test_get_ssh_command_returns_none_without_info(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_instances.return_value = [{"id": 1}]
        mgr._sdk = mock_sdk
        assert mgr.get_ssh_command(1) is None

    def test_launch_by_offer_id_returns_contract(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.create_instance.return_value = {"success": True, "new_contract": 99}
        mock_sdk.show_instances.return_value = []
        mgr._sdk = mock_sdk
        result = mgr.launch_by_offer_id(offer_id=55, image="ubuntu:latest")
        assert result.get("new_contract") == 99

    def test_start_instance_calls_sdk(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.start_instance.return_value = {"success": True}
        mgr._sdk = mock_sdk
        mgr.start_instance(7)
        mock_sdk.start_instance.assert_called_once_with(id=7)

    def test_stop_instance_calls_sdk(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.stop_instance.return_value = {"success": True}
        mgr._sdk = mock_sdk
        mgr.stop_instance(8)
        mock_sdk.stop_instance.assert_called_once_with(id=8)

    def test_get_balance_parses_json_string(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        import json

        mock_sdk.show_user.return_value = json.dumps({"credit": 5.0, "total_spend": -2.0})
        mgr._sdk = mock_sdk
        bal = mgr.get_balance()
        assert bal["credit"] == 5.0

    def test_get_balance_handles_unexpected_type(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.show_user.return_value = 12345
        mgr._sdk = mock_sdk
        bal = mgr.get_balance()
        assert isinstance(bal, dict)

    def test_search_gpus_empty_sdk_result_returns_empty(self):
        mgr = self._manager()
        mock_sdk = MagicMock()
        mock_sdk.search_offers.return_value = None
        mock_sdk.show_instances.return_value = []
        mgr._sdk = mock_sdk
        with patch(
            "services.vast.gpu_manager.manager.VastGPUManager._search_offers_rest", return_value=[]
        ):
            results = mgr.search_gpus()
        assert results == []


class TestErrors:
    def test_api_key_error_is_vast_ai_error(self):
        assert issubclass(APIKeyError, VastAIError)

    def test_instance_not_found_is_vast_ai_error(self):
        assert issubclass(InstanceNotFoundError, VastAIError)


def test_quick_launch_calls_manager():
    with patch("services.vast.gpu_manager.manager.VastGPUManager") as MockMgr:
        mock_instance = MagicMock()
        mock_instance.launch_instance.return_value = {"new_contract": 1}
        MockMgr.return_value = mock_instance
        quick_launch(gpu_name="RTX_3080", api_key="test-key")
        mock_instance.launch_instance.assert_called_once()
