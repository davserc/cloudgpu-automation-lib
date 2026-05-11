import pytest
from unittest.mock import MagicMock, patch
from services.vast.service.offers import (
    _rank_offers,
    _is_offer_blacklisted,
    _add_offer_blacklist,
    _prune_offer_blacklist,
    _load_offer_blacklist,
    _format_offer_line,
    _format_offer_header,
)


def _make_offer(**kwargs):
    defaults = {
        "id": 1,
        "gpu_name": "RTX 3080",
        "num_gpus": 1,
        "gpu_ram": 10240,
        "dph_total": 0.03,
        "dlperf": 27.0,
        "total_flops": 29.0,
        "reliability": 0.98,
        "cuda_max_good": 12.4,
        "geolocation": "Washington, US",
        "cpu_ram": 32768,
        "disk_space": 50.0,
        "score": 100.0,
    }
    defaults.update(kwargs)
    return defaults


class TestRankOffers:
    def _make_manager(self, offers):
        mgr = MagicMock()
        mgr.search_gpus.return_value = offers
        return mgr

    def test_returns_sorted_by_value(self):
        offers = [
            _make_offer(id=1, dph_total=0.02, dlperf=10.0),
            _make_offer(id=2, dph_total=0.01, dlperf=20.0),
        ]
        mgr = self._make_manager(offers)
        result = _rank_offers(mgr, max_price=0.04)
        assert result[0]["id"] == 2

    def test_filters_by_max_cuda(self, monkeypatch):
        monkeypatch.delenv("VAST_BLOCKED_COUNTRIES", raising=False)
        offers = [
            _make_offer(id=1, cuda_max_good=12.0),
            _make_offer(id=2, cuda_max_good=13.0),
        ]
        mgr = self._make_manager(offers)
        result = _rank_offers(mgr, max_cuda=12.9)
        ids = [o["id"] for o in result]
        assert 1 in ids
        assert 2 not in ids

    def test_filters_by_min_cuda(self, monkeypatch):
        monkeypatch.delenv("VAST_BLOCKED_COUNTRIES", raising=False)
        offers = [
            _make_offer(id=1, cuda_max_good=11.0),
            _make_offer(id=2, cuda_max_good=12.4),
        ]
        mgr = self._make_manager(offers)
        result = _rank_offers(mgr, min_cuda=12.0)
        ids = [o["id"] for o in result]
        assert 2 in ids
        assert 1 not in ids

    def test_filters_discouraged_gpus(self, monkeypatch):
        monkeypatch.delenv("VAST_BLOCKED_COUNTRIES", raising=False)
        offers = [
            _make_offer(id=1, gpu_name="GTX 1080"),
            _make_offer(id=2, gpu_name="RTX 3080"),
        ]
        mgr = self._make_manager(offers)
        result = _rank_offers(mgr)
        ids = [o["id"] for o in result]
        assert 2 in ids
        assert 1 not in ids

    def test_filters_blocked_countries(self, monkeypatch):
        monkeypatch.setenv("VAST_BLOCKED_COUNTRIES", "CN")
        offers = [
            _make_offer(id=1, geolocation=", CN"),
            _make_offer(id=2, geolocation="Washington, US"),
        ]
        mgr = self._make_manager(offers)
        result = _rank_offers(mgr)
        ids = [o["id"] for o in result]
        assert 2 in ids
        assert 1 not in ids

    def test_returns_empty_when_no_offers(self):
        mgr = self._make_manager([])
        mgr.search_gpus.side_effect = [[], []]
        result = _rank_offers(mgr)
        assert result == []

    def test_filters_zero_price(self, monkeypatch):
        monkeypatch.delenv("VAST_BLOCKED_COUNTRIES", raising=False)
        offers = [
            _make_offer(id=1, dph_total=0),
            _make_offer(id=2, dph_total=0.02),
        ]
        mgr = self._make_manager(offers)
        result = _rank_offers(mgr)
        ids = [o["id"] for o in result]
        assert 2 in ids
        assert 1 not in ids


class TestBlacklist:
    def test_add_and_check_blacklisted(self):
        bl = {}
        bl = _add_offer_blacklist(bl, 42, ttl_sec=3600)
        assert _is_offer_blacklisted(bl, 42) is True

    def test_not_blacklisted(self):
        assert _is_offer_blacklisted({}, 99) is False

    def test_prune_expired(self):
        import time
        bl = {}
        bl = _add_offer_blacklist(bl, 1, ttl_sec=3600)
        bl["offers"]["1"] = time.time() - 1  # already expired
        bl = _prune_offer_blacklist(bl, ttl_sec=3600)
        assert _is_offer_blacklisted(bl, 1) is False

    def test_load_missing_file(self, tmp_path):
        bl = _load_offer_blacklist(tmp_path / "nonexistent.json")
        assert bl == {}


def test_format_offer_header_contains_gpu():
    header = _format_offer_header()
    assert "GPU" in header


def test_format_offer_line():
    offer = _make_offer()
    line = _format_offer_line(offer)
    assert "RTX 3080" in line
