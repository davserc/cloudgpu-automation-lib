import pytest

from services.vast.gpu_manager.env import get_env, get_env_float, get_env_int


def test_get_env_returns_value(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "hello")
    assert get_env("TEST_VAR") == "hello"


def test_get_env_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_VAR", raising=False)
    assert get_env("TEST_VAR", "default") == "default"


def test_get_env_float_valid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "3.14")
    assert get_env_float("TEST_FLOAT", 0.0) == pytest.approx(3.14)


def test_get_env_float_invalid_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "not_a_float")
    assert get_env_float("TEST_FLOAT", 9.9) == pytest.approx(9.9)


def test_get_env_float_missing_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    assert get_env_float("TEST_FLOAT", 1.5) == pytest.approx(1.5)


def test_get_env_int_valid(monkeypatch):
    monkeypatch.setenv("TEST_INT", "42")
    assert get_env_int("TEST_INT", 0) == 42


def test_get_env_int_invalid_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_INT", "bad")
    assert get_env_int("TEST_INT", 7) == 7


def test_get_env_int_missing_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)
    assert get_env_int("TEST_INT", 3) == 3
