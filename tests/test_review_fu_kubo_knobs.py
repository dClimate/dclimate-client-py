"""Regression tests for exposing KuboCAS connection configuration."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import subprocess
from typing import Any

import pytest

from dclimate_client_py.dclimate_client import dClimateClient


dclimate_client_module = importlib.import_module("dclimate_client_py.dclimate_client")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KUBO_OPTION_NAMES = (
    "concurrency",
    "headers",
    "auth",
    "max_retries",
    "initial_delay",
    "backoff_factor",
    "client_factory",
)


def _install_recording_kubo(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    recorded_calls: list[dict[str, Any]] = []

    class RecordingKuboCAS:
        def __init__(self, **kwargs: Any) -> None:
            recorded_calls.append(kwargs)

        async def __aenter__(self) -> "RecordingKuboCAS":
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            return None

    monkeypatch.setattr(dclimate_client_module, "KuboCAS", RecordingKuboCAS)
    return recorded_calls


async def test_kubo_connection_options_are_forwarded_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_calls = _install_recording_kubo(monkeypatch)

    async with dClimateClient(
        concurrency=64,
        headers={"Authorization": "Bearer x"},
        auth=("user", "pass"),
        max_retries=5,
        initial_delay=0.5,
        backoff_factor=3.0,
    ):
        pass

    assert recorded_calls == [
        {
            "gateway_base_url": "https://ipfs-gateway.dclimate.net",
            "rpc_base_url": "https://ipfs-gateway.dclimate.net",
            "concurrency": 64,
            "headers": {"Authorization": "Bearer x"},
            "auth": ("user", "pass"),
            "max_retries": 5,
            "initial_delay": 0.5,
            "backoff_factor": 3.0,
        }
    ]


async def test_default_construction_preserves_kubo_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(dClimateClient.__init__)
    missing_options = set(KUBO_OPTION_NAMES) - set(signature.parameters)
    assert not missing_options, f"missing Kubo options: {sorted(missing_options)}"
    for option_name in KUBO_OPTION_NAMES:
        assert signature.parameters[option_name].default is None

    recorded_calls = _install_recording_kubo(monkeypatch)
    async with dClimateClient():
        pass

    assert recorded_calls == [
        {
            "gateway_base_url": "https://ipfs-gateway.dclimate.net",
            "rpc_base_url": "https://ipfs-gateway.dclimate.net",
        }
    ]


async def test_client_factory_is_forwarded_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_calls = _install_recording_kubo(monkeypatch)

    def client_factory() -> object:
        return object()

    async with dClimateClient(client_factory=client_factory):
        pass

    assert recorded_calls == [
        {
            "gateway_base_url": "https://ipfs-gateway.dclimate.net",
            "rpc_base_url": "https://ipfs-gateway.dclimate.net",
            "client_factory": client_factory,
        }
    ]


@pytest.mark.parametrize(
    "conflicting_option",
    [
        {"headers": {"Authorization": "Bearer x"}},
        {"auth": ("user", "pass")},
    ],
)
def test_client_factory_rejects_headers_and_auth_at_construction(
    conflicting_option: dict[str, Any],
) -> None:
    def client_factory() -> object:
        return object()

    with pytest.raises(ValueError, match="client_factory"):
        dClimateClient(client_factory=client_factory, **conflicting_option)


@pytest.mark.parametrize(
    "standalone_option",
    [
        {"client_factory": lambda: object()},
        {"headers": {"Authorization": "Bearer x"}},
        {"auth": ("user", "pass")},
    ],
)
def test_client_factory_headers_and_auth_are_valid_alone(
    standalone_option: dict[str, Any],
) -> None:
    dClimateClient(**standalone_option)


def test_gateway_benchmark_script_exposes_tuning_flags() -> None:
    completed = subprocess.run(
        ["uv", "run", "python", "scripts/benchmark_gateway.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    for flag in ("--http2", "--no-http2", "--concurrency", "--repetitions"):
        assert flag in output
