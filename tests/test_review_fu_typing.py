"""Regression tests for the package's PEP 561 typing contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TYPECHECK_META_ENV = "DCLIMATE_TYPECHECK_META_TEST"

pytestmark = pytest.mark.skipif(
    os.environ.get(TYPECHECK_META_ENV) == "1",
    reason="typing meta-tests do not run recursively",
)


def test_package_passes_mypy() -> None:
    env = os.environ.copy()
    env[TYPECHECK_META_ENV] = "1"
    env["UV_CACHE_DIR"] = str(PROJECT_ROOT / ".uv-cache")
    env["UV_TOOL_DIR"] = str(PROJECT_ROOT / ".uv-cache" / "tools")
    result = subprocess.run(
        [
            "uvx",
            "mypy",
            "dclimate_client_py",
            "--ignore-missing-imports",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"mypy exited with status {result.returncode}:\n{result.stdout}"
    )


def test_package_has_pep561_marker() -> None:
    marker = PROJECT_ROOT / "dclimate_client_py" / "py.typed"

    assert marker.is_file(), (
        "dclimate_client_py/py.typed must exist so the package ships its type "
        "information in built distributions"
    )
