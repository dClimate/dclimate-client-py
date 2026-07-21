"""Regression tests for the package's PEP 561 typing contract."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import zipfile

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
    result = subprocess.run(
        [
            "uv",
            "run",
            "mypy",
            "dclimate_client_py",
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
    source_count_match = re.search(
        r"no issues found in (\d+) source files", result.stdout, re.IGNORECASE
    )
    assert source_count_match is not None, (
        f"mypy did not report its checked source-file count:\n{result.stdout}"
    )
    assert int(source_count_match.group(1)) >= 16, (
        "mypy checked fewer than the expected 16 package source files:\n"
        f"{result.stdout}"
    )


def test_package_has_pep561_marker() -> None:
    marker = PROJECT_ROOT / "dclimate_client_py" / "py.typed"

    assert marker.is_file(), (
        "dclimate_client_py/py.typed must exist so the package ships its type "
        "information in built distributions"
    )


def test_built_wheel_contains_pep561_marker(tmp_path: Path) -> None:
    env = os.environ.copy()
    env[TYPECHECK_META_ENV] = "1"
    env["UV_CACHE_DIR"] = str(PROJECT_ROOT / ".uv-cache")
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f"uv build exited with status {result.returncode}:\n{result.stdout}"
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels!r}"
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert "dclimate_client_py/py.typed" in wheel.namelist()
