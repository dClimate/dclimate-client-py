"""Regression tests for the pytest configuration and test-suite boundaries."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
META_TEST_ENV = "DCLIMATE_META_TEST"
SUMMARY_COUNT = re.compile(r"(?P<count>\d+) (?P<outcome>passed|skipped)\b")

pytestmark = pytest.mark.skipif(
    os.environ.get(META_TEST_ENV) == "1",
    reason="pytest infrastructure meta-tests do not run in child pytest processes",
)


@pytest.fixture(scope="session", autouse=True)
def check_ipfs_connection():
    """Keep these infrastructure checks independent of the global IPFS gate."""


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["IPFS_GATEWAY_URI_STEM"] = "http://127.0.0.1:9"
    env[META_TEST_ENV] = "1"
    return subprocess.run(
        ["uv", "run", "pytest", *args, "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )


def _summary_counts(output: str) -> tuple[int, int]:
    summary_line = next(
        (
            line
            for line in reversed(output.splitlines())
            if SUMMARY_COUNT.search(line)
        ),
        None,
    )
    assert summary_line is not None, f"pytest summary line not found:\n{output}"

    counts = {
        match.group("outcome"): int(match.group("count"))
        for match in SUMMARY_COUNT.finditer(summary_line)
    }
    return counts.get("passed", 0), counts.get("skipped", 0)


def test_offline_unit_tests_run_without_ipfs_gateway():
    result = _run_pytest(
        "tests/test_zarr_metadata.py",
        "tests/test_siren.py",
        "-q",
    )
    passed, skipped = _summary_counts(result.stdout)

    assert passed > 0 and skipped == 0, (
        "offline unit tests must run without an IPFS gateway; "
        f"observed passed={passed}, skipped={skipped}\n{result.stdout}"
    )


def test_unmarked_async_tests_execute():
    result = _run_pytest("test_stac_integration.py", "-q")

    unsupported_message = "async def functions are not natively supported"
    assert unsupported_message not in result.stdout, result.stdout

    _, skipped = _summary_counts(result.stdout)
    assert skipped == 0, f"async tests were skipped instead of executed:\n{result.stdout}"


def test_root_collection_is_confined_to_tests_directory():
    result = _run_pytest("--collect-only", "-q")
    assert result.returncode == 0, result.stdout

    nodeids = [line.strip() for line in result.stdout.splitlines() if "::" in line]
    stray_nodeids = [
        nodeid
        for nodeid in nodeids
        if nodeid.startswith(("test_stac_integration.py", "examples/"))
    ]
    assert not stray_nodeids, f"pytest collected tests outside tests/: {stray_nodeids}"


def test_debug_tests_contain_a_real_assertion():
    debug_test = PROJECT_ROOT / "tests/test_debug.py"
    if not debug_test.exists():
        return

    tree = ast.parse(debug_test.read_text(encoding="utf-8"), filename=str(debug_test))
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert assertions, "tests/test_debug.py contains no assert statements"
