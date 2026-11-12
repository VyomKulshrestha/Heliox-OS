"""Which Python the first-run setup may use: one range, read by the app, the setup scripts and CI.

An existing Python is used only inside the range: a newer one may have no prebuilt wheels for the core's
dependencies, so pip compiles them and fails without a Fortran compiler (CI on macOS, 12 November: Homebrew's
Python 3.15 and scipy). Outside the range Heliox downloads its own.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "tauri-app" / "src-tauri" / "setup"
VERSIONS = json.loads((SETUP / "python-versions.json").read_text(encoding="utf-8"))


def _version(text: str) -> tuple[int, int]:
    major, minor = text.split(".")[:2]
    return int(major), int(minor)


def _check_code(script: str, pattern: str) -> str:
    match = re.search(pattern, (SETUP / script).read_text(encoding="utf-8"))
    assert match, f"no range check in {script}"
    code = match.group(1)
    for name in ("$PY_MIN", "$PyMin"):
        code = code.replace(name, VERSIONS["min"])
    for name in ("$PY_MAX", "$PyMax"):
        code = code.replace(name, VERSIONS["max"])
    return code


@pytest.mark.parametrize(
    ("script", "pattern"),
    [("first-run.sh", r'CHECK="(import sys[^"]+)"'), ("first-run.ps1", r'\$Check = "(import sys[^"]+)"')],
)
@pytest.mark.parametrize(
    ("version", "accepted"),
    [((3, 10, 14), False), ((3, 11, 0), True), ((3, 12, 6), True), ((3, 13, 9), True), ((3, 15, 2), False)],
)
def test_existing_python_is_used_only_inside_the_tested_range(monkeypatch, script, pattern, version, accepted):
    code = _check_code(script, pattern)
    monkeypatch.setattr(sys, "version_info", version)
    with pytest.raises(SystemExit) as stop:
        exec(code, {})  # noqa: S102 - the scripts' own one-line check
    assert (stop.value.code == 0) is accepted


def test_the_range_is_sane_and_holds_the_private_python():
    low, high, private = (_version(VERSIONS[key]) for key in ("min", "max", "private"))
    assert low <= private <= high


def test_ci_tests_the_daemon_inside_the_range():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matrix = re.search(r"python-version: \[([^\]]+)\]", ci)
    assert matrix, "the python job's matrix moved"
    versions = [_version(v.strip().strip("'\"")) for v in matrix.group(1).split(",")]
    low, high = _version(VERSIONS["min"]), _version(VERSIONS["max"])
    assert versions and all(low <= v <= high for v in versions)
    assert min(versions) == low  # the oldest Python people may have is the one CI tests


def test_pyproject_agrees_with_the_lower_bound():
    pyproject = (ROOT / "daemon" / "pyproject.toml").read_text(encoding="utf-8")
    assert f'requires-python = ">={VERSIONS["min"]}"' in pyproject
