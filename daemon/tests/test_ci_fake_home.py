"""Every CI step that runs a script outside pytest which builds a Heliox server, a listener or the bench gives it
folders that are not the runner's own: either the script moves every folder itself (use_scratch_folders), or the
step completes the fake home for each system it runs on. pilot.real_folders refuses anything else, and these steps
hide failures (continue-on-error, set +e), so a refusal there would go unseen. Also where_folders_are() for the
runner cases.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pilot.real_folders import classify

ROOT = Path(__file__).resolve().parents[2]
DAEMON = ROOT / "daemon"
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
BUILDS = re.compile(r"PilotServer\((?!\s*\))|PilotServer\(\s*\w|Harness\(|ContinuousVoiceListener\(")
SCRIPT = re.compile(r"(?:daemon/)?((?:tests/platform_real|benchmarks|scripts)/[\w/]+\.py)|-m (benchmarks(?:\.\w+)+)")


def _script_file(match: re.Match) -> Path:
    if match.group(1):
        return DAEMON / match.group(1)
    module = DAEMON / Path(*match.group(2).split("."))
    return module / "__main__.py" if module.is_dir() else module.with_suffix(".py")


def _imported_bench_modules(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8")
    names = re.findall(r"from \.(\w+) import|from benchmarks\.conversation_bench\.(\w+) import", text)
    return [path.parent / f"{a or b}.py" for a, b in names if (path.parent / f"{a or b}.py").exists()]


def _builds(path: Path) -> bool:
    return bool(BUILDS.search(path.read_text(encoding="utf-8")))


def _moves_its_folders(path: Path) -> bool:
    """The script, or the bench module that builds for it, calls use_scratch_folders."""
    files = [path, *_imported_bench_modules(path)]
    return any("use_scratch_folders(" in f.read_text(encoding="utf-8") for f in files)


def _steps(workflow: Path) -> list[tuple[str, list[str], str]]:
    """(job name, systems, run text) for every step with a run block. Systems from the job's matrix or runs-on."""
    lines = workflow.read_text(encoding="utf-8").splitlines()
    steps: list[tuple[str, list[str], str]] = []
    job, systems = "", ["ubuntu"]
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^  [\w-]+:\s*$", line):
            job, systems = line.strip()[:-1], ["ubuntu"]
        found = re.search(r"\bos:\s*\[(.*?)\]", line) or re.search(r"runs-on:\s*([\w.-]+)\s*$", line)
        if found:
            systems = [s.strip().strip("'\"") for s in found.group(1).split(",")]
        run = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if run:
            indent, text = len(run.group(1)), [run.group(2)]
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > indent
            ):
                text.append(lines[index])
                index += 1
            steps.append((job, systems, "\n".join(text)))
            continue
        index += 1
    return steps


def _kinds(systems: list[str]) -> set[str]:
    kinds = set()
    for system in systems:
        low = system.lower()
        kinds.add("windows" if "windows" in low else "macos" if "macos" in low else "linux")
    return kinds


def _completes_fake_home(text: str, kinds: set[str]) -> list[str]:
    missing = []
    if not re.search(r'export HOME="\$RUNNER_TEMP', text):
        missing.append("HOME at a folder of the run")
    if "windows" in kinds and not re.search(r'LOCALAPPDATA="\$HOME/', text):
        missing.append('LOCALAPPDATA="$HOME/..." (Windows)')
    if "windows" in kinds and not re.search(r'USERPROFILE="\$RUNNER_TEMP', text):
        missing.append("USERPROFILE with HOME (Windows)")
    if "linux" in kinds and not re.search(r'XDG_RUNTIME_DIR="\$HOME/', text):
        missing.append('XDG_RUNTIME_DIR="$HOME/..." (Linux)')
    if "linux" in kinds and not re.search(r"unset XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME", text):
        missing.append("the runner's XDG folders cleared (Linux)")
    return missing


def _builders_in_ci() -> list[tuple[str, str, Path, list[str]]]:
    found = []
    for workflow in WORKFLOWS:
        for job, systems, text in _steps(workflow):
            for match in SCRIPT.finditer(text):
                path = _script_file(match)
                if not path.exists() or "pytest" in text.split(match.group(0))[0].splitlines()[-1]:
                    continue
                if _builds(path):
                    found.append((workflow.name, job, path, _completes_fake_home(text, _kinds(systems))))
                    if _moves_its_folders(path):
                        found[-1] = (*found[-1][:3], [])
    return found


def test_the_check_finds_the_steps_it_is_about():
    names = {path.name for _w, _j, path, _m in _builders_in_ci()}
    assert {"kill_checks.py", "seed_ledger.py", "ci_timing.py"} <= names


def test_every_step_that_builds_outside_the_daemon_moves_its_folders_or_completes_the_fake_home():
    problems = [
        f"{workflow} / {job}: {path.relative_to(ROOT)} lacks {', '.join(missing)}"
        for workflow, job, path, missing in _builders_in_ci()
        if missing
    ]
    assert problems == []


RUNNER = Path("/home/runner")
FAKE = Path("/home/runner/work/_temp/kill")


def _linux_folders(runtime: Path) -> dict[str, Path]:
    return {
        "config": FAKE / ".config" / "heliox-os",
        "data": FAKE / ".local" / "share" / "heliox-os",
        "state": FAKE / ".local" / "state" / "heliox-os",
        "runtime": runtime,
        "cache": FAKE / ".cache" / "heliox",
        "setup": FAKE / ".config" / "heliox-os",
    }


def test_a_fake_home_with_the_linux_runtime_folder_outside_is_the_real_folders():
    runtime = Path("/run/user/1001/heliox-os")
    kind, outside = classify(FAKE, RUNNER, _linux_folders(runtime))
    assert kind == "real" and outside == [f"runtime {runtime}"]


def test_a_fake_home_with_the_runtime_folder_inside_is_a_fake_home():
    assert classify(FAKE, RUNNER, _linux_folders(FAKE / ".runtime" / "heliox-os")) == ("fake home", [])


def test_the_accounts_own_home_is_the_real_folders_whatever_the_folders():
    assert classify(RUNNER, RUNNER, _linux_folders(RUNNER / ".runtime"))[0] == "real"


def _where_in_a_new_process(env_changes: dict[str, str | None], home: Path) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("HELIOX_TEST_ROOT",)}
    env.update({"HOME": str(home), "USERPROFILE": str(home)})
    for key, value in env_changes.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    code = "from pilot.real_folders import where_folders_are; print(where_folders_are()[0])"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True
    ).stdout.strip()


@pytest.mark.skipif(sys.platform != "win32", reason="the per-user Windows folder")
def test_windows_with_the_runners_own_localappdata_is_the_real_folders(tmp_path):
    own = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    assert _where_in_a_new_process({"LOCALAPPDATA": own, "XDG_RUNTIME_DIR": None}, tmp_path) == "real"


@pytest.mark.skipif(sys.platform != "win32", reason="the per-user Windows folder")
def test_windows_with_localappdata_inside_the_fake_home_is_a_fake_home(tmp_path):
    inside = str(tmp_path / "AppData" / "Local")
    changes = {"LOCALAPPDATA": inside, "XDG_RUNTIME_DIR": None, "XDG_CONFIG_HOME": None, "XDG_DATA_HOME": None}
    assert _where_in_a_new_process({**changes, "XDG_STATE_HOME": None}, tmp_path) == "fake home"
