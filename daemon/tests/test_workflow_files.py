"""Every GitHub workflow file is one GitHub can run: it parses, has its triggers and jobs, every step does something,
and every bash script in it parses.

voice-memory.yml was handed over with two step names holding an unquoted colon and a space; the file did not parse,
so the workflow could never have started, and every test passed, because none parsed every file. This one does.
YAML reads the key ``on`` as True, so both spellings count. PyYAML comes with the daemon's own dependencies (caldav).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
FILES = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
NOT_BASH = {"pwsh", "powershell", "cmd", "python"}
EXPRESSION = re.compile(r"\$\{\{.*?\}\}", re.DOTALL)


def _bash() -> str | None:
    """Git's bash on Windows (a bare "bash" there can be WSL's launcher, which depends on WSL being up), else bash."""
    if sys.platform == "win32":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        return str(git_bash) if git_bash.exists() else None
    return shutil.which("bash")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_there_are_workflows_to_check():
    assert len(FILES) >= 10, FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_the_workflow_parses_and_every_job_has_steps_that_do_something(path):
    doc = _load(path)  # a file GitHub cannot parse fails here, with the line YAML names
    assert isinstance(doc, dict), f"{path.name}: not a mapping"
    assert "on" in doc or True in doc, f"{path.name}: no triggers (on)"
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{path.name}: no jobs"
    for name, job in jobs.items():
        if "uses" in job:  # a job that calls a reusable workflow has no steps of its own
            continue
        steps = job.get("steps")
        assert isinstance(steps, list) and steps, f"{path.name}: job {name} has no steps"
        for number, step in enumerate(steps, 1):
            assert isinstance(step, dict) and ("uses" in step or "run" in step), (
                f"{path.name}: job {name}, step {number} ({step.get('name', '?') if isinstance(step, dict) else step!r}) "
                "has neither uses nor run"
            )


def _shell(doc: dict, job: dict, step: dict) -> str:
    """The shell GitHub runs this step with: the step's, the job's defaults, the workflow's, else bash (a step on a
    Windows-only job without one runs in pwsh)."""
    for source in (step, (job.get("defaults") or {}).get("run") or {}, (doc.get("defaults") or {}).get("run") or {}):
        if source.get("shell"):
            return str(source["shell"]).split()[0]
    runs_on = str(job.get("runs-on", ""))
    return "pwsh" if "windows" in runs_on and "matrix" not in runs_on else "bash"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_bash_script_parses(path):
    bash = _bash()
    if bash is None:
        pytest.skip("no bash on this computer")
    doc = _load(path)
    problems = []
    for name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            script = step.get("run") if isinstance(step, dict) else None
            if not script or _shell(doc, job, step) in NOT_BASH:
                continue
            if "windows" in str(step.get("if", "")).casefold() and "!=" not in str(step.get("if", "")):
                continue  # a Windows-only step on a job without an explicit bash shell runs in pwsh there
            # GitHub fills ${{ ... }} in before bash sees the script: a plain word stands in for it.
            text = EXPRESSION.sub("GITHUB_EXPRESSION", script)
            done = subprocess.run([bash, "-n"], input=text.encode("utf-8"), capture_output=True, timeout=30)
            if done.returncode != 0:
                problems.append(f"job {name}, step {step.get('name', '?')!r}: {done.stderr.decode()[:200]}")
    assert problems == [], f"{path.name}: " + "; ".join(problems)
