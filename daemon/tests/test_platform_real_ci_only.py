"""The real checks that change something outside their own temporary folder refuse to run off a CI runner.

Each is started as a person would start it by hand, in an empty home and without GITHUB_ACTIONS or RUNNER_TEMP: it
must print one NOT RUN sentence, exit with 3 (neither a pass nor a failure) and leave the home folder empty.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PLATFORM_REAL = Path(__file__).resolve().parent / "platform_real"
GUARDED = (
    "keychain_locked_check.py",
    "storage_faults_check.py",
    "stop_timing_check.py",
    "soak_check.py",
    "two_copies_check.py",
    "seed_ledger.py",
    "repair_data_check.py",
    "protected_case_check.py",
    "audio_check.py",
    "credential_store_check.py",
    "learner_key_reboot_check.py",
    "voice_worker_stop_check.py",
    "network_drop_check.py",
    "frozen_daemon_check.py",
    "run_record_check.py",
    "deb_install_check.py",  # installs a package and creates a user, in a container
    "windows_package_check.py",  # installs and uninstalls the app in the real profile
    "end_heliox_check.py",  # runs the uninstaller's script, which ends processes
    "appimage_check.py",  # installs packages and creates a user, in a container
    # the desktop drivers: they open windows and type, click or show notifications
    "app_launch_check.py",
    "apps_check.py",
    "apps_fields_check.py",
    "apps_fields_macos_check.py",
    "click_hang_check.py",
    "linux_atspi_check.py",
    "linux_portal_check.py",
    "linux_windows_check.py",
    "live_end_to_end_check.py",
    "live_more_checks.py",
    "macos_ax_check.py",
    "notify_check.py",
    "ocr_check.py",
    "toolkit_check.py",
    "windows_fields_check.py",
)


def _person_env(home: Path, **extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in {"GITHUB_ACTIONS", "RUNNER_TEMP"}}
    for key in ("HOME", "USERPROFILE"):
        env[key] = str(home)
    for key in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
        env[key] = str(home / key.lower())
    env["PYTHONPATH"] = str(PLATFORM_REAL.parents[1])
    return {**env, **extra}


@pytest.mark.parametrize("script", GUARDED)
def test_off_a_runner_the_check_says_why_and_changes_nothing(script, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    done = subprocess.run(
        [sys.executable, str(PLATFORM_REAL / script), "before", str(tmp_path / "state.json")],
        env=_person_env(home),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 3, done.stdout + done.stderr
    assert done.stdout.startswith("NOT RUN: this check "), done.stdout
    assert "only on a CI runner" in done.stdout
    assert list(home.iterdir()) == []
    assert not (tmp_path / "state.json").exists()


def test_github_actions_alone_is_not_a_runner(tmp_path):
    """GITHUB_ACTIONS set by hand without RUNNER_TEMP still refuses."""
    home = tmp_path / "home"
    home.mkdir()
    done = subprocess.run(
        [sys.executable, str(PLATFORM_REAL / "keychain_locked_check.py")],
        env=_person_env(home, GITHUB_ACTIONS="true"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 3 and "NOT RUN" in done.stdout


@pytest.mark.parametrize(
    "script",
    ("real_removal.sh", "with_keyring.sh", "deb_install_check.sh", "rpm_install_check.sh", "appimage_check.sh"),
)
def test_the_shell_scripts_refuse_too(script):
    text = (PLATFORM_REAL / script).read_text(encoding="utf-8")
    guard = text.index('if [ "${GITHUB_ACTIONS:-}" != "true" ] || [ -z "${RUNNER_TEMP:-}" ]; then')
    commands = [match.start() for match in re.finditer(r"^(?!#|set |if |  |fi$)\S", text, flags=re.MULTILINE)]
    first_effect = next(index for index in commands if index > text.index("\n"))  # after the #! line
    assert guard < first_effect
    assert "exit 3" in text[guard : guard + 400]
