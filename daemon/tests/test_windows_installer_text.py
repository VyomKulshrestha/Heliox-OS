"""The Windows installer's text and hooks (tauri-app/src-tauri/windows), read on every system.

Tauri uses windows/English.nsh instead of its own English strings, so the file must define every string Tauri's
installer script names (TAURI_STRINGS, from Tauri 2's languages/English.nsh): a missing one would show an empty
message. The uninstaller's checkbox must say what it deletes (the app's web view data, never the person's settings
and data), and the message at the end of an interactive uninstall must name what is left, in FIRST_RUN.md's three
kinds. The hooks end the app and the daemon before uninstalling and stay out of updates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WINDOWS = Path(__file__).resolve().parents[2] / "tauri-app" / "src-tauri" / "windows"
CONFIG = WINDOWS.parent / "tauri.conf.json"

# Tauri 2's English strings, by name. On a Tauri upgrade, compare with its languages/English.nsh.
TAURI_STRINGS = (
    "addOrReinstall", "alreadyInstalled", "alreadyInstalledLong", "appRunning", "noPermissionToCloseApp",
    "appRunningOkKill", "chooseMaintenanceOption", "chooseHowToInstall", "createDesktop", "dontUninstall",
    "dontUninstallDowngrade", "failedToKillApp", "installingWebview2", "newerVersionInstalled", "older",
    "olderOrUnknownVersionInstalled", "silentDowngrades", "unableToUninstall", "uninstallApp",
    "uninstallBeforeInstalling", "unknown", "webview2AbortError", "webview2DownloadError", "webview2DownloadSuccess",
    "webview2Downloading", "webview2InstallError", "webview2InstallSuccess", "deleteAppData",
)  # fmt: skip


def strings() -> dict[str, str]:
    text = (WINDOWS / "English.nsh").read_text(encoding="utf-8")
    found = re.findall(r'^LangString (\w+) \$\{LANG_ENGLISH\} "(.*)"$', text, flags=re.MULTILINE)
    names = [name for name, _ in found]
    assert len(names) == len(set(names)), "a string is defined twice"
    return dict(found)


def test_every_string_tauri_names_is_defined_and_only_those_plus_helioxs_own():
    assert set(strings()) == {*TAURI_STRINGS, "helioxLeftBehind"}


def test_no_template_placeholders_that_a_custom_file_does_not_get_filled_in():
    assert "{{" not in (WINDOWS / "English.nsh").read_text(encoding="utf-8")


def test_the_checkbox_says_it_deletes_the_web_view_data_and_keeps_the_persons():
    label = strings()["deleteAppData"]
    assert "web view data" in label and "settings and data are kept" in label


def test_the_message_names_what_is_left_in_the_three_kinds():
    message = strings()["helioxLeftBehind"]
    assert "Nothing of yours was deleted" in message
    assert "Program files are still in $PROFILE\\.config\\heliox-os" in message
    for part in ("env", "python", "tools", "uv-cache", "setup"):
        assert part in message
    assert "Working files" in message and "safe to delete" in message
    for folder in ("$PROFILE\\.local\\state\\heliox-os", "$LOCALAPPDATA\\heliox-os", "$LOCALAPPDATA\\com.helioxos.app"):
        assert folder in message
    assert "settings, memory and receipts are kept" in message and "$PROFILE\\.local\\share\\heliox-os" in message


def test_the_hooks_end_heliox_first_and_speak_only_in_an_interactive_uninstall():
    hooks = (WINDOWS / "installer-hooks.nsh").read_text(encoding="utf-8")
    before = hooks[hooks.index("!macro NSIS_HOOK_PREUNINSTALL") : hooks.index("!macro NSIS_HOOK_POSTUNINSTALL")]
    after = hooks[hooks.index("!macro NSIS_HOOK_POSTUNINSTALL") :]
    assert "${If} $UpdateMode <> 1" in before and "${If} $UpdateMode <> 1" in after
    assert "end-heliox.ps1" in before and "HELIOX_APP" in before and "${MAINBINARYNAME}.exe" in before
    assert "${AndIfNot} ${Silent}" in after and '"$(helioxLeftBehind)"' in after


def test_the_end_script_ends_the_app_then_the_daemon_and_deletes_nothing():
    script = (WINDOWS / "end-heliox.ps1").read_text(encoding="utf-8")
    assert script.index("$env:HELIOX_APP") < script.index("-m pilot.server")
    # Only Heliox's own environment folder: a checkout cloned as "Heliox-OS" must not match.
    assert "'*\\.config\\heliox-os\\env\\*'" in script and "'*\\heliox-os\\*'" not in script
    assert "taskkill.exe /F /T /PID" in script
    assert not re.search(r"Remove-Item|\bdel\b|\brd\b|rmdir", script, flags=re.IGNORECASE)


def test_the_bundle_settings_point_at_these_files():
    nsis = json.loads(CONFIG.read_text(encoding="utf-8"))["bundle"]["windows"]["nsis"]
    assert nsis["installerHooks"] == "windows/installer-hooks.nsh"
    assert nsis["customLanguageFiles"] == {"English": "windows/English.nsh"}
    assert "English" in nsis["languages"]
    for path in (nsis["installerHooks"], nsis["customLanguageFiles"]["English"]):
        assert (CONFIG.parent / path).is_file()
