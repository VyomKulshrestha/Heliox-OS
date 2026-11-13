"""The build-time config change (TAURI_CONFIG) that lets msedgedriver attach to the app's WebView2, for the app-launch
job's Windows test build only; the released app never gets it.

msedgedriver asks WebView2 for its debugging port through the environment (WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS,
with --remote-debugging-port=0). In CI on 2d24504 (run 43352314827) the app had both variables, and its WebView2
browser process took the environment's user data folder but not the arguments: no debugging port, and no
DevToolsActivePort file anywhere. The app passes browser arguments of its own (wry's default,
--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection), which is the likely reason the environment's were
not used. So the test build passes the port itself, in the window's additionalBrowserArgs, after wry's default.

Prints the JSON merge patch: the main window as tauri.conf.json defines it, plus additionalBrowserArgs (a JSON merge
replaces arrays whole, so the window is repeated from the file rather than written out here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "tauri-app" / "src-tauri" / "tauri.conf.json"
WRY_DEFAULT = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection"


def patch(config: dict) -> dict:
    windows = [dict(window) for window in config["app"]["windows"]]
    windows[0]["additionalBrowserArgs"] = f"{WRY_DEFAULT} --remote-debugging-port=0"
    return {"app": {"windows": windows}}


if __name__ == "__main__":
    sys.stdout.write(json.dumps(patch(json.loads(CONFIG.read_text(encoding="utf-8")))))
