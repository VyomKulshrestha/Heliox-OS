"""export_logs.py — Bundles logs, config, and audit trails into a zip for bug reports."""

from __future__ import annotations

import re
import zipfile
from datetime import datetime
from pathlib import Path

from pilot.config import CONFIG_FILE, LOG_FILE, STATE_DIR


def _redact_config(config_text: str) -> str:
    """Redact API keys and secrets from config before including in zip."""
    return re.sub(
        r'(api_key|secret|token|password)\s*=\s*"[^"]*"',
        r'\1 = "[REDACTED]"',
        config_text,
        flags=re.IGNORECASE,
    )


def export_logs() -> Path:
    """Package logs, config.toml, and audit trails into a zip on the Desktop."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    zip_path = desktop / f"heliox_bugreport_{timestamp}.zip"

    log_dir = STATE_DIR
    config_file = CONFIG_FILE

    files_added = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add all log files
        if log_dir.exists():
            for log_file in sorted(log_dir.rglob("*")):
                if log_file.is_file():
                    arcname = "logs/" + log_file.relative_to(log_dir).as_posix()
                    zf.write(log_file, arcname=arcname)
                    files_added += 1
        else:
            print(f"[warn] Log directory not found: {log_dir}")

        # Add redacted config.toml
        if config_file.exists():
            raw = config_file.read_text(encoding="utf-8", errors="replace")
            zf.writestr("config.toml", _redact_config(raw))
            files_added += 1
        else:
            print(f"[warn] Config file not found: {config_file}")

    print(f"[heliox] Bug report created: {zip_path} ({files_added} files)")
    return zip_path
