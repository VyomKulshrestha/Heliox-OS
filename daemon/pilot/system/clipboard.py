"""Clipboard operations — read and write system clipboard.

Cross-platform: uses pyperclip if available, falls back to OS commands.
"""

from __future__ import annotations

import logging

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.clipboard")


async def clipboard_read() -> str:
    """Read text from the system clipboard."""
    try:
        import pyperclip

        return pyperclip.paste() or "(clipboard is empty)"
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell("Get-Clipboard")
        if code != 0:
            raise RuntimeError(f"Clipboard read failed: {err.strip()}")
        return out.strip() or "(clipboard is empty)"

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["pbpaste"])
        if code != 0:
            raise RuntimeError(f"Clipboard read failed: {err.strip()}")
        return out or "(clipboard is empty)"

    else:  # Linux
        # Try xclip, then xsel
        code, out, err = await run_command(["xclip", "-selection", "clipboard", "-o"])
        if code == 0:
            return out or "(clipboard is empty)"
        code, out, err = await run_command(["xsel", "--clipboard", "--output"])
        if code == 0:
            return out or "(clipboard is empty)"
        raise RuntimeError("No clipboard tool available (install xclip or xsel)")


async def clipboard_write(content: str) -> str:
    """Write text to the system clipboard."""
    try:
        import pyperclip

        pyperclip.copy(content)
        return f"Copied {len(content)} characters to clipboard"
    except ImportError:
        pass

    if CURRENT_PLATFORM == Platform.WINDOWS:
        escaped = content.replace("'", "''")
        code, out, err = await run_powershell(f"Set-Clipboard -Value '{escaped}'")
        if code != 0:
            raise RuntimeError(f"Clipboard write failed: {err.strip()}")
        return f"Copied {len(content)} characters to clipboard"

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["pbcopy"], input_data=content.encode("utf-8"))
        if code != 0:
            raise RuntimeError(f"Clipboard write failed: {err.strip()}")
        return f"Copied {len(content)} characters to clipboard"

    else:  # Linux
        code, out, err = await run_command(["xclip", "-selection", "clipboard"], input_data=content.encode("utf-8"))
        if code == 0:
            return f"Copied {len(content)} characters to clipboard"
        code, out, err = await run_command(["xsel", "--clipboard", "--input"], input_data=content.encode("utf-8"))
        if code == 0:
            return f"Copied {len(content)} characters to clipboard"
        raise RuntimeError("No clipboard tool available (install xclip or xsel)")
