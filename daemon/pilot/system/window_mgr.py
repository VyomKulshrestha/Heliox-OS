"""Window management — list, focus, close, minimize, maximize windows.

Cross-platform: Windows (PowerShell/user32), Linux (wmctrl/xdotool),
macOS (osascript/AppleScript).
"""

from __future__ import annotations

import logging

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.windows")


async def window_list() -> str:
    """List all open windows."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
            "Select-Object Id, ProcessName, MainWindowTitle | "
            "Format-Table -AutoSize | Out-String -Width 300"
        )
        if code != 0:
            raise RuntimeError(f"Window list failed: {err.strip()}")
        return out.strip()

    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get the name of every window of every process whose visible is true',
            ]
        )
        if code != 0:
            # Fallback
            code, out, err = await run_command(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of every process whose visible is true',
                ]
            )
        return out.strip() if code == 0 else "Could not list windows"

    else:  # Linux
        code, out, err = await run_command(["wmctrl", "-l"])
        if code != 0:
            # Fallback to xdotool
            code, out, err = await run_command(["xdotool", "search", "--onlyvisible", "--name", ""])
            if code != 0:
                raise RuntimeError("Window list failed (install wmctrl or xdotool)")
        return out.strip()


async def window_focus(window_id: str | None = None, title: str | None = None, process_name: str | None = None) -> str:
    """Focus/activate a window by ID, title, or process name."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        if process_name:
            code, out, err = await run_powershell(
                f"$p = Get-Process -Name '{process_name}' -ErrorAction SilentlyContinue | "
                f"Where-Object {{$_.MainWindowHandle -ne 0}} | Select-Object -First 1; "
                f"if ($p) {{ "
                f"  Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
                f'  public class Win32 {{ [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd); }}\'; '
                f"  [Win32]::SetForegroundWindow($p.MainWindowHandle) "
                f"}} else {{ Write-Error 'Process not found' }}"
            )
        elif title:
            code, out, err = await run_powershell(
                f"$p = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{title}*'}} | Select-Object -First 1; "
                f"if ($p) {{ "
                f"  Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
                f'  public class Win32 {{ [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd); }}\'; '
                f"  [Win32]::SetForegroundWindow($p.MainWindowHandle) "
                f"}} else {{ Write-Error 'Window not found' }}"
            )
        else:
            raise ValueError("Provide process_name or title to focus a window")

    elif CURRENT_PLATFORM == Platform.MACOS:
        target = process_name or title or ""
        code, out, err = await run_command(["osascript", "-e", f'tell application "{target}" to activate'])

    else:  # Linux
        if window_id:
            code, out, err = await run_command(["wmctrl", "-i", "-a", window_id])
        elif title:
            code, out, err = await run_command(["wmctrl", "-a", title])
        else:
            raise ValueError("Provide window_id or title to focus a window")

    if code != 0:
        raise RuntimeError(f"Window focus failed: {err.strip()}")
    return f"Focused window: {title or process_name or window_id}"


async def window_close(window_id: str | None = None, title: str | None = None, process_name: str | None = None) -> str:
    """Close a window."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        if process_name:
            code, out, err = await run_powershell(
                f"$p = Get-Process -Name '{process_name}' -ErrorAction SilentlyContinue; "
                f"if ($p) {{ $p | ForEach-Object {{ $_.CloseMainWindow() }} }} "
                f"else {{ Write-Error 'Process not found' }}"
            )
        elif title:
            code, out, err = await run_powershell(
                f"Get-Process | Where-Object {{$_.MainWindowTitle -like '*{title}*'}} | "
                f"ForEach-Object {{ $_.CloseMainWindow() }}"
            )
        else:
            raise ValueError("Provide process_name or title")

    elif CURRENT_PLATFORM == Platform.MACOS:
        target = process_name or title or ""
        code, out, err = await run_command(["osascript", "-e", f'tell application "{target}" to quit'])

    else:  # Linux
        if window_id:
            code, out, err = await run_command(["wmctrl", "-i", "-c", window_id])
        elif title:
            code, out, err = await run_command(["wmctrl", "-c", title])
        else:
            raise ValueError("Provide window_id or title")

    if code != 0:
        raise RuntimeError(f"Window close failed: {err.strip()}")
    return f"Closed window: {title or process_name or window_id}"


async def window_minimize(title: str | None = None, process_name: str | None = None) -> str:
    """Minimize a window."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        target = process_name or title
        code, out, err = await run_powershell(
            f"$p = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{target}*'}} | Select-Object -First 1; "
            f"if ($p) {{ "
            f"  Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
            f'  public class Win32 {{ [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow); }}\'; '
            f"  [Win32]::ShowWindow($p.MainWindowHandle, 6) "  # SW_MINIMIZE = 6
            f"}}"
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        target = process_name or title or ""
        code, out, err = await run_command(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to set miniaturized of first window of '
                f'(first process whose name is "{target}") to true',
            ]
        )
    else:
        if title:
            code, out, err = await run_command(["xdotool", "search", "--name", title, "windowminimize"])
        else:
            raise ValueError("Provide title to minimize")

    if code != 0:
        raise RuntimeError(f"Minimize failed: {err.strip()}")
    return f"Minimized window: {title or process_name}"


async def window_maximize(title: str | None = None, process_name: str | None = None) -> str:
    """Maximize a window."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        target = process_name or title
        code, out, err = await run_powershell(
            f"$p = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{target}*'}} | Select-Object -First 1; "
            f"if ($p) {{ "
            f"  Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
            f'  public class Win32 {{ [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow); }}\'; '
            f"  [Win32]::ShowWindow($p.MainWindowHandle, 3) "  # SW_MAXIMIZE = 3
            f"}}"
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        target = process_name or title or ""
        code, out, err = await run_command(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to set size of first window of '
                f'(first process whose name is "{target}") to {{1920, 1080}}',
            ]
        )
    else:
        if title:
            code, out, err = await run_command(["wmctrl", "-r", title, "-b", "add,maximized_vert,maximized_horz"])
        else:
            raise ValueError("Provide title to maximize")

    if code != 0:
        raise RuntimeError(f"Maximize failed: {err.strip()}")
    return f"Maximized window: {title or process_name}"
