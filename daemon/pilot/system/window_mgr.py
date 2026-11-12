"""Window management — list, focus, close, minimize, maximize windows.

Cross-platform: Windows (PowerShell/user32), Linux (wmctrl/xdotool),
macOS (the Accessibility API through pilot.system.macos_windows; AppleScript
through System Events only when pyobjc is not installed).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell
from pilot.system.script_values import osascript_command

logger = logging.getLogger("pilot.system.windows")

_NOT_HANDLED = object()


async def _mac_accessibility(action: str, process_name: str | None, title: str | None) -> Any:
    """Run a window action through the Accessibility API on macOS; ``_NOT_HANDLED`` when pyobjc is missing."""
    from pilot.system import macos_windows

    windows = macos_windows.available()
    if windows is None:
        return _NOT_HANDLED

    def run() -> Any:
        if action == "list":
            return windows.list()
        app = process_name or windows.owner_of(title or "")
        return getattr(windows, action)(app, title)

    return await asyncio.to_thread(run)


# Window titles and process names come from plans, and titles often from the screen (a web page chooses its own),
# so they are never part of the code text: PowerShell reads them from $HelioxArgs (run_powershell values=), and
# AppleScript from its arguments (script_values.osascript_command). A title is matched as text, its wildcard
# characters escaped.
_SELECT = (
    "$pattern = '*' + [WildcardPattern]::Escape([string]$HelioxArgs['title']) + '*'; "
    "if ($HelioxArgs['process']) { $p = Get-Process -Name $HelioxArgs['process'] -ErrorAction SilentlyContinue | "
    "Where-Object {$_.MainWindowHandle -ne 0} | Select-Object -First 1 } "
    "else { $p = Get-Process | Where-Object {$_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like $pattern} | "
    "Select-Object -First 1 }; "
)


def _target(*, title: str | None, process_name: str | None) -> dict[str, str]:
    """The values ``_SELECT`` reads: the process name when given, else the title."""
    if process_name:
        return {"process": process_name, "title": ""}
    if title:
        return {"process": "", "title": title}
    raise ValueError("Provide process_name or title")


_WIN32_FOCUS = (
    "Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
    'public class Win32 { [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow); '
    '[DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow(); '
    '[DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid); }\'; '
)
_FOCUS = (
    _SELECT
    + "if (-not $p) { Write-Error 'Window not found'; exit 1 }; "
    + _WIN32_FOCUS
    + "[void][Win32]::ShowWindowAsync($p.MainWindowHandle, 9); "
    + "$shell = New-Object -ComObject WScript.Shell; "
    + "$activated = $shell.AppActivate([int]$p.Id); Start-Sleep -Milliseconds 120; "
    + "$foregroundPid = 0; [void][Win32]::GetWindowThreadProcessId([Win32]::GetForegroundWindow(), [ref]$foregroundPid); "
    + "if (-not $activated -or $foregroundPid -ne $p.Id) { Write-Error 'Window activation was rejected'; exit 1 }; $p.Id"
)
_CLOSE = (
    "if ($HelioxArgs['process']) { $p = Get-Process -Name $HelioxArgs['process'] -ErrorAction SilentlyContinue; "
    "if ($p) { $p | ForEach-Object { $_.CloseMainWindow() } } else { Write-Error 'Process not found' } } "
    "else { $pattern = '*' + [WildcardPattern]::Escape([string]$HelioxArgs['title']) + '*'; "
    "Get-Process | Where-Object {$_.MainWindowTitle -like $pattern} | ForEach-Object { $_.CloseMainWindow() } }"
)
# Minimize and maximize match the name or title given against window titles, as before.
_SHOW_WINDOW = (
    "$pattern = '*' + [WildcardPattern]::Escape([string]$HelioxArgs['title']) + '*'; "
    "$p = Get-Process | Where-Object {$_.MainWindowTitle -like $pattern} | Select-Object -First 1; "
    "if ($p) { Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; "
    'public class Win32 { [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow); }\'; '
    "[Win32]::ShowWindow($p.MainWindowHandle, [int]$HelioxArgs['show']) }"
)
_MAC_EXISTS = (
    'tell application "System Events"\n'
    "set matches to every process whose name is (item 1 of heliox_args)\n"
    "if (count of matches) is 0 then return 0\n"
    "return count windows of first item of matches\n"
    "end tell"
)
_MAC_ACTIVATE = "tell application (item 1 of heliox_args) to activate"
_MAC_QUIT = "tell application (item 1 of heliox_args) to quit"
_MAC_MINIMIZE = (
    'tell application "System Events" to set miniaturized of first window of '
    "(first process whose name is (item 1 of heliox_args)) to true"
)
_MAC_MAXIMIZE = (
    'tell application "System Events" to set size of first window of '
    "(first process whose name is (item 1 of heliox_args)) to {1920, 1080}"
)


def _windows_text_element_script() -> str:
    """PowerShell that binds ``$element`` to the target window's active editor."""
    return (
        "Add-Type -AssemblyName UIAutomationClient; Add-Type -AssemblyName UIAutomationTypes; "
        "Add-Type -TypeDefinition 'using System; using System.Text; using System.Runtime.InteropServices; "
        "public class NativeText { "
        '[DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, IntPtr l); '
        '[DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, StringBuilder l); '
        '[DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, string l); }\'; '
        "$root = [System.Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle); "
        "$focused = New-Object System.Windows.Automation.PropertyCondition("
        "[System.Windows.Automation.AutomationElement]::HasKeyboardFocusProperty, $true); "
        "$element = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $focused); "
        "if (-not $element) { "
        "$document = New-Object System.Windows.Automation.PropertyCondition("
        "[System.Windows.Automation.AutomationElement]::ControlTypeProperty, "
        "[System.Windows.Automation.ControlType]::Document); "
        "$element = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $document); "
        "if (-not $element) { "
        "$edit = New-Object System.Windows.Automation.PropertyCondition("
        "[System.Windows.Automation.AutomationElement]::ControlTypeProperty, "
        "[System.Windows.Automation.ControlType]::Edit); "
        "$element = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $edit); "
        "if (-not $element) { "
        "$pane = New-Object System.Windows.Automation.PropertyCondition("
        "[System.Windows.Automation.AutomationElement]::ControlTypeProperty, "
        "[System.Windows.Automation.ControlType]::Pane); "
        "$element = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $pane); "
        "} "
        "} } "
    )


async def window_read_text(*, title: str | None = None, process_name: str | None = None) -> str:
    """Read exact text from a target window without requiring foreground focus."""
    if CURRENT_PLATFORM != Platform.WINDOWS:
        raise RuntimeError("Background window text reading is currently available on Windows only")
    values = _target(title=title, process_name=process_name)
    script = (
        "$ErrorActionPreference = 'Stop'; "
        + _SELECT
        + "if (-not $p) { Write-Error 'Window not found'; exit 1 }; "
        + _windows_text_element_script()
        + "$textPattern = $null; $valuePattern = $null; $text = $null; "
        + "if ($element -and $element.TryGetCurrentPattern("
        + "[System.Windows.Automation.TextPattern]::Pattern, [ref]$textPattern)) { "
        + "$text = $textPattern.DocumentRange.GetText(-1) } "
        + "elseif ($element -and $element.TryGetCurrentPattern("
        + "[System.Windows.Automation.ValuePattern]::Pattern, [ref]$valuePattern)) { "
        + "$text = $valuePattern.Current.Value } "
        + "elseif ($element -and $element.Current.NativeWindowHandle -ne 0) { "
        + "$handle = [IntPtr]$element.Current.NativeWindowHandle; "
        + "$length = [int][NativeText]::SendMessage($handle, 14, [IntPtr]::Zero, [IntPtr]::Zero); "
        + "$buffer = New-Object System.Text.StringBuilder ($length + 1); "
        + "[void][NativeText]::SendMessage($handle, 13, [IntPtr]($length + 1), $buffer); $text = $buffer.ToString() } "
        + "else { Write-Error 'No readable text control found'; exit 1 }; "
        + "$bytes = [System.Text.Encoding]::UTF8.GetBytes([string]$text); "
        + "[Convert]::ToBase64String($bytes)"
    )
    code, out, err = await run_powershell(script, values=values)
    if code != 0:
        raise RuntimeError(f"Window text read failed: {err.strip()}")
    try:
        return base64.b64decode(out.strip()).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError("Window text read returned invalid data") from error


async def window_set_text(
    text: str,
    *,
    title: str | None = None,
    process_name: str | None = None,
) -> str:
    """Set and verify a target window's active editor through Windows UIA."""
    if CURRENT_PLATFORM != Platform.WINDOWS:
        raise RuntimeError("Background window text entry is currently available on Windows only")
    values = {**_target(title=title, process_name=process_name), "text": text}
    script = (
        "$ErrorActionPreference = 'Stop'; "
        + _SELECT
        + "if (-not $p) { Write-Error 'Window not found'; exit 1 }; "
        + _windows_text_element_script()
        + "$valuePattern = $null; "
        + "$hasValue = $element -and $element.TryGetCurrentPattern("
        + "[System.Windows.Automation.ValuePattern]::Pattern, [ref]$valuePattern); "
        + "$nativeHandle = if ($element) { [IntPtr]$element.Current.NativeWindowHandle } else { [IntPtr]::Zero }; "
        + "if (-not $hasValue -and $nativeHandle -eq [IntPtr]::Zero) { "
        + "Write-Error 'No writable text control found'; exit 1 }; "
        + "$expected = [string]$HelioxArgs['text']; "
        + "if ($hasValue) { $valuePattern.SetValue($expected); $actual = $valuePattern.Current.Value } "
        + "else { [void][NativeText]::SendMessage($nativeHandle, 12, [IntPtr]::Zero, $expected); "
        + "$length = [int][NativeText]::SendMessage($nativeHandle, 14, [IntPtr]::Zero, [IntPtr]::Zero); "
        + "$buffer = New-Object System.Text.StringBuilder ($length + 1); "
        + "[void][NativeText]::SendMessage($nativeHandle, 13, [IntPtr]($length + 1), $buffer); "
        + "$actual = $buffer.ToString() }; "
        + "if ($actual -ne $expected) { "
        + "Write-Error 'Text verification mismatch'; exit 1 }; 'ok'"
    )
    code, out, err = await run_powershell(script, values=values)
    if code != 0:
        raise RuntimeError(f"Window text entry failed: {err.strip()}")
    return f"Set exact text in window: {title or process_name}"


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
        rows = await _mac_accessibility("list", None, None)
        if rows is not _NOT_HANDLED:
            return "\n".join(f"{app} - {title}" if title else app for app, title in rows)
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


async def window_exists(
    *,
    window_id: str | None = None,
    title: str | None = None,
    process_name: str | None = None,
) -> bool:
    """Observe whether the exact close target still has a visible window."""
    if not any((window_id, title, process_name)):
        raise ValueError("Provide window_id, process_name, or title")

    if CURRENT_PLATFORM == Platform.WINDOWS:
        if not (process_name or title):
            raise ValueError("Windows window observation requires process_name or title")
        code, out, err = await run_powershell(
            _SELECT + "if ($p) { 'present' }", values=_target(title=title, process_name=process_name)
        )
        if code != 0:
            raise RuntimeError(f"Window observation failed: {err.strip()}")
        return out.strip() == "present"

    if CURRENT_PLATFORM == Platform.MACOS:
        present = await _mac_accessibility("exists", process_name, None if process_name else title)
        if present is not _NOT_HANDLED:
            return bool(present)
        code, out, err = await run_command(osascript_command(_MAC_EXISTS, [process_name or title or ""]))
        if code != 0:
            raise RuntimeError(f"Window observation failed: {err.strip()}")
        try:
            return int(out.strip()) > 0
        except ValueError as error:
            raise RuntimeError("Window observation returned invalid data") from error

    code, out, err = await run_command(["wmctrl", "-l"])
    if code != 0:
        raise RuntimeError(f"Window observation failed: {err.strip()}")
    rows = [line for line in out.splitlines() if line.strip()]
    if window_id:
        expected = window_id.casefold()
        return any(row.split(maxsplit=1)[0].casefold() == expected for row in rows)
    expected = str(title or process_name).casefold()
    return any(expected in row.casefold() for row in rows)


async def window_focus(window_id: str | None = None, title: str | None = None, process_name: str | None = None) -> str:
    """Focus/activate a window by ID, title, or process name."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        if not (process_name or title):
            raise ValueError("Provide process_name or title to focus a window")
        code, out, err = await run_powershell(_FOCUS, values=_target(title=title, process_name=process_name))

    elif CURRENT_PLATFORM == Platform.MACOS:
        if await _mac_accessibility("focus", process_name, title) is not _NOT_HANDLED:
            return f"Focused window: {title or process_name or window_id}"
        code, out, err = await run_command(osascript_command(_MAC_ACTIVATE, [process_name or title or ""]))

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
        if not (process_name or title):
            raise ValueError("Provide process_name or title")
        code, out, err = await run_powershell(_CLOSE, values=_target(title=title, process_name=process_name))

    elif CURRENT_PLATFORM == Platform.MACOS:
        # Closes the window, as its close button does; the AppleScript fallback can only quit the app.
        if await _mac_accessibility("close", process_name, title) is not _NOT_HANDLED:
            return f"Closed window: {title or process_name or window_id}"
        code, out, err = await run_command(osascript_command(_MAC_QUIT, [process_name or title or ""]))

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
        code, out, err = await run_powershell(
            _SHOW_WINDOW,
            values={"title": process_name or title or "", "show": "6"},  # SW_MINIMIZE
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        if await _mac_accessibility("minimize", process_name, title) is not _NOT_HANDLED:
            return f"Minimized window: {title or process_name}"
        code, out, err = await run_command(osascript_command(_MAC_MINIMIZE, [process_name or title or ""]))
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
        code, out, err = await run_powershell(
            _SHOW_WINDOW,
            values={"title": process_name or title or "", "show": "3"},  # SW_MAXIMIZE
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        if await _mac_accessibility("maximize", process_name, title) is not _NOT_HANDLED:
            return f"Maximized window: {title or process_name}"
        code, out, err = await run_command(osascript_command(_MAC_MAXIMIZE, [process_name or title or ""]))
    else:
        if title:
            code, out, err = await run_command(["wmctrl", "-r", title, "-b", "add,maximized_vert,maximized_horz"])
        else:
            raise ValueError("Provide title to maximize")

    if code != 0:
        raise RuntimeError(f"Maximize failed: {err.strip()}")
    return f"Maximized window: {title or process_name}"
