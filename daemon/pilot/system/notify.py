"""Desktop notifications from the daemon's ``notify`` action, on each system, reporting what really happened.

Windows shows a balloon through PowerShell, macOS a notification through ``osascript``, Linux through
``notify-send`` (libnotify; it needs a notification daemon, which every desktop session has). The title and the
text are data on every system, never part of the code that shows them: PowerShell reads them from
``$HelioxArgs`` and AppleScript from its arguments (``script_values``); notify-send takes them as arguments after
``--``. When the command fails, the result says the notification was not shown and why, and carries the message so
it still reaches the person in the conversation.
"""

from __future__ import annotations

from pilot.system.script_values import osascript_command

# Constant code; the title and text arrive as $HelioxArgs['summary'] and $HelioxArgs['body'].
WINDOWS_SCRIPT = (
    "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
    "$notify = New-Object System.Windows.Forms.NotifyIcon; "
    "$notify.Icon = [System.Drawing.SystemIcons]::Information; "
    "$notify.Visible = $true; "
    "$notify.ShowBalloonTip(5000, $HelioxArgs['summary'], $HelioxArgs['body'], "
    "[System.Windows.Forms.ToolTipIcon]::Info); "
    "Start-Sleep -Seconds 3; $notify.Dispose()"
)
MAC_SCRIPT = "display notification (item 2 of heliox_args) with title (item 1 of heliox_args)"


def notification_command(platform: str, summary: str, body: str) -> list[str]:
    """The command that shows a notification on macOS or Linux. Windows goes through ``show``, as its values travel in
    the environment (``run_powershell(WINDOWS_SCRIPT, values=...)``)."""
    if platform == "macos":
        return osascript_command(MAC_SCRIPT, [summary, body])
    return ["notify-send", "--app-name=Heliox", "--", summary, body]


async def show(platform: str, summary: str, body: str) -> tuple[int, str]:
    """Show a notification; (exit code, error text)."""
    from pilot.system.platform_detect import run_command, run_powershell

    if platform == "windows":
        code, _out, err = await run_powershell(WINDOWS_SCRIPT, values={"summary": summary, "body": body})
    else:
        code, _out, err = await run_command(notification_command(platform, summary, body))
    return code, err


def notification_result(platform: str, code: int, error: str, summary: str, body: str) -> str:
    """What the action reports: "Notification sent" only when the command succeeded."""
    if code == 0:
        return f"Notification sent: {summary}"
    if platform == "windows":
        return f"Notification: {summary} — {body}"  # unchanged: the balloon failed, the text carries the message
    reason = (error or "").strip().splitlines()[0] if (error or "").strip() else f"exit code {code}"
    if "not found" in reason.casefold() and platform == "linux":
        reason = "notify-send is not installed (package libnotify-bin or libnotify)"
    return f"Could not show a desktop notification ({reason[:160]}). {summary}: {body}".rstrip(": ")
