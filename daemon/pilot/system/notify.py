"""Desktop notifications from the daemon's ``notify`` action, on each system, reporting what really happened.

Windows shows a balloon through PowerShell, macOS a notification through ``osascript``, Linux through
``notify-send`` (libnotify; it needs a notification daemon, which every desktop session has). Text is passed as
data, never spliced into a script unescaped. When the command fails, the result says the notification was not
shown and why, and carries the message so it still reaches the person in the conversation.
"""

from __future__ import annotations


def powershell_quoted(text: str) -> str:
    """A PowerShell single-quoted string literal: only ' needs escaping, by doubling it."""
    return "'" + text.replace("'", "''") + "'"


def applescript_quoted(text: str) -> str:
    """An AppleScript string literal: backslash and double quote are escaped."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notification_command(platform: str, summary: str, body: str) -> list[str]:
    """The command that shows a notification on ``platform`` ("windows", "macos" or "linux")."""
    if platform == "windows":
        script = (
            "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
            "$notify = New-Object System.Windows.Forms.NotifyIcon; "
            "$notify.Icon = [System.Drawing.SystemIcons]::Information; "
            "$notify.Visible = $true; "
            f"$notify.ShowBalloonTip(5000, {powershell_quoted(summary)}, {powershell_quoted(body)}, "
            "[System.Windows.Forms.ToolTipIcon]::Info); "
            "Start-Sleep -Seconds 3; $notify.Dispose()"
        )
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    if platform == "macos":
        script = f"display notification {applescript_quoted(body)} with title {applescript_quoted(summary)}"
        return ["osascript", "-e", script]
    return ["notify-send", "--app-name=Heliox", "--", summary, body]


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
