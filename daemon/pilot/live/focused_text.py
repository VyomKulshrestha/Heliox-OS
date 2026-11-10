"""Read the text of the field the person is typing in, through UI Automation.

One hidden PowerShell process stays alive and answers each ``read`` request with the focused element's visible
text, so a read costs milliseconds instead of a new process. Only the focused element is read, only its visible
text (the last part of the field for long ones), and only when it is somewhere the person writes: a writable
text field or document, or a terminal. Page text, labels and buttons that happen to have focus are not read, so
an article or a chat transcript on screen is never treated as what the person typed. A password field is never
read either: UI Automation marks those and the helper refuses them before touching their value. Nothing it
returns is logged or stored.

This is the live layer's privacy boundary on Windows. Other platforms return nothing here.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform

logger = logging.getLogger("pilot.live.focused_text")

MAX_CHARS = 6000
READ_TIMEOUT_SECONDS = 1.5
STARTUP_TIMEOUT_SECONDS = 12.0  # PowerShell start plus loading the UI Automation assemblies
RESTART_BACKOFF_SECONDS = (2.0, 5.0, 15.0, 60.0)

# Terminal hosts whose text control is the console buffer: the prompt line at the end is what the person types.
TERMINAL_PROCESSES = ("WindowsTerminal", "OpenConsole", "conhost")

# Whether the focused element is somewhere the person writes, decided before any text is read:
# - a terminal (Windows Terminal's TermControl, or a console host) with a text pattern: kind "terminal";
# - an Edit or Document control whose ValuePattern says it is not read-only: kind "field". This is the signal
#   that holds in Chromium and Electron, where a web page's Document reports read-only and a text box or
#   contenteditable editor (a chat input) reports writable. The text selection's IsReadOnly attribute is not
#   used there: Chromium reports it as writable even on plain page text;
# - an Edit or Document without a ValuePattern (Word, some rich editors) whose whole text is marked not
#   read-only by the TextPattern: kind "field". A mixed or missing answer counts as read-only.
# Anything else (a page, a label, a button, a list) is not the person's writing, and its text is never read.
_HELPER = rf"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$terminals = @({", ".join(f"'{name}'" for name in TERMINAL_PROCESSES)})
$names = @{{}}
function ProcName($id) {{
  if (-not $names.ContainsKey($id)) {{
    if ($names.Count -gt 200) {{ $names.Clear() }}
    try {{ $names[$id] = (Get-Process -Id $id).ProcessName }} catch {{ $names[$id] = '' }}
  }}
  return $names[$id]
}}
while ($true) {{
  $line = [Console]::In.ReadLine()
  if ($line -eq $null -or $line -eq 'quit') {{ break }}
  $out = @{{ ok = $false }}
  try {{
    $el = [System.Windows.Automation.AutomationElement]::FocusedElement
    if ($el -ne $null) {{
      $c = $el.Current
      $out.ok = $true
      $out.type = $c.ControlType.ProgrammaticName
      $out.password = [bool]$c.IsPassword
      $out.pid = $c.ProcessId
      $out.editable = $false
      $out.kind = ''
      if (-not $c.IsPassword) {{
        $tp = $null
        $vp = $null
        $hasText = $el.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$tp)
        $hasValue = $el.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$vp)
        if ($hasText -and ($c.ClassName -eq 'TermControl' -or $terminals -contains (ProcName $c.ProcessId))) {{
          $out.kind = 'terminal'
        }} elseif ($out.type -eq 'ControlType.Edit' -or $out.type -eq 'ControlType.Document') {{
          if ($hasValue) {{
            if (-not $vp.Current.IsReadOnly) {{ $out.kind = 'field' }}
          }} elseif ($hasText) {{
            $ro = $tp.DocumentRange.GetAttributeValue([System.Windows.Automation.TextPattern]::IsReadOnlyAttribute)
            if (($ro -is [bool]) -and (-not $ro)) {{ $out.kind = 'field' }}
          }}
        }}
        if ($out.kind -ne '') {{
          $out.editable = $true
          $out.field = ($el.GetRuntimeId() -join '.')
          $text = $null
          if ($hasText) {{
            $parts = @()
            try {{ foreach ($r in $tp.GetVisibleRanges()) {{ $parts += $r.GetText({MAX_CHARS}) }} }} catch {{ }}
            if ($parts.Count -gt 0) {{ $text = ($parts -join "`n") }} else {{ $text = $tp.DocumentRange.GetText({MAX_CHARS}) }}
          }} elseif ($hasValue) {{
            $text = $vp.Current.Value
          }}
          if ($text -ne $null) {{
            if ($text.Length -gt {MAX_CHARS}) {{ $text = $text.Substring($text.Length - {MAX_CHARS}) }}
            $out.text = $text
          }}
        }}
      }}
    }}
  }} catch {{ $out.ok = $false }}
  [Console]::Out.WriteLine(($out | ConvertTo-Json -Compress))
  [Console]::Out.Flush()
}}
"""


@dataclass(frozen=True)
class FocusedText:
    text: str
    control_type: str
    pid: int
    password: bool
    # True only when the focused element is a writable text field, document or terminal; text is empty otherwise.
    editable: bool = False
    kind: str = ""  # "field", "terminal", or "" when not editable
    field: str = ""  # UI Automation runtime id: a new id means a different field, even in the same window


def window_handle(field: str) -> int | None:
    """The window a field lives in, from its UI Automation runtime id, or None when the id does not say.

    An element hosted in a window has a runtime id that starts with 42 and the window handle ("42.30540886" for
    Notepad's editor, "42.88540834.4.7.4.212" for a text box inside a browser's render window).
    """
    parts = field.split(".")
    if len(parts) < 2 or parts[0] != "42":
        return None
    try:
        handle = int(parts[1])
    except ValueError:
        return None
    return handle % 2**32 or None  # UI Automation prints the handle as a signed 32-bit number


def window_exists(handle: int) -> bool:
    """Whether a window is still open (minimised or in the background counts). True where it cannot be checked."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        return bool(ctypes.windll.user32.IsWindow(ctypes.c_void_p(handle)))  # type: ignore[attr-defined]
    except Exception:
        return True


def parse_reply(data: dict[str, Any]) -> FocusedText | None:
    """The helper's answer for one read. Text is kept only for a writable, non-password element."""
    if not data.get("ok"):
        return None
    password = bool(data.get("password"))
    kind = str(data.get("kind") or "")
    editable = bool(data.get("editable")) and not password and kind in ("field", "terminal")
    return FocusedText(
        text=str(data.get("text") or "") if editable else "",
        control_type=str(data.get("type") or "").removeprefix("ControlType."),
        pid=int(data.get("pid") or 0),
        password=password,
        editable=editable,
        kind=kind if editable else "",
        field=str(data.get("field") or "") if editable else "",
    )


class FocusedTextReader:
    """A long-lived UI Automation helper. Safe to construct anywhere; it starts on the first read."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._failures = 0
        self._retry_at = 0.0
        self._fresh = False
        self.available = CURRENT_PLATFORM == Platform.WINDOWS

    async def read(self) -> FocusedText | None:
        if not self.available or time.monotonic() < self._retry_at:
            return None
        async with self._lock:
            try:
                proc = await self._ensure()
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(b"read\n")
                await proc.stdin.drain()
                timeout = STARTUP_TIMEOUT_SECONDS if self._fresh else READ_TIMEOUT_SECONDS
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
                self._fresh = False
                if not raw:
                    raise ConnectionError("UI Automation helper exited")
                data = json.loads(raw.decode("utf-8", errors="replace"))
                self._failures = 0
            except (OSError, ConnectionError, TimeoutError, ValueError, AssertionError):
                await self._fail()
                return None
        return parse_reply(data) if isinstance(data, dict) else None

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self._proc is not None and self._proc.returncode is None:
            return self._proc
        encoded = base64.b64encode(_HELPER.encode("utf-16-le")).decode("ascii")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._proc = await asyncio.create_subprocess_exec(
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=flags,
        )
        self._fresh = True
        return self._proc

    async def _fail(self) -> None:
        self._failures += 1
        delay = RESTART_BACKOFF_SECONDS[min(self._failures - 1, len(RESTART_BACKOFF_SECONDS) - 1)]
        self._retry_at = time.monotonic() + delay
        logger.debug("Focused-text helper failed (%d in a row); retrying in %.0fs", self._failures, delay)
        await self.close()

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(Exception):
            if proc.stdin:
                proc.stdin.write(b"quit\n")
                await proc.stdin.drain()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        if proc.returncode is None:
            with contextlib.suppress(Exception):
                proc.kill()
