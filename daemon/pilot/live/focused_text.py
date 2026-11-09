"""Read the text of the field the person is typing in, through UI Automation.

One hidden PowerShell process stays alive and answers each ``read`` request with the focused element's visible
text, so a read costs milliseconds instead of a new process. Only the focused element is read, only its visible
text (the last part of the field for long ones), and never a password field: UI Automation marks those and the
helper refuses them before touching their value. Nothing it returns is logged or stored.

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

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform

logger = logging.getLogger("pilot.live.focused_text")

MAX_CHARS = 6000
READ_TIMEOUT_SECONDS = 1.5
STARTUP_TIMEOUT_SECONDS = 12.0  # PowerShell start plus loading the UI Automation assemblies
RESTART_BACKOFF_SECONDS = (2.0, 5.0, 15.0, 60.0)

_HELPER = rf"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
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
      if (-not $c.IsPassword) {{
        $text = $null
        $tp = $null
        if ($el.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$tp)) {{
          $parts = @()
          try {{ foreach ($r in $tp.GetVisibleRanges()) {{ $parts += $r.GetText({MAX_CHARS}) }} }} catch {{ }}
          if ($parts.Count -gt 0) {{ $text = ($parts -join "`n") }} else {{ $text = $tp.DocumentRange.GetText({MAX_CHARS}) }}
        }} else {{
          $vp = $null
          if ($el.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$vp)) {{ $text = $vp.Current.Value }}
        }}
        if ($text -ne $null) {{
          if ($text.Length -gt {MAX_CHARS}) {{ $text = $text.Substring($text.Length - {MAX_CHARS}) }}
          $out.text = $text
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
        if not data.get("ok"):
            return None
        return FocusedText(
            text=str(data.get("text") or "") if not data.get("password") else "",
            control_type=str(data.get("type") or "").removeprefix("ControlType."),
            pid=int(data.get("pid") or 0),
            password=bool(data.get("password")),
        )

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
