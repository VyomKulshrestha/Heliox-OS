"""Read a Windows registry value back in a new PowerShell process, for the registry_write post-condition.

The key, the value name and the requested data are data (``$HelioxArgs``), never part of the script text. The
answer is JSON, so nothing depends on the language Windows speaks.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

from pilot.system.platform_detect import run_powershell

# Reads the value as stored (environment names in REG_EXPAND_SZ left unexpanded) and its kind, and converts the
# requested text the way the registry provider converts -Value for that kind (LanguagePrimitives.ConvertTo), so a
# number written as "0x10" or a single byte compares as the number PowerShell stored. "expected" is null when that
# conversion fails.
READ_BACK_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$key = Get-Item -LiteralPath ('Registry::' + [string]$HelioxArgs['key']) -ErrorAction SilentlyContinue
if (-not $key) { '{"key":false,"present":false}'; exit 0 }
$name = [string]$HelioxArgs['name']
if (-not ($key.GetValueNames() | Where-Object { $_ -eq $name })) { '{"key":true,"present":false}'; exit 0 }
$kind = $key.GetValueKind($name).ToString()
$data = $key.GetValue($name, $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
if ($data -is [byte[]]) { $data = [int[]]$data }
$target = @{ 'DWord' = [int]; 'QWord' = [long]; 'Binary' = [byte[]]; 'MultiString' = [string[]] }[[string]$HelioxArgs['kind']]
$expected = $null
if ($target) {
    try { $expected = [System.Management.Automation.LanguagePrimitives]::ConvertTo($HelioxArgs['value'], $target) }
    catch {
        if ($target -eq [int]) {
            try { $expected = [System.Management.Automation.LanguagePrimitives]::ConvertTo($HelioxArgs['value'], [uint32]) }
            catch { $expected = $null }
        }
    }
    if ($expected -is [byte[]]) { $expected = [int[]]$expected }
}
[pscustomobject]@{ key = $true; present = $true; kind = $kind; data = @($data); expected = @($expected) } |
    ConvertTo-Json -Compress -Depth 3
"""


class RegistryValue(NamedTuple):
    key_exists: bool
    present: bool
    kind: str | None = None  # RegistryValueKind name: String, ExpandString, Binary, DWord, MultiString, QWord
    data: list[Any] | None = None  # always a list: one item for scalars
    expected: list[Any] | None = None  # the requested text as PowerShell converts it for the kind, when it could


async def read_registry_value(key_path: str, value_name: str, kind: str, requested: str) -> RegistryValue:
    """The value as the registry holds it now, read in a new process. Raises when PowerShell itself fails."""
    code, out, err = await run_powershell(
        READ_BACK_SCRIPT,
        values={"key": key_path, "name": value_name, "kind": kind, "value": requested},
        timeout=20,
    )
    if code != 0:
        raise RuntimeError(f"Registry read-back failed: {err.strip() or 'PowerShell exited ' + str(code)}")
    text = out.strip().splitlines()[-1] if out.strip() else ""
    try:
        answer = json.loads(text)
    except ValueError as error:
        raise RuntimeError("Registry read-back gave no readable answer") from error
    if not answer.get("present"):
        return RegistryValue(bool(answer.get("key")), False)

    def as_list(value: Any) -> list[Any] | None:
        if value is None:
            return None
        values = value if isinstance(value, list) else [value]
        return None if values == [None] else values

    return RegistryValue(
        True, True, str(answer.get("kind")), as_list(answer.get("data")), as_list(answer.get("expected"))
    )
