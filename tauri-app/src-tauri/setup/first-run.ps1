# Heliox first-run setup for Windows: the same steps and output as first-run.sh (read that file's header).
# Run by the desktop app (src/setup.rs) and by CI from an empty profile. Never touches the system Python and never
# installs anything the person did not choose.
param([string]$Mode = "run")
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$UvVersion = "0.13.5"
$Dir = $env:HELIOX_SETUP_DIR
if (-not $Dir) { Write-Output "@@step python failed HELIOX_SETUP_DIR is required"; exit 1 }
$Check = "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 11) else 1)"

function Say([string]$line) { Write-Output $line }
function Fail([string]$id, [string]$message, [string]$hint = "") {
  Say "@@step $id failed $message"
  if ($hint) { Say "@@hint $id $hint" }
  exit 1
}

function Get-UvAsset {
  $arch = $env:PROCESSOR_ARCHITECTURE
  if ($arch -eq "ARM64") {
    return @("uv-aarch64-pc-windows-msvc.zip", 15439089, "09f47edf90726084a70f9426f6e3ca4f46e9bf7cbf23b1716ec64d60b055fae3")
  }
  return @("uv-x86_64-pc-windows-msvc.zip", 15264056, "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a")
}

# A Python 3.11 or later with venv and ensurepip. The "python.exe" in WindowsApps is the Store's installer stub:
# with arguments it only prints a message, so the check below rejects it without opening the Store.
function Find-Python {
  $candidates = @()
  if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.13", "3.12", "3.11")) {
      $exe = (& py "-$v" -c "import sys; print(sys.executable)" 2>$null)
      if ($LASTEXITCODE -eq 0 -and $exe) { $candidates += $exe.Trim() }
    }
  }
  foreach ($name in @("python", "python3")) {
    foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
      if ($cmd.Source -notlike "*\WindowsApps\*") { $candidates += $cmd.Source }
    }
  }
  foreach ($exe in $candidates) {
    & $exe -c $Check 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { return $exe }
  }
  return $null
}

$Py = $null
$PythonMode = $env:HELIOX_PYTHON
if (-not $PythonMode) { $PythonMode = "auto" }
if ($PythonMode -eq "auto") { $Py = Find-Python }
$Asset = Get-UvAsset
$Url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$($Asset[0])"

if ($Mode -eq "probe") {
  if ($Py) {
    $version = (& $Py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])").Trim()
    Say "@@python system $version $Py"
  } else {
    Say "@@python download $($Asset[1]) $Url"
  }
  exit 0
}

New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$EnvDir = Join-Path $Dir "env"
$EnvPython = Join-Path $EnvDir "Scripts\python.exe"

# ── Python ──
if (Test-Path $EnvPython) {
  Say "@@step python skipped Heliox's environment already exists."
} elseif ($Py) {
  Say "@@step python running Using $Py"
  & $Py -m venv $EnvDir 2>&1 | ForEach-Object { "$_" }
  if (-not (Test-Path $EnvPython)) { Fail python "Could not create the environment with $Py." "Press Retry." }
  Say "@@step python done Using $((& $EnvPython -c 'import sys; print(sys.version.split()[0])').Trim()) from $Py"
} else {
  $Tools = Join-Path $Dir "tools"
  New-Item -ItemType Directory -Force -Path $Tools | Out-Null
  $Archive = Join-Path $Tools $Asset[0]
  Say "@@step uv running Downloading uv $UvVersion ($($Asset[1]) bytes) from github.com/astral-sh/uv"
  $have = (Test-Path $Archive) -and ((Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLower() -eq $Asset[2])
  if (-not $have) {
    Remove-Item -Force -ErrorAction SilentlyContinue $Archive
    try { Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Archive } catch { }
  }
  if (-not (Test-Path $Archive)) { Fail uv "The download from github.com did not arrive." "Check the internet connection and press Retry." }
  $got = (Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLower()
  if ($got -ne $Asset[2]) {
    Remove-Item -Force $Archive
    Fail uv "The downloaded uv does not match its published checksum ($got)." "Press Retry; if it happens again, report it."
  }
  $UvDir = Join-Path $Tools "uv-$UvVersion"
  Expand-Archive -Force -Path $Archive -DestinationPath $UvDir
  $Uv = Get-ChildItem -Recurse -Path $UvDir -Filter uv.exe | Select-Object -First 1
  if (-not $Uv) { Fail uv "uv was not found in the download." "Press Retry." }
  Say "@@step uv done uv $UvVersion, checksum verified"
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $Dir "python"
  $env:UV_CACHE_DIR = Join-Path $Dir "uv-cache"
  $env:UV_NO_CONFIG = "1"
  Say "@@step python running Downloading a private Python 3.12 for Heliox (python-build-standalone, about 30 MB)"
  & $Uv.FullName python install 3.12 2>&1 | ForEach-Object { "$_" }
  if ($LASTEXITCODE -ne 0) { Fail python "uv could not download Python." "Check the internet connection and press Retry." }
  & $Uv.FullName venv --seed --python 3.12 --python-preference only-managed $EnvDir 2>&1 | ForEach-Object { "$_" }
  if (-not (Test-Path $EnvPython)) { Fail python "Could not create the environment." "Press Retry." }
  Say "@@step python done Private Python $((& $EnvPython -c 'import sys; print(sys.version.split()[0])').Trim()) in $(Join-Path $Dir 'python')"
}

# Its output goes straight to the console so that the function's only value is whether pip succeeded.
function Pip-Install([string]$requirement) {
  & $EnvPython -m pip --disable-pip-version-check install --progress-bar off $requirement 2>&1 |
    ForEach-Object { [Console]::Out.WriteLine("$_") }
  return ($LASTEXITCODE -eq 0)
}

# ── Core ──
Say "@@step core running Installing Heliox's core"
if (-not (Pip-Install $env:HELIOX_CORE_REQ)) {
  Fail core "The core could not be installed." "Check the internet connection and press Retry. The full log is in the setup details."
}
Say "@@step core done Core installed"

# ── Optional groups: a failure is reported and the others carry on ──
$status = 0
foreach ($pair in ("$env:HELIOX_GROUPS" -split ";")) {
  if (-not $pair) { continue }
  $id, $req = $pair -split "=", 2
  Say "@@step $id running Installing"
  if (Pip-Install $req) {
    if ($id -eq "browser") {
      Say "Downloading Chromium for browser automation (playwright.azureedge.net, about 170 MB)"
      & $EnvPython -m playwright install chromium 2>&1 | ForEach-Object { "$_" }
      if ($LASTEXITCODE -eq 0) { Say "@@step $id done Installed, with Chromium" }
      else { Say "@@step $id failed Chromium could not be downloaded."; Say "@@hint $id Press Retry; the rest of Heliox works without it."; $status = 2 }
    } else {
      Say "@@step $id done Installed"
    }
  } else {
    Say "@@step $id failed Could not be installed."
    Say "@@hint $id Press Retry; the rest of Heliox works without it."
    $status = 2
  }
}
exit $status
