# setup.ps1 - Heliox OS Windows Development Environment Setup
#
# This script automates the setup of the Heliox OS development environment.
#
# It installs and configures:
#   - Chocolatey
#   - Python 3.11+
#   - Node.js 20+
#   - Microsoft C++ Build Tools
#   - WebView2 Runtime
#   - Rust (stable-msvc toolchain)
#   - Tauri CLI
#
# USAGE:
#   .\setup.ps1
#
# NOTE:
#   If PowerShell blocks this script from running, please check the
#   "Windows Setup" section inside CONTRIBUTING.md.

$ErrorActionPreference = "Stop"

# Constants

$REQUIRED_PYTHON_MAJOR = 3
$REQUIRED_PYTHON_MINOR = 11
$REQUIRED_PYTHON_MAX_MINOR = 12  # tribev2/torch wheels are not yet available for 3.13+

$REQUIRED_NODE_MAJOR = 20

$CHOCOLATEY_DIR = Join-Path $env:ProgramData 'chocolatey'
$CHOCOLATEY_BIN = Join-Path $CHOCOLATEY_DIR 'bin\choco.exe'

# Logging
function Write-Info($msg)    { Write-Host "[info]  $msg" -ForegroundColor Cyan }
function Write-Success($msg) { Write-Host "[ok]    $msg" -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)     { Write-Host "[error] $msg" -ForegroundColor Red }

# Utilities

function Update-EnvPath {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path    = "$machinePath;$userPath"
}

# Admin Check

function Confirm-AdminPrivileges {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdmin) {
        Write-Err "Administrator privileges required."
        Write-Info "To fix this:"
        Write-Warn "  1. Close this PowerShell window."
        Write-Warn "  2. Right-click PowerShell and select 'Run as Administrator'."
        Write-Warn "  3. Re-run: .\setup.ps1"
        exit 1
    }

    Write-Success "Running as administrator."
}

# Chocolatey

function Install-Chocolatey {
    Write-Info "Installing Chocolatey..."

    Set-ExecutionPolicy Bypass -Scope Process -Force

    # Force TLS 1.2 for compatibility with older Windows versions.
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

    Invoke-RestMethod https://community.chocolatey.org/install.ps1 | Invoke-Expression

    Update-EnvPath

    if (-not (Get-Command choco.exe -ErrorAction SilentlyContinue)) {
        Write-Err "Chocolatey installation failed. 'choco.exe' not found after install."
        exit 1
    }

    Write-Success "Chocolatey installed."
}

function Initialize-Chocolatey {
    # Case 1: Chocolatey is installed and on PATH.
    if (Get-Command choco.exe -ErrorAction SilentlyContinue) {
        Write-Success "Chocolatey is already installed."
        return
    }

    # Case 2: Chocolatey is on disk but missing from PATH.
    if (Test-Path $CHOCOLATEY_BIN) {
        Write-Warn "Chocolatey found on disk but missing from PATH. Repairing..."
        $env:Path = "$(Join-Path $CHOCOLATEY_DIR 'bin');$env:Path"
        Write-Success "Chocolatey added to PATH for this session."
        return
    }

    # Case 3: Chocolatey directory exists but the installation is broken.
    if (Test-Path $CHOCOLATEY_DIR) {
        Write-Warn "Incomplete Chocolatey installation detected. Reinstalling..."
        Remove-Item -LiteralPath $CHOCOLATEY_DIR -Recurse -Force -ErrorAction SilentlyContinue
        Install-Chocolatey
        return
    }

    # Case 4: Chocolatey is not installed at all.
    Install-Chocolatey
}

# Python

function Initialize-Python {
    Write-Host ""
    Write-Info "Checking Python installation..."

    $pyRange     = "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR - $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MAX_MINOR"
    $pyLauncher  = Get-Command py.exe -ErrorAction SilentlyContinue

    if ($pyLauncher) {
        $versions = py -0p 2>$null

        if ($versions) {
            Write-Info "Detected Python installations:"
            $versions | ForEach-Object { Write-Info "  $_" }

            $compatible = $versions | ForEach-Object {
                if ($_ -match '-V:(\d+)\.(\d+)') {
                    [PSCustomObject]@{
                        Major = [int]$Matches[1]
                        Minor = [int]$Matches[2]
                    }
                }
            } | Where-Object {
                $_.Major -eq $REQUIRED_PYTHON_MAJOR -and
                $_.Minor -ge $REQUIRED_PYTHON_MINOR -and
                $_.Minor -le $REQUIRED_PYTHON_MAX_MINOR
            } | Sort-Object Minor -Descending | Select-Object -First 1

            if ($compatible) {
                Write-Success "Python $($compatible.Major).$($compatible.Minor) satisfies the requirement ($pyRange)."
                return
            }

            Write-Warn "No compatible Python version found. Required: $pyRange."
            Write-Warn "Python 3.13+ is not supported; torch/tribev2 wheels are unavailable for it."
        }
        else {
            Write-Warn "Python launcher (py.exe) is present but reports no installed versions."
        }
    }
    else {
        Write-Warn "Python launcher (py.exe) not found."
    }

    Write-Info "Installing Python via Chocolatey..."

    choco install "python$REQUIRED_PYTHON_MAJOR$REQUIRED_PYTHON_MAX_MINOR" -y

    Update-EnvPath
    Write-Success "Python installed."
}

# Python Virtual Environment & Dependencies

function Initialize-PythonVenv {
    Write-Host ""
    Write-Info "Setting up Python virtual environment for the daemon..."

    $daemonDir = Join-Path $PSScriptRoot 'daemon'

    if (-not (Test-Path $daemonDir)) {
        Write-Err "Daemon directory not found at: $daemonDir"
        Write-Err "Please run this script from the root of the Heliox OS repository."
        exit 1
    }

    $venvDir = Join-Path $daemonDir '.env'

    # Create the virtual environment only if it does not already exist, or if it was built with the wrong Python version.
    $needsCreate = $true

    if (Test-Path $venvDir) {
        $venvPython = Join-Path $venvDir 'Scripts\python.exe'
        $venvVersionOutput = & $venvPython --version 2>$null

        if ($venvVersionOutput -match "Python $REQUIRED_PYTHON_MAJOR\.$REQUIRED_PYTHON_MAX_MINOR\.") {
            Write-Warn "Virtual environment already exists at '$venvDir'. Skipping creation."
            $needsCreate = $false
        }
        else {
            Write-Warn "Virtual environment at '$venvDir' uses wrong Python ($venvVersionOutput). Recreating..."
            Remove-Item -LiteralPath $venvDir -Recurse -Force
        }
    }

    if ($needsCreate) {
        Write-Info "Creating virtual environment at '$venvDir'..."

        py "-$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MAX_MINOR" -m venv $venvDir
        Write-Success "Virtual environment created."
    }

    # Resolve the pip executable inside the .env
    $venvPip = Join-Path $venvDir 'Scripts\pip.exe'

    if (-not (Test-Path $venvPip)) {
        Write-Err "pip not found inside the virtual environment. The venv may be corrupted."
        Write-Err "Delete '$venvDir' and re-run setup.ps1 to recreate it."
        exit 1
    }

    # Always (re-)install dependencies so the .env is up to date even when it already existed.
    Write-Info "Installing Python dependencies (this may take a few minutes)..."
    & $venvPython -m pip install --upgrade pip
    & $venvPip install -e "$daemonDir[full,dev]"
    Write-Success "Python dependencies installed."
}

# Node.js

function Initialize-Node {
    Write-Host ""
    Write-Info "Checking Node.js installation..."

    $nodeCmd = Get-Command node.exe -ErrorAction SilentlyContinue

    if ($nodeCmd) {
        $version = node --version 2>$null

        if ($version -match 'v(\d+)\.') {
            $major = [int]$Matches[1]

            if ($major -ge $REQUIRED_NODE_MAJOR) {
                Write-Success "Node.js $version satisfies the requirement (>= v$REQUIRED_NODE_MAJOR)."
                return
            }

            Write-Warn "Node.js $version is below the required version (>= v$REQUIRED_NODE_MAJOR). Upgrading..."
        }
    }
    else {
        Write-Warn "Node.js not found."
    }

    Write-Info "Installing Node.js LTS via Chocolatey..."

    choco install nodejs-lts -y

    Update-EnvPath
    Write-Success "Node.js installed."
}

# npm Packages

function Initialize-NpmPackages {
    Write-Host ""
    Write-Info "Installing npm packages for the Tauri UI..."

    $uiDir = Join-Path $PSScriptRoot 'tauri-app\ui'

    if (-not (Test-Path $uiDir)) {
        Write-Err "UI directory not found at: $uiDir"
        Write-Err "Please run this script from the root of the Heliox OS repository."
        exit 1
    }

    $packageJson = Join-Path $uiDir 'package.json'

    if (-not (Test-Path $packageJson)) {
        Write-Err "No package.json found in '$uiDir'. Cannot install npm packages."
        exit 1
    }

    Write-Info "Running 'npm install' in '$uiDir'..."
    Push-Location $uiDir
    try {
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    Write-Success "npm packages installed."
}

# Microsoft C++ Build Tools

function Initialize-CppBuildTools {
    Write-Host ""
    Write-Info "Checking Microsoft C++ Build Tools..."

    $vsWherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    if (Test-Path $vsWherePath) {
        $hasVcTools = & $vsWherePath `
            -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -nologo `
            2>$null

        if ($hasVcTools) {
            Write-Success "Microsoft C++ Build Tools are installed."
            return
        }

        Write-Warn "Visual Studio is installed but the C++ workload is missing."
    }
    else {
        Write-Warn "Microsoft C++ Build Tools not found."
    }

    Write-Info "Installing Microsoft C++ Build Tools via Chocolatey..."
    Write-Warn "This step may take several minutes."

    choco install visualstudio2022buildtools `
        --package-parameters `
        "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive" `
        -y

    Update-EnvPath
    Write-Success "Microsoft C++ Build Tools installed."
}

# WebView2

function Initialize-WebView2 {
    Write-Host ""
    Write-Info "Checking WebView2 Runtime..."

    $regPaths = @(
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
        'HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    )

    foreach ($path in $regPaths) {
        if (Test-Path $path) {
            Write-Success "WebView2 Runtime is already installed."
            return
        }
    }

    Write-Warn "WebView2 Runtime not found."
    Write-Info "Installing WebView2 Runtime via Chocolatey..."

    choco install webview2-runtime -y

    Update-EnvPath
    Write-Success "WebView2 Runtime installed."
}

# Rust

function Initialize-Rust {
    Write-Host ""
    Write-Info "Checking Rust installation..."

    $cargoCmd = Get-Command cargo.exe -ErrorAction SilentlyContinue
    $rustupCmd = Get-Command rustup.exe -ErrorAction SilentlyContinue

    if ($cargoCmd -and $rustupCmd) {
        $rustVersion = rustc --version 2>$null
        Write-Success "Rust is already installed: $rustVersion"
        Write-Info "Ensuring the stable-msvc toolchain is active..."
        rustup default stable-msvc
        Write-Success "Rust toolchain set to stable-msvc."
        return
    }

    Write-Warn "Rust not found."
    Write-Info "Installing Rust via Chocolatey..."

    choco install rustup.install -y

    Update-EnvPath

    Write-Info "Configuring Rust toolchain to stable-msvc..."

    rustup default stable-msvc

    Write-Success "Rust installed and configured."
}

# Tauri CLI

function Initialize-TauriCli {
    Write-Host ""
    Write-Info "Checking Tauri CLI installation..."

    try {
        $tauriVersion = cargo tauri --version 2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Tauri CLI is already installed: $tauriVersion"
            return
        }
    }
    catch {
        # cargo tauri --version throws when tauri-cli is not installed; fall through to install.
    }

    Write-Warn "Tauri CLI not found."
    Write-Info "Installing Tauri CLI via Cargo (compiling from source)..."
    Write-Warn "This step may take several minutes."

    cargo install tauri-cli --version "^2" --locked

    Write-Success "Tauri CLI installed."
}

# Main

Confirm-AdminPrivileges
Initialize-Chocolatey

Write-Info "Starting Heliox OS development environment setup..."

Initialize-Python
Initialize-PythonVenv
Initialize-Node
Initialize-NpmPackages
Initialize-CppBuildTools
Initialize-WebView2
Initialize-Rust
Initialize-TauriCli

Write-Host ""
Write-Success "All dependencies installed. Environment ready."
