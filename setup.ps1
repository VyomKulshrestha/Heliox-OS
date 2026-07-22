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
$REQUIRED_PYTHON_MAX_MINOR = 12  # torch wheels are not yet available for 3.13+
$script:PYTHON_VERSION = "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MAX_MINOR"
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

function Get-CompatiblePythonVersion {
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if (-not $pyLauncher) {
        return $null
    }

    $versions = py -0p 2>$null
    if (-not $versions) {
        return $null
    }

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
        return "$($compatible.Major).$($compatible.Minor)"
    }

    return $null
}

function Initialize-Python {
    Write-Host ""
    Write-Info "Checking Python installation..."

    $pyRange = "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR - $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MAX_MINOR"
    $foundVersion = Get-CompatiblePythonVersion

    if ($foundVersion) {
        $script:PYTHON_VERSION = $foundVersion
        Write-Success "Python $PYTHON_VERSION satisfies the requirement ($pyRange)."
        return
    }

    Write-Warn "No compatible Python version found. Required: $pyRange."
    Write-Warn "Python 3.13+ is not supported; torch wheels are unavailable for it."

    Write-Info "Installing Python via Chocolatey..."
    choco install "python$REQUIRED_PYTHON_MAJOR$REQUIRED_PYTHON_MAX_MINOR" -y
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python. choco exited with code $LASTEXITCODE"
    }

    Update-EnvPath
    $script:PYTHON_VERSION = "$REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MAX_MINOR"
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

    $venvDir = Join-Path $daemonDir '.venv'
    $venvPython = $null
    $needsCreate = $true

    if (Test-Path $venvDir) {
        $venvPython = Join-Path $venvDir 'Scripts\python.exe'
        $venvVersionOutput = & $venvPython --version 2>$null

        if ($venvVersionOutput -match 'Python (\d+)\.(\d+)') {
            $venvMajor = [int]$Matches[1]
            $venvMinor = [int]$Matches[2]

            if ($venvMajor -eq $REQUIRED_PYTHON_MAJOR -and
                $venvMinor -ge $REQUIRED_PYTHON_MINOR -and
                $venvMinor -le $REQUIRED_PYTHON_MAX_MINOR) {
                Write-Warn "Virtual environment already exists at '$venvDir' with compatible Python $venvMajor.$venvMinor. Skipping creation."
                $needsCreate = $false
            }
            else {
                Write-Warn "Virtual environment at '$venvDir' uses wrong Python ($venvVersionOutput). Recreating..."
                Remove-Item -LiteralPath $venvDir -Recurse -Force
                $venvPython = $null
            }
        }
        else {
            Write-Warn "Could not determine the Python version in existing virtual environment. Recreating..."
            Remove-Item -LiteralPath $venvDir -Recurse -Force
            $venvPython = $null
        }
    }

    if ($needsCreate) {
        Write-Info "Creating virtual environment at '$venvDir'..."
        py "-$PYTHON_VERSION" -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment. py.exe exited with code $LASTEXITCODE"
        }
        Write-Success "Virtual environment created."
        $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    }

    if (-not $venvPython) {
        $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    }

    $venvPip = Join-Path $venvDir 'Scripts\pip.exe'

    if (-not (Test-Path $venvPip)) {
        Write-Err "pip not found inside the virtual environment. The venv may be corrupted."
        Write-Err "Delete '$venvDir' and re-run setup.ps1 to recreate it."
        exit 1
    }

    Write-Info "Installing Python dependencies (this may take a few minutes)..."
    Push-Location $daemonDir
    try {
        & $venvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to upgrade pip. pip exited with code $LASTEXITCODE"
        }
        & $venvPython -m pip install -e ".[full,dev]"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install daemon dependencies. pip exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Node.js. choco exited with code $LASTEXITCODE"
    }

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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Microsoft C++ Build Tools. choco exited with code $LASTEXITCODE"
    }

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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install WebView2 Runtime. choco exited with code $LASTEXITCODE"
    }

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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Rust. choco exited with code $LASTEXITCODE"
    }

    Update-EnvPath

    Write-Info "Configuring Rust toolchain to stable-msvc..."

    rustup default stable-msvc
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to configure Rust toolchain. rustup exited with code $LASTEXITCODE"
    }

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
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Tauri CLI. cargo exited with code $LASTEXITCODE"
    }

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
