# Contributing to Heliox OS

Thanks for your interest in contributing to Heliox OS! This guide will help you get started.

## 🏗️ Architecture Overview

```
heliox-os/
├── daemon/                  # Python backend (AI agent system)
│   └── pilot/
│       ├── agents/          # Planner, Executor, Verifier, and specialist mesh
│       ├── models/          # LLM routing (Gemini, OpenAI, Claude, Ollama)
│       ├── intelligence/    # Ledger, learning, replay, strategy, and evolution
│       ├── memory/          # Temporal facts and bounded context assembly
│       ├── neural/          # Acquisition, calibrated decoding, intent gate, recording
│       ├── plugins/         # Capability validation and native/WASM brokers
│       ├── security/        # Permission, gateway, audit, risk, and plugin policy
│       ├── workflows/       # Durable tasks and voice/gesture workflows
│       └── system/          # OS interfaces behind the 158-action catalog
├── tauri-app/               # Desktop GUI
│   ├── ui/                  # Svelte 5 + Vite frontend
│   │   └── src/lib/
│   │       ├── components/  # UI components (VoiceControl, GestureControl, etc.)
│   │       └── stores/      # Svelte stores (session, settings)
│   └── src-tauri/           # Rust backend (Tauri v2)
└── schemas/                 # Shared JSON schemas for action validation
```

### How it works

1. **Input and ledger** → Text, voice, gesture, gaze, and screen context become typed causal events
2. **Context and planning** → Bounded temporal memory informs a structured action plan
3. **Prediction and policy** → The hybrid world model may add caution; deterministic policy and approval stay authoritative
4. **Durable orchestration** → The task journal and capability mesh route actions across 21 specialists
5. **Execution and verification** → Guarded adapters execute, then real environment state is checked
6. **Learning and recovery** → Verified outcomes feed replay, bounded online adaptation, reflection, and explicit recovery

Read [Architecture](docs/ARCHITECTURE.md) before changing execution,
intelligence, security, plugins, or agent routing.

## 🚀 Dev Environment Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **Rust toolchain** (for Tauri)
- **Git**

## Windows Setup

For Windows users, after cloning the repository you can use the provided `setup.ps1` script to automatically set up the development environment.

### Run the setup script

Open PowerShell as administrator in the project root directory and run:

```powershell
.\setup.ps1
```

### If PowerShell blocks the script

Some Windows systems restrict PowerShell scripts by default.  
If you see an execution policy error, run the following command in PowerShell as administrator:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

After running the command:

1. Close the terminal
2. Reopen PowerShell
3. Navigate back to the project directory
4. Run the setup script again:

```powershell
.\setup.ps1
```

> Note: The `setup.ps1` script is intended for Windows users only.

### 1. Clone the repo

```bash
git clone https://github.com/VyomKulshrestha/Heliox-OS.git
cd Heliox-OS
```

### 2. Set up the Python daemon

```bash
cd daemon
pip install -e ".[all,dev]"
```

### 3. Set up the frontend

```bash
cd tauri-app/ui
npm ci
```

### 4. Run in development mode

**Terminal 1 — Python daemon:**
```bash
cd daemon
python -m pilot.server
```

**Terminal 2 — Svelte frontend:**
```bash
cd tauri-app/ui
npm run dev
```

The app will be available at `http://localhost:1420`.

### 5. (Optional) Run the full Tauri desktop app

```bash
cd tauri-app
npm ci
npm run tauri dev
```

## 📝 Code Style

### Python (daemon/)
- Formatter: **Ruff** (`ruff format .`)
- Linter: **Ruff** (`ruff check .`)
- Type hints are encouraged for all public functions

### Svelte/TypeScript (tauri-app/ui/)
- Formatter: **Prettier** (`npx prettier --write "src/**/*.{svelte,ts,js,css}" "vite.config.ts"`)
- Type and component check: **svelte-check** (`npm run check`)
- Tests: **Vitest** (`npm run test:unit -- --run`) and **Playwright** (`npm run test:visual`)
- Use Svelte 5 runes (`$state`, `$derived`, `$effect`)

### Svelte Component Naming Conventions
- Use PascalCase for all Svelte component filenames.
  Example: `VoiceControl.svelte`, `SettingsModal.svelte`
- Component names should clearly describe their functionality and purpose.
- Avoid generic or temporary names such as `Component.svelte`, `Temp.svelte`, or `Test.svelte`.
- Keep one primary component per file.
- Prefer reusable and modular component design whenever possible.
- Follow existing naming patterns already used in the project for consistency.

### Frontend Folder Organization
- Store reusable UI components inside `src/lib/components/`.
- Store shared Svelte stores inside `src/lib/stores/`.
- Group related components into subfolders when a feature grows in complexity.
- Keep component-specific helper utilities close to their related feature/module.
- Avoid deeply nested folder structures unless necessary.

### CSS & Styling Standards
- Prefer scoped styles inside Svelte components whenever possible.
- Maintain consistent spacing, typography, and layout patterns across the UI.
- Use meaningful and readable class names.
- Avoid excessive CSS nesting and overly complex selectors.
- Minimize inline styles unless required for dynamic behavior.
- Reuse existing design patterns and utility styles before creating new ones.
- Ensure responsive layouts and proper alignment across different screen sizes.

### Reusable UI Components
- Design components to be modular and reusable.
- Use props for configurable behavior instead of duplicating components.
- Keep components focused on a single responsibility.
- Avoid tightly coupling components with unrelated business logic.
- Maintain consistent behavior and styling across reusable UI elements.
- Follow Svelte 5 conventions and use runes consistently where applicable.

### Rust (tauri-app/src-tauri/)
- Formatter: **rustfmt** (`cargo fmt`)
- Linter: **Clippy** (`cargo clippy`)

## 🤝 GSSoC '26 Guidelines & Issue Assignment

To ensure a fair and organized environment for all GirlScript Summer of Code (GSSoC) 2026 contributors, we strictly enforce the following rules:

1. **One Active Issue Per Person**: You may only be assigned to **one** issue at a time. 
2. **Complete Before Requesting**: You **cannot** request to be assigned to a second issue until you have submitted a Pull Request for your currently assigned issue.
3. **No Spamming**: Do not spam "please assign me" on every open issue. Contributors who spam issue threads or attempt to hoard issues will be reported to GSSoC management and banned from the repository.

## 🔀 Submitting a Pull Request

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feat/my-feature`
3. **Make your changes** and ensure they pass linting
4. **Test your changes** locally
5. **Commit** with a descriptive message:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `ci:` for CI/CD changes
   - `refactor:` for code refactoring
6. **Push** and open a Pull Request against `main`

### PR Checklist

- [ ] Code follows the style guidelines
- [ ] Self-reviewed the code
- [ ] Added comments for complex logic
- [ ] No API keys or secrets committed
- [ ] Tested on at least one OS (Windows/macOS/Linux)

## 🐛 Reporting Bugs

Please use the [Bug Report template](https://github.com/VyomKulshrestha/Heliox-OS/issues/new?template=bug_report.md) and include:

- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Console logs / error messages

## 💡 Feature Requests

Use the [Feature Request template](https://github.com/VyomKulshrestha/Heliox-OS/issues/new?template=feature_request.md).

## 🔌 Writing Plugins

Heliox supports signed local plugins and a reviewed public marketplace. Start
with the [Plugin Marketplace guide](docs/PLUGIN_MARKETPLACE.md). Marketplace
packages live under `plugins/<plugin-name>`, declare an explicit capability
manifest, and must pass:

```bash
python scripts/validate_marketplace.py --write
python scripts/validate_marketplace.py
```

Native Python plugins run in a constrained child broker; WASM plugins run
through the WASI broker. Do not bypass these paths or add undeclared authority.

## Full validation

Run the gates relevant to your change before opening a pull request:

```bash
cd daemon
python -m ruff check pilot tests
python -m ruff format --check pilot tests
python -m pytest

cd ../tauri-app/ui
npx prettier --check "src/**/*.{svelte,ts,js,css}" "vite.config.ts"
npm run check
npm run test:static
npm run test:unit -- --run
npm run build
npm audit --audit-level=high
npm run test:visual

cd ../src-tauri
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

If a change touches `.github/workflows/`, run `actionlint` as well. Changes to
browser or desktop execution must test missing and ambiguous targets,
no-progress termination, real post-action verification, and the relevant
platform adapters. Do not make completion depend only on a command returning
without an exception.

Changes to execution, intelligence, security, plugins, or routing must also
update or assess README, [Architecture](docs/ARCHITECTURE.md),
[Security](SECURITY.md), and [IPC](IPC_MESSAGE_FORMATS.md).

Neural changes must additionally preserve role separation, strict/replay-safe
contracts, bounded buffers, abstention, explicit non-neural arming, preview and
cancellation, the fixed Tier 0/1 goal registry, raw-data consent/retention, and
physical-control denial. Run the neural protocol/acquisition/decoder/gate/
controller/recording tests and the paired UI/sidecar end-to-end test. Synthetic
evidence must never be described as live-headset or human accuracy evidence;
record physical hardware results separately. See
[Neural Intent Research Controls](docs/NEURAL_INTENT.md).

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
