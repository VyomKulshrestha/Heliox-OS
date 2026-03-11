# Contributing to Heliox OS

Thanks for your interest in contributing to Heliox OS! This guide will help you get started.

## 🏗️ Architecture Overview

```
heliox-os/
├── daemon/                  # Python backend (AI agent system)
│   └── pilot/
│       ├── agents/          # Planner, Executor, Verifier, Code Sanitizer
│       ├── models/          # LLM routing (Gemini, OpenAI, Claude, Ollama)
│       ├── security/        # Encrypted vault, permissions, audit log
│       └── system/          # OS interface modules (50+ action types)
├── tauri-app/               # Desktop GUI
│   ├── ui/                  # Svelte 5 + Vite frontend
│   │   └── src/lib/
│   │       ├── components/  # UI components (VoiceControl, GestureControl, etc.)
│   │       └── stores/      # Svelte stores (session, settings)
│   └── src-tauri/           # Rust backend (Tauri v2)
└── schemas/                 # Shared JSON schemas for action validation
```

### How it works

1. **User Input** → Voice, Text, or Hand Gesture
2. **Planner** (LLM) → Converts input to a structured JSON action plan
3. **Security Gate** → Validates actions against permission tiers
4. **Executor** → Runs each action via native OS APIs
5. **Verifier** → Confirms the action succeeded
6. **Auto-Fix** → If code fails, the LLM rewrites and retries

## 🚀 Dev Environment Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **Rust toolchain** (for Tauri)
- **Git**

### 1. Clone the repo

```bash
git clone https://github.com/VyomKulshrestha/Heliox-OS.git
cd Heliox OS
```

### 2. Set up the Python daemon

```bash
cd daemon
pip install -e ".[full,dev]"
```

### 3. Set up the frontend

```bash
cd tauri-app/ui
npm install
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
cd tauri-app/src-tauri
cargo tauri dev
```

## 📝 Code Style

### Python (daemon/)
- Formatter: **Ruff** (`ruff format .`)
- Linter: **Ruff** (`ruff check .`)
- Type hints are encouraged for all public functions

### Svelte/TypeScript (tauri-app/ui/)
- Formatter: **Prettier** (`npx prettier --write .`)
- Linter: **ESLint** (`npx eslint .`)
- Use Svelte 5 runes (`$state`, `$derived`, `$effect`)

### Rust (tauri-app/src-tauri/)
- Formatter: **rustfmt** (`cargo fmt`)
- Linter: **Clippy** (`cargo clippy`)

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

Heliox OS supports community plugins! See `daemon/pilot/system/plugins.py` for the plugin API. Plugins are Python modules placed in `~/.config/heliox-os/plugins/`.

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
