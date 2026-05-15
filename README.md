# Heliox OS — AI System Control Agent

<p align="center">
  <a href="https://gssoc.girlscript.org/"><img src="https://img.shields.io/badge/GSSoC-2026-F96F59?style=for-the-badge" alt="GSSoC 2026"></a>
  <a href="https://github.com/VyomKulshrestha/Heliox-OS/releases"><img src="https://img.shields.io/github/v/release/VyomKulshrestha/Heliox-OS?style=for-the-badge&color=00f0ff&label=Release" alt="Release"></a>
  <a href="https://github.com/VyomKulshrestha/Heliox-OS/releases"><img src="https://img.shields.io/github/downloads/VyomKulshrestha/Heliox-OS/total?style=for-the-badge&color=7c6fe0&label=Downloads" alt="Downloads"></a>
  <a href="https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/VyomKulshrestha/Heliox-OS/release.yml?style=for-the-badge&label=Build" alt="Build Status"></a>
  <a href="https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/VyomKulshrestha/Heliox-OS/ci.yml?style=for-the-badge&label=CI&color=44cc11" alt="CI"></a>
  <a href="https://github.com/VyomKulshrestha/Heliox-OS/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22"><img src="https://img.shields.io/github/issues/VyomKulshrestha/Heliox-OS/good%20first%20issue?style=for-the-badge&color=purple&label=Good%20First%20Issues" alt="Good First Issues"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/VyomKulshrestha/Heliox-OS?style=for-the-badge&color=blue" alt="License"></a>
  <img src="https://img.shields.io/badge/Platform-Windows%20|%20macOS%20|%20Linux-333?style=for-the-badge" alt="Platform">
</p>

<p align="center">
  <!-- Replace with your actual demo GIF path once recorded -->
  <img src="https://raw.githubusercontent.com/VyomKulshrestha/Heliox-OS/main/assets/demo.gif" alt="Heliox OS Jarvis Demo" width="800">
</p>


<p align="center">
  <strong>Control your entire computer with natural language, voice, and hand gestures.</strong><br>
  An open-source, privacy-first AI agent that plans, executes, and verifies complex multi-step tasks.
</p>

<p align="center">
  🌐 <b><a href="https://helioxos.dev">Visit the Official Website (helioxos.dev)</a></b> 🌐
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#-jarvis-mode-new">JARVIS Mode</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#security">Security</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## Why Heliox OS?

Unlike simple command runners, Heliox OS is a **true agentic system** inspired by robust autonomous architectures like OpenClaw, running a continuous ReAct loop with a **modular multi-agent orchestrator**:

1. **Gateway Hub & Memory** — LLM evaluates persistent memory context before reasoning.
2. **Planner** — Converts natural language into a structured multi-step action plan.
3. **Agent Orchestrator** — Routes each action to the correct specialist agent (System, Code, Web, Monitor, Communication).
4. **Specialist Agents** — Five domain experts execute actions via native OS APIs (never flimsy GUI automation).
5. **Verifier** — Post-execution verification confirms the action succeeded and feeds results back into the loop.
6. **Reflector** — Self-improvement engine learns from successes and failures.
7. **Security** — Five-tier permission system with confirmation gates and rollback support.

## 🤖 JARVIS Autonomy (v0.7.1)

Heliox OS has achieved true proactive autonomy, transitioning from a reactive assistant to an invisible, always-on background intelligence system natively integrated into your OS:

- 🧠 **Proactive Suggestion Engine**: Learns your daily workflows and pattern-matches your screen context to silently surface UI action suggestions (e.g., offering to summarize a long thread or launch an IDE when browsing issues) *before* you ask.
- ⚡ **Fire-and-Forget Autonomous Jobs**: Spawn complex multi-step background tasks that decompose, execute, and verify completely independent of the UI or main event loop.
- 👁️ **Always-On Screen Awareness**: Automatically bootstrapped computer vision that tracks your contextual state cross-platform, natively bridging exactly what you see into the LLM planner. 
- 🎤 **Continuous Voice Listener**: Real-time push-free 'Hey Heliox' ambient wake-word dispatch for frictionless task execution.
- 🤚 **30+ Hand Gestures & Air Drawing**: Control your PC via webcam with static poses (Palm, Pinch) and motion gestures (Two-Finger Swipe).
- 🌀 **Arc Reactor UI & Ambient HUD**: Animated, immersive Tauri overlays responding contextually to system actions.

## 🧠 TRIBE v2 Cognitive Engine Integrations (v0.7.1)

Heliox OS is now fully neuro-adaptive, integrating Meta's **TRIBE v2 Cognitive Engine** directly into the operating logic:

1. **Neural Cognitive HUD:** Tracks real-time *Saliency* and *Brain Load* via screen vision buffer, mapped onto the Svelte UI.
2. **Dynamic TTS Stress-Pacing:** JARVIS automatically slows down voice generation if you are engaged in high cognitive-load tasks.
3. **Neuro-Safe Destructive Gate:** High-risk actions (e.g., recursive deletes) evaluate cognitive stress first. If you are distracted, a strict 10-second auditory confirmation gate holds the process.
4. **Subconscious Persona Fingerprint:** The Subconscious background loop learns your neural visual plasticity constraints and encodes them to `persona.md` to align UIs to your brain.
5. **Attention-Optimized Notifications:** The notification pipeline dynamically buffers trivial background alerts until the system detects a "cortical transition" (a low-load resting state).
6. **ReAct Neural Cost Estimator:** Tasks predict aggregate cognitive demand proactively logic. If executing a plan risks exceeding mental bandwidth, JARVIS pauses.
7. **JARVIS Intent Classifier:** Fully integrated native intent fusion classifying spoken commands against current workload intensity.

## 🚀 Revolutionary TRIBE v2 Cognitive Features (v0.6.0)

Heliox OS now pushes the boundaries with **7 revolutionary biologically-inspired AI features** powered by Meta's **TRIBE v2** neural model:

### 1. Adaptive Biometric Learning Loop
Track user's physiological patterns over weeks (time-of-day productivity, stress cycles). Create personalized "cognitive fingerprints" that predict optimal interaction times. Implements a closed-loop feedback system where user responses refine the TRIBE predictions.

```python
from pilot.cognitive.biometric_loop import BiometricLearningLoop

loop = BiometricLearningLoop(user_id="user")
loop.record_cognitive_sample(attention=0.7, stress=0.3, load=0.5)
recommendation = loop.get_interaction_recommendation()
print(recommendation.recommended, recommendation.interaction_type)
```

### 2. Ambient Intelligence Mode
Instead of reactive commands, Heliox proactively suggests actions based on predicted cognitive state. Example: *"You've been on this task for 2 hours with increasing stress — want me to schedule a break?"*

```python
from pilot.cognitive.ambient_intelligence import AmbientIntelligenceEngine

ambient = AmbientIntelligenceEngine(biometric_loop)
await ambient.update_cognitive_state(attention=0.8, stress=0.6, load=0.7)
# Automatically generates proactive suggestions based on trends
```

### 3. Multi-Modal Neural Bridge
Extend beyond screen vision — integrate webcam eye-tracking, audio tone analysis, keyboard/mouse dynamics. Build a unified "neural workspace" that predicts what the user will need before they ask.

```python
from pilot.cognitive.neural_bridge import NeuralBridge

bridge = NeuralBridge()
bridge.record_keystroke()
bridge.record_mouse_move(x, y)
workspace = bridge.compute_workspace()
print(workspace.cognitive_state, workspace.predicted_need)
```

### 4. Cognitive Offloading
When load > 80%, automatically surface "memory anchors" — key info from recent actions. Let Heliox absorb cognitive burden by remembering complex multi-step workflows.

```python
from pilot.cognitive.cognitive_offload import CognitiveOffloader

offloader = CognitiveOffloader()
offloader.update_load(0.85)  # Triggers offload surface
surface = offloader.get_offload_surface()
print(surface["anchors"], surface["workflows"])
```

### 5. Evolving Persona Architecture
Move from static persona.md to a living "neural avatar" that changes daily based on cognitive patterns. The AI's communication style adapts: concise when stressed, detailed when relaxed.

```python
from pilot.cognitive.evolving_persona import EvolvingPersonaEngine

persona = EvolvingPersonaEngine(user_id="user")
persona.record_interaction(attention=0.7, stress=0.3, load=0.5)
greeting = persona.get_greeting()  # "Good afternoon! Ready to tackle anything?"
print(persona.get_ui_config())  # Adapts UI based on state
```

### 6. Cross-Device Cognitive Handoff
If TRIBE detects high load on desktop, suggest continuing on mobile with context transfer. Build a "cognitive state cloud" that follows the user across devices.

```python
from pilot.cognitive.cognitive_handoff import CognitiveHandoffEngine

handoff = CognitiveHandoffEngine(device_name="desktop")
handoff.capture_snapshot(attention=0.8, stress=0.6, load=0.7)
handoff.sync_to_cloud()
suggestion = handoff.get_handoff_suggestion(load=0.85, stress=0.3)
```

### 7. Quantum-Ready Architecture
Design the cognitive pipeline to be model-agnostic — swap TRIBE for future neural models. Create standard cognitive APIs that other developers can build on.

```python
from pilot.cognitive.quantum_cognitive import create_pipeline, QuantumCognitivePipeline

pipeline = create_pipeline()
output = await pipeline.predict("user working on complex task")
print(output.attention_score, output.stress_level, output.cognitive_load)

# Switch models at runtime
pipeline.set_active_model("gpt_neuro")  # Future model support
```

### Unified Cognitive Hub

All features are wrapped in a single interface:

```python
from pilot.cognitive.hub import CognitiveHub

hub = CognitiveHub()
state = await hub.analyze("user is coding")

# All features accessible
print(f"Attention: {state.attention}, Stress: {state.stress}")
print(f"Optimal: {state.optimal_interaction}")
print(f"Overloaded: {state.is_overloaded}")
print(hub.get_greeting())  # Adaptive greeting
print(hub.get_offload_surface())  # Memory anchors
```

### TRIBE-Powered

The **QuantumCognitivePipeline** automatically uses TRIBE v2 when available:

```
Available Models:
  - Meta TRIBE v2 (tribe_v2)
    Available: True
    Capabilities: ATTENTION_PREDICTION, STRESS_DETECTION, LOAD_ESTIMATION
    Avg Latency: ~50ms
```

When TRIBE is unavailable, it gracefully falls back to heuristic models.

---

## 🧠 Multi-Agent Orchestrator

Heliox OS uses a **modular multi-agent architecture** where specialized agents collaborate to solve complex tasks:

| Agent | Domain | Key Skills |
|-------|--------|------------|
| 🖥️ **System Agent** | OS Operations | Files, processes, services, power, input control, screen vision, triggers |
| 💻 **Code Agent** | Development | Code generation, execution, debugging, dev tooling (git, pip, npm) |
| 🌐 **Web Agent** | Web & APIs | Browser automation, scraping, HTTP requests, downloads |
| 📊 **Monitor Agent** | Monitoring | CPU, RAM, disk, network monitoring with threshold alerts |
| 📡 **Communication Agent** | Messaging | Email, Slack, Discord, webhooks, desktop notifications |

**How it works:** The Planner generates an action plan → the Orchestrator analyzes each action type → routes to the correct specialist → agents execute in sequence → results merge for verification.

**Dynamic Spawning:** Agents can be created on-demand at runtime via the `agent_spawn` API endpoint.

## 🧪 Tested With 10 Complex Tasks — 80%+ Pass Rate

| Task | Type | Status |
|------|------|--------|
| Web scrape Wikipedia + word frequency analysis | Web + Code | ✅ |
| Background CPU trigger with voice alert | System Monitor | ✅ |
| Screenshot OCR + text reversal + file tree | Vision + Code | ✅ |
| Multi-page web comparison (Python vs JS) | Web + Analysis | ✅ |
| Create project scaffold + run unit tests | File + Code | ✅ |
| REST API fetch + JSON parse + formatted table | API + Code | ✅ |
| CSV data pipeline + financial analysis | Data + Code | ✅ |
| And more... | | ✅ |

## 🖥️ Cross-Platform Support

| Platform | Status |
|----------|--------|
| Windows 10/11 | ✅ Full support |
| Ubuntu / Debian | ✅ Full support |
| macOS | ✅ Full support |
| Fedora / Arch | ✅ Via dnf/pacman |

## ⚡ 50+ Action Types

### File Operations
`file_read` · `file_write` · `file_delete` · `file_move` · `file_copy` · `file_list` · `file_search` · `directory_summary` · `file_permissions`

### Process Management
`process_list` · `process_kill` · `process_info`

### Shell Execution
`shell_command` · `shell_script` (multi-line bash/powershell/python)

### Code Execution
`code_execute` — Run Python, PowerShell, Bash, or JavaScript with auto-fix on failure

### Browser & Web
`browser_navigate` · `browser_extract` · `browser_extract_table` · `browser_extract_links`

### Screen & Vision
`screenshot` · `screen_ocr` · `screen_analyze`

### Package Management
`package_install` · `package_remove` · `package_update` · `package_search`
Auto-detects: winget, choco, brew, apt, dnf, pacman

### System Information
`system_info` · `cpu_usage` · `memory_usage` · `disk_usage` · `network_info` · `battery_info`

### Window Management
`window_list` · `window_focus` · `window_close` · `window_minimize` · `window_maximize`

### Audio / Volume
`volume_get` · `volume_set` · `volume_mute`

### Display / Screen
`brightness_get` · `brightness_set` · `screenshot`

### Power Management
`power_shutdown` · `power_restart` · `power_sleep` · `power_lock` · `power_logout`

### Network / WiFi
`wifi_list` · `wifi_connect` · `wifi_disconnect`

### Clipboard
`clipboard_read` · `clipboard_write`

### Scheduled Tasks & Triggers
`schedule_create` · `schedule_list` · `schedule_delete` · `trigger_create`

### Environment Variables
`env_get` · `env_set` · `env_list`

### Downloads
`download_file`

### Service Management (Linux)
`service_start` · `service_stop` · `service_restart` · `service_enable` · `service_disable` · `service_status`

### GNOME / Desktop (Linux)
`gnome_setting_read` · `gnome_setting_write` · `dbus_call`

### Windows Registry
`registry_read` · `registry_write`

### Open / Launch / Notify
`open_url` · `open_application` · `notify`

## Architecture

```mermaid
graph TD
    User(["User Input: Voice, Text, Gestures"]) --> Gateway

    subgraph "Frontend Gateway - Tauri + Svelte"
        Gateway["WebSocket Hub"]
        GUI["Desktop Window"]
        HUD["Ambient System HUD"]
        VC["Voice Controller"]
        GC["Hand Gesture Controller"]
        RPV["ReAct Pipeline Visualizer"]
        TVS["Thought Visualization 🧠"]
        Gateway --- GUI
        GUI --- HUD
        GUI --- VC
        GUI --- GC
        GUI --- RPV
        RPV --- TVS
    end

    Gateway --> Fusion

    subgraph "Multimodal Fusion"
        Fusion["Intent Fusion Engine"]
        Fusion --> |"voice + gesture"| FusedIntent["Fused Intent"]
    end

    FusedIntent --> Daemon

    subgraph "Agent Runtime - Python"
        Daemon["Agent Server / ReAct Loop"] --> Memory[("Long-term Memory")]
        Memory --> |"Vector + Semantic"| ChromaDB[("ChromaDB")]
        Daemon --> Decomposer["Task Decomposer"]
        Decomposer --> Planner["LLM Planner"]
        Planner --> PromptImprover["Prompt Improver"]
        PromptImprover --> |"reuse strategies"| Planner
        Planner --> Router{"Model Router"}
        Router --> Ext_LLM("Gemini / OpenAI / Claude")
        Router --> Int_LLM("Ollama")
        Planner --> Sandbox["Simulation Sandbox"]
        Sandbox --> |"risk report"| Security["Security Gate"]
        Security --> Orchestrator["Agent Orchestrator"]
    end

    subgraph "Multi-Agent System"
        Orchestrator --> SA["System Agent"]
        Orchestrator --> CA["Code Agent"]
        Orchestrator --> WA["Web Agent"]
        Orchestrator --> MA["Monitor Agent"]
        Orchestrator --> COMM["Communication Agent"]
    end

    subgraph "Plugin Ecosystem"
        PluginReg["Plugin Registry"]
        PluginReg -->|"tools"| Orchestrator
        P1["developer-tools"]
        P2["media-control"]
        P3["home-assistant"]
        P1 --- PluginReg
        P2 --- PluginReg
        P3 --- PluginReg
    end

    SA --> Executor["System Executor"]
    CA --> Executor
    WA --> Executor
    COMM --> Executor
    MA --> BG["Background Tasks"]

    Executor --> Verifier["Verifier"]
    BG --> Verifier
    Verifier --> Reflector["Reflector"]
    Reflector --> PromptImprover
    Reflector --> Memory
    Reflector --> SkillReg["Skill Registry"]

    subgraph "Subconscious Layer"
        SubAgent["Subconscious Agent"]
        SubAgent -->|"persona rules"| Planner
        Reflector --> SubAgent
        SubAgent --> PersonaFile["~/.heliox/persona.md"]
    end

    subgraph "Screen Awareness"
        ScreenVision["Screen Vision Agent"]
        ScreenVision -->|"context"| Planner
        ScreenVision --> AppDetect["Active App Detector"]
        ScreenVision --> DiffEngine["Screenshot Diff"]
    end


    subgraph "Reasoning Telemetry"
        Daemon -.-> |"events"| ReasoningEmitter["Reasoning Emitter"]
        ReasoningEmitter -.-> |"WebSocket"| TVS
    end
```

## 🧠 Research-Level AI Architecture

Heliox OS implements **13 research-level features** that push beyond typical AI agents:

| # | Feature | Status | Module |
|---|---------|--------|--------|
| 1 | Persistent Long-Term Memory (Vector + Semantic) | ✅ | `memory/store.py` + ChromaDB |
| 2 | Self-Reflection Loop | ✅ | `agents/reflector.py` |
| 3 | Tool Discovery / Skill Registry | ✅ | `agents/reflector.py` (skill_registry table) |
| 4 | Task Decomposition Engine | ✅ | `agents/decomposer.py` |
| 5 | Autonomous Background Agents | ✅ | `agents/background.py` + `monitor_agent.py` |
| 6 | Multi-Agent Collaboration | ✅ | `agents/orchestrator.py` (5 specialists) |
| 7 | Real-Time Reasoning Visualization | ✅ | `reasoning/events.py` + `ReActPipeline.svelte` |
| 8 | Simulation Sandbox | ✅ | `agents/sandbox.py` |
| 9 | Self-Improving Prompt System | ✅ | `agents/prompt_improver.py` |
| 10 | Plugin Ecosystem | ✅ | `plugins/__init__.py` |
| 11 | Flagship Plugins (Developer, Media, IoT) | ✅ | `plugins/developer/`, `plugins/media/`, `plugins/homeassistant/` |
| 12 | Subconscious Agent (Persona Learning) | ✅ | `agents/subconscious.py` |
| 13 | Screen Vision (Continuous Screen Awareness) | ✅ | `agents/screen_vision.py` |

### 🔧 Task Decomposition Engine

Complex goals are automatically broken into dependency-aware subtask trees:

```
User: "Build a Flask API for todo list"
 → 1. [system] Create project folder
 → 2. [system] Install Flask            (depends: 1)
 → 3. [code]   Generate API code         (depends: 1)
 → 4. [code]   Create requirements.txt   (depends: 2)
 → 5. [code]   Run tests                 (depends: 3, 4)
```

### 🛡️ Simulation Sandbox

Before executing dangerous commands, the sandbox produces an **impact report**:

```
⚠️ Simulation Report:
  Risk: HIGH
  Impact: 154 files affected (wildcard)
  Warnings:
    - ⚠️ Plan contains destructive actions
    - 🔐 Plan requires elevated privileges
    - ♻️ 2 action(s) are NOT reversible
  Recommendation: ⚠️ HIGH RISK — Confirm impact
```

### 🧬 Self-Improving Prompt System

Successful reasoning chains are stored and reused:
- Keyword-indexed prompt templates with success/failure rates
- Automatic strategy matching for similar future tasks
- Rolling improvement — the agent gets better over time

### 🔌 Plugin Ecosystem

Heliox OS ships with **3 flagship plugins** and supports community-built extensions:

| Plugin | Type | Capabilities |
|--------|------|--------------|
| **developer-tools** | Code | Jira tickets, `git clone`, branch, commit, push, GitHub PRs |
| **media-control** | System | Spotify (play/pause/skip), system volume, YouTube, media keys |
| **home-assistant** | IoT | Smart lights, switches, thermostats, scenes, device discovery |

Drop custom plugins into `~/.heliox/plugins/` — they're auto-discovered at startup.

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "tools": [{"name": "my_tool", "inputs": ["arg1"], "action_type": "api_call"}]
}
```

### 🧠 Subconscious Agent (Persona Learning)

A background agent that runs every 30 minutes to review the day's actions and learn user preferences:

- Clusters behavioral patterns ("always writes Python", "prefers dark mode")
- Extracts actionable rules with confidence scores
- Writes a `~/.heliox/persona.md` that is injected into planner context
- Supports manual preference setting via `persona_add_preference` API
- Categories: `preference`, `habit`, `constraint`, `style`

### 👁️ Screen Vision Agent

Continuous computer-vision loop that gives the agent awareness of what the user sees:

- Takes screenshots every 2 seconds, hashes for change detection
- Detects the active application and window title cross-platform
- Maintains a rolling context buffer of recent screen states
- When user says "summarize this" or "close that", the planner already knows the target
- Optional LLM-powered screen description for advanced awareness

## 🚀 Installation

### Option 1: Download Compiled Desktop App (Recommended)

The easiest way to get started is to download the pre-compiled installer for your operating system.

1. Go to the [GitHub Releases page](https://github.com/VyomKulshrestha/Heliox-OS/releases).
2. Download the installer for your OS:
   - **Windows**: `Heliox OS_x64-setup.exe`
   - **macOS (Apple Silicon)**: `Heliox OS_aarch64.dmg`
   - **macOS (Intel)**: `Heliox OS_x86_64.dmg`
   - **Linux**: `.AppImage` or `.deb`
3. Install the app.
4. Open Heliox OS and enter your API Key (e.g., Gemini, OpenAI, Claude) in the Settings tab.

*Note: The Python backend requires Python 3.11+ installed on your system. You must start the local daemon manually for now.*

### Option 2: Build from Source (For Developers)

If you want to contribute or modify Heliox OS, build it from the source code:

**1. Install the Python daemon:**
```bash
git clone https://github.com/VyomKulshrestha/Heliox-OS.git
cd Heliox OS/daemon
pip install -e ".[full,dev]"
```

**2. Choose your LLM:**
*   Local (Ollama): `ollama pull llama3.1:8b` -> `ollama serve`
*   Cloud (Gemini/OpenAI/Claude): Add your API key in the app GUI.

**3. Run the daemon:**
```bash
cd daemon
python -m pilot.server
```

**4. Run the frontend:**
```bash
cd tauri-app/ui
npm install
npm run dev
```

## Example Commands

```
"Show me my system info"
"Take a screenshot and read the text on screen"
"Go to Wikipedia's page on AI and summarize the first 3 paragraphs"
"Create a Python project with tests and run them"
"Kill the process using the most CPU"
"Monitor my CPU and alert me when it goes above 80%"
"Download a file and show me a tree of the folder"
"List all .py files on my Desktop"
"Set my volume to 50%"
"Create a CSV with sales data and analyze it"
"What's my IP address?"
"Install Firefox"
```

## 🛡️ Security

> [!WARNING]
> **PLEASE READ BEFORE USE: SYSTEM COMPROMISE RISK**
> Heliox OS is an autonomous agent with the ability to execute code, delete files, and run terminal commands directly on your host operating system. While we have provided sandbox measures, the AI has real system access. **Do NOT run Heliox OS with root/Administrator privileges** unless absolutely necessary. We are not responsible for accidental data loss caused by LLM hallucinations.

- All AI outputs pass through structured schema validation before execution
- Five-tier permission system (read-only through root-level)
- Confirmation required for system-modifying and destructive actions
- Snapshot-based rollback via Btrfs or Timeshift (Linux)
- Append-only audit log for all executed actions
- Command whitelist with optional unrestricted mode
- **Encrypted API key storage** via platform keyring (GNOME Keyring / Windows Credential Manager)
- API keys are NEVER logged, included in plans, or sent to local LLMs

### Permission Tiers

| Tier | Level | Auto-Execute | Examples |
|------|-------|-------------|----------|
| 0 - Read Only | 🟢 | Yes | file_read, system_info, clipboard_read |
| 1 - User Write | 🟡 | Yes | file_write, clipboard_write, env_set |
| 2 - System Modify | 🟠 | Needs Confirm | package_install, service_restart, wifi_connect |
| 3 - Destructive | 🔴 | Needs Confirm | file_delete, process_kill, power_shutdown |
| 4 - Root Critical | ⛔ | Needs Confirm | root operations, disk operations |

## Configuration

Config file: `~/.config/pilot/config.toml`

```toml
[model]
provider = "ollama"           # "ollama" | "cloud"
ollama_model = "llama3.1:8b"
cloud_provider = "gemini"     # "gemini" | "openai" | "claude"

[security]
root_enabled = false
confirm_tier2 = true
unrestricted_shell = false
snapshot_on_destructive = true

[server]
host = "127.0.0.1"
port = 8785
```

## 🤝 Contributing

We love contributions! Whether it's adding a new gesture, fixing a bug, or building a new plugin, check out our guides to get started.

1. Read our [Contributing Guide](CONTRIBUTING.md) to set up your dev environment.
2. Check the [Good First Issues](https://github.com/VyomKulshrestha/Heliox-OS/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) tab to find beginner-friendly tasks.
3. Review our [Code of Conduct](CODE_OF_CONDUCT.md).
4. Join the community discussions in [GitHub Discussions](https://github.com/VyomKulshrestha/Heliox-OS/discussions).

## License

MIT
