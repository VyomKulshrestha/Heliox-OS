# Heliox OS — IPC Message Formats

The Heliox OS UI and daemon communicate over a local WebSocket using the [JSON-RPC 2.0](https://www.jsonrpc.org/specification) protocol.

## Protocol Overview

| Property | Value |
|----------|-------|
| Transport | WebSocket |
| URL | `ws://127.0.0.1:8785` |
| Encoding | JSON-RPC 2.0 |
| Request timeout | 5 minutes |
| Reconnect interval | 3 seconds (auto-reconnect on close) |

### Envelope formats

**Request** (UI → Daemon):
```json
{ "jsonrpc": "2.0", "method": "...", "params": {}, "id": 1 }
```

**Response** (Daemon → UI, matched by `id`):
```json
{ "jsonrpc": "2.0", "result": {}, "id": 1 }
```

**Error response**:
```json
{ "jsonrpc": "2.0", "error": { "code": -32601, "message": "Method not found: foo" }, "id": 1 }
```

**Notification** (Daemon → UI broadcast, no `id`, no response expected):
```json
{ "jsonrpc": "2.0", "method": "...", "params": {} }
```

Standard JSON-RPC error codes: `-32700` parse error, `-32600` invalid request, `-32601` method not found, `-32603` internal error.

---

## 1. UI → Daemon Requests

### Core Pipeline

#### `execute`
Run the full ReAct pipeline for a user command.

**Params:**
```json
{
  "input": "open Firefox and navigate to github.com",
  "dry_run": false
}
```
`dry_run` is optional (defaults to the daemon's configured value).

**Result:**
```json
{
  "status": "success",
  "dry_run": false,
  "explanation": "Opened Firefox and navigated to github.com",
  "results": [
    {
      "action_type": "open_application",
      "target": "firefox",
      "success": true,
      "output": "Firefox launched",
      "error": null
    }
  ],
  "verification": {
    "passed": true,
    "details": ["Firefox window detected"],
    "failed_actions": [],
    "rollback_triggered": false
  },
  "agent_routing": {
    "assigned_agents": ["system_agent"],
    "is_multi_agent": false
  }
}
```
`status` is one of `"success"`, `"partial_failure"`, `"error"`, or `"cancelled"`.

#### `confirm`
Resolve a pending confirmation gate (see `confirm_required` notification).

**Params:**
```json
{ "plan_id": "a3b2c1f5", "confirmed": true }
```

**Result:**
```json
{ "status": "ok", "confirmed": true }
```

---

### Configuration

#### `get_config`
Return the daemon's full runtime configuration.

**Params:** `{}`

**Result:**
```json
{
  "model": {
    "provider": "ollama",
    "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_model": "llama3.1:8b",
    "mode": "lightweight",
    "gpu_memory_limit_mb": 0,
    "cloud_provider": "",
    "cloud_model": ""
  },
  "security": {
    "root_enabled": false,
    "confirm_tier2": true,
    "dry_run": false,
    "snapshot_on_destructive": true,
    "snapshot_backend": "auto",
    "snapshot_retention_count": 10,
    "snapshot_retention_days": 7,
    "unrestricted_shell": false
  },
  "restrictions": {
    "protected_folders": [],
    "protected_packages": [],
    "blocked_commands": []
  },
  "first_run_complete": true
}
```

#### `update_config`
Update one config section.

**Params:**
```json
{
  "section": "security",
  "values": { "dry_run": true, "confirm_tier2": false }
}
```
Use `section: ""` with `values: { "first_run_complete": true }` to set top-level fields.

**Result:** `{ "status": "ok" }` or `{ "status": "error", "message": "..." }`

---

### History & Memory

#### `get_history`
Retrieve past interactions from the memory store.

**Params:**
```json
{ "limit": 50, "offset": 0 }
```
Both params are optional.

**Result:**
```json
{
  "entries": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "input": "install vim",
      "plan": {},
      "results": [],
      "notes": ""
    }
  ]
}
```

---

### API Key Management

#### `store_api_key`
Store an API key in the encrypted vault.

**Params:** `{ "provider": "openai", "api_key": "sk-..." }`
**Result:** `{ "status": "ok" }`

#### `delete_api_key`
Delete a stored API key.

**Params:** `{ "provider": "openai" }`
**Result:** `{ "status": "ok" }`

#### `list_api_keys`
List all providers with stored keys.

**Params:** `{}`
**Result:** `{ "providers": ["openai", "claude"] }`

---

### Health & Discovery

#### `ping`
Check connectivity.

**Params:** `{}`
**Result:** `{ "pong": true, "version": "0.7.1" }`

#### `health`
Check all model backend health.

**Params:** `{}`
**Result:** `{ "backends": { "ollama": true, "cloud": false } }`

#### `system_status`
Return platform information.

**Params:** `{}`
**Result:** `{ "platform": { ... }, "capabilities_count": 120 }`

#### `capabilities`
List all available action types.

**Params:** `{}`
**Result:** `{ "action_types": ["file_read", "file_write", ...], "count": 120 }`

#### `list_ollama_models`
Discover locally available Ollama models.

**Params:** `{}`
**Result:** `{ "models": ["llama3.1:8b", "mistral:7b"], "available": true }`

---

### Agent Routing & Orchestration

#### `agent_routing`
Dry-run routing analysis: which specialist agent(s) would handle a given input.

**Params:** `{ "input": "write a Python script" }`
**Result:**
```json
{
  "assigned_agents": ["code_agent"],
  "is_multi_agent": false,
  "orchestrator": { ... }
}
```

#### `agent_stats`
Performance statistics for all registered specialist agents.

**Params:** `{}`
**Result:** agent performance statistics dict.

#### `agent_capabilities`
List capabilities of every registered agent.

**Params:** `{}`
**Result:** capabilities dict grouped by agent role.

#### `agent_spawn`
Dynamically spawn a new specialist agent.

**Params:** `{ "role": "code_agent" }`
**Result:** `{ "status": "spawned", "agent_id": "abc123" }` or `{ "status": "error", "message": "..." }`

---

### Multimodal Input

#### `voice_event`
Feed a voice transcript to the fusion engine.

**Params:**
```json
{ "transcript": "open terminal", "confidence": 0.95, "is_final": true }
```

**Result:**
```json
{ "status": "fused", "intent": { "command": "open terminal", "confidence": 0.95 } }
```
`status` is `"fused"` when a complete intent was produced, or `"buffered"` when the engine is waiting for more input.

#### `gesture_event`
Feed a gesture event to the fusion engine.

**Params:**
```json
{
  "gesture": "thumbs_up",
  "confidence": 0.88,
  "data": { "x": 320, "y": 240, "velocity": 0.5 }
}
```
**Result:** same shape as `voice_event`.

#### `multimodal_stats`
Return fusion engine statistics.

**Params:** `{}`
**Result:** fusion engine stats dict.

---

### Reasoning Visualization

#### `reasoning_log`
Return the full reasoning event log for the current session.

**Params:** `{}`
**Result:** `{ "events": [ <ReasoningEvent>, ... ] }`

See the `reasoning_event` notification for the `ReasoningEvent` object shape.

#### `reasoning_stats`
Return reasoning emitter statistics.

**Params:** `{}`
**Result:** stats dict.

---

### Task Decomposition & Simulation

#### `decompose_task`
Break a complex goal into a dependency-ordered subtask tree.

**Params:** `{ "goal": "set up a Python web server with nginx" }`
**Result:** decomposed task tree dict.

#### `simulate_plan`
Dry-analyze a pending plan for impact (no execution).

**Params:** `{ "plan_id": "a3b2c1f5" }` (must be a plan currently awaiting confirmation)
**Result:** impact report dict.

---

### Prompt Improvement

#### `prompt_strategies`
Return proven prompt strategies for a task type.

**Params:** `{ "query": "file operations" }`
**Result:** `{ "strategies": "Use absolute paths. Check permissions before writing..." }`

#### `prompt_stats`
Return prompt improvement statistics.

**Params:** `{}`
**Result:** stats dict.

---

### Plugin Ecosystem

#### `plugin_list`
List all loaded plugins and their stats.

**Params:** `{}`
**Result:** plugin registry stats dict.

#### `plugin_tools`
List all tools exposed by loaded plugins.

**Params:** `{}`
**Result:** `{ "tools": [ { "name": "...", "description": "..." }, ... ] }`

#### `plugin_toggle`
Enable or disable a plugin by name.

**Params:** `{ "name": "my_plugin", "enabled": true }`
**Result:** `{ "success": true, "plugin": "my_plugin", "enabled": true }`

---

### Subconscious / Persona Agent

#### `persona_rules`
Return all learned persona rules and preferences.

**Params:** `{}`
**Result:** persona context and statistics dict.

#### `persona_consolidate`
Force a consolidation cycle to extract rules from recent history.

**Params:** `{}`
**Result:** consolidation result dict.

#### `persona_add_preference`
Manually record a user preference.

**Params:** `{ "key": "editor", "value": "neovim" }`
**Result:** `{ "status": "ok", "key": "editor", "value": "neovim" }`

#### `subconscious_stats`
Return subconscious agent statistics.

**Params:** `{}`
**Result:** stats dict.

---

### Screen Vision

#### `screen_context`
Return the current screen context summary.

**Params:** `{}`
**Result:** `{ "summary": "VS Code is open with main.py", ... }`

#### `screen_current_app`
Return the currently active application name.

**Params:** `{}`
**Result:** `{ "active_app": "code" }`

#### `screen_vision_stats`
Return screen vision statistics.

**Params:** `{}`
**Result:** stats dict.

#### `screen_vision_toggle`
Start or stop the screen vision agent.

**Params:** `{ "enabled": true, "interval_seconds": 3.0, "enable_describe": false }`
**Result:** `{ "status": "ok", "enabled": true }`

---

### Cognitive Intelligence (TRIBE v2)

#### `cognitive_stats`
Return statistics for all cognitive subsystems.

**Params:** `{}`
**Result:**
```json
{
  "tribe_engine": { ... },
  "attention_ui": { ... },
  "stress_gate": { ... },
  "intent_predictor": { ... }
}
```

#### `cognitive_state`
Return the current predicted cognitive state.

**Params:** `{ "stimulus": "optional description" }`
**Result:** cognitive state dict.

#### `attention_toggle`
Enable or disable attention-aware notification scoring.

**Params:** `{ "enabled": true }` (`enabled` is optional; omit to toggle)
**Result:** `{ "enabled": true }`

#### `stress_gate_toggle`
Enable or disable stress-aware task gating.

**Params:** `{ "enabled": true }`
**Result:** `{ "enabled": true }`

#### `intent_predictor_toggle`
Enable or disable JARVIS-mode intent prediction.

**Params:** `{ "enabled": true }`
**Result:** `{ "enabled": true }`

#### `tribe_model_toggle`
Load, unload, or query the TRIBE v2 local model.

**Params:** `{ "action": "load" }` — `action` is `"load"`, `"unload"`, or `"status"` (default)
**Result:**
```json
{ "loaded": true, "fallback": false, "available": true }
```

---

### Voice Listener (JARVIS Mode)

#### `voice_listener_start`
Start the continuous wake-word voice listener.

**Params:** `{ "wake_words": ["hey heliox", "heliox", "hey pilot"] }`
**Result:** `{ "status": "started", "message": "...", "wake_words": ["hey heliox", ...] }`

#### `voice_listener_stop`
Stop the voice listener.

**Params:** `{}`
**Result:** `{ "status": "stopped", "message": "..." }`

#### `voice_listener_stats`
Return voice listener statistics.

**Params:** `{}`
**Result:** stats dict.

---

### Autonomous Executor (Background Jobs)

#### `autonomous_submit`
Submit a goal for fire-and-forget autonomous background execution.

**Params:** `{ "goal": "organize my Downloads folder", "source": "text" }`
**Result:**
```json
{
  "status": "submitted",
  "job": {
    "job_id": "abc123",
    "goal": "organize my Downloads folder",
    "status": "pending",
    "steps": [],
    "source": "text"
  }
}
```

#### `autonomous_cancel`
Cancel a running autonomous job.

**Params:** `{ "job_id": "abc123" }`
**Result:** `{ "cancelled": true, "job_id": "abc123" }`

#### `autonomous_jobs`
List all autonomous jobs.

**Params:** `{}`
**Result:** `{ "jobs": [ <job>, ... ] }`

#### `autonomous_job`
Get a single autonomous job by ID.

**Params:** `{ "job_id": "abc123" }`
**Result:** job object dict (same shape as the `job` field in `autonomous_submit`).

---

### Proactive Suggestions

#### `proactive_start` / `proactive_stop`
Start or stop the proactive suggestion engine.

**Params:** `{}`
**Result:** `{ "status": "started"|"stopped", "message": "..." }`

#### `proactive_stats`
Return proactive engine statistics.

**Params:** `{}`
**Result:** stats dict.

#### `proactive_accept`
Accept and execute a proactive suggestion.

**Params:** `{ "suggestion_id": "sug_xyz" }`
**Result:**
```json
{ "status": "executing", "action": "clear temp files", "job": { ... } }
```

#### `proactive_dismiss`
Dismiss a proactive suggestion without acting on it.

**Params:** `{ "suggestion_id": "sug_xyz" }`
**Result:** `{ "dismissed": true, "suggestion_id": "sug_xyz" }`

---

### Background Tasks

#### `background_tasks`
List all registered background monitoring tasks.

**Params:** `{}`
**Result:** `{ "tasks": [ { "task_id": "...", ... } ] }`

#### `background_start` / `background_stop`
Start or stop a background monitoring task.

**Params:** `{ "task_id": "cpu_monitor" }`
**Result:** `{ "status": "started"|"stopped"|"error", "task_id": "cpu_monitor" }`

#### `reflection_stats`
Return self-improvement reflection statistics.

**Params:** `{}`
**Result:** stats dict.

---

### Interactive Git Conflict Resolver

#### `resolve_git_conflict`
Parse the file at the given path, locate conflict blocks, run LLM routing to get candidate resolutions structured per `schemas/git_conflict_resolution.json`, and return the blocks with original + suggested resolution.

**Params:**
```json
{
  "filepath": "conflict_demo.py"
}
```

**Result:**
```json
{
  "status": "success",
  "conflicts": [
    {
      "id": "conflict_0",
      "original_block": "<<<<<<< HEAD\ndef hello():\n    return 'local'\n=======\ndef hello():\n    return 'incoming'\n>>>>>>> feature-branch",
      "our_code": "def hello():\n    return 'local'",
      "their_code": "def hello():\n    return 'incoming'",
      "our_branch": "HEAD",
      "their_branch": "feature-branch",
      "resolved_code": "def hello():\n    return 'local'"
    }
  ]
}
```

#### `apply_git_resolution`
Apply a git conflict resolution block by safely, atomically, and securely writing/replacing the resolved code block inside the file.

**Params:**
```json
{
  "path": "conflict_demo.py",
  "full_block": "<<<<<<< HEAD\ndef hello():\n    return 'local'\n=======\ndef hello():\n    return 'incoming'\n>>>>>>> feature-branch",
  "resolved_code": "def hello():\n    return 'local'"
}
```

**Result:**
```json
{
  "status": "success",
  "message": "Git conflict resolution applied successfully"
}
```

---

## 2. Daemon → UI Notifications

Notifications are broadcast to **all** connected clients with no `id` field. The UI should not send a response.

### `status`
Current pipeline stage during `execute`.

```json
{ "phase": "planning" }
```

`phase` values (in order): `"receiving input"`, `"recalling memory"`, `"routing agents"`, `"planning"`, `"re-planning (attempt 2)"`, `"executing"`, `"verifying"`, `"retrying — previous attempt failed"`.

---

### `agent_routing`
Which specialist agent(s) were selected to handle the request.

```json
{
  "assigned_agents": ["system_agent", "code_agent"],
  "is_multi_agent": true
}
```

---

### `plan_preview`
Full plan generated by the planner, sent before execution begins.

```json
{
  "plan_id": "a3b2c1f5",
  "explanation": "Install vim using apt",
  "actions": [
    {
      "action_type": "package_install",
      "target": "vim",
      "requires_confirmation": false,
      "requires_root": true,
      "destructive": false,
      "reversible": true,
      "rollback_action": null,
      "parameters": { "name": "vim" }
    }
  ],
  "dry_run": false,
  "source": "text"
}
```

`source` is `"voice"` when the plan was triggered by the voice listener, otherwise absent.

---

### `confirm_required`
Sent when one or more actions in the plan require explicit user approval (Tier 2+ actions). The `execute` handler blocks until a matching `confirm` request arrives or the 5-minute timeout expires.

```json
{
  "plan_id": "a3b2c1f5",
  "actions": [
    {
      "action_type": "package_remove",
      "target": "python3",
      "requires_confirmation": true,
      "destructive": true
    }
  ]
}
```

The UI should present an approval dialog and call `confirm` with the `plan_id`.

---

### `action_start`
Fired immediately before each action is executed.

```json
{
  "action": {
    "action_type": "file_write",
    "target": "/etc/hosts",
    "parameters": { "path": "/etc/hosts", "content": "..." },
    "requires_confirmation": false,
    "dry_run": false
  }
}
```

---

### `action_complete`
Fired after each action finishes.

```json
{
  "result": {
    "action_type": "file_write",
    "target": "/etc/hosts",
    "success": true,
    "output": "File written (256 bytes)",
    "error": null,
    "dry_run": false
  }
}
```

---

### `orchestrator_routing`
Multi-agent orchestrator assignment — which specialist agents will handle which parts of the plan.

```json
{
  "assigned_agents": [
    { "role": "system_agent", "capability": "package management" }
  ],
  "total_agents": 1
}
```

---

### `reasoning_event`
Granular thought-visualization telemetry. Every `execute` call emits a stream of these events so the UI can render a live execution graph.

```json
{
  "event_id": "3f7a2b9e1c4d",
  "event_type": "phase_start",
  "event_name": "planner_started",
  "stage": "planning",
  "timestamp": 1705316400.123,
  "duration_ms": 0,
  "data": { "attempt": 1 },
  "parent_id": "",
  "sequence": 5
}
```

**`event_type` values:**

| Value | Meaning |
|-------|---------|
| `phase_start` | A pipeline stage began |
| `phase_complete` | A pipeline stage finished successfully |
| `phase_error` | A pipeline stage failed |
| `thought` | An LLM inner-reasoning text snippet |
| `decision` | A decision point with options and the chosen path |
| `data` | A data payload (plan, results, routing info) |
| `progress` | Progress update within a stage |
| `metric` | A performance metric (e.g. duration) |

**`stage` values:** `user_input`, `memory_recall`, `agent_routing`, `planning`, `confirmation`, `orchestration`, `execution`, `verification`, `reflection`, `memory_update`.

**`data` field contents by event type:**

```jsonc
// thought
{ "text": "Analyzing the user's request to find the right approach..." }

// decision
{ "description": "Which specialist agent to route to", "chosen": "code_agent" }

// progress
{ "percent": 50, "label": "file_write" }

// metric
{ "name": "total_duration_ms", "value": 4523, "unit": "ms" }

// phase_complete / phase_error carry stage-specific data, e.g.:
{ "plan_id": "a3b2c1f5", "action_count": 3, "explanation": "Install vim..." }
```

---

### `voice_command`
Emitted when the JARVIS voice listener recognizes a command and begins executing it.

```json
{ "command": "open the terminal", "status": "executing" }
```

---

### `voice_status`
Voice listener lifecycle updates.

```json
{ "status": "listening" }
```

`status` is one of `"listening"`, `"processing"`, `"stopped"`, or `"error"`.

---

### `voice_result`
Result of a voice-triggered command execution.

```json
{ "command": "open the terminal", "status": "success", "result": "Terminal opened." }
```

On error: `{ "command": "...", "status": "error", "message": "..." }`

---

### `multimodal_intent`
A fused voice + gesture intent from the fusion engine.

```json
{
  "command": "scroll down slowly",
  "voice_component": "scroll down",
  "gesture_component": "swipe_down",
  "gesture_modifier": "slow",
  "fusion_type": "voice_gesture",
  "confidence": 0.91,
  "timestamp": 1705316400123,
  "metadata": {}
}
```

`fusion_type` is one of `"voice_gesture"`, `"voice_only"`, `"gesture_only"`, or `"single"`.

---

### `feature_announcement`
Emitted once on startup when a new daemon version introduces new capabilities.

```json
{ "message": "New: TRIBE v2 cognitive intelligence is now available!", "version": "0.6.0" }
```

---

## 3. Shared Object Schemas

### Action object
Defined in `schemas/action_plan.schema.json` and `daemon/pilot/actions.py`.

```json
{
  "action_type": "file_write",
  "target": "/home/user/notes.txt",
  "parameters": { "path": "/home/user/notes.txt", "content": "hello" },
  "requires_confirmation": false,
  "requires_root": false,
  "destructive": false,
  "reversible": true,
  "rollback_action": null
}
```

**`action_type` values** — grouped by permission tier:

| Tier | Action types |
|------|-------------|
| Read-only | `file_read`, `file_list`, `file_search`, `directory_summary`, `package_search`, `service_status`, `gnome_setting_read`, `open_url`, `open_application`, `notify`, `process_list`, `process_info`, `clipboard_read`, `system_info`, `disk_usage`, `memory_usage`, `cpu_usage`, `network_info`, `battery_info`, `env_get`, `env_list`, `window_list`, `volume_get`, `brightness_get`, `screenshot`, `wifi_list`, `disk_list`, `user_list`, `user_info`, `schedule_list`, `mouse_position`, `screen_ocr`, `screen_find_text`, `screen_analyze`, `screen_element_map`, `browser_extract`, `browser_extract_table`, `browser_extract_links`, `browser_screenshot`, `browser_list_tabs`, `browser_page_info`, `trigger_list`, `file_parse`, `file_search_content`, `api_scrape`, `registry_read` |
| User write | `file_write`, `file_move`, `file_copy`, `git_resolve`, `clipboard_write`, `keyboard_type`, `keyboard_press`, `keyboard_hotkey`, `keyboard_hold`, `mouse_click`, `mouse_double_click`, `mouse_right_click`, `mouse_move`, `mouse_drag`, `mouse_scroll`, `volume_set`, `volume_mute`, `brightness_set`, `window_focus`, `window_minimize`, `window_maximize`, `browser_navigate`, `browser_click`, `browser_type`, `browser_select`, `browser_hover`, `browser_scroll`, `browser_execute_js`, `browser_fill_form`, `browser_new_tab`, `browser_close_tab`, `browser_switch_tab`, `browser_back`, `browser_forward`, `browser_refresh`, `browser_wait`, `browser_close`, `env_set`, `download_file`, `api_request`, `api_github`, `code_execute`, `code_generate_and_run`, `trigger_create`, `trigger_start`, `trigger_stop` |
| System modify | `package_install`, `package_update`, `service_start`, `service_stop`, `service_restart`, `service_enable`, `service_disable`, `gnome_setting_write`, `shell_command`, `shell_script`, `schedule_create`, `file_permissions`, `wifi_connect`, `wifi_disconnect`, `disk_mount`, `registry_write`, `api_send_email`, `api_webhook`, `api_slack`, `api_discord` |
| Destructive | `file_delete`, `package_remove`, `process_kill`, `power_shutdown`, `power_restart`, `power_logout`, `schedule_delete`, `disk_unmount`, `window_close`, `trigger_delete`, `browser_click_text` |
| Root critical | `power_sleep`, `power_lock`, `user_info` (with elevation), `dbus_call` |

Actions at **Tier 2 (System modify) and above** set `requires_confirmation: true` when `confirm_tier2` is enabled in the security config (the default).

---

### ActionResult object

```json
{
  "action_type": "file_write",
  "target": "/home/user/notes.txt",
  "success": true,
  "output": "File written (42 bytes)",
  "error": null,
  "snapshot_id": null
}
```

`snapshot_id` is set when a filesystem snapshot was taken before a destructive action.

---

### Verification object

```json
{
  "passed": true,
  "details": ["File /home/user/notes.txt exists and has expected content"],
  "failed_actions": [],
  "rollback_triggered": false
}
```

---

### Config object (from `get_config`)

See the `get_config` response above. The `server` section (host, port, auth_token) is stripped before sending.

---

### Cognitive metadata (`_cognitive`)

When the **Attention-Aware UI** feature is enabled, the daemon injects a `_cognitive` field into notification `params` before broadcasting. UI components can use this to decide how to render the notification.

```json
{
  "_cognitive": {
    "priority": "high",
    "attention_score": 0.72,
    "should_animate": true,
    "display_duration_ms": 4000,
    "flushed": false
  }
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `priority` | `string` | `"critical"`, `"high"`, `"normal"`, or `"low"` |
| `attention_score` | `number` | 0.0 – 1.0; higher = user more focused/busy |
| `should_animate` | `boolean` | Whether the notification should animate into view |
| `display_duration_ms` | `number` | Suggested on-screen duration in milliseconds |
| `flushed` | `boolean` | `true` when a previously buffered notification is released during a low-attention window |

When `attention_score` is high and `priority` is not `"critical"`, the notification may be **buffered** and delivered later; it will arrive with `"flushed": true`.

---

## 4. End-to-End Execution Flow

The following sequence shows all messages for a typical `execute` call that contains a dangerous action requiring user confirmation.

```
UI                                          Daemon
│                                               │
│── execute {input, dry_run} ──────────────────►│
│                                               │
│◄── notification: status {phase: "receiving input"}
│◄── notification: status {phase: "recalling memory"}
│◄── notification: status {phase: "routing agents"}
│◄── notification: agent_routing {assigned_agents, is_multi_agent}
│◄── notification: status {phase: "planning"}
│                                               │  (LLM generates plan)
│◄── notification: plan_preview {plan_id, actions, explanation}
│                                               │
│◄── notification: confirm_required {plan_id, actions}
│         (UI shows approval dialog)            │
│── confirm {plan_id, confirmed: true} ────────►│
│                                               │
│◄── notification: status {phase: "executing"}
│◄── notification: orchestrator_routing {assigned_agents}
│                                               │  (for each action:)
│◄── notification: action_start {action}        │
│◄── notification: action_complete {result}     │
│                                               │
│◄── notification: status {phase: "verifying"}
│                                               │
│◄── response: execute result ─────────────────┤
│   {status, results, verification, explanation}│
```

Throughout the call, `reasoning_event` notifications stream in parallel with the stage notifications, providing granular thought-graph telemetry.

If verification fails, the daemon re-plans and the cycle repeats (up to 2 retries), broadcasting `status: "re-planning (attempt 2)"` before the next `plan_preview`.

---

## Source References

| File | Contents |
|------|----------|
| `daemon/pilot/server.py` | All request handlers and notification senders |
| `daemon/pilot/actions.py` | `ActionType` enum and parameter models |
| `daemon/pilot/config.py` | `PilotConfig`, `ModelConfig`, `SecurityConfig` |
| `daemon/pilot/reasoning/events.py` | `ReasoningEvent` schema and event name constants |
| `tauri-app/ui/src/lib/api/daemon.ts` | WebSocket client (`connect`, `call`, `onNotification`) |
| `tauri-app/ui/src/lib/stores/session.ts` | Notification handlers for the core pipeline |
| `tauri-app/ui/src/lib/stores/multimodal.ts` | Multimodal fusion state and notification handler |
| `schemas/action_plan.schema.json` | JSON Schema for the `ActionPlan` object |
| `schemas/responses/execution_result.json` | JSON Schema for `ExecutionResult` |
| `schemas/actions/*.json` | Per-action-type parameter schemas |
