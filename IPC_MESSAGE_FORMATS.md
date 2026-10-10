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

The daemon currently registers **197 WebSocket RPC methods**. That is the API
surface count, not the action catalog: Heliox exposes **158 action
types** through the guarded planner/executor system. This document names every
registered RPC method; grouped tables are used where several methods share one
contract.

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
Run the shared text/voice interaction path for a user command: context,
planning, companion review, safety and approval, execution, environment
verification, and a terminal result. The removed thought-graph/ReAct UI is not
part of this contract; progress is exposed through `status`,
`interaction_state`, action notifications, and `task_complete`.

**Params:**
```json
{
  "input": "open Firefox and navigate to github.com",
  "dry_run": false,
  "session_id": "chat_123",
  "source": "text"
}
```
`dry_run` is optional (defaults to the daemon's configured value).
`session_id` scopes working memory, recent companion context, and the durable
task to one chat. `source` is `text` or `voice`; voice dispatch calls this same
handler.

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
Common terminal `status` values are `"success"`, `"partial_failure"`,
`"error"`, `"cancelled"`, `"blocked_by_companion"`, and
`"blocked_by_critic"`. Clients must preserve the daemon's explanatory
`message` or `explanation` instead of reducing these outcomes to a generic
failure.

A conversational request may validly return no actions:

```json
{
  "status": "success",
  "explanation": "Good morning. What would you like to work on?",
  "results": [],
  "conversational": true
}
```

#### `interject`
Correct, stop, or ask about the active interactive task out of band. A correction cancels
the current step, preserves already observed results, and replans. It remains
available even when optional step narration is disabled.

**Params:** `{ "input": "Use Hermes, not Notepad", "mode": "correct" }`.
Use `mode: "stop"` (or an explicit stop/cancel phrase) to abort. Use
`mode: "ask"` to receive a bounded read-only answer without cancelling or
approving the task. Explicit “read my screen” uses local OCR on demand.

**Result:** `{ "status": "revising", "message": "Live correction accepted." }`,
`{ "status": "answered", "message": "..." }`, `{ "status": "aborted" }`,
or a clear disabled/no-active-task response.

#### `queue_execute` / `cancel_queued_execute`
The authenticated UI may queue up to ten additional interactive tasks while
one is active. Queued tasks run serially through the ordinary planner,
approval, execution, and verification path; they are not parallel desktop
controllers. Pending tasks are marked interrupted after daemon restart rather
than silently executed later.

**Queue params:** `{ "input": "Summarize the report", "session_id": "chat_123" }`.
**Queue result:** `{ "status": "queued", "task_id": "...", "resume_token": "...", "position": 1 }`.
**Cancel params:** `{ "task_id": "...", "resume_token": "..." }`.
**Cancel result:** `{ "status": "cancelled", "task_id": "..." }` when the
task has not started; use `abort` for the currently running task.

#### `resume_task`
Authenticate and resume a non-terminal durable task after a reconnect or daemon
restart.

**Params:** `{ "task_id": "task_...", "resume_token": "..." }`

**Result:** the terminal response with `replayed: true`, an
`awaiting_approval` object, or the resumed execution result with
`resumed: true`. The raw token is never stored; the task journal stores its
hash. Durable plans cannot be resumed with `resume_plan` alone.

#### `resume_plan`
Resume a legacy/non-durable checkpoint from its first incomplete action.

**Params:** `{ "plan_id": "a3b2c1f5" }`

**Result:** an execution result with `resumed`, `skipped_actions`,
`executed_actions`, and combined results. If the checkpoint belongs to a
durable task, the daemon rejects this method and requires `resume_task` plus
the resume token.

#### `export_session_chat`
Export the messages supplied by the active UI chat to the user's Downloads
folder (or the daemon export directory if Downloads is unavailable).

**Params:** `{ "messages": [...], "format": "json"|"csv"|"markdown" }`

**Result:** `{ "status": "ok", "path": "...", "count": 12, "format": "json" }`.

#### `abort`
Stop the current execution — both cooperatively and, where possible, by
really killing whatever is in flight right now. Sets the per-session
`cancel_event` (so the Orchestrator/Executor halt at the next action
boundary, as before), **and** cancels the currently tracked interactive
execution task — cascading all the way down to a mid-flight shell
subprocess's `proc.kill()` — **and** interrupts every live PTY session
(`pty_exec` can't be stopped by task cancellation alone; see
`PtySession.interrupt()`). See **Mid-flight Cancellation** in `SECURITY.md`
for the full design and known scope limits.

Returns immediately — cancellation propagates asynchronously. The in-flight
`execute`/`resume_plan` call observes it on its own and resolves with
`{"status": "cancelled", ...}` (see `execute` above).

**Params:** `{}` (none)

**Result:**
```json
{ "status": "aborted" }
```
`{ "status": "no_active_execution" }` if nothing was running to stop.

#### `confirm`
Resolve a pending confirmation gate (see `confirm_required` notification).

**Params:**
```json
{ "plan_id": "a3b2c1f5", "confirmed": true, "approved_indices": [0, 2] }
```
`approved_indices` is optional — a list of `plan.actions` indices (matching
the `index` field on each action in the `confirm_required` payload) the user
approved out of those requiring confirmation, for per-action granular
approval. Omit it (or send `confirmed: false`) for the old all-or-nothing
behavior: omitting it while `confirmed: true` approves every action that
required confirmation.

**Result:**
```json
{ "status": "ok", "confirmed": true }
```

#### `rollback_plan`
Roll back the filesystem snapshot taken before a plan executed (see
`ActionResult.snapshot_id` / the `rollback_complete` notification). This is
**filesystem-wide**, not per-action — it reverts everything since the
snapshot, including unrelated changes made after it. The UI must gate this
behind its own explicit confirmation; the daemon does not re-confirm.

**Params:** `{ "plan_id": "a3b2c1f5" }`

**Result:**
```json
{ "status": "ok", "message": "Rollback snapshot created from ... . Reboot to apply." }
```
`{ "status": "error", "message": "..." }` if no snapshot is on record for that `plan_id`, or the rollback itself fails.

#### `list_permission_events`
List recent tamper-evident permission-escalation audit events.

**Params:** `{ "limit": 50, "plan_id": "a3b2c1f5" }` (both optional; `plan_id` filters to one plan)

**Result:** `{ "status": "ok", "events": [ <PermissionAuditEvent>, ... ] }` — see the `PermissionAuditEvent` schema below.

#### `verify_permission_audit`
Verify the HMAC hash-chain integrity of the permission audit log — detects whether any row was tampered with, reordered, or deleted.

**Params:** `{}`

**Result:**
```json
{ "status": "ok", "valid": true, "checked_entries": 42, "error": "" }
```

---

### Agent Gateway (source-scoped permissions, dry-run, audit)

Source-scoped permission floors for shell/browsing/system-control actions, layered alongside the tier-based `PermissionChecker` (see SECURITY.md's "Agent Gateway" section for the full threat model and design). `pilot.security.gateway.AgentGateway.authorize()` is checked inside `Executor.execute()` before dispatch; these RPCs are read-only observability plus a policy editor, not the enforcement point itself.

#### `list_gateway_events`
List recent tamper-evident Agent Gateway audit events.

**Params:** `{ "limit": 50, "plan_id": null, "source_profile": null, "action_family": null, "decision": null }` (all optional filters)

**Result:** `{ "status": "ok", "events": [ <GatewayAuditEvent>, ... ] }` — see the `GatewayAuditEvent` schema below.

#### `verify_gateway_audit`
Verify the HMAC hash-chain integrity of the Agent Gateway audit log — a separate chain from `verify_permission_audit`'s (different database/key file), so a compromise of one doesn't help forge the other.

**Params:** `{}`

**Result:**
```json
{ "status": "ok", "valid": true, "checked_entries": 17, "error": "" }
```

#### `gateway_policy_get`
Return the current per-`InvocationSource` enforced floors. The shipped set has
25 profiles:

`interactive`, `autonomous`, `voice`, `gesture`, `self_healing`,
`system_agent`, `ssh_agent`, `code_agent`, `web_agent`, `monitor_agent`,
`comm_agent`, `rss_agent`, `calendar_agent`, `forensics_agent`,
`semantic_search_agent`, `file_agent`, `package_agent`, `service_agent`,
`desktop_agent`, `automation_agent`, `integration_agent`, `vision_agent`,
`plugin_runtime_agent`, `network_agent`, and `git_agent`.

Communication and Email are separate runtime providers that intentionally share
the `comm_agent` gateway profile.

**Params:** `{}`

**Result:**
```json
{
  "status": "ok",
  "enabled": true,
  "profiles": {
    "autonomous": {
      "max_tier": { "shell": 2, "browsing": 2, "system_control": 1, "other": 2 },
      "deny_action_types": ["browser_execute_js", "power_shutdown", "power_restart", "registry_write"],
      "allow_root": false
    }
  }
}
```

#### `gateway_policy_update`
Update one source profile's enforced floor. Only edits the persisted floor — per-task overrides (e.g. `autonomous_submit`'s `scope_override`) are never settable here, only supplied per-submission by the caller, and can only narrow this floor further, never widen it.

**Params:** `{ "profile": "autonomous", "max_tier": { "shell": 0 }, "deny_action_types": [...], "allow_root": false }` (`max_tier` is merged onto the existing floor — only the families you include are changed; `deny_action_types`/`allow_root` are optional and replace their prior value when present)

**Result:** `{ "status": "ok", "profile": "autonomous", "policy": { "max_tier": {...}, "deny_action_types": [...], "allow_root": false } }`, or `{ "status": "error", "message": "Unknown source profile: ..." }`.

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
    "cloud_model": "",
    "subscription_provider": "codex",
    "subscription_model": "",
    "subscription_timeout_seconds": 120,
    "subscription_max_prompt_chars": 48000
  },
  "security": {
    "root_enabled": false,
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
  "gesture_cursor": {
    "enabled": false,
    "sensitivity": 1.0,
    "prediction_ms": 80.0,
    "blend": 0.3
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
  "values": { "dry_run": true }
}
```
Use `section: ""` with `values: { "first_run_complete": true }` to set top-level fields.

**Result:** `{ "status": "ok" }` or `{ "status": "error", "message": "..." }`

#### `get_security_status`
Return both the configured Heliox root policy and whether the current daemon
process actually has elevated OS privileges.

**Params:** `{}`

#### `get_snapshot_status`
Probe the configured pre-destructive-action snapshot backend and report live
readiness. This distinguishes an enabled preference from a usable restore
point/Timeshift/Btrfs backend.

**Params:** `{}`

#### `restart_elevated`
On Windows, request a UAC handoff to a replacement Administrator daemon. The
request is blocked unless `security.root_enabled` is already true. Non-Windows
platforms return `unsupported`; declining UAC returns an error rather than
pretending elevation succeeded.

**Params:** `{}`

#### `reset_config`
Reset persisted configuration fields to `PilotConfig` defaults.

**Params:** `{}`
**Result:** `{ "status": "ok" }`

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

Chat transcripts and durable user memory are deliberately different stores.
The frontend chat-session dialog keeps per-chat UI records locally;
`get_history` returns daemon interaction history used for memory and audit
surfaces.

#### `memory_checkpoint`
Request a SQLite WAL checkpoint for the memory store.

**Params:** `{}`
**Result:** checkpoint status and WAL statistics, or an initialization error.

#### `temporal_memory_status`
Return up to 200 provenance-labelled active/candidate facts, episodes, and
working-memory items for user review.

**Params:** `{ "limit": 50 }`

#### `temporal_memory_retract`
Retract one active or candidate fact without rewriting the append-only
experience ledger.

**Params:** `{ "fact_id": "fact_...", "reason": "Incorrect preference" }`
**Result:** `{ "status": "ok", "fact_id": "fact_...", "fact_status": "retracted" }`

---

### API Key Management

#### `store_api_key`
Store an API key in the operating-system credential store. Heliox uses Windows
Credential Manager, macOS Keychain, or a Secret-Service-compatible Linux
keyring and fails closed if the platform store is unavailable.

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
**Result:** `{ "pong": true, "version": "0.10.1" }`

#### `health`
Return daemon-process health: uptime, RSS memory, active WebSocket connections,
and loaded agent roles.

**Params:** `{}`
**Result:** `{ "uptime": 42.1, "memory_usage_mb": 180.4, "active_connections": 1, "loaded_agents": [...] }`

#### `ready`
Return `{ "ready": true }` only when the orchestrator has registered agents
and every registered agent is running without an error state.

#### `system_info`
Return exact HUD metrics for CPU, memory, disk, hostname, and uptime.

#### `get_uptime`
Return formatted host uptime such as `"3h 12m"` or `"2d 4h 8m"`.

#### `system_status`
Return platform information.

**Params:** `{}`
**Result:** `{ "platform": { ... }, "capabilities_count": 158 }`

#### `capabilities`
List all available action types.

**Params:** `{}`
**Result:** `{ "action_types": ["file_read", "file_write", ...], "count": 158 }`

#### `list_ollama_models`
Discover locally available Ollama models.

**Params:** `{}`
**Result:** `{ "models": ["llama3.1:8b", "mistral:7b"], "available": true }`

#### `subscription_status`

Probe one official local CLI without exposing account identity or credentials.
Codex counts as a subscription only when `codex login status` reports a ChatGPT
login; an API-key login is deliberately not relabelled as subscription access.

**Params:** `{ "provider": "codex"|"claude", "refresh": true }`

**Result:** installation, authentication, subscription, version, safe status
message, last-call usage, and daemon-session usage. Usage separates provider
input, cached input, uncached input, output, Heliox prompt characters, and the
Heliox prompt-token estimate when the CLI reports those fields.

#### `subscription_login`

Start the provider's official browser login command in the background. Heliox
does not receive the browser callback or credential. The UI must call
`subscription_status` after the user completes the provider-owned flow.

**Params:** `{ "provider": "codex"|"claude" }`

**Result:** `{ "status": "started"|"missing"|"error", "message": "..." }`

#### `extract_file_text`
Extract text from a user-selected file for UI context injection.

**Params:** `{ "path": "C:/path/to/document.pdf" }`
**Result:** `{ "status": "ok", "text": "...", ... }` or a clear unsupported/read error.

### Secure productivity integrations

These desktop-only methods configure integrations without putting credentials
in plans or configuration files. Secrets are stored in the operating-system
keyring and are unavailable to the MCP and neural sidecar roles.

| Method | Contract |
|---|---|
| `calendar_test_connection` | Authenticate to the enabled HTTPS CalDAV account and list calendar names without changing events. |
| `email_test_connection` | Authenticate to the enabled IMAP account without reading or changing messages. |
| `ssh_list_hosts` | Return allowlisted aliases and credential-readiness booleans; never key material or provider identifiers. |
| `ssh_set_enabled` | Persist the global SSH integration switch without deleting saved hosts. |
| `ssh_save_host` | Validate and persist one host alias; store its private key/passphrase only in the OS keyring. |
| `ssh_delete_host` | Remove one alias and its associated keyring credentials. |
| `ssh_test_connection` | Authenticate to one allowlisted alias without executing a remote command. |

SSH commands and scripts remain approval-gated system-modification actions even
after a connection test succeeds.

### Air Handoff

Air Handoff is an opt-in, same-LAN, one-target transfer surface. Pairing uses an
ephemeral QR payload; content is encrypted per paired device, and the receiver
exposes no computer-control tools.

| Method | Contract |
|---|---|
| `air_handoff_status` | Return receiver state, paired devices, and the current draft metadata. |
| `air_handoff_set_enabled` | Atomically persist the switch and start/stop the local receiver. |
| `air_handoff_start_pairing` | Create a short-lived QR pairing payload. |
| `air_handoff_cancel_pairing` | Expire the current pairing offer. |
| `air_handoff_revoke_device` | Revoke one paired receiver by daemon-issued device ID. |
| `air_handoff_grab` | Stage a screenshot, bounded text value, or immutable file snapshot. |
| `air_handoff_drop` | Encrypt and deliver the staged draft to one explicitly selected device. |
| `air_handoff_cancel` | Discard the staged draft without sending it. |

---

### Agent Routing & Orchestration

#### `agent_routing`
Dry-run routing analysis: which specialist agent(s) would handle a given input.

**Params:** `{ "input": "write a Python script" }`
**Result:**
```json
{
  "input": "write a Python script",
  "assigned_agents": ["code_agent"],
  "enhanced_prompt": "...",
  "is_multi_agent": false,
  "orchestrator": {
    "input": "write a Python script",
    "assigned_agents": ["code_agent"],
    "assigned_specialists": [
      {
        "agent_key": "CodeAgent",
        "role": "code_agent",
        "display_name": "Code Agent",
        "matches": 2,
        "quality_score": 0.5
      }
    ],
    "is_multi_agent": false,
    "routing_basis": "capability_contract"
  }
}
```

Preview keywords and descriptions explain routing but do not create execution
authority. Actual provider selection also requires exact capability support and
uses persisted callback-observed outcome quality.

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

**Preferred params:** `{ "agent_name": "VisionAgent" }`

`agent_name` is a concrete class discovered by `AgentRegistry`. The legacy
`{ "role": "code_agent" }` form remains for the original compatibility roles
but is not the extensibility path.

**Result:** `{ "status": "spawned", "agent_id": "vision_agent_ab12cd", "agent_name": "VisionAgent" }` or `{ "status": "error", "message": "..." }`

#### `agent_mesh_status`

Return the live provider, authority, budget, routing-quality, and action
coverage contract.

**Params:** `{}`

The current `main` runtime reports 21 executable specialists, 158 declared and
available action types, and an empty `uncovered_action_types` list. Clients must
surface exact uncovered names as a release-blocking warning rather than hiding
coverage regression.

**Result (abbreviated):**

```json
{
  "enabled": true,
  "total_specialists": 21,
  "executable_specialists": 21,
  "registered_action_types": 158,
  "available_action_types": 158,
  "coverage_complete": true,
  "uncovered_action_types": [],
  "specialists": []
}
```

---

### Intelligence, durability, and evolution

These RPCs expose the implemented intelligence layers. They do not bypass the
normal execution or promotion gates.

| Method | Purpose |
|--------|---------|
| `resume_task` | Resume an interrupted durable task using its hashed-capability token |
| `temporal_memory_status` | List provenance-labelled active/candidate facts, episodes, and working state |
| `temporal_memory_retract` | Explicitly retract one fact by `fact_id` |
| `companion_speech_status` | Inspect the single priority speech channel and current owner |
| `risk_gate_status` | Inspect structured, learned-risk, and optional UI-JEPA availability/evidence |
| `online_learning_status` | Inspect verified labels, transitions, replay, drift, routines, corrections, and privacy state |
| `online_learning_reset` | Forget online model state without deleting or mutating the experience ledger |
| `strategy_evolution_status` / `strategy_candidates` | Inspect inert strategy candidates and active assignments |
| `strategy_propose` / `strategy_reflect` | Create inert strategy candidates |
| `strategy_record_isolated` / `strategy_start_shadow` / `strategy_record_shadow` | Supply isolated and shadow evidence |
| `strategy_start_canary` / `strategy_record_canary` | Start and record an explicitly consented canary |
| `strategy_promote` / `strategy_rollback` | Exact-ID human promotion or reversible rollback |
| `evolution_status` / `evolution_runs` / `evolution_candidates` | Inspect the isolated engineering harness |
| `evolution_create_run` / `evolution_generate_candidates` / `evolution_evaluate` | Create and evaluate two-to-eight inert patches in detached Docker worktrees |
| `evolution_request_promotion` | Archive an exact-candidate promotion request for external review; it cannot merge or push |

`resume_task` requires `resume_token`; unauthenticated durable-plan resume is
rejected. `strategy_start_canary` requires an identified `actor` and
`consent_confirmed: true`. Strategy promotion/rollback and evolution promotion
requests require the exact candidate ID in `confirmation`. Automatic promotion
is disabled.

---

### Local MCP Bridge

The local stdio MCP server authenticates to this WebSocket API as the
`mcp_local` role using a separate rotated runtime token. It has an explicit
ten-method allowlist: `health`, `ready`, `system_status`, `capabilities`,
`agent_capabilities`, and the five methods below. It cannot call `execute`,
`confirm`, configuration, credential, neural, or arbitrary RPC methods.

| Method | Purpose |
| --- | --- |
| `mcp_plan_task` | Produce a non-binding, side-effect-free task preview. |
| `mcp_typed_decision` | Optionally score a bounded dynamic yes/no, choice, or ordered-option request with the local research model; never authorizes actions. |
| `mcp_submit_task` | Replan and enqueue an MCP-owned durable task; return immediately. |
| `mcp_task_status` | Read approval, progress, and terminal result for an MCP-owned task. |
| `mcp_cancel_task` | Cancel an MCP-owned task and deny any pending approval wait. |

`mcp_typed_decision` accepts a JSON-object `state`, `kind` (`noul`, `choice`,
or `score`), `question`, and two to 64 unique `options`. It returns a local
research-only score with `execution_authority: false`; it is disabled by
default, requires explicit local artifact paths and a supported CUDA runtime,
never downloads weights, and does not write the request to disk. It cannot feed
the planner or approval pipeline.

`mcp_submit_task` accepts `input` plus an optional `session_id`. It returns a
`task_id` and an accepted/planning state, not an execution result. Every action
is presented in the Heliox UI for confirmation. There is deliberately no MCP
RPC for approval. `mcp_task_status` and `mcp_cancel_task` reject tasks that are
not owned by `mcp-local`.

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

#### `gaze_event`
Feed a coarse gaze-region reading to the fusion engine (see GESTURES.md's
"Gaze Tracking" section). Only ever a region label + confidence — never
raw face landmarks or video.

**Params:**
```json
{ "region": "left", "confidence": 0.72 }
```
`region` is one of `"center"`, `"left"`, `"right"`, `"up"`, `"down"`.

**Result:**
```json
{ "status": "ingested" }
```
`status` is `"ingested"` (the reading was buffered — gaze never itself
produces a fused intent, only voice/gesture do, using gaze as
disambiguating context), `"ignored"` when `config.vision.gaze_tracking_enabled`
is off, or `"error"` when the fusion engine isn't initialized.

#### `multimodal_stats`
Return fusion engine statistics.

**Params:** `{}`
**Result:** fusion engine stats dict.

---

### Neural Intent Research Controls

Daemon RPC uses three authenticated roles. The UI retains the ordinary daemon
token. The one allowed `pilot-neurod` connection authenticates with a separate
short-lived sidecar token and can call only the methods marked Sidecar below.
The local MCP bridge uses its own token and the narrow allowlist documented
above. Role violations return method-not-available before a handler runs.

| Method | Role | Purpose |
|--------|------|---------|
| `neural_status` | UI, Sidecar | Read state, capabilities, fixed goals, quality and bounded audit status |
| `neural_stage_task` | UI | Stage one explicit text-authored goal for later neural focus/select |
| `neural_remove_staged_task` | UI | Remove an unlaunched staged goal by daemon-issued UUID |
| `neural_connect` | Sidecar | Register one strict `NeuralStreamDescriptorV1` session |
| `neural_begin_calibration` | UI | Move the current session into explicit calibration |
| `neural_finish_calibration` | Sidecar | Bind verified artifact, subject pseudonym, decoder version and held-block metrics |
| `neural_arm` | UI | Arm `navigate` or `safe_desktop` with explicit non-neural authorization |
| `neural_intent_preview` | Sidecar | Submit one signed, expiring `NeuralIntentV1` candidate |
| `neural_observation` | Sidecar | Publish a bounded quality/artifact/buffer summary; no samples |
| `neural_commit` | UI | Commit the current preview by UUID and expected state revision |
| `neural_disarm` | UI, Sidecar | Immediately return to observe-only/disconnected safety state |
| `neural_stimulus_marker` | UI | Write one daemon-stamped grid/target marker |
| `neural_stimulus_markers` | Sidecar | Read new markers after a sequence for local recording/epoching |

`neural_connect` params contain `descriptor` with schema version, UUID,
`evidence_kind` (`synthetic`, `recorded_eeg`, `live_eeg`, or legacy `unknown`),
pseudonymous source, transport, sample rate, channel metadata, reference,
optional calibration hash, sequence start, and monotonic start. Unknown fields,
invalid ranges, inconsistent channels, and unsupported versions reject.

`neural_stage_task` accepts a 3-2000 character `goal`, an optional 1-80
character `label`, and an optional bounded conversation `session_id`. At most
eight tasks are held in memory. The response exposes their opaque task and
command IDs, but the sidecar cannot call either staging method or place goal
text in a signed intent. Neural selection therefore launches prior explicit
text rather than treating EEG as a natural-language channel. The launched job
uses the normal autonomous decomposition, specialist routing, permissions,
confirmation, verification, cancellation, and progress notifications.

`neural_finish_calibration`:

```json
{
  "session_id": "6eaf...",
  "calibration_id": "4ad7...",
  "subject_key": "local-subject-1",
  "decoder_version": "34bc...",
  "metrics": {
    "epoch_count": 24,
    "block_count": 6,
    "balanced_accuracy": 0.84,
    "expected_calibration_error": 0.08,
    "per_class_recall": {
      "focus_left": 0.83,
      "focus_right": 0.85,
      "select": 0.82,
      "cancel": 0.86
    }
  }
}
```

`neural_arm` accepts only a current `session_id`, `scope` (`navigate` or
`safe_desktop`), and `user_authorized: true`. `physical_goal` always rejects.

`neural_intent_preview` carries the strict signed envelope under `intent`.
Accepted candidates produce `status: "previewed"` with `preview_id`,
`intent_id`, resolved canonical goal, scope, state revision, cancellation/
expiry timestamps, world-model assessment, non-neural approval requirement,
and a raw-media-free fusion summary. Cancel returns `status: "cancelled"` and
disarms without creating a preview.

`neural_commit`:

```json
{
  "preview_id": "2d7e...",
  "expected_revision": 5,
  "world_model_approved": false
}
```

The UI calls commit only after the preview's cancellation interval. A world-
model warning requires a distinct UI approval and `world_model_approved: true`;
neural repetition is not approval. A missing, stale, expired, raced, replayed,
or already-consumed preview returns `status: "rejected"`. There is no automatic
retry.

`neural_observation` accepts `quality`, `buffered_samples`, `dropped_samples`,
and `observed_at_ns`. `quality` contains only the enum, artifact flags, bounded
metrics, and reasons defined in `SignalQualitySummary`—never raw EEG or feature
vectors.

`neural_stimulus_marker` accepts active `session_id`, optional registered
`target_id`, event (`grid_shown`, `grid_hidden`, `target_on`, `target_off`), and
the client's display-performance timestamp. The daemon assigns authoritative
order and `received_monotonic_ns`. Sidecar reads use
`{"after_sequence": 12}`. The queue is bounded.

See [Neural Intent Research Controls](docs/NEURAL_INTENT.md) for the complete
state machine, security boundary, and honest N0-N3 evidence.

---

### Gesture Cursor Control (browser/dev-mode fallback)

`cursor_move`/`cursor_click` are the **degraded fallback path** for the
continuous gesture-to-cursor bridge (see GESTURES.md) — used when testing
the wiring in a plain browser (`npm run dev`) without a compiled Tauri
binary. The primary, real-time path never touches these RPCs at all: it's a
native Rust Tauri command (`move_gesture_cursor`/`click_gesture_cursor` in
`tauri-app/src-tauri/src/commands.rs`, backed by the `enigo` crate) invoked
directly over the Tauri IPC bridge, in-process, with no WebSocket round-trip.
The daemon path exists only because the daemon's `mouse_move` (pyautogui,
300ms tween + 50ms pause per call) plus a fresh WebSocket connection per
invocation cannot sustain the bridge's ~30fps update rate.

Both bypass Planner/Executor/confirmation entirely — `MOUSE_MOVE`/
`MOUSE_CLICK` are Tier 1 (USER_WRITE), already never requiring confirmation.

#### `cursor_move`
Move the OS mouse cursor to an absolute screen position via
`pilot.system.input_control.mouse_move(x, y, duration=0.0)`.

**Params:** `{ "x": 640, "y": 400 }`
**Result:** `{ "status": "ok", "message": "Moved mouse to (640, 400) [absolute]" }`, or `{ "status": "error", "message": "x/y must be integers" }`.

#### `cursor_click`
Click at a screen position via `pilot.system.input_control.mouse_click(x, y, button="left")` — the gesture-cursor bridge passes the same coordinates it last sent to `cursor_move`.

**Params:** `{ "x": 640, "y": 400 }`
**Result:** `{ "status": "ok", "message": "Clicked (left, 1x) at (640, 400)" }`

---

### Interaction state and daemon speech

#### `interaction_status`
Return the one current text/voice interaction snapshot: `interaction_id`,
`source`, `phase`, redacted/bounded `user_input`, display `message`, `active`,
`sequence`, elapsed time, and update time.

#### `list_audio_input_devices`
List microphone inputs compatible with Heliox's recording format. Stable
identifiers are suitable for `voice.input_device`; unavailable devices return
a clear error instead of silently selecting another microphone.

#### `speak_text`
Send UI text through the daemon's single speech coordinator and configured
Kokoro/Pocket/OS-native engine.

**Params:** `{ "text": "Done", "channel": "final_answer", "dedupe_key": "task:123" }`
**Result:** a `SpeechOutcome` with accepted/completed/interrupted/suppressed status.

#### `stop_speech`
Immediately stop daemon-side speech playback and release the speech channel.

**Params:** `{}`

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

#### `plugin_market_list`
Load the approved GitHub catalog and merge it with installed local-only
plugins. Each item reports `installed`, `local_only`, catalog `source`, and the
submission URL. A catalog/network failure returns an empty list plus `error`;
it never turns an unapproved package into a marketplace item.

#### `plugin_install`
Install one exact catalog-approved, hash-verified package.

**Params:** `{ "plugin_name": "weather" }`
**Result:** the verified installation record, or `{ "error": "..." }`.

#### `plugin_uninstall`
Remove one validated local plugin directory and refresh planner/mesh inventory.

**Params:** `{ "plugin_name": "weather" }`

#### `plugin_create`
Create and locally sign a development plugin. It remains `local_only` until a
reviewed marketplace pull request is merged.

**Params:** `{ "name": "my-plugin", "version": "1.0.0", "description": "...", "author": "...", "tools": [...], "code": "..." }`

#### `plugin_run_tool`
Run an installed tool through the plugin registry's capability broker.

**Params:** `{ "tool_name": "get_weather", "args": { "city": "Delhi" } }`
**Result:** `{ "result": ... }` or a clear registry/tool error.

See [Plugin Marketplace](docs/PLUGIN_MARKETPLACE.md) for moderation,
capabilities, package hashes, and runtime isolation.

---

### Skill Registry

#### `skills_list`
Return all currently loaded declarative skills.

#### `skills_reload`
Reload configured skill search directories, refresh planner context, and return
one success/error record per source file.

#### `skills_load_report`
Return the last skill-load records and the exact search directories used.

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

### Cognitive Intelligence

#### `cognitive_stats`
Return statistics for all cognitive subsystems.

**Params:** `{}`
**Result:**
```json
{
  "cognitive_engine": { ... },
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

#### `cognitive_model_toggle`
Load, unload, or query the cognitive engine.

**Params:** `{ "action": "load" }` — `action` is `"load"`, `"unload"`, or `"status"` (default)
**Result:**
```json
{ "loaded": true, "fallback": false, "available": true }
```

---

### Voice Listener (JARVIS Mode)

#### `voice_listener_start`
Start the continuous voice listener. Wake words remain supported, but
`voice.continuous_conversation_enabled` defaults to true, so complete
utterances can be routed without repeating the wake phrase while listening is
on. The listener suppresses Heliox's own TTS and uses a bounded follow-up
window after answers.

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

#### `voice_recognition_test`
Capture one utterance and return its transcript without executing it. If
continuous listening is off, a temporary listener is started for the test and
stopped afterwards.

**Params:** `{ "timeout_seconds": 12 }` (clamped to 3-30 seconds)
**Result:** `{ "status": "ok", ... }` with the listener's diagnostic transcript
fields, or `{ "status": "error", "message": "..." }` when nothing was recognized
before the timeout.

Recognized commands enter the same `execute` handler and observable
`interaction_state` phases as typed commands. During active work, ordinary
speech is a live correction; “quick question …” receives an answer without
interrupting, and “next task …” queues a new guarded task.

---

### Voice Calibration (on-device continual learning)

`reset_wake_calibration`/`list_wake_variants` back the Settings → Voice
Calibration UI (see GESTURES.md's gesture-side "Gesture Calibration" for the
parallel frontend feature). Both are direct handlers, bypassing Planner/
Executor entirely — same pattern as `cursor_move`/`cursor_click` above.

The underlying `WakeWordCalibrator` (`daemon/pilot/system/voice_calibration.py`)
is a fallback tried only after `_listen_loop()`'s fixed exact-substring
wake-word match misses: it learns accent/mic-specific near-miss transcripts
that are followed shortly after by a real wake-word hit, and once a variant
has accumulated `PROMOTION_THRESHOLD` (5) such confirmations it's trusted
going forward. Storage is a local JSON file at
`~/.cache/heliox/voice_calibration/wake_variants.json` — no audio, no general
transcripts, nothing transmitted anywhere.

#### `reset_wake_calibration`
Clear all learned wake-word variants and delete the on-device store. If a
voice listener is currently running, its live calibrator is reset in place
so the change takes effect without restarting the listener.

**Params:** `{}`
**Result:** `{ "status": "ok" }`

#### `list_wake_variants`
List learned wake-word variants for the Settings transparency view.

**Params:** `{}`
**Result:**
```json
{
  "status": "ok",
  "variants": [
    { "text": "hey iliox", "confirmed_count": 3, "first_seen": 1234567890.0, "last_confirmed": 1234567999.0 }
  ],
  "promotion_threshold": 5
}
```

---

### Autonomous Executor (Background Jobs)

#### `autonomous_submit`
Submit a goal for fire-and-forget autonomous background execution.

**Params:** `{ "goal": "organize my Downloads folder", "source": "text", "session_id": "chat_123", "scope_override": null }`
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

Each interactive browser/desktop step uses the bounded adaptive app loop: a
fresh observation, one grounded action or pair, real verification, then
replanning when the goal is not complete. Jobs stop on verified completion,
cancellation, repeated no progress, or six rounds per step.

---

### Recorded Workflows

Opt-in recording and guarded replay of desktop workflows (see
[`docs/RECORDED_WORKFLOWS.md`](docs/RECORDED_WORKFLOWS.md)). Recording needs
explicit consent, observing manual desktop input needs a second consent, and
replay is submitted to the guarded autonomous executor. Errors return
`{ "status": "error", "message": "..." }`.

| Method | Params | Result |
| --- | --- | --- |
| `recorded_workflow_status` | `{}` | `active`, `name`, `capture_visual`, `capture_manual`, `step_count`, `sample_count`, `omitted_actions`, `visual_gaps`, `manual_hook_healthy` |
| `recorded_workflow_start` | `{ "consent": true, "name": "...", "goal": "...", "capture_visual": false, "capture_manual": false, "manual_consent": false }` | the new recording session |
| `recorded_workflow_stop` | `{ "save": true }` | `saved` plus the stored workflow, or the discarded session's status |
| `recorded_workflow_list` | `{}` | `workflows` |
| `recorded_workflow_get` | `{ "workflow_id": "..." }` | `workflow` |
| `recorded_workflow_delete` | `{ "workflow_id": "..." }` | `{ "status": "ok" }` |
| `recorded_workflow_run` | `{ "workflow_id": "...", "reviewed": false, "text_values": {}, "capture_visual": false }` | `{ "status": "submitted", "job": {...} }` |
| `recorded_workflow_feedback` | `{ "workflow_id": "...", "job_id": "...", "outcome": "..." }` | updated `workflow`, accepted only after that run finished |
| `recorded_workflow_schedule` | `{ "workflow_id": "...", "run_at": "<future ISO 8601 time with zone>", "repeat_seconds": 0, "reviewed": false }` | updated `workflow`; repeats need 5 minutes to 1 year |
| `recorded_workflow_unschedule` | `{ "workflow_id": "..." }` | updated `workflow` |

Replay is refused while a recording is active, for incomplete recordings, and
for workflows with manual or coordinate-based steps unless `reviewed` is true.

### Durable Voice/Gesture Workflows

These workflows persist multi-step state, expose pause/resume/cancel controls,
and delegate application steps to the same adaptive loop as autonomous jobs.
Spoken instructions while one is running can revise or stop it.

#### `voice_gesture_workflow_submit`
**Params:** `{ "goal": "open Hermes and draft a note", "invocation_source": "voice"|"gesture", "scope_override": null }`
**Result:** `{ "status": "submitted", "workflow": { ... } }`

#### `voice_gesture_workflow_list`
**Params:** `{ "include_terminal": false }`
**Result:** `{ "workflows": [...] }`

#### `voice_gesture_workflow_get`
**Params:** `{ "workflow_id": "vgw_..." }`

#### `voice_gesture_workflow_pause`
Pause at the next step boundary.

**Params:** `{ "workflow_id": "vgw_..." }`

#### `voice_gesture_workflow_resume`
Resume a paused or trigger-waiting workflow.

**Params:** `{ "workflow_id": "vgw_..." }`

#### `voice_gesture_workflow_cancel`
Cancel the workflow and clear its task-scoped working memory.

**Params:** `{ "workflow_id": "vgw_..." }`

#### `gesture_workflow_bindings_get`
Return the Settings policy: global enabled state, supported gestures, and the
current gesture-to-goal templates.

#### `gesture_workflow_bindings_update`
Validate and persist the global enabled state and binding list. Unsupported or
duplicate gesture names, empty goals, and malformed bindings fail closed.

**Params:** `{ "enabled": true, "bindings": [{ "gesture_name": "peace", "goal_template": "Open my calendar", "enabled": true }] }`

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

#### `proactive_learning_status`
Return per-pattern accept/dismiss evidence, learned priority/timing, and
temporary suppression state stored on-device.

#### `proactive_learning_reset`
Forget learned proactive-suggestion preferences and return the reset status.
This does not delete the immutable experience ledger.

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

### Hybrid Risk World Model

#### `risk_gate_status`
Return whether evaluation is enabled, whether validated learned weights and
optional UI-JEPA artifacts loaded, training sample/action coverage, model
version, deterministic fallback state, and the latest plan evaluation.

#### `risk_gate_config_update`
Enable or disable learned/structured risk evaluation and persist the choice.
The deterministic permission and safety floor remains active either way.

**Params:** `{ "enabled": true }`
**Result:** the updated `risk_gate_status` object.

---

### Autonomous Healing Engine

Passive system-health monitoring (see `pilot.agents.autonomous_healing`) that plans a remediation goal when a `monitor_cpu`/`monitor_memory`/`monitor_disk` background task above triggers. Off by default (`self_healing.enabled`). Low-tier/reversible plans auto-execute; anything else is broadcast as a `self_healing_confirmation_required` notification and resolved via the existing `confirm` RPC (same `plan_id`/`PendingConfirmation` mechanism threat containment uses) — there is no dedicated approve/reject RPC.

#### `self_healing_status`
Report current config plus recent remediation attempts.

**Params:** `{}`
**Result:**
```json
{
  "enabled": false,
  "auto_execute_max_tier": 1,
  "watched_metrics": ["cpu", "memory", "disk"],
  "attempts": [
    {
      "attempt_id": "heal_a1b2c3d4",
      "metric": "disk",
      "trigger": { "disk_percent": 92.1, "triggered": true, "message": "Disk usage at 92.1%!" },
      "goal": "The system is running low on disk space ...",
      "plan_id": "heal_a1b2c3d4",
      "outcome": "auto_executed",
      "max_tier": 0,
      "irreversible": false,
      "explanation": "...",
      "created_at": 1731000000.0,
      "resolved_at": 1731000001.2
    }
  ]
}
```
`outcome` is one of `auto_executed`, `proposed`, `confirmed`, `denied`, `timed_out`, `no_action`, `plan_error`.

#### `self_healing_config_update`
Update self-healing config. Persists via `config.save()`.

**Params:** any of `{ "enabled": true, "auto_execute_max_tier": 1, "cooldown_seconds": 600.0, "confirm_timeout_seconds": 300.0, "watched_metrics": ["cpu", "memory", "disk"] }`
**Result:** `{ "status": "ok", "enabled": true, "auto_execute_max_tier": 1, "cooldown_seconds": 600.0, "confirm_timeout_seconds": 300.0, "watched_metrics": [...] }`

#### Notifications
- `self_healing_auto_executing` / `self_healing_complete` — broadcast around the auto-exec path, attempt dict payload.
- `self_healing_confirmation_required` — attempt dict plus `actions` (full action list) and `timeout_seconds`; resolve via `confirm` with the attempt's `plan_id`.
- `self_healing_denied` / `self_healing_timeout` — the propose-and-wait branch ended without executing.

---

### Live Execution Narrator

Narrates plan execution as it happens and can pre-emptively pause a plan or a single browser action flagged as risky, right before it runs (see `pilot.agents.narrator`). Off by default (`narration.enabled`). Ambient narration (`execution_narration`) is always non-blocking; risk-triggered interrupts reuse the existing `confirm` RPC / `PendingConfirmation` mechanism, same as Autonomous Healing above — there is no dedicated approve/reject RPC for this either.

#### `narration_status`
Report current narration config.

**Params:** `{}`
**Result:**
```json
{
  "enabled": false,
  "narrate_steps": true,
  "interrupt_on_risk": true,
  "confirm_timeout_seconds": 120.0
}
```

#### `narration_config_update`
Update narration config. Persists via `config.save()`.

**Params:** any of `{ "enabled": true, "narrate_steps": true, "interrupt_on_risk": true, "confirm_timeout_seconds": 120.0 }`
**Result:** `{ "status": "ok", "enabled": true, "narrate_steps": true, "interrupt_on_risk": true, "confirm_timeout_seconds": 120.0 }`

#### Notifications
- `execution_narration` — non-blocking, fired around each action's start/completion. Payload: `{ "text": "...", "plan_id": "..." }`.
- `execution_interrupt` — a plan (via the Agent Gateway's critic verdict), a single browser action (via the pre-execution target assessment), or an autonomous action's "simulate before executing" preview was flagged/surfaced before running. Payload: `{ "plan_id": "interrupt_xxxxxxxx", "reason": "...", "kind": "plan_risk"|"target_assessment"|"action_preview", "timeout_seconds": 120.0 }`; resolve via `confirm` with the same `plan_id`. When `kind == "action_preview"` (see **Simulate Before Executing** in SECURITY.md), the payload also carries a `preview` object: `{ "screenshot_base64": "...", "bbox": { "x": 10, "y": 20, "w": 80, "h": 30 } | null, "target_label": "Submit button" | null, "caption": "About to click: Save", "dom_diff": { "change_score": 0.42, "summary": "...", ... } | null }` — `bbox` is in pixel coordinates relative to the screenshot image; `dom_diff` is only present for the 5 browser action types and only when a browser session was already open.
- `execution_interrupt_timeout` / `execution_interrupt_denied` — the interrupt-and-wait ended without confirmation (timed out, or the user chose Stop); `plan_id` matches the originating `execution_interrupt`.

---

### User Manual Supervision

Watches the user's OWN independent screen/keyboard/mouse activity — never anything Heliox itself executes, see `pilot.agents.user_supervision`. Off by default (`supervision.enabled`), with `supervision.keyboard_mouse_hook_enabled` as a separate, starker opt-in for the global keyboard/mouse hook. Advisory only — unlike the Narrator's `execution_interrupt`, there is no blocking gate here and no dedicated approve/reject RPC, since Heliox cannot intercept the user's own OS-level input.

#### `supervision_status`
Report current supervision config plus whether the keyboard/mouse hook (if enabled) is actually still alive.

**Params:** `{}`
**Result:**
```json
{
  "enabled": false,
  "keyboard_mouse_hook_enabled": false,
  "cognitive_coaching_enabled": true,
  "risk_pattern_detection_enabled": true,
  "hook_healthy": false
}
```

#### `supervision_config_update`
Update supervision config. Unlike `narration_config_update`/`self_healing_config_update`, this handler actually starts/stops the background task and the keyboard/mouse hook on an `enabled`/`keyboard_mouse_hook_enabled` transition, not just a config-field flip — the thing being gated has real cost and privacy weight even when idle.

**Params:** any of `{ "enabled": true, "keyboard_mouse_hook_enabled": true, "cognitive_coaching_enabled": true, "risk_pattern_detection_enabled": true, "tick_interval_seconds": 1.5, "ocr_interval_seconds": 8.0, "stress_coaching_threshold": 0.75, "cognitive_load_coaching_threshold": 0.8, "coaching_cooldown_seconds": 900.0, "risk_cooldown_seconds": 30.0, "keystroke_buffer_max_chars": 256, "ocr_snippet_max_chars": 400 }`
**Result:** `{ "status": "ok", "enabled": true, "keyboard_mouse_hook_enabled": true, "cognitive_coaching_enabled": true, "risk_pattern_detection_enabled": true, "hook_healthy": true }`

#### Notifications
- `supervision_cognitive_checkin` — a sustained stress/cognitive-load threshold crossing from a real OCR snippet + window-title stimulus fed to `CognitiveEngine.predict_cognitive_state()`. Payload: `{ "message": "...", "attention_score": 0.4, "stress_level": 0.9, "cognitive_load": 0.5 }`.
- `supervision_risk_warning` — the OCR snippet or the keystroke hook's buffer matched a known destructive-action pattern. Payload: `{ "pattern": "destructive_shell_command", "source": "ocr"|"keystroke", "message": "..." }` — deliberately never includes the matched text itself, only the pattern's name.

---

### Operations and audit status

#### `budget_stats`
Return the current month's recorded model token and cost summary.

#### `budget_reset`
Delete current-month token-usage records. This is a destructive settings
operation in the UI even though it does not affect task memory.

#### `mesh_peers`
Return connected LAN peers with hostname, execution capability, CPU load, and
plugin count. When mesh networking is unavailable, returns
`{ "enabled": false, "peers": [] }`.

#### `mesh_status`
Return LAN mesh configuration, node identity, connection count, and readiness.

#### `mesh_configure`
Persist and apply Peer Mesh settings in one step. A supplied `shared_secret`
must be 32-512 bytes with no leading or trailing whitespace and is stored in the
OS vault; enabling the mesh requires a saved secret.

**Params:** `{ "enabled": true, "skill_sync_enabled": false, "collab_exec_enabled": false, "shared_secret": "<optional>" }`

#### `mesh_generate_secret`
Return a new random secret for the user to copy to trusted peers. Nothing is
stored until `mesh_configure` saves it.

**Params:** `{}`
**Result:** `{ "status": "ok", "shared_secret": "..." }`

#### `mesh_clear_secret`
Delete the stored mesh secret. Refused while Peer Mesh is enabled.

**Params:** `{}`
**Result:** `{ "status": "ok" }`

#### `get_plan_history`
Return paginated plan-audit summaries, distinct from chat history.

**Params:** `{ "limit": 50, "offset": 0, "status": "success" }`

#### `get_plan_detail`
Return the complete stored plan, critic, result, and verification record for
one `plan_id`.

**Params:** `{ "plan_id": "a3b2c1f5" }`

#### `threat_containment_stats`
Return whether the `ThreatContainmentBridge` is wired and the count of pending
confirmation gates.

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

Common `phase` values are `"receiving input"`, `"recalling memory"`,
`"routing agents"`, `"planning"`, `"companion reviewing plan"`,
`"companion revising plan (...)"`, `"critic review"`, `"executing"`,
`"verifying"`, `"re-planning (attempt ...)"`,
`"retrying — previous attempt failed"`, `"revising after your correction"`,
`"resuming"`, and `"aborted"`. Clients must tolerate new human-readable
phase strings and use `interaction_state.phase` for stable state-machine logic.

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
Sent when one or more actions in the plan require explicit user approval (Tier 2+ actions, or any action flagged `irreversible` regardless of tier). The `execute` handler blocks until a matching `confirm` request arrives or the 5-minute timeout expires.

```json
{
  "plan_id": "a3b2c1f5",
  "actions": [
    {
      "action_type": "package_remove",
      "target": "python3",
      "requires_confirmation": true,
      "destructive": true,
      "irreversible": true,
      "index": 0
    }
  ]
}
```

`index` is the action's position in `plan.actions` — pass it back (as part of `approved_indices`) in the `confirm` call for per-action granular approval. `irreversible` is true when the action can't be undone via snapshot rollback even if it isn't tier-"destructive" (e.g. an email send).

The UI should present an approval dialog and call `confirm` with the `plan_id`.

---

### `critic_verdict`
Sent when a Tier 4 (root-critical), Tier 3 (destructive), or irreversible-flagged plan was reviewed by the secondary LLM safety critic — broadcast before the `confirm_required` gate. A `BLOCK` verdict aborts the plan entirely (the `execute` response is `{"status": "blocked_by_critic", ...}` and no `confirm_required`/execution ever fires); `WARN`/`APPROVE` fall through to the normal confirmation gate.

```json
{
  "verdict": "WARN",
  "risk_score": 0.55,
  "issues": ["Plan requests root access broader than the stated task needs"],
  "safe_actions": ["file_delete"],
  "flagged_actions": ["service_restart"],
  "recommendation": "Proceed with caution — root scope is wider than necessary."
}
```

`verdict` is `"APPROVE"`, `"WARN"`, or `"BLOCK"`. A low-risk Tier 3 plan (no Tier 4 actions, heuristic risk score below threshold) skips the LLM round-trip entirely; in that case the payload instead looks like:

```json
{
  "verdict": "SKIPPED",
  "risk_score": 0.1,
  "issues": [],
  "safe_actions": [],
  "flagged_actions": [],
  "recommendation": "Low-risk heuristic — LLM safety review was skipped.",
  "critic_skipped": "low_risk_heuristic"
}
```

---

### `rollback_complete`
Sent after a successful `rollback_plan` call.

```json
{
  "plan_id": "a3b2c1f5",
  "snapshot_id": "/.snapshots/pilot-a3b2c1f5-20260717-120000",
  "message": "Rollback snapshot created from ... . Reboot to apply."
}
```

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

### `interaction_state`
The current observable state shared by typed and spoken interaction. This
replaces the removed thought-graph/ReAct visualizer and never exposes hidden
model chain-of-thought.

```json
{
  "interaction_id": "5e8a...",
  "source": "voice",
  "phase": "acting",
  "user_input": "open Hermes and draft a note",
  "message": "Executing verified actions",
  "active": true,
  "sequence": 5,
  "elapsed_ms": 1240,
  "updated_at": 1785820000.2
}
```

`phase` is one of `idle`, `listening`, `understanding`, `planning`,
`awaiting_approval`, `acting`, `verifying`, `correcting`, `speaking`,
`completed`, `interrupted`, or `failed`.

### `task_complete`
Terminal, sanitized summary for an interactive task.

```json
{
  "status": "success",
  "summary": "Opened Hermes and verified the requested text.",
  "duration_ms": 4310,
  "dry_run": false,
  "plan_id": "a3b2c1f5"
}
```

### Companion notifications

- `companion_plan_review` — independent `APPROVE`, `WARN`, `REVISE`, or
  `STOP` assessment before execution.
- `companion_plan_intervention` — a warning, automatic bounded revision, or
  terminal stop caused by that review.
- `companion_interjection` — confirms that a typed/spoken live correction or
  stop request was accepted.
- `companion_revision_started` / `companion_revision_rejected` — reports
  bounded replanning or exhaustion of the correction limit.
- `companion_follow_up` — session-scoped grounded next ideas generated after
  the verified result; it is delivered asynchronously and never delays the
  task's terminal response.
- `conversation_response` — a valid zero-action conversational answer.
- `queued_task_added` / `queued_task_started` / `queued_task_result` — queue
  position, execution start, and terminal result for an additional interactive
  task. The result carries the normal action and verification response but
  excludes the resume token.

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
{ "status": "wake_detected", "transcript": "hey heliox open the terminal" }
```

`status` is one of:
- `"wake_detected"` — a wake word was heard; `transcript` is the full utterance it came from.
- `"listening"` — no command followed the wake word in the same utterance; waiting for a follow-up (`message` is a user-facing prompt).
- `"timeout"` — no follow-up command was heard within the wait window (`message` is a user-facing note).
- `"interrupted"` — the user started talking while Heliox was still speaking its response, so playback was cut off (barge-in — see `pilot.system.voice.speak_interruptible`). Only fires when `config.voice.barge_in_enabled` is on and the continuous VAD recorder is active.

---

### `voice_result`
Result of a voice-triggered command execution.

```json
{ "command": "open the terminal", "status": "success", "result": "Terminal opened." }
```

On error: `{ "command": "...", "status": "error", "message": "..." }`

For durable voice workflows, `status` may be `submitted`, `revising`, or
`cancelled` and the payload may include `workflow` or
`coordinated_correction: true`.

### Autonomous and workflow progress

- `autonomous_started`, `autonomous_decomposed`, `autonomous_step_start`,
  `autonomous_step_complete`, `autonomous_complete`, and
  `autonomous_cancelled` carry the current autonomous job object.
- `voice_gesture_workflow_state` carries the complete persisted workflow state
  whenever it changes.
- `gesture_workflow_bindings_updated` carries the validated binding policy.
- `proactive_suggestion` carries a visible optional suggestion; it does not
  execute until accepted through the guarded path.

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

### Neural notifications

- `neural_status` — active state/session/scope and fixed capability summary.
- `neural_observation` — bounded signal quality, artifacts, buffered and
  dropped counts.
- `neural_preview` — visible candidate, cancellation/expiry timestamps,
  resolved goal, fusion summary, and world-model caution.
- `neural_navigation` — committed dedicated-UI focus/select result.
- `neural_result` — fixed safe-desktop plan ID, verified action results, and
  `retry_allowed: false`.
- `neural_disarmed` — terminal safety transition and reason.

The daemon never broadcasts sidecar credentials, signatures, raw EEG, feature
vectors, or recording keys. The sidecar itself is excluded from ordinary UI
broadcast recipients.

---

### `feature_announcement`
Emitted once on startup when a new daemon version introduces new capabilities.

```json
{ "message": "New Heliox capabilities are now available.", "version": "0.13.0" }
```

The exact message is assembled from the daemon changelog; clients must treat
`version` as authoritative instead of parsing a version from `message`.

---

### `daemon_speech`
Display-only pairing for text accepted by the daemon's
`CompanionSpeechCoordinator` (for example, a cognitive-stress pause or
autonomous-job completion). The coordinator serializes daemon callers under the
shared priority contract. The frontend renders `text` as a chat bubble and
must not replay it through browser `speechSynthesis`; that path is only a
fallback when daemon speech did not own the utterance.

```json
{ "text": "Your focus state is low. Confirming in 10 seconds.", "source": "stress_gate" }
```

`source` is one of `stress_gate` (executor.py's cognitive-stress-gate pause) or `autonomous_job` (AutonomousExecutor's end-of-job announcement).

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
  "rollback_action": null,
  "dangerous_flags": []
}
```
`dangerous_flags` is populated by `ActionValidator` when a shell command matches a high-risk argument pattern (recursive+force delete, wildcard/root path target, etc.) even though the base command itself was already allowed — a non-empty list escalates the action to irreversible (see below) regardless of tier. This is defense-in-depth, not a guarantee (argv pattern-matching is inherently incomplete).

**`action_type` values** — grouped by permission tier:

| Tier | Action types |
|------|-------------|
| Read-only (63) | `file_read`, `file_list`, `file_search`, `directory_summary`, `directory_size`, `file_hash`, `file_compare`, `git_status`, `git_diff`, `git_log`, `package_search`, `service_status`, `gnome_setting_read`, `open_url`, `open_application`, `notify`, `process_list`, `process_info`, `clipboard_read`, `system_info`, `disk_usage`, `memory_usage`, `cpu_usage`, `network_info`, `battery_info`, `schedule_list`, `env_get`, `env_list`, `window_list`, `volume_get`, `brightness_get`, `screenshot`, `wifi_list`, `disk_list`, `user_list`, `user_info`, `registry_read`, `log_analyze`, `mouse_position`, `screen_ocr`, `screen_find_text`, `screen_analyze`, `screen_element_map`, `browser_extract`, `browser_extract_table`, `browser_extract_links`, `browser_screenshot`, `browser_list_tabs`, `browser_wait`, `browser_page_info`, `trigger_list`, `calendar_parse`, `calendar_list_events`, `file_parse`, `file_search_content`, `api_scrape`, `workspace_index`, `workspace_search`, `email_fetch`, `email_summarize`, `calendar_fetch`, `calendar_reconcile`, `screen_detect_elements` |
| User write (47) | `file_write`, `file_move`, `file_copy`, `git_resolve`, `git_branch`, `git_stage`, `dbus_call`, `pty_exec`, `clipboard_write`, `power_sleep`, `power_lock`, `env_set`, `window_focus`, `window_minimize`, `window_maximize`, `volume_set`, `volume_mute`, `brightness_set`, `download_file`, `mouse_click`, `mouse_double_click`, `mouse_right_click`, `mouse_move`, `mouse_drag`, `mouse_scroll`, `keyboard_type`, `keyboard_press`, `keyboard_hotkey`, `keyboard_hold`, `browser_hover`, `browser_scroll`, `browser_new_tab`, `browser_close_tab`, `browser_switch_tab`, `browser_back`, `browser_forward`, `browser_refresh`, `browser_close`, `trigger_create`, `trigger_delete`, `trigger_start`, `trigger_stop`, `code_generate_and_run`, `api_request`, `api_github`, `wasm_call`, `skill_run` |
| System modify (35) | `file_permissions`, `git_commit`, `git_push`, `package_install`, `package_update`, `service_start`, `service_stop`, `service_restart`, `service_enable`, `service_disable`, `gnome_setting_write`, `shell_command`, `shell_script`, `schedule_create`, `wifi_connect`, `wifi_disconnect`, `disk_mount`, `registry_write`, `browser_navigate`, `browser_click`, `browser_click_text`, `browser_type`, `browser_select`, `browser_fill_form`, `code_execute`, `calendar_sync`, `calendar_create_event`, `api_send_email`, `api_webhook`, `api_slack`, `api_discord`, `email_reply`, `ssh_command`, `ssh_script`, `plugin_call` |
| Destructive (11) | `file_delete`, `package_remove`, `process_kill`, `power_shutdown`, `power_restart`, `power_logout`, `schedule_delete`, `window_close`, `disk_unmount`, `browser_execute_js`, `calendar_delete_event` |
| Root critical | Any action carrying validated `requires_root: true`; root critical is an action-instance elevation, not a fixed enum list |

Actions at **Tier 2 (System modify) and above** always set `requires_confirmation: true`. This safety boundary is not configurable.

**Irreversibility is orthogonal to tier.** `Action.is_irreversible` (computed, not sent as a raw field on `plan_preview`/`action_start` — surfaced explicitly as `irreversible` in the `confirm_required` payload) is `true` when the action cannot be undone via `rollback_plan`, regardless of its tier:
- Any Tier 4 (root-critical) action
- `api_send_email`, `api_webhook`, `api_slack`, `api_discord`, `email_reply` — external communications, Tier 2, can't be recalled once sent
- `ssh_command`, `ssh_script` — Tier 2, remote hosts outside the local snapshot's reach
- `power_shutdown`, `power_restart`, `power_logout` — Tier 3, can't be "rolled back" once they take effect
- `package_remove` — Tier 3, can lose config/data a simple reinstall won't restore
- `browser_execute_js` — Tier 3, arbitrary page-context code can exfiltrate before local rollback
- `git_push` — Tier 2, remote repository state is outside the local snapshot
- Any action with a non-empty `dangerous_flags` list (see above)

These always require confirmation even if some future policy toggle allowed auto-approving lower tiers.

---

### PermissionAuditEvent object
Returned by `list_permission_events`. Defined in `daemon/pilot/security/permission_audit.py` (`PermissionEscalationAuditStore`).

```json
{
  "id": 42,
  "timestamp": "2026-07-17T12:00:00+00:00",
  "plan_id": "a3b2c1f5",
  "action_index": 0,
  "action_type": "package_remove",
  "target": "python3",
  "permission_tier": "DESTRUCTIVE",
  "requires_root": false,
  "destructive": true,
  "confirmation_decision": "approved",
  "critic_verdict": { "verdict": "APPROVE", "risk_score": 0.2 },
  "execution_success": true,
  "execution_error": ""
}
```
`confirmation_decision` is one of `"approved"`, `"partially_approved"`, `"denied"`, `"blocked_by_critic"`, or `"n/a"` (dry-run). Every row commits to the previous row's HMAC in an append-only chain — `verify_permission_audit` detects any row that was modified, reordered, or deleted after the fact.

---

### GatewayAuditEvent object
Returned by `list_gateway_events`. Defined in `daemon/pilot/security/gateway_audit.py` (`AgentGatewayAuditStore`) — a separate HMAC chain from `PermissionAuditEvent` above.

```json
{
  "id": 17,
  "timestamp": "2026-07-18T12:00:00+00:00",
  "plan_id": "a3b2c1f5",
  "action_index": 0,
  "action_type": "browser_execute_js",
  "action_family": "browsing",
  "target": "document.cookie",
  "source_profile": "autonomous",
  "permission_tier": "DESTRUCTIVE",
  "override_applied": false,
  "override_restricted": false,
  "decision": "denied",
  "denial_reason": "browser_execute_js is denied for source 'autonomous' by gateway policy.",
  "dry_run": false,
  "execution_success": null,
  "execution_error": "",
  "policy_snapshot": { "max_tier": {...}, "deny_action_types": [...], "allow_root": false }
}
```
`action_index` is `-1` for a plan-level row (currently only the `DestructiveCriticAgent` BLOCK verdict, tagged `action_type: "__critic_review__"`) rather than a specific action. `decision` is `"allowed"` or `"denied"`. `override_restricted` is `true` only when a per-task `scope_override` actually narrowed the source's floor for this action (not merely present). Every row commits to the previous row's HMAC — `verify_gateway_audit` detects tampering the same way `verify_permission_audit` does for its own chain.

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

The following sequence shows all messages for a typical `execute` call that contains a Tier 3+ (destructive) action requiring the safety critic and user confirmation.

```
UI                                          Daemon
│                                               │
│── execute {input, dry_run} ──────────────────►│
│                                               │
│◄── notification: status {phase: "receiving input"}
│◄── notification: interaction_state {phase: "understanding"}
│◄── notification: status {phase: "recalling memory"}
│◄── notification: status {phase: "routing agents"}
│◄── notification: agent_routing {assigned_agents, is_multi_agent}
│◄── notification: status {phase: "planning"}
│                                               │  (LLM generates plan)
│◄── notification: plan_preview {plan_id, actions, explanation}
│                                               │
│◄── notification: status {phase: "critic review"}
│                                               │  (Tier 3+/irreversible plan;
│                                               │   low-risk Tier 3 skips this)
│◄── notification: critic_verdict {verdict, risk_score, ...}
│         (BLOCK aborts here — execute response is "blocked_by_critic")
│                                               │
│◄── notification: confirm_required {plan_id, actions}
│         (UI shows approval dialog, per-action checkboxes)
│── confirm {plan_id, confirmed: true, approved_indices: [...]} ►│
│                                               │
│◄── notification: status {phase: "executing"}
│◄── notification: interaction_state {phase: "acting"}
│◄── notification: orchestrator_routing {assigned_agents}
│                                               │  (snapshot taken first if
│                                               │   plan_requires_snapshot)
│                                               │  (for each approved action:)
│◄── notification: action_start {action}        │
│◄── notification: action_complete {result}     │  (result.snapshot_id set
│                                               │   if a snapshot was taken)
│                                               │
│◄── notification: status {phase: "verifying"}
│◄── notification: interaction_state {phase: "verifying"}
│                                               │
│◄── notification: task_complete {status, summary, duration_ms}
│◄── response: execute result ─────────────────┤
│   {status, results, verification, explanation}│
│                                               │
│  ... later, if the user wants to undo ...     │
│── rollback_plan {plan_id} ───────────────────►│
│◄── notification: rollback_complete {plan_id, snapshot_id, message}
```

The UI receives observable stage and outcome telemetry, not hidden model
chain-of-thought. A voice request follows this same sequence and additionally
emits `voice_command`/`voice_result` plus listening/speaking state.

If verification fails, the daemon re-plans and the cycle repeats (up to 2 retries), broadcasting `status: "re-planning (attempt 2)"` before the next `plan_preview`.

---

## Source References

| File | Contents |
|------|----------|
| `daemon/pilot/server.py` | All request handlers and notification senders |
| `daemon/pilot/actions.py` | `ActionType` enum, parameter models, `is_irreversible`/`dangerous_flags` |
| `daemon/pilot/config.py` | `PilotConfig`, `ModelConfig`, `SecurityConfig`, `GestureCursorConfig`, `AdaptiveCalibrationConfig` |
| `daemon/pilot/system/input_control.py` | `mouse_move`/`mouse_click` — backing implementation for the `cursor_move`/`cursor_click` fallback |
| `daemon/pilot/system/voice_calibration.py` | `WakeWordCalibrator`, `VoiceCalibrationStore` — backing implementation for `reset_wake_calibration`/`list_wake_variants` |
| `tauri-app/src-tauri/src/commands.rs` | `move_gesture_cursor`/`click_gesture_cursor` — the primary, real-time gesture-cursor path (enigo) |
| `tauri-app/ui/src/lib/gesture/spatialModel.ts` | `predictAhead()`/`predictCursorTarget()` — the kinematic prediction feeding the cursor bridge |
| `daemon/pilot/security/permissions.py` | `PermissionChecker` — single source of truth for confirmation/snapshot policy |
| `daemon/pilot/security/sanitizer.py` | Command whitelist, dangerous-argument pattern detection |
| `daemon/pilot/agents/destructive_critic.py` | `DestructiveCriticAgent`, `heuristic_risk()` (Tier-3 LLM-review skip heuristic) |
| `daemon/pilot/system/snapshots.py` | `SnapshotManager` — create/rollback/list snapshots |
| `daemon/pilot/security/permission_audit.py` | `PermissionEscalationAuditStore` — tamper-evident HMAC-chained audit log |
| `daemon/pilot/security/gateway.py` | `AgentGateway`, `InvocationSource`, `SourceProfile`, `resolve_effective_profile()` — source-scoped permission floors |
| `daemon/pilot/security/gateway_audit.py` | `AgentGatewayAuditStore` — separate tamper-evident HMAC-chained audit log for gateway decisions |
| `daemon/pilot/intelligence/experience.py` | Append-only event schema, redaction, causality, and idempotency |
| `daemon/pilot/workflows/durable_tasks.py` | Durable task states, resume capabilities, approvals, and execution claims |
| `daemon/pilot/memory/store.py` | Temporal facts, evidence, contradiction, retraction, and bounded context |
| `daemon/pilot/system/companion_speech.py` | Shared daemon speech priority and ownership coordinator |
| `daemon/pilot/intelligence/world_model.py` | Structured prediction contract and hybrid risk fusion |
| `daemon/pilot/intelligence/online_learning.py` | Verified River learning, replay, drift, and reset |
| `daemon/pilot/intelligence/strategy_evolution.py` | Inert strategy candidate lifecycle and assignments |
| `daemon/pilot/intelligence/evolution_harness.py` | Detached-worktree/Docker engineering evaluation archive |
| `daemon/pilot/agents/agent_mesh.py` | Specialist capability, resource, budget, routing-quality, delegation, and coverage contracts |
| `daemon/pilot/neural/` | Neural contracts, acquisition, calibration/decoder, intent gate, controller, provenance, recording, and sidecar bridge |
| `daemon/pilot/security/rpc_identity.py` | Separate UI/sidecar authentication and sidecar method allow-list |
| `tauri-app/src-tauri/src/commands.rs` | Native neural sidecar lifecycle and recording export commands |
| `tauri-app/ui/src/lib/stores/neural.ts` | Neural UI state, lifecycle, calibration, arming, commit, and disarm |
| `daemon/pilot/plugins/` | Capability validation and constrained native/WASM execution |
| `daemon/pilot/system/interaction.py` | Shared text/voice interaction phases and acknowledgement contract |
| `daemon/pilot/agents/autonomous.py` | Bounded adaptive observe/act/verify application loop |
| `daemon/pilot/system/applications.py` | Fail-closed cross-platform installed-application resolution and launch |
| `daemon/pilot/system/window_mgr.py` | Target-window focus and editable-text verification |
| `tauri-app/ui/src/lib/api/daemon.ts` | WebSocket client (`connect`, `call`, `onNotification`) |
| `tauri-app/ui/src/lib/stores/session.ts` | Notification handlers for the core pipeline, confirm/rollback state |
| `tauri-app/ui/src/lib/stores/multimodal.ts` | Multimodal fusion state and notification handler |
| `tauri-app/ui/src/lib/components/ConfirmDialog.svelte` | Per-action approve/deny confirmation UI |
| `tauri-app/ui/src/lib/components/RollbackDialog.svelte` | Undo confirmation UI |
| `tauri-app/ui/src/lib/components/PermissionAuditLog.svelte` | Audit log viewer + integrity verification UI |
| `tauri-app/ui/src/lib/components/GatewayPolicyEditor.svelte` | Agent Gateway source-profile floor editor |
| `tauri-app/ui/src/lib/components/GatewayAuditLog.svelte` | Agent Gateway audit log viewer + integrity verification UI |
| `schemas/action_plan.schema.json` | JSON Schema for the `ActionPlan` object |
| `schemas/responses/execution_result.json` | JSON Schema for `ExecutionResult` |
| `schemas/actions/*.json` | Per-action-type parameter schemas |
