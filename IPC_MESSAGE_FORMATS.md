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

The daemon currently registers **276 WebSocket RPC methods**. That is the API
surface count, not the action catalog: Heliox exposes **162 action
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

While the active task waits for approval, typed or spoken text never answers the
approval and never becomes a correction of it (except with `mode: "stop"`, which
stops the task as usual). The text is matched after lowercasing and turning
punctuation into spaces, the same way spoken commands are matched:

- An approval phrase ("yes", "go ahead", "approve", "confirm it", ...) returns
  `{ "status": "approval_requires_dialog", "plan_id": "...", "message": "Approve
  in the visible confirmation dialog. Speech and typed text can deny but never
  approve." }`. The approval stays pending and nothing is logged as a correction.
- A denial phrase ("no", "deny it", "don't do that", "cancel that", ...) denies
  the approval under its decision lock (`input_channel: "voice_deny"`,
  `resolution_path: "interject_deny"`) and returns `{ "status": "denied",
  "plan_id": "...", "message": "Okay, I won't run that." }` only when the denial
  took effect, so the plan does not run. When the approval was already answered
  and consumed, the reply is `{ "status": "not_denied", "plan_id": "...",
  "message": "That approval was already answered, so nothing was denied. Say or
  press Stop to cancel the task." }`.

The same guard applies while a step of the running task waits on a narrator
`execution_interrupt`: an approval phrase leaves the interrupt pending, a denial
phrase denies it, and the reply's `plan_id` is the interrupt's `interrupt_...` id.

Neither branch writes a `USER_CORRECTION`, sets a live correction or replans. Any
other text keeps the correction behaviour above, which denies the pending
approval with `resolution_path: "interject_correction"`.

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

A plan resumes only from an APPROVED durable approval, and only with the actions
it approved: every action that needed approval and is not in the approval's
`approved_indices` is dropped before anything runs. An approval that is not
APPROVED (denied or expired), or that approved none of the actions that needed
approval, returns `{ "status": "cancelled", "message": "...", "task_id": "...",
"plan_id": "..." }` and finalizes the task. When an action index of the approval
is missing from the plan's checkpoint, the approval does not match the
checkpoint: the reply is `{ "status": "error", ... }` and nothing runs.

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

A pending approval of the active plan is denied under its decision lock
(`resolution_path: "abort"`). A durable approval row that a `confirm` already
marked APPROVED but that the waiting request had not yet acted on is turned into
DENIED, so the aborted plan never resumes after a restart. If that durable write
fails, the denial still takes effect: the waiting request reads it at once, the
running step is cancelled, live PTY sessions are interrupted, and the durable task
is marked `cancellation_requested`, so a restart finalizes it as cancelled instead
of resuming it.

#### `confirm`
Resolve a pending confirmation gate (see `confirm_required` notification).

**Params:**
```json
{ "plan_id": "a3b2c1f5", "confirmed": true, "approved_indices": [0, 2], "input_channel": "dialog" }
```
`approved_indices` is optional — a list of `plan.actions` indices (matching
the `index` field on each action in the `confirm_required` payload) the user
approved out of those requiring confirmation, for per-action granular
approval. Omit it (or send `confirmed: false`) for the old all-or-nothing
behavior: omitting it while `confirmed: true` approves every action that
required confirmation.

- `confirmed` must be a JSON boolean. `"false"`, `1` and other values return
  `{ "status": "error", "message": "confirmed must be a boolean" }`.
- `input_channel` is optional and names the authority channel that answered:
  `dialog`, `gesture`, `voice_deny`, `internal` or `unknown` (the default). Any
  other value returns `{ "status": "error", "message": "input_channel must be one
  of dialog, gesture, voice_deny, internal, unknown" }`. The channel never says
  whether a pointer or a keyboard pressed the dialog's button.
- `confirmed: true` with `input_channel: "voice_deny"` returns `{ "status":
  "error", "message": "voice_deny can only deny" }`. Voice never approves.
- For a plan approval (an interactive or autonomous plan, or a durable approval
  after a restart), `confirmed: true` with an `input_channel` and without
  `approved_indices` returns `{ "status": "error", "message": "approved_indices is
  required" }`. A client that names its channel must say exactly which actions it
  approves. Narrator, self-healing and threat-containment confirmations keep the
  behaviour above, and so do clients that send no channel.
- A confirmation that is already answered, or that a denial already resolved,
  returns `Confirmation already resolved for plan_id: ...`.

The channel and the resolution path are recorded in the `APPROVAL_RESOLVED`
ledger event as `input_channel` and `resolution_path` (`confirm_rpc`,
`voice_deny`, `interject_deny`, `interject_correction` or `abort`). A plan
approval that ended without an answer the plan may act on is recorded with
`decision: "expired"`, `input_channel: "internal"` and one of three paths:
`timeout` (nobody answered in time), `shutdown` (the daemon stopped first; the
durable approval is left EXPIRED, not DENIED) or `unacknowledged` (an approval
whose `confirm` reply never reached the UI, so the person was never shown that it
was accepted; it expires and the plan does not run). Decision latency is not
recorded.

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

#### `undo_list`
Recent tasks with how much of each can still be undone (newest first).

**Params:** `{ "limit": 20 }` (optional, 1–200)

**Result:**
```json
{
  "status": "ok",
  "enabled": true,
  "tasks": [
    {
      "task_key": "4f0c2a9e",
      "plan_ids": ["a3b2c1f5"],
      "created_at": 1792690000.1,
      "updated_at": 1792690004.7,
      "undone": false,
      "undoable": 2,
      "not_undoable": 1,
      "targets": ["C:/Users/me/notes.txt", "C:/Users/me/inbox/report.pdf"]
    }
  ]
}
```

#### `undo_preview`
What undoing one task would do, item by item, checked against the disk now. Items are
listed newest first. `state` is `ready` (will be undone), `conflict` (changed since
Heliox touched it, so it is left alone) or `unavailable` (the saved copy is gone, the
earlier clipboard text was lost on restart, or the path is inside Heliox's own
folders). `not_undoable` lists the task's actions that cannot be reversed at all, with
the reason.

**Params:** `{ "task_key": "4f0c2a9e" }`

**Result:**
```json
{
  "status": "ok",
  "preview": {
    "task_key": "4f0c2a9e",
    "plan_ids": ["a3b2c1f5"],
    "undone": false,
    "ready": 1,
    "items": [
      { "seq": 2, "type": "file_move", "target": "C:/Users/me/report.pdf", "state": "ready",
        "description": "Move report.pdf back to C:/Users/me." },
      { "seq": 1, "type": "file_write", "target": "C:/Users/me/notes.txt", "state": "conflict",
        "description": "notes.txt changed after Heliox wrote it, so it is left as it is." }
    ],
    "not_undoable": [
      { "type": "api_send_email", "target": "team@example.com",
        "reason": "This reaches outside your computer (a message, a web page or a remote command); Heliox cannot take it back." }
    ],
    "preview_token": "9c1d0b7e5a3f2e6d8b4a1c0f7e9d2b3a"
  }
}
```

#### `undo_task`
Undo a finished task. `confirmed` must be exactly `true` and `preview_token` must match
the preview the person saw; if anything changed since, the daemon returns
`{"status": "changed", "message": ..., "preview": <fresh preview>}` and undoes nothing.
Refused while an interactive task or an autonomous job is running. Entries are reversed
newest first and each is re-checked just before it is applied. Undo restores recorded
state on recorded paths only; it does not plan, call a model or go through the executor.

**Params:** `{ "task_key": "4f0c2a9e", "preview_token": "9c1d…", "confirmed": true }`

**Result:**
```json
{
  "status": "ok",
  "task_key": "4f0c2a9e",
  "undone": 1,
  "skipped": 1,
  "failed": 0,
  "results": [
    { "seq": 2, "target": "C:/Users/me/report.pdf", "state": "undone", "description": "Move report.pdf back to C:/Users/me." },
    { "seq": 1, "target": "C:/Users/me/notes.txt", "state": "conflict", "description": "notes.txt changed after Heliox wrote it, so it is left as it is." }
  ],
  "not_undoable": []
}
```
`status` is `partial` when at least one entry failed while being applied.

#### `undo_clear`
Delete all undo history and every saved copy. **Params:** `{ "confirmed": true }`.
Refused while tasks run. **Result:** `{ "status": "ok" }`.

#### `receipts_list`
Recent signed task receipts (newest first) and this computer's signing key.

**Params:** `{ "limit": 50 }` (optional)

**Result:**
```json
{
  "status": "ok",
  "enabled": true,
  "key": { "available": true, "fingerprint": "3f9a:0c21:77de:a410:9b2e:5c13:e8f0:41aa", "public_key": "base64..." },
  "receipts": [
    { "receipt_id": "c0ffee...", "task_key": "4f0c2a9e", "finished_at": "2026-10-23T09:14:02+00:00",
      "status": "success", "summary": "Moved 3 files.", "request": "tidy my downloads",
      "source": "voice", "action_count": 3, "signed": true }
  ]
}
```

#### `receipt_get`
The full receipt envelope: `{"receipt": {...}, "signature": "...", "public_key": "...", "algorithm": "ed25519"}`.
The receipt lists each action with `action_type`, `target`, `parameters_sha256`,
`output_sha256`, `user_confirmed`, `executed`, `success`, `error` and `undo`, plus the
request, status, summary, `heliox_version`, `key_fingerprint` and
`previous_receipt_sha256` (the hash of the previous receipt envelope). Parameters and
outputs appear only as hashes. With `"export": true` the envelope is also written to
the Downloads folder and the result includes `path`.

**Params:** `{ "receipt_id": "c0ffee...", "export": false }`

#### `receipt_verify`
Check a stored receipt (`receipt_id`) or a pasted envelope (`envelope`).
**Result:** `{"status": "ok", "result": {"valid": true, "signed": true, "fingerprint": "...", "this_computer": true, "reasons": []}}`.
Anyone can check an exported receipt offline with
`python -m pilot.workflows.task_receipts verify receipt.json`.

#### `receipts_verify_chain`
Check every stored receipt's signature and its link to the previous one.
**Result:** `{"status": "ok", "result": {"valid": false, "count": 41, "problems": [{"file": "...", "reason": "The receipt before this one was changed or removed."}]}}`.

#### `timeline_status`
**Result:** `{"status": "ok", "enabled": true, "paused_until": null, "span_count": 812, "retention_days": 7}`.

#### `timeline_search`
Spans whose app or window title contains `query`, newest first.
**Params:** `{ "query": "quarterly report", "limit": 50 }`
**Result:** `{"status": "ok", "spans": [{"start": 1792690000.0, "end": 1792690420.0, "app": "chrome", "title": "Quarterly report - Google Docs"}]}`.

#### `timeline_summary`
Time per app for a window given as a phrase (`when`: "this morning", "yesterday afternoon",
"at 3 pm", "an hour ago") or as `start`/`end` epoch seconds.
**Result:** `{"status": "ok", "label": "this morning", "summary": {"total_seconds": 9120, "apps": [{"app": "Code", "seconds": 4320, "top_titles": ["server.py - Heliox"]}]}, "spans": [...]}`.

#### `timeline_pause`
Stop recording for `minutes` (0 to 1440; 0 resumes). **Result:** `{"status": "ok", "paused_until": 1792693600.0}`.

#### `timeline_clear`
Delete everything the timeline recorded. **Params:** `{ "confirmed": true }`.

#### `today_overview`
One day at a glance, composed in code from local records.
**Params:** `{ "day": "today" }` (`today` or `yesterday`)
**Result:** `{"status": "ok", "day": "2026-10-23", "label": "today", "tasks": [<receipt short form>], "timeline_enabled": true, "usage": {"total_seconds": 9120, "apps": [...]}, "undoable": [<undo_list row>], "brief": "Today Heliox ran 3 tasks; ..."}`.
`usage` is null while the activity timeline is off.

#### `privacy_inventory` / `privacy_clear` / `timeline_forget`
`privacy_inventory` lists each personal store under `DATA_DIR` (activity timeline, task
receipts, undo copies, notes, reminders, voice shortcuts, setups, usage nudges, quiet
hours) as `{"key", "label", "about", "bytes", "size", "items"}`. `privacy_clear` takes
`key` and `confirmed: true` and returns the new inventory. `timeline_forget` removes the
last `minutes` (1 to 1440) of the timeline; "forget my activity from the last hour" (or
"from this morning") does the same by voice and in chat.

#### `backup_create` / `backup_list` / `backup_restore`
`backup_create` zips notes, reminders, shortcuts, setups, nudges, quiet hours and task
receipts (plus the timeline when `include_timeline` is true) into the Downloads folder
as `heliox-backup-YYYYMMDD-HHMMSS.zip`. It never contains configuration, the daemon token,
API keys or signing keys, and undo copies are left out. `backup_list` lists those files;
`backup_restore` takes a listed `name` and `confirmed: true`, restores only the known
store files, and never overwrites a receipt already on this computer.

#### `command_guide`
Everything Heliox answers or does right away, without planning, grouped with examples:
`{"status": "ok", "groups": [{"group": "Reminders and timers", "about": "...", "examples": ["remind me in 20 minutes to stretch", ...]}]}`.
Every example works as written (`tests/test_command_guide.py` runs each one) and none can
be used as a voice shortcut. "What can I say?" speaks a short summary and broadcasts
`open_command_palette`.

#### `notes_list` / `note_add` / `note_delete`
Quick notes (`{"id", "text", "created_at"}`, newest first; text 1 to 500 characters).
"Note that the plumber comes Tuesday", "read my notes", "what did I note about the
bank" and "delete my last note" work by voice and in chat without the planner.

#### `nudge_list` / `nudge_add` / `nudge_delete`
Usage nudges (`{"id", "match", "minutes", "last_fired_day"}`): once a day, when today's
time in apps or window titles containing `match` passes `minutes` (1 to 1440), Heliox
sends a `nudge` notification and says it. They need the activity timeline on and never
block anything. "Tell me if I spend more than an hour on YouTube today", "warn me after
30 minutes of Twitter", "what nudges do I have" and "stop the nudge for twitter" work by
voice and in chat.

Everyday settings can also change by voice: "turn on/off the activity timeline", "start
/ stop recording my activity", "turn on/off the daily brief", "send me a daily brief at
7 pm", "read the brief aloud" / "don't read the brief aloud". They go through
`update_config` validation and broadcast `config_updated`. Security, gateway, approval
and model settings never change by voice.

#### `workspace_list` / `workspace_save` / `workspace_delete`
Named sets of apps. `workspace_save` records the apps that have windows open now
(friendly names only; Heliox itself and desktop components such as Explorer are left
out) under `name`. Saying "open my coding setup" becomes an ordinary request ("open
Visual Studio Code, Slack and Google Chrome") with the normal approvals and a
`request_expanded` note. "Save this as my coding setup", "what setups do I have" and
"delete my coding setup" work by voice and in chat.

#### `receipts_search`
Receipts whose request or summary contains every word of `query` (1 to 200 characters),
newest first, in the short form `receipts_list` uses. "When did I ask you to email the
invoice?" is answered from the same search by voice and in chat.

#### `focus_start` / `focus_stop` / `focus_status`
Focus sessions. `focus_start` takes `minutes` (1 to 480) and returns
`{"active": true, "started_at", "until", "remaining_seconds", "held"}`; starting again
while a session runs moves its end. While a session runs, reminders and the scheduled
daily brief are held and proactive/background speech is suppressed; approvals, task
results and answers are not affected. `focus_stop` ends the session early and returns
its `summary`. "Focus for 45 minutes", "do not disturb for an hour" and "stop focusing"
work by voice and in chat. Every change broadcasts `focus_status`.

`focus_start` also takes `break_minutes` (0 to 60): when the session runs to its end,
a reminder fires after the break ("start a pomodoro" uses 25 + 5 minutes).

#### `focus_schedule_list` / `focus_schedule_add` / `focus_schedule_delete`
Quiet hours: `{"id", "days": [0, 1, 2, 3, 4], "start": 540, "end": 720}` (0 = Monday;
minutes after midnight). Inside a window Heliox starts a focus session that ends with
the window; stopping it early keeps that window off until it ends. "Focus every weekday
from 9 to 12", "what is my focus schedule" and "clear my quiet hours" work by voice and
in chat. "Say that again" repeats the last thing Heliox said aloud.

#### `report_export`
Write a Markdown report (brief, every task with its receipt id, app time, what can still
be undone) for `day` = `today`, `yesterday` or `week` to the Downloads folder.
**Result:** `{"status": "ok", "path": "C:/Users/me/Downloads/heliox-report-2026-10-17-week.md"}`.
`today_overview` also accepts `day: "week"` (the last seven days).

#### `undo_diff`
Line-by-line changes for each text file a task wrote, from the saved copy to the file as
it is now (unified diff, at most 400 lines per file; files that are not UTF-8 text or are
over 512 KB are listed with a `note` instead). **Params:** `{ "task_key": "..." }`
**Result:** `{"status": "ok", "changes": [{"seq": 2, "path": "C:/Users/me/notes.txt", "created": false, "diff": "--- before
+++ now
-eggs
+bread", "truncated": false}]}`.

#### `reminder_list` / `reminder_add` / `reminder_cancel`
Local reminders. `reminder_add` takes `text` (1 to 300 characters) and `in_minutes`
(1 to 10080) or `at` (epoch seconds); `reminder_cancel` takes the reminder `id`. Each
returns the current `reminders` list (`{"id", "due", "text", "created_at"}`, soonest
first). "Remind me in 20 minutes to stretch", "remind me tomorrow at 9 to send the
report", "what are my reminders" and "cancel the reminder about rent" work by voice and
in chat without the planner. "Set a timer for 10 minutes" (10 seconds to 24 hours) is a
reminder whose text says the timer is done.

#### `routine_suggest`
A routine made from a finished task's receipt: `name`, `goal` (the task's request) and
`allowed_actions`, the action types that task ran successfully, plus a default
`daily_at` (the next full hour). Create it with `scheduled_run_create`; the run refuses
any other action. **Params:** `{ "receipt_id": "..." }`

#### `shortcut_list` / `shortcut_add` / `shortcut_remove`
Voice shortcuts: `{ "phrase": "start work", "command": "open VS Code and Slack" }`. A
phrase is 2 to 60 characters and up to six words; phrases that already mean something
(stop, cancel, approve, no, undo, history questions, "when I say …") are refused. Saying
or typing the phrase later submits `command` as an ordinary `execute` request. Each call
returns the current `shortcuts` list and broadcasts `shortcuts_changed`.

#### `daily_brief_now`
Send today's brief now as a `daily_brief` notification (it does not replace the scheduled
one). **Result:** `{"status": "ok", "text": "..."}`.

#### `phone_request_list`
Requests paired phones sent. **Result:** `{"status": "ok", "accept_requests": true, "run_read_only_requests": false, "requests": [<PhoneRequest>]}`
where a PhoneRequest is `{"request_id", "device_id", "device_name", "text", "created_at", "status", "job_id", "result", "read_only"}`
and `status` is `waiting`, `running`, `done`, `failed` or `dismissed`.

#### `phone_request_run`
Start a `waiting` request as a background job with the normal gateway floor and approvals.
**Params:** `{ "request_id": "..." }` **Result:** `{"status": "ok", "request": <PhoneRequest>, "job_id": "..."}`.

#### `phone_request_dismiss`
Drop a `waiting` request without running it. **Params:** `{ "request_id": "..." }`.

#### `air_handoff_requests_configure`
**Params:** `{ "accept_requests": true, "run_read_only_requests": false }` (both booleans;
`run_read_only_requests` needs `accept_requests`). Persists the choice and broadcasts
`air_handoff_state`, which now also carries `accept_requests`, `run_read_only_requests`,
`accepting_requests` and the ten newest `requests`.

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
  "first_run_complete": true,
  "load_errors": []
}
```

`load_errors` lists problems Heliox hit reading `config.toml` or
`restrictions.toml` at startup (for example a TOML syntax error). The daemon
keeps a copy of the unreadable file next to it as
`config.toml.unreadable-YYYYmmdd-HHMMSS` and runs on defaults for the broken
part. While `load_errors` is non-empty every settings write is refused, so the
defaults are never saved over the person's file; fix the file or use
`reset_config` to start over.

Settings RPCs (`get_config`, `update_config`, `reset_config`, the dedicated
`*_config_update` / `*_status` RPCs listed below, `store_api_key`,
`delete_api_key`, `list_api_keys`) are dispatched out of band, so they answer
while an `execute` on the same socket is waiting for approval. Writes run one at
a time under a daemon-wide lock. Before each write the daemon reloads
`config.toml` if it was edited by hand since the last read; values set only for
this run (`--dry-run`, an automatic Ollama model fallback) are re-applied on top
and are never written to disk. Comments in `config.toml` are kept on save.

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

**Result:** `{ "status": "ok" }`, `{ "status": "ok", "restart_required": true }`,
or `{ "status": "error", "message": "..." }`. Nothing is saved on error, and a
failed disk write puts the previous values back in memory.

- `restart_required: true` means the value was saved but is read only at
  startup: any key in `memory`, `proxy` or `redis`; `screen_vision`
  `capture_timeout_seconds`, `max_consecutive_timeouts` and
  `auto_resume_after_seconds`; `cognitive.enabled`; `rss.enabled`.
- Numbers must be finite (`NaN` and infinities are refused);
  `model.gpu_memory_limit_mb` must be 0 or more.
- `calendar.caldav_password_provider` may only be `""` or `"caldav"`, and
  `email.password_provider` only `""` or `"email"`, so a setting cannot point a
  feature at another feature's stored secret.
- Sections that have their own RPC are refused here, because the generic path
  would skip their validation and runtime side effects: `server` (holds the
  daemon's auth token; edit `config.toml` while Heliox is stopped), `narration`,
  `supervision`, `self_healing`, `gateway`, `ssh`, `gesture_workflows`,
  `co_adaptation` (use `co_adaptation_config_update`), `air_handoff` and
  `network` (use `mesh_configure`).
- `section: "restrictions"` accepts only `protected_folders`,
  `protected_packages` and `blocked_commands`, each a list of at most 500
  non-empty strings of at most 1024 characters (duplicates are dropped). It
  returns `{ "status": "ok", "values": { ...cleaned lists } }`.
  `sandbox_allowed_commands` cannot be changed over IPC.

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
**Result:** `{ "status": "ok", "runtime_reconciled": true, "kept": ["server", "restrictions"] }`

`server` (the auth token and port), `restrictions` and `first_run_complete` are
kept. The reset is refused with `{ "status": "error", "message": "Stop the
running tasks before resetting settings." }` while a request or an autonomous
job is running. After saving, the daemon rebuilds the model clients, the circuit
breaker and the voice listener from the new values and releases the Gnomon
model, and clears `load_errors`. The `co_adaptation` defaults turn assistance
co-adaptation off and end any open study session; the reset never deletes
co-adaptation participants, study data or assistance rules.

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

Saving the key for the selected cloud provider rebuilds the cloud client through
the model router; generations already using the old client finish before it is
closed. Entries owned by other features (`heliox_mesh`, and names starting
with `ssh:` or `air_handoff:`) are refused with
`{ "status": "error", "message": "<name> is managed by its own feature" }`.

#### `delete_api_key`
Delete a stored API key.

**Params:** `{ "provider": "openai" }`
**Result:** `{ "status": "ok" }`

The same reserved names are refused.

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
**Result:** `{ "platform": { ... }, "capabilities_count": 162 }`

#### `capabilities`
List all available action types.

**Params:** `{}`
**Result:** `{ "action_types": ["file_read", "file_write", ...], "count": 162 }`

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
exposes no computer-control tools. With `accept_requests` on, a paired phone may also
`POST /api/requests` an AES-GCM-encrypted (`request-v1`), HMAC-signed request of up to
500 characters and read its requests' status from `GET /api/requests` (`requests-v1`).
A request never runs on the phone's say-so: see `phone_request_run`.

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

The current `main` runtime reports 21 executable specialists, 162 declared and
available action types, and an empty `uncovered_action_types` list. Clients must
surface exact uncovered names as a release-blocking warning rather than hiding
coverage regression.

**Result (abbreviated):**

```json
{
  "enabled": true,
  "total_specialists": 21,
  "executable_specialists": 21,
  "registered_action_types": 162,
  "available_action_types": 162,
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
Cancel a running autonomous job, or dismiss an interrupted one.

**Params:** `{ "job_id": "abc123" }`
**Result:** `{ "cancelled": true, "job_id": "abc123" }`

#### `autonomous_resume`
Continue a job that a daemon shutdown or crash interrupted. Jobs are saved in
`autonomous_jobs.db` in the data folder; after a restart, any job that was
queued, planning or running comes back with `"status": "interrupted"` and
`"resumable": true`, and nothing resumes on its own. Resuming skips finished
steps, re-observes the screen and tells the planner the step was interrupted;
approval and human-takeover rules apply as usual.

**Params:** `{ "job_id": "abc123" }`
**Result:** `{ "status": "resumed", "job": <job> }`, or
`{ "status": "error", "message": "Only an interrupted job can be resumed" }`.

#### `autonomy_review_status`
Whether the world model and Gnomon take part in autonomous decisions right now.

**Params:** `{}`
**Result:** `{ "status": "ok", "world_model": { "enabled": true },
"gnomon": { "state": "cold" | "loading" | "ready" | "unavailable" | "disabled",
"last_error": "", "execution_authority": false } }`.

Before each autonomous round the world model scores the plan and, when it is
loaded, Gnomon reviews plans that change something. Either can only add an
approval requirement. Each review is broadcast as `world_model_assessment`
with the plan's `job_id` and a `gnomon` object (`escalate`, `reasons`,
`scores`), and an escalated approval's `confirm_required` carries `reason`,
`risk_assessment` (when the world model asked) and `gnomon`.

#### Scheduled runs

A scheduled run is a goal Heliox works on by itself on a timetable, limited to
an approved list of action types. Each run is an ordinary autonomous job
(`source: "scheduled"`, `allowed_actions` set): the gateway denies every other
action through the job's scope override, and the autonomous loop refuses a
plan that uses one even when the gateway is switched off. Approval rules are
unchanged, so actions that need approval still wait for the person.
Schedules live in `scheduled_runs.db`. A run that came due while Heliox was off
for more than 15 minutes is skipped (`last_status: "missed"`), and a schedule
never starts a run while its previous one is still going.

| Method | Params | Result |
| --- | --- | --- |
| `scheduled_run_list` | `{}` | `{ "status": "ok", "runs": [<run>] }` |
| `scheduled_run_create` | `{ "name", "goal", "allowed_actions": ["disk_usage", "notify"], "every_minutes": 60 }` or `{ ..., "daily_at": "09:00", "weekdays": [0, 1, 2, 3, 4] }` | `{ "status": "ok", "run": <run> }` or `{ "status": "error", "message" }` |
| `scheduled_run_set_enabled` | `{ "run_id", "enabled": false }` | `{ "status": "ok", "run": <run> }` |
| `scheduled_run_delete` | `{ "run_id" }` | `{ "status": "ok", "deleted": true }` |
| `scheduled_run_now` | `{ "run_id" }` | `{ "status": "ok", "run": <run> }`; the timetable is unchanged |

`<run>`: `{ "run_id", "name", "goal", "allowed_actions", "every_minutes",
"daily_at", "weekdays", "enabled", "next_run_at", "last_run_at", "last_job_id",
"last_status", "last_job_status" }`. `every_minutes` is 5 to 10080; `weekdays`
are 0 (Monday) to 6 (Sunday) and apply only to `daily_at`.

#### `autonomous_jobs`
List all autonomous jobs, including interrupted ones from before a restart.

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

After a successful `recorded_workflow_start` or `recorded_workflow_stop` the
daemon sends `recorded_workflow_status_changed` to the other windows (see
section 2); a failed start or stop sends nothing. On Windows, manual-input
recording skips input while the desktop app's window, titled "Heliox-OS", is
in the foreground. The check compares the foreground window's whole title (case,
hyphens and underscores ignored) and, when Windows names the owning process,
requires the Heliox executable, so a page or editor whose title only mentions the
project is recorded. It does not cover the UI opened in a browser (the page is
titled "Pilot"), and on Linux and macOS nothing is skipped.

### Human Control

The person can take the desktop back from Heliox at any time. While they hold
control, desktop input that Heliox starts on its own (autonomous jobs,
MCP-submitted work) is refused rather than queued, and autonomous jobs pause.
The person's own typed, spoken or gestured commands still run. During
autonomous work, physical mouse clicks, wheel turns and key presses also take
control automatically on Windows; synthetic input, including Heliox's own, is
ignored. Handing control back discards earlier accessibility snapshots, so the
agent observes again before acting.

| Method | Params | Result |
| --- | --- | --- |
| `take_control` | `{ "reason": "my turn" }` (optional) | control status |
| `release_control` | `{}` | control status with a new `generation` |
| `human_control_status` | `{}` | control status |

Control status: `{ "status": "ok", "holder": "agent" or "human", "reason": "...",
"since": 1791640000.0, "generation": 3, "armed": true,
"physical_input_detection": true }`.

### MCP Client

Heliox can use tools from MCP servers the person lists in `mcp_servers.json`
inside Heliox's config folder, in the usual
`{ "mcpServers": { "name": { "command", "args", "env", "disabled" } } }`
format. Each server runs as a local stdio process; actions cannot read or
change that file. The planner sees the tools through `mcp_list_tools` and
calls them with `mcp_call_tool`. A call runs without approval only when the
server marks the tool `readOnlyHint: true`.

| Method | Params | Result |
| --- | --- | --- |
| `mcp_client_status` | `{}` | configured servers and tool counts |
| `mcp_client_refresh` | `{}` | the same, after restarting every server |

Status: `{ "status": "ok", "tool_count": 12, "config_path": "...mcp_servers.json",
"servers": [{ "name": "github", "running": true, "tools": 12, "error": "",
"server_info": { ... } }] }`. Settings shows this list with a Reload button.

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
Forget learned proactive-suggestion preferences and return the reset status
(`{ "status": "ok", "enabled": true, "patterns": {} }`, or `status: "error"`
when the proactive engine is not running). This does not delete the immutable
experience ledger.

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

### Assistance co-adaptation

Opt-in research on assisted selection inside Heliox's own window (the Selection
Study tab; see [`docs/COADAPTATION.md`](docs/COADAPTATION.md)). It is off by
default (`co_adaptation.enabled = false`), and nothing is stored until a
participant has an active, purpose-bound consent record. It changes only how
study targets are presented and what a selection needs from the person. Every
recommendation carries `"authority": "presentation_only"` and
`"execution_authority": false`. No co-adaptation RPC reads, answers or records an
approval: there is no `co_adaptation_record_approval`, and a call leaves every
pending confirmation and the experience ledger unchanged. None of these methods
is open to the neural sidecar or the local MCP bridge.

| Method | Dispatch set | Purpose |
| --- | --- | --- |
| `co_adaptation_status` | settings | Settings, store size, participants, rules, fallbacks |
| `co_adaptation_config_update` | config-mutating | Change `co_adaptation` settings |
| `co_adaptation_consent_grant` | config-mutating | Create a participant or update its consent |
| `co_adaptation_consent_revoke` | config-mutating | Withdraw consent and erase the participant |
| `co_adaptation_preference_set` | settings | Save an assistance rule |
| `co_adaptation_preference_retract` | settings | Remove an assistance rule |
| `co_adaptation_learning_reset` | settings | Reset what was learned for a participant |
| `co_adaptation_erase` | settings | Erase today's study data of a participant |
| `co_adaptation_results` | settings | The participant's own results, from stored rows |
| `co_adaptation_export` | study | Write an encrypted export file |
| `co_adaptation_session_begin` | study | Start a study session |
| `co_adaptation_session_resume` | study | Continue a paused session |
| `co_adaptation_recommend` | study | A block regime, an ambiguity table or an offer |
| `co_adaptation_record_trial` | study | Store and verify one trial |
| `co_adaptation_report_burden` | study | Store one burden rating |
| `co_adaptation_session_end` | study | End a session and summarise it |

- **config-mutating** (`CONFIG_MUTATING_RPC_METHODS`): run one at a time under the
  config lock and are followed by `config_updated`.
- **settings** (`SETTINGS_RPC_METHODS`): out of band, without the config lock.
- **study** (`CO_ADAPTATION_RPC_METHODS`): out of band, so a study trial never
  waits behind an `execute` on the same socket.

`update_config` refuses `section: "co_adaptation"` ("Use co_adaptation_config_update
to change co_adaptation settings"). `reset_config` restores the section's
defaults (which turns the feature off and ends open sessions) and never deletes
participants, study data or assistance rules.

When the daemon could not start the co-adaptation service, `co_adaptation_status`
returns `{ "status": "ok", "available": false, "config": {...}, "message":
"Co-adaptation is unavailable on this daemon.", "authority": "presentation_only",
"execution_authority": false }` and every other method returns `{ "status":
"error", "message": "Co-adaptation is unavailable on this daemon." }`. When the
OS keyring is unavailable the service starts degraded: status, revoke, erase and
the retention prune work, and every other method returns `{ "status": "refused",
"reason": "keyring_unavailable" }`. A revoke or prune in this state deletes the
rows and export files and leaves the participant's key for a later retry (see
`co_adaptation_consent_revoke`).

The retention prune runs at service start, at the start of `co_adaptation_status`
and `co_adaptation_session_begin`, when a call on an open session finds its
consent expired, and otherwise every 24 hours. An expired consent is refused at
once, but its rows, exports and key can remain for up to 24 hours after expiry
while the daemon keeps running and none of those calls arrives.

Identifiers used below:

- Modalities: `mouse`, `touchpad`, `pen`, `touch`, `keyboard`, `hand_pointer`.
- L1 regime ids: `assistance_off` (the plain interface: the literal click
  commits and nothing is interpreted), `act`, `enlarge_target:1.5`,
  `enlarge_target:2.0`, `confirm_selection:second_click`,
  `confirm_selection:hold_400` (hand pointer only), `confirm_selection:hold_700`.
- L2 responses (`L2_OPTIONS`, always in this order): `act`,
  `confirm_selection:second_click`, `wait_for_evidence`, `ask_clarification:2`,
  `ask_clarification:3`.
- L3 offers: `no_offer`, `offer_modality:keyboard`, `offer_modality:pointer`.
- Consent purposes: `selection_study` (required), `adaptive_assistance`,
  `cross_session_learning`, `gaze_prior`, `hand_pointer`, `research_export`.
- Participant keys are generated by the daemon and match `^p-[0-9a-f]{8}$`.

#### `co_adaptation_status`
**Params:** `{}`. Runs the lazy retention prune first. Never fails.

**Result:**
```json
{
  "status": "ok",
  "available": true,
  "config": { "enabled": false, "adaptive_enabled": false, "...": "every CoAdaptationConfig field" },
  "keyring_available": true,
  "erasure_pending": false,
  "store": { "exists": true, "bytes": 123456, "participants": 1, "sessions": 3, "decisions": 420,
             "trials": 330, "burden_reports": 40, "oldest_recorded_at": "2026-10-15T10:00:00+00:00",
             "pruned_sessions": 0 },
  "participants": [{
    "participant_key": "p-3fa29c1e",
    "label": "Lab A",
    "consent": { "consent_id": "...", "purposes": ["selection_study"], "granted_at": "...",
                 "expires_at": "...", "retention_days": 90, "consent_text_version": "coadapt-consent-v1",
                 "consent_text_sha256": "...", "protocol_id": "coadapt-study-v1", "active": true },
    "model": { "trials_seen": 330, "sessions_seen": 3, "last_day_index": 740000, "drift_flags": [],
               "calibration_suspended": false, "learn_across_sessions_effective": false },
    "fallbacks": { "mouse": { "...": "FallbacksV1" } },
    "sessions": 3,
    "last_session_at": "..."
  }],
  "preferences": ["PreferenceV1", "..."],
  "proposals": [{ "participant_key": "p-3fa29c1e", "kind": "forbid_variant",
                  "variant": "ask_clarification:3", "reason": "three_vetoes" }],
  "protocol": { "protocol_id": "coadapt-study-v1", "sha256": "..." },
  "versions": { "strategy": "coadapt-strategies-v1", "feature_schema": "coadapt-features-v1",
                "model": "coadapt-owm-v1", "prior": "literature-v1", "policy": "coadapt-policy-v1",
                "sdk": "0.1.0", "consent_text": "coadapt-consent-v1" },
  "authority": "presentation_only",
  "execution_authority": false
}
```
`label` is the decrypted display label, or null (always null while the keyring
is unavailable). `fallbacks` is given per modality for every participant with an
active consent. A proposal is only a suggestion: nothing is written until the
person accepts it. `erasure_pending` is true while the pending-erasure list
described under `co_adaptation_consent_revoke` still holds a keyring key, an
export file or a participant's rules after the prune this call ran; it is the
only sign of an expiry erasure that could not finish, so a client should show it.

#### `co_adaptation_config_update`
**Params:** any subset of the `co_adaptation` settings:

| Key | Type | Default | Range |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | |
| `adaptive_enabled` | bool | `false` | also needs purpose `adaptive_assistance` |
| `learn_across_sessions` | bool | `false` | also needs purpose `cross_session_learning` |
| `gaze_prior_enabled` | bool | `false` | also needs `vision.gaze_tracking_enabled` and purpose `gaze_prior` |
| `hand_pointer_enabled` | bool | `false` | also needs purpose `hand_pointer` and a running camera |
| `ambiguity_threshold` | number | `0.25` | 0.05 to 0.25 (can only be made stricter) |
| `unintended_activation_tolerance` | number | `0.05` | 0.0 to 0.2; a study session uses at most 0.05 (stricter only) |
| `noninferiority_margin` | number | `0.01` | 0.0 to 0.05; a study session uses at most 0.01 (stricter only) |
| `upper_bound_credibility` | number | `0.95` | 0.8 to 0.99; a study session uses at least 0.95 (stricter only) |
| `exploration_floor` | number | `0.1` | 0.0 to 0.3; study sessions always use the protocol's 0.1 |
| `switching_penalty` | number | `0.1` | 0.0 to 1.0; study sessions always use the protocol's 0.1 |
| `burden_prompt_every_blocks` | integer | `1` | 1 to 8 |
| `day_boundary_hour` | integer | `4` | 0 to 23 |

**Result:** `{ "status": "ok", "config": {...}, "restart_required": false }`.

**Errors:** `Unknown co-adaptation setting: <key>`; a strict type error such as
`enabled must be a boolean`, `ambiguity_threshold must be numeric` or
`day_boundary_hour must be an integer`; a range error such as
`ambiguity_threshold must be at most 0.25`; `Grant consent before turning
co-adaptation on.` when `enabled: true` arrives while no participant has an
active consent. A failed save puts the previous values back and returns the save
error. Nothing needs a restart. A change reaches the study at the next
`co_adaptation_session_begin`, which stores the effective settings in the new
session; an open session keeps the settings it started with. Only `enabled:
false` acts at once: it ends open sessions. A hand-edited `config.toml` value
outside a range is clamped into it when the file is loaded.

Study sessions follow the frozen protocol (`coadapt-study-v1`).
`exploration_floor` and `switching_penalty` always take the protocol's values,
and `ambiguity_threshold`, `unintended_activation_tolerance`,
`noninferiority_margin` and `upper_bound_credibility` take the stricter of the
saved value and the protocol's. A looser value is accepted, saved and returned
by `co_adaptation_status`, but it does not change what a session runs with. The
stored session row lists in `protocol_deviations` every setting whose saved
value differs from the protocol's, also when the effective value did not
change.

#### `co_adaptation_consent_grant`
**Params:**
```json
{
  "participant_key": null,
  "label": "Lab A",
  "purposes": ["selection_study", "gaze_prior"],
  "retention_days": 90,
  "consent_text_version": "coadapt-consent-v1",
  "consent_text_sha256": "<64 hex>",
  "protocol_id": "coadapt-study-v1",
  "protocol_sha256": "<64 hex>",
  "authorized": true,
  "enable": true
}
```
- `participant_key: null` creates a participant with a new daemon-generated key,
  its encryption key in the OS keyring, the optional encrypted `label` (at most 32
  characters, new participants only) and the consent record. The database file is
  created at this point, not before.
- The key of a participant whose consent is active updates that consent. The
  earlier record is kept with `replaced_at`, the new expiry is the earlier of the
  old expiry and `granted_at + retention_days` (an update never extends
  retention), and a purpose that is removed deletes what was stored only for it:
  export files for `research_export`, the gaze-check accuracy and every stored gaze
  region for `gaze_prior`, the hand summaries for `hand_pointer` (followed by a
  model rebuild). An update that removes a purpose while the participant has an
  open session is refused.
- Any other key is refused (`unknown_participant`); keys are never reused.
- After a successful grant the daemon sets `co_adaptation.enabled = true` unless
  `enable` is `false` (it must be a JSON boolean), and saves; a failed save turns
  it back off and returns the save error.

**Result:**
```json
{
  "status": "ok",
  "participant_key": "p-3fa29c1e",
  "consent": { "...": "ConsentSummary, as in co_adaptation_status" },
  "config": { "...": "co_adaptation settings" },
  "removed": { "purposes": [], "export_files": 0, "trials_rewritten": 0, "sessions_rewritten": 0,
               "export_files_left": 0 }
}
```
`removed` lists the purposes an update removed and the counts of deleted export
files and rewritten trial and session rows (empty lists and zeros for a first
grant or an update that removes nothing). `export_files_left` counts export files
that could not be deleted yet; they go on the pending-erasure list described
under `co_adaptation_consent_revoke`.

**Errors:** `authorized` not literally `true`; `selection_study` missing; a bad
hash; `retention_days` outside 1 to 365; a malformed or unknown key; `Secure OS
key storage is unavailable; co-adaptation stays off.`; and
`{ "status": "refused", "reason": "session_active", "message": "Stop the running
session before removing a permission." }`.

#### `co_adaptation_consent_revoke`
**Params:** `{ "participant_key": "p-3fa29c1e" }`

Withdrawal always erases: the participant's in-memory state is evicted, every row
and export file is deleted, the keyring key is destroyed (so encrypted columns of
leftover database pages become unreadable), and the participant's own rules are
deleted. Rules set for everyone on the device stay. When no active consent
remains, the daemon sets `co_adaptation.enabled = false` and saves.

**Result:** `{ "status": "ok", "erased": { "rows": 412, "key_destroyed": true,
"export_files": 1, "rules_deleted": 2, "export_files_left": 0, "pending": false },
"config": {...} }`

`key_destroyed` is `false` when the OS keyring could not be reached (degraded
mode), and `export_files_left` counts export files that could not be deleted, for
example because another program holds them open. `pending` is true when either
happened or when the participant's rules could not be removed from an unreadable
rules file. The rows are gone in every case. What is left stays on the
pending-erasure list `co_adaptation_pending_erasure.json` next to the database,
which holds only pseudonymous participant keys and export file names. The daemon
retries that list at every service start, `co_adaptation_status`
and prune, and on a repeated revoke, which completes what is left instead of
returning `ignored`. Until the key is deleted it decrypts the encrypted columns of
database backups and leftover pages and any export copied elsewhere, so a client
should tell the person when `pending` is true.

#### `co_adaptation_preference_set`
**Params:** `{ "kind": "...", "scope": { "context": "study"|"all", "modality":
"<modality>"|null, "participant_key": "p-xxxxxxxx"|null }, "value": ... }`

| `kind` | `value` |
| --- | --- |
| `forbid_variant` | a variant id or a family (`confirm_selection`, `ask_clarification`, `enlarge_target`, `offer_modality`) |
| `pin_regime` | an L1 regime id other than `assistance_off` |
| `pin_act` | `true` ("Always select exactly what I click") |
| `min_caution` | a variant id in caution order |
| `max_interruptions_per_block` | an integer 0 to 8 |
| `priorities` | `{ "fewer_mistakes": f, "faster": f, "fewer_interruptions": f }`, each in [0, 1] |
| `freeze_adaptation` | `{ "regime": "<L1 regime id, assistance_off included>" }` |

Works without consent (rules are settings, kept in
`co_adaptation_preferences.json`). A participant key in the scope must belong to
an existing participant; labels are never accepted.

**Result:**
```json
{
  "status": "ok",
  "preference": { "pref_id": "...", "kind": "forbid_variant", "scope": {...}, "value": "act",
                  "origin": "explicit_user", "created_at": "...", "retracted_at": null },
  "notes": [],
  "refresh_required": true,
  "refresh_seq": 4,
  "fallbacks": { "...": "FallbacksV1 of the participant in the scope" }
}
```
`refresh_required`, `refresh_seq` and `fallbacks` describe the participant named
in the scope; for a device-wide rule they are `false`, `null` and `null`, and a
running study learns of the change from `co_adaptation_updated`. `notes` explains
an accepted combination (a cap against a minimum caution that leaves only confirm
regimes).

**Errors:** a rule that conflicts with an active rule whose scope overlaps it is
rejected and nothing is written:
```json
{
  "status": "error",
  "reason": "preference_conflict",
  "message": "You already pinned ...; pinning ... would contradict it. Remove that rule and save this one, or keep it.",
  "conflicting": [{ "pref_id": "...", "summary": "..." }]
}
```
Conflicts include a pin or freeze of a forbidden regime or of one below an active
minimum caution, a cap below the block's 8 trials against a frozen or pinned
confirm or hold regime, a freeze and a pin naming different regimes, and two pins
or two freezes naming different regimes. The message names each conflicting rule;
the UI's "Remove that rule" button retracts the rules in `conflicting` and then
retries. Overlapping minimum cautions combine to the strictest and overlapping
caps to the smallest; they are not conflicts. Forbidding `wait_for_evidence` is
rejected, and so is any rule that would leave a modality with no allowed regime.

#### `co_adaptation_preference_retract`
**Params:** `{ "pref_id": "..." }`

**Result:** `{ "status": "ok", "refresh_required": false, "refresh_seq": null,
"fallbacks": null }` (filled in as for `preference_set` when the rule applied to a
participant), or `{ "status": "error", "message": "Unknown rule" }`.

#### `co_adaptation_learning_reset`
**Params:** `{ "participant_key": "p-3fa29c1e" }`

Ends the participant's open session first, then puts the model, spread and
practice counters back to the prior. Data and consent are kept; a later rebuild
ignores rows recorded before the reset.

**Result:** `{ "status": "ok", "model": {...}, "session_ended": true }`

#### `co_adaptation_erase`
**Params:** `{ "participant_key": "p-3fa29c1e", "scope": "today" }`

Ends the participant's open session first, then deletes today's sessions with
their decisions, trials, burden reports and events, and every export file written
since the earliest erased session started (older exports cannot contain those
rows). Full erasure is `co_adaptation_consent_revoke`.

**Result:** `{ "status": "ok", "erased": { "rows": 96, "export_files": 1,
"export_files_left": 0 }, "session_ended": false }`. `export_files_left` counts
export files that could not be deleted yet and are retried from the
pending-erasure list described under `co_adaptation_consent_revoke`.

#### `co_adaptation_results`
**Params:** `{ "participant_key": "p-3fa29c1e", "scope": "today"|"all" }`

**Result:** computed from stored rows only, never from simulated data:
```json
{
  "status": "ok",
  "evidence_kind": "live_human",
  "claim": "Your data on this device; not a general result.",
  "per_regime": [{ "variant": "act", "trials": 40, "completion": 0.97, "ua_per_100": 3.0,
                   "idle_ua_per_min": 0.0, "corrections_per_trial": 0.1, "median_time_ms": 1450,
                   "p95_time_ms": 3100, "throughput_bits_s": 4.1,
                   "burden": { "effort": 3.2, "intrusiveness": 2.0, "control": 5.5 },
                   "verification": { "ok": 38, "fallback": 1, "draw_mismatch": 0,
                                     "posterior_mismatch": 0, "interrupted": 1 } }],
  "calibration": { "ua": { "brier": 0.03, "ece": 0.02, "log_score": -0.15, "n": 330, "bins": [...] },
                   "complete": {...}, "log_time": { "coverage90": 0.88, "n": 330 } },
  "probe_trend": [{ "session_number": 2, "throughput_bits_s": 4.0, "ua_per_100": 5.0 }],
  "drift_flags": [],
  "allowed_now": { "eligible": ["act", "enlarge_target:1.5"],
                   "excluded": { "confirm_selection:hold_700": "forbidden_by_you" } },
  "predictions_now": { "act": "PredictionV1" }
}
```

#### `co_adaptation_export`
**Params:** `{ "participant_key": "p-3fa29c1e" }`. Needs the `research_export`
purpose.

Writes `co_adaptation_exports/coadapt_<participant_key>_<UTC yyyymmddTHHMMSSZ>.enc`
under the data directory: a plaintext JSON header line, then the AES-256-GCM
encrypted bundle. The display label is never exported, and there is no upload
path.

**Result:** `{ "status": "ok", "path": "...", "sha256": "...", "bytes": 48213 }`,
or `{ "status": "refused", "reason": "purpose_missing" }`.

#### Study RPCs

Status values: `ok`, `recorded`, `ignored` (reasons `disabled`, `no_consent`,
`consent_expired`, `purpose_missing`, `unknown_session`, `session_closed`),
`refused` (reasons below) and `error` (with `message`).

The daemon re-checks three start refusals before `co_adaptation_session_begin`,
`co_adaptation_session_resume`, `co_adaptation_recommend` for `block_regime` and
`ambiguity_table`, and `co_adaptation_record_trial`, and answers `{ "status":
"refused", "reason": ... }` without reaching the study service:

- `recording_active`: a recorded workflow is being captured;
- `task_running`: an interactive task is running;
- `neural_armed`: the neural gate is in `armed_safe_ui`, `armed_safe_desktop`,
  `candidate_intent`, `previewed`, `committed` or `cooldown`.

A refused trial is not stored; the study marks it interrupted
(`interruption_reason: "refused"`), pauses, and sends it again after a successful
resume. The study service itself may refuse with `disabled`, `no_consent`,
`consent_expired`, `purpose_missing`, `consent_text_changed`,
`protocol_changed`, `session_active` (one open session per participant),
`hand_pointer_not_allowed`, `no_allowed_regime`, `keyring_unavailable` or
`preferences_unreadable`.

`co_adaptation_session_begin` returns `preferences_unreadable` while
`co_adaptation_preferences.json` cannot be read: it is not valid JSON, has an
unknown `schema_version`, holds an invalid rule, or holds a rule whose `origin` is
not `explicit_user`. An unreadable rules file never loosens anything, so no study
session starts until the file is fixed or removed. Meanwhile
`co_adaptation_status` lists no rules (`preferences: []`) and no `model` or
`fallbacks`, and `co_adaptation_preference_set` and
`co_adaptation_preference_retract` return `{ "status": "error", "message": "The
co-adaptation rules file cannot be read." }`. The daemon reads the file again when
it starts, so restart Heliox after fixing or removing it. Removing the file
deletes every rule, device-wide rules included.

#### `co_adaptation_session_begin`
**Params:**
```json
{
  "participant_key": "p-3fa29c1e",
  "context": "study",
  "consent_text_sha256": "<64 hex>",
  "device": { "modality": "mouse", "pointer_types": ["mouse"], "dpr": 1.25, "surface_w": 700,
              "surface_h": 500, "gaze": false, "hand_pointer": false },
  "protocol_id": "coadapt-study-v1",
  "gaze_check": { "accuracy": 0.8 }
}
```
`gaze_check` is the result of the study's gaze check, or `null` when none ran.
The accuracy is stored only when the gaze purpose is in effect, and gaze is used
for the session only when it is in effect and the accuracy is at least 0.6.

**Result:**
```json
{
  "status": "ok",
  "session_id": "...",
  "participant_key": "p-3fa29c1e",
  "day_index": 740000,
  "session_number": 2,
  "plan": { "session_number": 2, "end_questionnaire": true,
            "blocks": [{ "index": 0, "phase": "practice", "task": "pointing", "trials": 8,
                         "idle_after": false, "burden_after": false }] },
  "layout_seed": 1234567,
  "spread": { "mouse": 1.02 },
  "floors": { "mouse": 1.0 },
  "thresholds": { "ambiguity": 0.25,
                  "bins": { "b1": [0.25, 0.4], "b2": [0.4, 0.6], "b3": [0.6, 1.0] } },
  "gaze_prior": false,
  "hand_pointer": false,
  "burden_prompt_every_blocks": 1,
  "evidence_kind": "live_human",
  "fallbacks": { "...": "FallbacksV1, issued_at = the session's start" },
  "refresh_seq": 0
}
```
`spread` and `floors` are the posterior parameters for the session's modality
until the first L1 recommendation; both are `{}` in a keyboard session, which has
no spread. `thresholds` come from the settings stored with the session and hold
for the whole session. `refresh_seq` is the participant's current refresh value,
which the study stores as its last applied value. The session's random salt is
never returned. Block phases are `practice`, `A`, `A_lite`, `B` (adaptive),
`probe` (`assistance_off`) and `comparator` (the best fixed assisted regime of
session 1).

#### `co_adaptation_session_resume`
**Params:** `{ "session_id": "..." }`. Called before the study continues a
paused session.

**Result:** `{ "status": "ok", "refresh_required": false, "refresh_seq": 3,
"fallbacks": {...} }`, or `ignored` with `unknown_session`, `session_closed`,
`no_consent` or `consent_expired`. With `refresh_required: true` the study ends
the current block before it shows another trial and requests fresh decisions
(see **Refresh sequence** below).

#### `co_adaptation_recommend`
**Params:** `{ "session_id": "...", "decision_id": "<uuid4>", "level":
"block_regime"|"ambiguity_table"|"offer", "context": {...} }`. Context keys:

- `block_regime`: `block_index`, `layouts` (one `LayoutSummaryV1` per planned
  trial), `previous_regime`, `session_minute`, `trials_today`,
  `hand_quality_recent`;
- `ambiguity_table`: the same plus `block_decision_id`, `start_trial_index` (0 at
  the block start; the next trial's index for a table requested after a veto) and
  `trials_in_block` (rows to draw, from `start_trial_index` to the block's end);
- `offer`: `block_index`, `trial_index`, `task`, `struggle_kind`,
  `keyboard_available`.

**Result:** `{ "status": "ok", "logged": true, "recommendation":
RecommendationV1 | AmbiguityTableV1 }`. Without consent, or while disabled:
`{ "status": "ignored", "reason": "...", "logged": false, "recommendation": ... }`
with the baseline (`{ "variant": "assistance_off" }` at L1, an empty table with
`cells: []` at L2, `{ "variant": "no_offer" }` at L3, each with the authority
fields). A reused `decision_id` returns `Duplicate decision_id`. When the
person's rules and the calibration suspension leave no regime, the reply is
`{ "status": "refused", "reason": "no_allowed_regime" }` and the block does not
start.

```json
RecommendationV1 = {
  "decision_id": "...", "level": "block_regime", "mode": "adaptive",
  "variant": "enlarge_target:1.5", "params": { "scale": 1.5 },
  "eligible": ["act", "enlarge_target:1.5", "enlarge_target:2.0", "confirm_selection:second_click"],
  "excluded": { "confirm_selection:hold_700": "forbidden_by_you", "assistance_off": "probe_schedule" },
  "propensities": { "assistance_off": 0, "act": 0.05, "enlarge_target:1.5": 0.82, "enlarge_target:2.0": 0.08,
                    "confirm_selection:second_click": 0.05, "confirm_selection:hold_700": 0 },
  "p_thompson": {...}, "exploration": 0.1, "fallback": false, "protocol_deviation": false,
  "round_index": null,
  "spread": { "r": 1.02, "floor": 1.0 },
  "priorities": { "fewer_mistakes": 0.5, "faster": 0.5, "fewer_interruptions": 0.5 },
  "predictions": { "act": "PredictionV1", "...": "..." },
  "explanation": "ExplanationV1", "seed": "9f2c...", "versions": {...},
  "authority": "presentation_only", "execution_authority": false
}

PredictionV1 = {
  "variant": "act", "p_complete": { "mean": 0.97, "lo90": 0.93, "hi90": 0.99 }, "p_unintended": {...},
  "ucb_unintended": 0.06, "p_correction": {...}, "p_veto": {...},
  "time_ms": { "mean": 1400, "lo90": 800, "hi90": 2600 },
  "burden": { "effort": 3.1, "intrusiveness": 2.2, "control": 5.4 },
  "evidence": { "trials_person": 120, "trials_today": 30, "source": "today" },
  "authority": "presentation_only"
}

ExplanationV1 = {
  "summary_key": "coadaptation.explain.adaptive", "values": {...},
  "contrasts": [{ "metric": "slips", "variant": "enlarge_target:1.5", "baseline": "assistance_off",
                  "variant_mean": 0.04, "baseline_mean": 0.09, "delta_ci90": [-0.08, -0.01] }],
  "excluded": [["confirm_selection:hold_700", "forbidden_by_you"]],
  "evidence_source": "today", "trials_person": 120, "trials_today": 30
}

AmbiguityTableV1 = {
  "decision_id": "...", "level": "ambiguity_table", "block_decision_id": "...", "mode": "adaptive",
  "ambiguity_threshold": 0.25,
  "bins": [{ "id": "b1", "lo": 0.25, "hi": 0.4 }, { "id": "b2", "lo": 0.4, "hi": 0.6 },
           { "id": "b3", "lo": 0.6, "hi": 1.0 }],
  "cells": [{ "cell_id": "b1:p2:h1", "bin": "b1", "n_plausible": 2, "has_hit": true,
              "order": ["act", "confirm_selection:second_click", "wait_for_evidence",
                        "ask_clarification:2", "ask_clarification:3"],
              "propensities": { "act": 0, "confirm_selection:second_click": 0.45, "wait_for_evidence": 0.1,
                                "ask_clarification:2": 0.45, "ask_clarification:3": 0 },
              "eligible": ["confirm_selection:second_click", "wait_for_evidence", "ask_clarification:2"],
              "excluded": { "act": "ambiguous_activation", "ask_clarification:3": "not_applicable" } }],
  "start_trial_index": 0,
  "draws": [[0.12, 0.88, 0.40, 0.05]],
  "max_draws_per_trial": 4,
  "after_max": "wait_for_evidence",
  "explanation": "ExplanationV1", "seed": "...", "versions": {...},
  "authority": "presentation_only", "execution_authority": false
}
```
- `mode` is one of `randomized` (Phase A and A-lite permuted blocks), `adaptive`
  (Phase B learned choice), `probe` (`assistance_off`), `comparator` (the best fixed
  assisted regime from session 1, with the uniform L2 propensities of Phase A),
  `pinned` (a freeze, a pin or the `pin_act` collapse, propensity 1) and `baseline`
  (practice).
- `spread` (L1 only) is the `{r, floor}` pair the UI must use for every posterior
  of the block and echo in `TrialV1.spread`; it is `null` in a keyboard session.
- `priorities` (L1 only) are the weights in force at the decision; the analysis
  scores the block with them.
- `ambiguity_threshold` and `bins` are the session's own values.
- `draws` holds one row per trial from `start_trial_index` to the block's end.
- `order` is always `L2_OPTIONS` order. Outside a `pin_act` collapse `act` has
  propensity 0 and is excluded with `ambiguous_activation`, so no learned,
  exploratory or fallback choice commits an ambiguous activation.
- Propensities, draws, spread values and seeds are never rounded, so the UI's
  table selection and the daemon's re-derivation see identical floats.

#### `co_adaptation_record_trial`
**Params:** `{ "session_id": "...", "trial": TrialV1 }`

**Result:**
```json
{
  "status": "recorded",
  "verification": "ok",
  "labels": { "complete": true, "ua": false, "correction": false, "veto": false, "time_ms": 1380.0 },
  "preference_violation": false,
  "refresh_required": false,
  "refresh_seq": 3,
  "fallbacks": null
}
```
`verification` is `ok`, `draw_mismatch`, `posterior_mismatch`, `interrupted` or
`fallback`; `labels` is null when the trial is not learned from. Only `ok` trials
update the model. The call is idempotent per `trial_id`: a repeat returns the
stored result, which never reverts a trial again. `refresh_seq` is the
participant's current value; a drift alarm or calibration suspension fired by this
trial raises it by one and sets `refresh_required` with fresh `fallbacks`.

#### `co_adaptation_report_burden`
**Params:** `{ "session_id": "...", "report": BurdenV1 }`

**Result:** `{ "status": "recorded" }`

#### `co_adaptation_session_end`
**Params:** `{ "session_id": "...", "reason": "completed"|"stopped"|"abandoned"|"interrupted" }`

**Result:** `{ "status": "ok", "summary": SessionSummaryV1 }` with `trials`,
`completion`, `ua_rate`, `idle_ua_per_min`, `median_time_ms`, `burden_mean`,
`overrides`, `regime_entropy_bits`, `switch_rate`, `allowed_now`, `excluded_now`
and `predictions`.

#### Refresh sequence

A rule saved or retracted, a calibration suspension or a drift alarm marks the
participant's open decisions stale and refreshes the open session even when no
decision is open (a practice block, or a block whose regime request failed, runs
on fallbacks the change made stale). Each such change raises the participant's
`refresh_seq` once (a counter that starts at 0 when the daemon starts and only
increases while it runs). The `refresh_required` reply and the
`co_adaptation_updated` broadcast for one change carry the same value, a later
`co_adaptation_record_trial` or `co_adaptation_session_resume` reply that reports
the same stale state repeats it, and `co_adaptation_session_begin` and
`co_adaptation_session_resume` return the current value.

The study applies a refresh only when its `refresh_seq` is greater than the last
value it applied. It then pauses input until it holds fallbacks issued after the
change, reverts the current trial to the fresh `fallbacks.refresh` regime (or
interrupts it when that is `null`, or when the trial was already reverted for a
different change), and ends the block after that trial. So a trial is reverted or
interrupted at most once per `refresh_seq`. Ending the block is not deduplicated:
a refresh applied while no trial is in progress (paused, block intro, burden
prompt or idle epoch) ends the block before the next trial, and any reply with
`refresh_required` ends the block when the block's decisions were requested
before the last applied refresh. While a freeze or pin is in force,
`fallbacks.refresh` is the frozen or pinned regime, so a refresh never moves a
paused block off the regime the person kept.

#### Payload schemas

All study payloads are validated with `extra="forbid"` and reject `NaN` and
infinities.

```text
RectV1            cx, cy in [-10, 10000]; w, h in (0, 2000]
LayoutSummaryV1   task in {pointing, menu, idle}; n_candidates 1..32; w_px, h_px (0, 400]; gap_px [0, 200];
                  id_bar [0, 12]; menu_n 0..32; origin_x, origin_y [-10, 10000] (home or menu button centre);
                  presses_mean [1, 64] optional (pointing grids: the mean keyboard presses over the
                  candidates; without it the keyboard prior approximates the grid)
DeviceV1          modality; pointer_types: up to 4 of {mouse, pen, touch}; dpr (0, 8];
                  surface_w, surface_h (0, 20000]; gaze bool; hand_pointer bool
SpreadV1          r (0, 20]; floor [0, 50]
GazeCheckV1       accuracy [0, 1]
ActivationGazeV1  region in {center, left, right, up, down}; conf [0, 1]
L2AppliedV1       decision_id uuid; cell_id ^b[123]:p[123]:h[01]$; draw_index 0..4; variant in L2_OPTIONS;
                  source in {table, fixed_rule, fallback, cap}; fallback_reason in {late, rpc_error, stale_table}
                  when source = fallback, else null
ActivationV1      index 0..31; t_ms [0, 60000]; modality; x, y integers [-100, 20000]; hit_index int|null;
                  press_index int -1..31 (never null; -1 when the press hit no candidate); a_gate [0, 1];
                  n_plausible 1..3; expanded_index int|null (enlarge regimes only);
                  hit_rect RectV1|null (frozen at the press; null iff hit_index is null);
                  gaze ActivationGazeV1|null (the region and confidence used, gaze sessions only);
                  l2 L2AppliedV1|null; irreversible_rule bool;
                  result in {committed, confirming, held, asked, none} (the immediate response, never overwritten);
                  outcome in {committed, cancelled, aborted, open_at_end, none} (what finally happened to it);
                  committed_index int|null
RevertV1          reason in {veto, refresh}; from_activation 0..32; regime (an L1 id: fallbacks.plain for a veto,
                  fallbacks.refresh for a refresh); fallbacks_issued_at ISO-8601 UTC of the fallbacks applied
MovementSummaryV1 rt_ms, mt_ms [0, 60000]; path_ratio [1, 50]; submovements 0..50; peak_speed_px_s [0, 50000];
                  reentries 0..50; endpoint_dx_norm, endpoint_dy_norm [-20, 20]
OfferResponseV1   decision_id; variant in L3 offers; response in {accepted, dismissed, ignored};
                  fallback_reason in {late, rpc_error, null} (set: no offer was shown, variant no_offer, ignored)
VetoV1            decision_id; level in {block_regime, ambiguity_response}
HandSummaryV1     quality_mean, quality_min [0, 1]; tracking_losses 0..100
GazeSummaryV1     agreed_with_intended bool|null
TrialV1           trial_id uuid; block_index 0..64; trial_index 0..64; task; regime; regime_decision_id;
                  table_decision_ids up to 4; applied_regime; l1_fallback_reason in {late, rpc_error, null};
                  fallbacks_issued_at ISO-8601 UTC (the fallbacks held at the cue);
                  spread SpreadV1|null (null iff the session's modality is keyboard);
                  layout LayoutSummaryV1; candidates 1..32 RectV1 (cue-time); surface_w, surface_h, dpr;
                  reversible list[bool] (one per candidate); cue_at ISO-8601 UTC; intended_index int|null
                  (null iff task is idle); modality; activations up to 32; revert RevertV1|null;
                  committed_index int|null; commit_t_ms [0, 60000]|null;
                  commit_via in {direct, confirm, hold, ask, none}; corrections 0..50; time_ms [0, 60000];
                  limit_ms in {10000, 15000, 30000}; duration_ms [0, 60000]; movement MovementSummaryV1|null;
                  veto VetoV1|null; offers up to 2 OfferResponseV1; hand HandSummaryV1|null;
                  gaze GazeSummaryV1|null; interrupted bool;
                  interruption_reason in {modal, hidden, blur, resize, palm_exit, paused, daemon, refused,
                  refresh, no_allowed_regime, null}; explanation_views 0..20
BurdenV1          block_index 0..64|null; scope in {block, session}; effort, intrusiveness, control 1..7|null;
                  tlx {mental, physical, temporal, performance, effort, frustration: 0..100|null}|null;
                  predictability 1..7|null; preferred_regime L1 id|null; skipped list of item names
FallbacksV1       l1, plain, refresh, practice: L1 id|null; freezable: up to 6 L1 ids in caution order
                  (may be empty); freeze_conflict {message, conflicting: [pref_id]}|null; l2: up to 3 L2 ids;
                  interruption_cap 0..8|null; issued_at ISO-8601 UTC
PreferenceV1      pref_id, kind, scope, value, origin = "explicit_user", created_at, retracted_at
```

- **Keyboard activations** (`modality: "keyboard"`): `x` and `y` are the focused
  candidate's cue-time centre rounded with `floor(c + 0.5)`; `hit_index` and
  `press_index` are the focused candidate; `a_gate` 0; `n_plausible` 1;
  `expanded_index` null; `hit_rect` that candidate's cue-time rect; `gaze` null.
  Keyboard sessions carry no spread.
- **`ActivationV1.result`** is set when the activation is applied: `committed`
  (committed a candidate on this activation), `confirming` (opened a confirm step,
  the irreversible-candidate rule included), `held` (started a hold), `asked`
  (showed a clarification) or `none` (`wait_for_evidence` from any source, the
  interruption cap, or no literal hit where one is needed). `outcome` is `none`
  exactly when `result` is `none`. The daemon recounts the interruption cap from
  `result`.
- **`L2AppliedV1.source`**: `table` (a table draw), `fixed_rule` (idle epochs and
  reverted trials), `fallback` (a late, failed or stale table, with
  `fallback_reason`) or `cap` (`wait_for_evidence` after the block reached
  `max_interruptions_per_block`). The daemon checks `fixed_rule`, `fallback` and
  `cap` responses against the fallbacks issued when they were applied. A
  `stale_table` fallback names the table requested after a veto, which had not
  arrived yet; the daemon relabels the table it names `baseline` unless the
  same trial then applied that table's draws.
- **`TrialV1.l1_fallback_reason`** is set only when the UI applied
  `fallbacks.l1` because the regime arrived late or the call failed; L2 and L3
  fallbacks are recorded on the activation and the offer.
- **`TrialV1.interruption_reason: "no_allowed_regime"`**: "Not this time" was
  pressed while `fallbacks.plain` was null; the trial records the veto, has no
  revert, and the block ends.
- **`FallbacksV1`**: `l1` is the fallback block regime (the frozen or pinned
  regime while one is in force, else the most cautious regime the rules allow and
  the calibration state permits); `plain` is the regime a "Not this time" revert
  applies (`assistance_off` when the rules allow it and calibration is not
  suspended); `refresh` is the regime a refresh revert applies (the frozen or
  pinned regime while one is in force, else `plain`); `practice` has the same
  value as `refresh` and is used for the practice block; `freezable` lists the
  regimes "Pause adapting" may freeze at; `freeze_conflict` is the conflict
  (`{message, conflicting}`) that blocks a freeze of the current regime, which the
  study shows with "Remove that rule" when `freezable` is empty;
  `interruption_cap` is the smallest active `max_interruptions_per_block`;
  `issued_at` is when the fallbacks were computed, which trials echo so each
  applied rule is judged under the rules the study held at that moment.

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
      "undo": { "kind": "not_undoable", "reason": "Heliox cannot reverse this kind of change automatically." },
      "index": 0
    }
  ]
}
```

`undo` says whether Heliox can undo the action afterwards, decided from the action type
alone: `undoable` (file write, move, copy or delete, clipboard writes, environment
variables, volume and brightness), `no_change`
(read-only) or `not_undoable` (anything that leaves the computer, and system changes
such as shell commands or package installs). It is `not_undoable` with "Undo is turned
off in Settings." when `[undo] enabled = false`.

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
  "plan_id": "a3b2c1f5",
  "undo": { "task_key": "4f0c2a9e", "available": true, "undoable": 2, "not_undoable": 1 }
}
```

`undo` is present when the task recorded anything in the undo journal (never for dry
runs). `available` is true while at least one change can still be undone; the UI then
offers **Review undo**, which calls `undo_preview`. The `execute` reply of the window
that ran the task carries the same `undo` and `receipt_id`, so the UI can show Undo and
**Verify receipt** on that task's own message.

Questions about the person's own history are answered in code, never by the planner or
a model, and return a normal conversational `execute` reply (and `voice_result` for
voice): "what was I working on …" and "when did I last open …" from the activity
timeline, and "what did you do today / yesterday / this morning" from task receipts.

### `undo_offer`
Sent when the person says or types "undo that", "undo the last task" or "undo the last
two tasks" (up to five) with no task running. `task_keys` lists the tasks that can still
be undone, newest first; tasks that changed nothing undoable are skipped and the reply
says so. The UI opens the preview for each in turn; every task still needs its own Undo
press, and Cancel stops the rest.

```json
{ "task_key": "4f0c2a9e", "task_keys": ["4f0c2a9e", "91b7d2c0"] }
```

### `open_command_palette`
`{}`: sent after "what can I say?". The UI opens the command palette (also Ctrl+K / Cmd+K),
which lists recent requests and the `command_guide` examples. Choosing one only fills
the command box; nothing runs until the person sends it.

### `nudge`
`{"id": "u1", "match": "youtube", "minutes": 60, "text": "You have spent 1 h 5 min on youtube today."}`.
Held during a focus session (then sent with `"held": true`).

### `focus_ended`
Broadcast when a focus session ends (on time or when stopped):
`{"text": "Focus session done: 45 minutes. You spent most of it in Code. While you focused: 1 reminder (stretch).", "duration": "45 minutes", "held": 1}`.
The held reminders follow as `reminder` notifications with `"held": true`, and a held
daily brief as `daily_brief`.

### `reminder`
Broadcast when a reminder falls due: `{"id": "a1b2c3", "due": 1792760400.0, "text": "stretch", "late": false}`.
`late` is true when it fell due while Heliox was not running (reminders more than a day
late are dropped). The daemon also speaks it.

### `routine_offer`
Sent when the person says or types "do that every day at 9" (also "every weekday at
8:30 am", "every Monday at 6 pm", "every 2 hours"). The payload is a `routine_suggest`
routine plus `daily_at`/`weekdays` or `every_minutes` and a readable `label`. Nothing
is scheduled until the UI calls `scheduled_run_create` after the person presses Create.

### `request_expanded`
Sent to the requesting window when a voice shortcut or "do that again" turned the input
into another request: `{"input": "open VS Code and Slack", "note": "Shortcut \"start work\": open VS Code and Slack"}`.
The expanded request then runs through the normal planning, gateway and approvals.

### `daily_brief`
Sent once a day at `[brief] time` when the brief is on (or right away after
`daily_brief_now`). `text` is the brief; `overview` is the same object `today_overview`
returns. The UI shows it in chat and as a desktop notification; with `[brief] speak`
the daemon also reads it aloud.

```json
{ "day": "2026-10-23", "text": "Today Heliox ran 3 tasks; the latest was \"tidy downloads\". ...", "overview": {} }
```

### `undo_complete`
Broadcast to every window after `undo_task` applied a task's undo.

```json
{ "task_key": "4f0c2a9e", "status": "ok", "undone": 1, "skipped": 1, "failed": 0 }
```

`task_complete` and `autonomous_complete` also carry `receipt_id` when the task ran at
least one action and a receipt was written.

### `phone_request`
Broadcast when a paired phone sends a request and whenever its status changes. The
payload is a PhoneRequest (see `phone_request_list`).

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

For durable voice workflows, `status` may be `submitted`, `revising`,
`answered`, `queued` or `cancelled`, with the spoken text in both `message`
and `result`, and the payload may include `workflow` or
`coordinated_correction: true`. A spoken request with two everyday desktop
verbs in sequence ("open Notepad, write a shopping list and save it"), or a
longer request naming an app, starts an adaptive workflow under the voice
permission ceiling. While it runs:

- "cancel" / "stop" cancels it (`cancelled`);
- "wait", "hold on" or "pause" pauses it at the next step boundary (`queued`);
- "what are you doing" or "status" is answered without touching the work
  (`answered`);
- an addition ("also save it as shopping.txt") amends the running step
  without restarting it (`queued`); Gnomon's latest-instruction check decides
  addition versus replacement when it is loaded, otherwise a leading "also",
  "and" or "then" marks an addition;
- anything else corrects the step and replans (`revising`).

Approval answers never change a workflow. "No", "deny" or "don't" denies the
pending approval (interactive, background, scheduled or workflow) through
`confirm` with `input_channel: "voice_deny"` and is reported as `answered`. The
message is "Okay, I won't run that." only when the denial took effect; when the
approval was already answered it says so instead. Unlike a denial sent through
`interject`, it does not override an approval the waiting task has not acted on
yet: `confirm` refuses any approval that is already resolved. "yes" or
"approve" returns `approval_required`, because approval stays in the visible
dialog. "Stop" with no interactive task
cancels running background jobs instead of being planned as a new task. A
paused workflow claims only continue and cancel, and a failed step says why
and waits about 90 seconds for continue, a correction or cancel. Multi-step
workflows announce each step as it starts.

### Autonomous and workflow progress

- `autonomous_started`, `autonomous_decomposed`, `autonomous_step_start`,
  `autonomous_step_complete`, `autonomous_complete`, and
  `autonomous_cancelled` carry the current autonomous job object.
- `scheduled_run_dispatched` carries `{ "run_id", "name", "job_id" }` when a
  schedule starts its job.
- `agent_tool_call` streams each action an autonomous job runs (background
  jobs, scheduled runs, MCP-submitted tasks, voice/gesture workflows), once
  with `"phase": "start"` and once with `"phase": "complete"` sharing a
  `call_id`: `{ "job_id", "goal", "source", "step", "total_steps", "round",
  "call_id", "phase", "action_type", "target", "arguments", "tier" }`, plus
  `success`, `executed`, `output`, `error` (each at most 600 characters) and
  `duration_ms` on completion. `arguments` is the JSON of the parameters the
  action set, redacted and capped at 400 characters. The chat shows one card
  per job with a row per call.
- `autonomous_paused_for_human` and `autonomous_resumed` carry the job object
  when a job waits for the person to hand the desktop back, and when it resumes.
- `human_control` carries the control status (see Human Control) whenever the
  holder changes.
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

### `co_adaptation_updated`
Broadcast by the co-adaptation service to every window when consent, rules or
learned state change.

```json
{ "jsonrpc": "2.0", "method": "co_adaptation_updated",
  "params": { "reason": "preferences", "participant_key": "p-3fa29c1e", "refresh_seq": 4 } }
```

`reason` is one of `consent`, `revoked`, `expired`, `preferences`, `reset`,
`erased`, `drift`, `calibration_suspended` and `proposal`. `refresh_seq` is set
only for `preferences`, `drift` and `calibration_suspended` when the change
marked that participant's open session stale, and equals the `refresh_seq` of
the `refresh_required` reply for the same change. A preference change sends one
notification per participant whose open session it marked stale; when it marked
none, it sends one notification with the rule's `participant_key` (null for a
device-wide rule) and `refresh_seq: null`, which only reloads status. During a
study the notification acts as `refresh_required` only for the running
participant and only when its `refresh_seq` is greater than the last one the
study applied (see **Refresh sequence** in section 1); otherwise it reloads
status. `revoked`, `expired`, `reset` and `erased` for the running participant
stop the study, because the daemon has already ended the session. The
notification is never deferred by attention-aware timing.

### `recorded_workflow_status_changed`
Sent after a successful `recorded_workflow_start` or `recorded_workflow_stop`,
with the new state. A running selection study pauses on it, and the daemon
re-checks for an active recording at the next study resume, block start or
recorded trial.

```json
{ "jsonrpc": "2.0", "method": "recorded_workflow_status_changed", "params": { "active": true } }
```

It goes to every window except the one whose request made the change, which
already has the new state in its reply.

### `config_updated`
Sent to every connected window after a settings write succeeds, whoever made it
(a Settings panel, another window, a voice command, `reset_config`). Windows
re-read the affected settings instead of saving a stale copy over the change.

```json
{ "jsonrpc": "2.0", "method": "config_updated", "params": { "method": "narration_config_update", "section": "" } }
```

`method` is the RPC that made the change; `section` is the `update_config`
section (empty for dedicated RPCs). `method: "reload"` means the daemon picked
up a hand edit of `config.toml` before a write. The notification is sent after
the reply to the request that caused it, and never for a refused write.

### `feature_announcement`
Emitted once on startup when a new daemon version introduces new capabilities.

```json
{ "message": "New Heliox capabilities are now available.", "version": "0.14.0" }
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
{ "text": "Pausing ten seconds before continuing. Say or press Stop to cancel.", "source": "stress_gate" }
```

`source` is one of `stress_gate` (executor.py's cognitive-stress-gate pause) or `autonomous_job` (AutonomousExecutor's end-of-job announcement). The stress-gate pause asks for nothing: the action continues after ten seconds unless the person stops the task.

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
│  ... later, if the user wants to undo the task │
│── undo_preview {task_key} ───────────────────►│
│◄── response: {preview: items, preview_token}  │
│── undo_task {task_key, preview_token, confirmed: true} ►│
│◄── notification: undo_complete {task_key, undone, skipped, failed}
│                                               │
│  ... or revert the whole disk snapshot ...    │
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
