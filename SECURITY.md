# 🔐 Security Policy

Helix OS is a privacy-first, autonomous AI agent that executes real actions on your system. Security isn't an afterthought here — it's core to how we build. If you've found a vulnerability, thank you for taking the time to report it responsibly. This document explains how.

---

## 🛡️ Supported Versions

We actively maintain and patch the `main` branch only.

| Version | Supported |
| ------- | --------- |
| `main` (latest) | ✅ Active |
| Older releases | ❌ Not supported |

---

## 🔍 Scope

This policy covers the following Helix OS components:

- **🧱 Sandbox Execution** — isolated environments where code and plans are executed safely
- **🔑 Permission Tiers** — the five-tier permission system with confirmation gates and rollback support
- **🧩 Plugin Loading** — how third-party plugins are installed, verified, and executed at startup
- **🌐 WebSocket IPC** — the communication bridge between the Tauri UI and Python Daemon
- **🤖 Python Daemon** — the core backend driving agent orchestration, planning, and verification
- **🖱️ Gesture Cursor Control** — the continuous gesture-to-cursor bridge (off by default, opt-in only) that drives the real OS mouse cursor; the one capability in Heliox OS that acts without a per-action confirmation gate, so its escape hatches (open palm, stop button, disabling the setting) and screen-bounds clamping are treated as security-relevant, not just UX
- **🚪 Agent Gateway** — source-scoped permission floors, tamper-evident audit logging, and dry-run/simulation coverage for shell, browsing, and system-control actions, layered alongside the tier-based `PermissionChecker` (see the dedicated section below)

**Out of scope:**
- Third-party plugins not maintained by the Helix OS team
- Vulnerabilities in upstream dependencies (please report to their maintainers)
- Social engineering attacks

---

## 🚨 Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Public issues expose the problem to everyone before it's fixed — which could put users at risk. Instead:

1. **GitHub Private Advisory** *(preferred)* — use the [Security tab](../../security/advisories/new) to submit a private report directly.
2. **Email the maintainer** — contact [@VyomKulshrestha](https://github.com/VyomKulshrestha) via the email listed on their GitHub profile.

### What to include in your report

- A clear description of the vulnerability and where it lives
- Steps to reproduce the issue
- The potential impact (e.g., sandbox escape, permission bypass, plugin hijack)
- Any suggested fixes or mitigations *(totally optional, but appreciated!)*

---

## ⏱️ Responsible Disclosure Timeline

Once we receive your report, here's what you can expect from us:

| Timeframe | Action |
|-----------|--------|
| **Within 48 hours** | We acknowledge your report |
| **Within 7 days** | We assess severity and confirm the vulnerability |
| **Within 30 days** | We develop and test a fix |
| **Within 45 days** | We release a patch |
| **After patch release** | Public disclosure, coordinated with you |

We ask that you:
- Give us a reasonable window to fix the issue before going public
- Avoid exploiting the vulnerability beyond what's needed to demonstrate it
- Not access or modify other users' data

We'll do our best to resolve it as quickly as possible.

---

## ⚠️ Severity Levels

| Level | Description |
|-------|-------------|
| **🔴 Critical** | Remote code execution, full system compromise |
| **🟠 High** | Sandbox escape, privilege escalation, plugin permission bypass |
| **🟡 Medium** | Information disclosure, partial access control bypass |
| **🟢 Low** | Minor issues with limited real-world impact |

---

## Linux Syscall Guard

Restricted sandbox mode can enable a Linux-only seccomp-BPF guard for generated
code. The first guard blocks `unlink` and `unlinkat`, which prevents sandboxed
code from deleting files even if it reaches the subprocess runtime. Unsupported
platforms and Linux architectures fall back to the existing restricted sandbox
without blocking startup.

Config keys:

- `security.sandbox_kernel_guard`: enable or disable the guard. Defaults to
  `true`.
- `security.sandbox_blocked_syscalls`: syscall denylist. Defaults to
  `["unlink", "unlinkat"]`.

Manual verification on a supported Linux host:

```bash
cd daemon
python -m pilot.system.linux_syscall_guard --block unlink,unlinkat -- \
  python -c 'import os, pathlib; p=pathlib.Path("/tmp/heliox-guard.txt"); p.write_text("x"); os.unlink(p)'
```

The command should fail with `PermissionError` and leave the file in place.

---

## 🚪 Agent Gateway: Scoped Permissions & Audit

**Threat model.** Before this feature, permission policy was entirely global: `PermissionChecker`'s five-tier system applied identically no matter whether an action came from an interactive, user-confirmed request or an unattended autonomous background job (`autonomous_submit`). Two concrete gaps made this exploitable:

1. **All 22 `BROWSER_*` action types were hardcoded into `actions.py`'s `ALWAYS_SAFE` set**, forcing them to Tier 1 (`USER_WRITE`) regardless of what they actually did. `browser_execute_js` (arbitrary JavaScript execution in page context — cookie/token exfiltration, CSP bypass), `browser_fill_form`, `browser_navigate`, and `browser_click`/`type`/`select` all ran with **zero confirmation, zero snapshot, and zero audit trail** — identical treatment to a harmless `browser_screenshot`.
2. **`DestructiveCriticAgent` only ran inside `server.py`'s interactive `execute` RPC handler**, before `Executor.execute()` was ever called. Any caller reaching `Executor.execute()` a different way — most importantly autonomous background jobs — never passed through the critic at all.

Combined, an autonomous or web-agent-sourced plan could drive the browser (navigate anywhere, execute arbitrary JS, extract and exfiltrate page data) with no confirmation gate, no independent safety review, and no centralized, tamper-evident record of what happened.

**The fix — `pilot.security.gateway.AgentGateway`** — is a second gate checked *alongside* (never replacing) `PermissionChecker` inside `Executor.execute()`:

- **Source-scoped floors.** Every plan is tagged with an `InvocationSource` (`interactive`, `autonomous`, `web_agent`, `voice`, `gesture`). Each source has an enforced ceiling per action family (`shell`, `browsing`, `system_control`, `other`) and an explicit deny list — e.g. the `autonomous` profile denies `browser_execute_js`, `power_shutdown`/`power_restart`, and `registry_write` outright, and cannot reach root/Tier-4 actions at all. The `interactive` floor is a strict no-op, since interactive traffic already goes through the full tier/confirmation/critic pipeline.
- **Per-task overrides can only narrow, never widen.** A caller (e.g. `autonomous_submit`'s optional `scope_override` parameter) may further restrict a source's floor for one specific task, but `resolve_effective_profile()` combines the floor and the override via `min()` on tiers and set-union on deny lists — an override attempting to claim a *wider* tier or re-enable root access is silently clamped back to the floor, never honored. This matters because the override itself is untrusted input from an RPC call.
- **Critic-bypass closed.** `AgentGateway.authorize()` re-implements the same heuristic-risk trigger predicate `server.py` uses for interactive requests, so a Tier 3/4 or irreversible plan arriving without `critic_already_reviewed=True` (i.e. any non-interactive path) still gets an independent `DestructiveCriticAgent` review before it can proceed.
- **Browser actions retiered by actual risk**, not blanket-safe: read-only extraction (`extract`/`screenshot`/`list_tabs`/`page_info`/`wait`) stays untouched; state-changing actions (`navigate`/`click`/`type`/`select`/`fill_form`) now require confirmation like any Tier 2 action; `execute_js` moved to `DESTRUCTIVE` + `IRREVERSIBLE` given its exfiltration potential. **This is an intentional breaking change** for existing interactive users — some browser actions that ran silently before will now prompt. There is deliberately no legacy "go back to silent" toggle: one would simply reopen the gap this feature closes.
- **Tamper-evident audit trail.** `pilot.security.gateway_audit.AgentGatewayAuditStore` is a separate HMAC-SHA256 hash-chained SQLite log (same chain-of-custody design as the existing `PermissionEscalationAuditStore`, but its own database/key file) recording every gateway decision — source profile, action family, tier, whether an override was applied/whether it actually narrowed anything, allow/deny outcome, and a full policy snapshot. `verify_gateway_audit` walks the chain and detects any row that was deleted, reordered, or modified after the fact. Kept as an independent chain so a compromise of one audit key doesn't help forge the other.
- **Dry-run/simulation extended.** `SimulationSandbox` previously modeled shell/file impacts only; it now produces meaningful risk assessments for browser (navigation targets, script previews) and system-control (mouse/keyboard, process, registry) actions too, so a dry-run plan touching these surfaces gets real impact analysis instead of a generic fallback description.

**Known scope limit.** Only the three call sites named above (`server.py`'s interactive handler, `AutonomousExecutor`, `WebAgent`) are explicitly tagged with their real `InvocationSource` in this pass. Roughly twenty other sub-agent call sites that invoke `Executor.execute()` directly (`chain_planner.py`, `code_agent.py`, `comm_agent.py`, `forensics_agent.py`, `system_agent.py`, `network/mesh.py`, `swarm_router_agent.py`, `self_heal.py`, etc.) still default to the unrestricted `interactive`-equivalent floor — fail-open, not fail-closed, so nothing existing breaks. This is tracked as a follow-up, not hidden: any untagged call executing a Tier ≥ `SYSTEM_MODIFY` action still gets recorded in the ordinary (non-chained) `AuditLogger`, so the gap is observable even where it isn't yet restricted.

Settings → Agent Gateway Policy shows the enforced floor per source and lets you tighten it (never loosen it beyond the shipped defaults' intent); Settings → Agent Gateway Audit Log shows every recorded decision with a one-click integrity check.

---

## 🧠 Learned Risk Gate (opt-in)

**What problem this solves.** `destructive_critic.py`'s `heuristic_risk()` — the cheap rule that decides whether a Tier-3-only/irreversible-only plan is worth an LLM critic round-trip — only looks at plan length, distinct-target count, `dangerous_flags`, and tier mixing. It has no visibility into two concrete failure modes: a plan that would push disk usage to exhaustion, or a plan that touches a user-configured protected folder/package (`config.restrictions.protected_folders`/`protected_packages`) despite being otherwise unremarkable by every heuristic signal above (short, single target, no flags).

**Design, modeled on [Ferrum-OS](https://github.com/VyomKulshrestha/Ferrum-OS)'s `cognitive/world_model` architecture, adapted for one critical difference.** Ferrum-OS trains its predictive model by running thousands of synthetic actions against a disposable, from-scratch kernel booted in a throwaway QEMU VM — real telemetry, safely, because the VM can be destroyed and rebuilt endlessly. Heliox runs on the user's actual machine, so this feature does **not** attempt the equivalent (repeatedly running real destructive/root-level actions against a real computer purely to collect training data). Instead:

- **`pilot.security.risk_observation`** captures real OS telemetry (process count, disk usage, memory usage) via `psutil` — no fabricated numbers.
- **`pilot.security.risk_model`** encodes (OS state, proposed action) into a small fixed vector and predicts two *concrete, interpretable* outcome fields — predicted disk usage after, predicted process-count delta — never an opaque risk scalar. A rule-based table (mirroring the hand-crafted defaults the reference design ships even without any learned component) is always available as the fallback; a small numpy MLP (no PyTorch, no new heavy dependency) refines the prediction only for the specific action types real training data was actually collected for (file write/copy/delete/download, and process/shell/service spawn-kill — see below).
- **`pilot.security.risk_safety`** scores the *predicted* outcome using hardcoded, human-readable rules — predicted disk usage > 95%, a process-count delta shaped like a fork bomb, or a match against `config.restrictions.protected_folders`/`protected_packages`. **The model never decides what counts as dangerous** — it only predicts what would happen; a rule anyone can read decides whether that's a problem. This split is deliberate: the actual block/allow-influencing decision stays fully auditable regardless of how good or bad the learned prediction turns out to be.
- **`pilot.security.risk_gate.RiskGate`** ties the above together per plan (worst-case action wins), and `destructive_critic.py`'s `risk_score()` combines it with `heuristic_risk()` via `max()` — the gate can only ever add a reason to run the critic that `heuristic_risk()` alone would have missed, never suppress one it already raised. **Tier 4 (`ROOT_CRITICAL`) plans always get critic review regardless of this score, unconditionally, at both call sites** — this feature has no ability to affect that floor.

**Training data — what's real and what isn't.** `scripts/collect_risk_training_data.py` runs the small set of action types that are genuinely safe to repeat thousands of times for real: file writes/copies/deletes confined to one throwaway temp directory this script creates and removes itself, and trivial (`python -c "..."`) process spawn/kill standing in for `SHELL_*`/`CODE_EXECUTE`/`OPEN_APPLICATION`/`SERVICE_START`/`SERVICE_STOP`/`PROCESS_KILL`'s effect on process count — since actually launching real applications or starting/stopping real system services thousands of times would not be safe to repeat. Every other ActionType never reaches the learned model at all (the rule-based fallback handles them, predicting "no change" — an honest default, not a guess). `scripts/train_risk_gate.py` trains the small MLP on this real (not synthetic, not fabricated) dataset and writes `pilot/security/risk_gate_weights.npz`, checked into the repo.

**Off by default.** Unlike the Agent Gateway (which only ever *restricts*), this is a genuinely new detection capability not yet validated against real-world usage at scale — `config.gateway.risk_gate_enabled` defaults to `False`, the same opt-in-first treatment `gesture_cursor` gets for the same reason (a new capability, not a tightening of an existing one). Even when enabled, this is a second, *predictive* check that runs before, and independently of, the existing reactive `PermissionChecker`/confirmation gate — both must still pass; neither replaces the other, and user confirmation for Tier ≥ `SYSTEM_MODIFY`/irreversible actions is untouched regardless of whether the critic itself was skipped.

**Known scope limits, stated plainly.** The plan-level aggregation takes the single worst action's risk, not a lookahead simulation of repeating an action several times to catch compounding risk (the reference design's own further-staged extension) — `heuristic_risk()`'s existing `len(plan.actions) > 3` check is a coarser proxy for the same concern today. The learned model's disk/process-count predictions are trained on a real but modest sample (a few thousand rows from a single-machine sandbox), not validated across diverse hardware/filesystems — treat its refinement over the rule-based defaults as approximate until used more broadly, the same caveat this project has applied to every other empirically-tuned threshold introduced without a large real-world validation pass.

---

## 🩺 Autonomous Healing Engine (opt-in)

**What this is, and the pattern it borrows.** `pilot.agents.autonomous_healing.AutonomousHealingEngine` gives the daemon a passive, always-watching mode — modeled on IBM Power Autonomous Operations' self-healing pattern — instead of acting only when asked. It listens to the CPU/memory/disk checks `BackgroundTaskManager` already runs on a timer, and when one crosses its threshold, generates a remediation goal and plans it through the exact same `Planner`/`Executor` pipeline every voice/text/gesture command already uses. **It introduces no new safety primitive** — the tiering decision below composes `PermissionChecker`, `AgentGateway`, and (for Tier 3/4 or irreversible plans) the Learned Risk Gate/critic exactly as `Executor.execute()` already enforces for every other caller.

**Tiered autonomy — the confirmed design.**

- If the resulting plan is entirely low-tier (`plan.max_tier <= config.self_healing.auto_execute_max_tier`, default `1` = `USER_WRITE`) **and** contains no irreversible action (`plan.needs_confirmation_unconditional`), it runs immediately, no prompt.
- Otherwise the plan is **proposed, not executed** — broadcast as a `self_healing_confirmation_required` notification and held in the same `PendingConfirmation`/`_pending_confirms` mechanism `ThreatContainmentBridge` already established for background-initiated (non-request-driven) confirm-then-execute flows. It resolves via the existing generic `confirm` RPC (no dedicated approve/reject RPC was added — one plan_id, one existing code path). A denial, or a timeout after `config.self_healing.confirm_timeout_seconds` (default 300s), means nothing executes.
- A per-metric cooldown (`config.self_healing.cooldown_seconds`, default 600s) prevents a single sustained alert (e.g. CPU pegged for ten minutes) from re-triggering planning/execution on every poll interval.

**A second, independent gate — not just the engine's own pre-check.** A new `InvocationSource.SELF_HEALING` was added to the Agent Gateway (`pilot.security.gateway`) with its own `SourceProfile` ceiling, deliberately wide enough to still reach genuinely useful remediation actions (kill a runaway process, restart a stuck service, clear temp files — all `SYSTEM_CONTROL`-family, capped at `DESTRUCTIVE`) once a human has approved them via the propose-and-wait branch above, while permanently denying `power_shutdown`/`power_restart`/`power_logout`/`registry_write`/`browser_execute_js` and never reaching root/Tier-4 — regardless of what the engine's own tiering check would otherwise allow. This mirrors why the Agent Gateway exists at all: a plan's originating trigger should never be trusted to police itself.

**Off by default.** `config.self_healing.enabled` defaults to `False` — generating (and potentially auto-executing) a remediation plan without being asked is a new autonomous capability, not a restriction, so it gets the same opt-in-first treatment as `gesture_cursor`/`gesture_workflows`/`risk_gate_enabled`.

**Known scope limits, stated plainly.** Remediation goals are currently three fixed, generic templates (`config.self_healing.goal_templates` can override them per metric) fed to the same LLM planner every other command uses — there is no dedicated "diagnose root cause" reasoning step, so the quality of the generated plan depends entirely on the planner's ordinary judgment for that prompt. The network-activity monitor (`monitor_network`) is intentionally left unwired: a generic "network usage is high" alert has no obvious safe remediation action, so wiring it would either do nothing useful or invite an overly broad goal template. Not verified against a real sustained-load scenario in this pass — the on-trigger wiring and tiering logic are covered by unit tests with a fake planner/executor, not a live high-CPU/low-disk machine.

---

## 🎯 Pre-Execution Target Assessment (browser actions)

**What problem this solves.** Before this, `SimulationSandbox`'s dry-run impact modeling for `browser_click`/`browser_click_text`/`browser_type`/`browser_select`/`browser_fill_form` was purely templated text (`f"Click a page element: {target}"`) — it had no way to say whether the target actually existed on the current page. A plan proposing to click `#submit-btn` on a page with no such element would get exactly the same dry-run description as one that would work fine.

**Why not a generative visual world model.** The natural-sounding fix — predict the future screen state from an action, the way World Labs' RTFM or NVIDIA Cosmos do for 3D scenes — was evaluated and rejected for this codebase: every self-hostable option in that space needs dedicated GPU hardware (RTFM: a single H100; Cosmos: H100/H200) and is built for photorealistic video generation, not "will this CSS selector resolve." The GUI-agent-specific research that *is* the right shape (Code2World's renderable-code prediction, CUWM's post-action UI-state reasoning) is unreleased academic work with no available model or weights. Ferrum-OS's own world model — the design this repo's Learned Risk Gate is already based on — was also checked directly: it folds a screen-text OCR read into its state embedding, but only as a single collapsed hash with its own authors explicitly excluding `mouse_click`/`keyboard_type` from having any modeled effect at all.

**What this does instead — structural, not generative.** `pilot.system.dom_diff.assess_target()` checks a click/type/select/fill_form target against the DOM snapshot already captured immediately before the action would run (the same `DomSnapshot` infrastructure `dom_diff.py` already used for *after*-the-fact self-correction, now also used *before*). It reports whether the target exists, is visible, and is unambiguous — using the live current page state, not a simulated future one:

- Simple selector forms (`#id`, `.class`, `tag`, and combinations) are resolved directly against the snapshot's flat node list.
- Combinators, attribute selectors, and pseudo-classes (`form > button`, `input[type=submit]`, `:has-text(...)`) are **not** guessed at — `TargetAssessment.matchable=False` is returned honestly rather than a wrong answer.
- If no browser session is open yet, the check is a complete no-op (`SimulationSandbox.simulate()` — now async — never launches a browser itself; a dry-run must stay a genuine no-op).
- When a target is missing, hidden, or resolves to multiple visible elements, the plan's risk for that action is raised to `HIGH` and a `predicted_issue` is attached to the impact item, surfaced in the dry-run report the user sees before confirming.

**Known scope limits, stated plainly.** This only ever inspects the *current* page — it cannot predict the DOM after a prior action in the same plan would run (e.g. action 2's target existing only after action 1 navigates), so multi-step plans only get an accurate assessment for their first browser action against the page open right now. `browser_fill_form` checks each field selector plus the submit selector independently, not whether the form as a whole is submittable. Not a replacement for the existing DOM-diff self-correction in `system/browser.py` — that still runs at execution time regardless of what this predicts.

---

## 🗣️ Live Execution Narrator (opt-in)

**What this is.** `pilot.agents.narrator.ExecutionNarrator` narrates plan execution as it happens (voice, via the frontend's browser `speechSynthesis`) and can pre-emptively pause a plan or a single browser action that gets flagged as risky *before* it runs — pairing the spoken interjection with a visual confirmation modal, always together, never one without the other. This is the "interrupts you, corrects you, talks to you while executing" capability, built on infrastructure this codebase already had rather than a new heavy model — see below for why.

**Why not a real-time interaction/interruption model.** The natural comparison is Thinking Machines Lab's "Interaction Models" (`TML-Interaction-Small`) — a model that treats audio/video/text as continuous streams and decides speaking/listening/interrupting at the token level. It's genuinely the right shape for this, and it's genuinely inaccessible: no open weights, no public API, invite-only research preview only, as of this writing. Ferrum-OS's own world model — already reused for this codebase's Learned Risk Gate — was checked directly and found to be the wrong substrate too: its "screen" signal is a single collapsed hash of OCR'd text with zero structure, and its own transition-model rule table explicitly excludes `mouse_click`/`keyboard_type` from having any modeled effect at all. Kyutai's Moshi was flagged (at the time this section was written) as a credible future full-duplex backbone. Research since found Moshi is no longer Kyutai's actively-promoted product — see **Kyutai Pocket TTS** below for what was actually adopted from that research, and why Moshi's genuine full-duplex/barge-in capability remains out of scope (GPU-only, English-only).

**What this does instead — composition, not a new safety primitive.** Two independent trigger sources, both wired automatically the moment `Executor.set_narrator()` is called (no changes needed in `AutonomousExecutor`, the Voice/Gesture Workflow Engine, or the interactive path):

- **Ambient narration** (`on_action_start`/`on_action_complete`) — always non-blocking, speaks a short description of each action as it starts/finishes.
- **Risk-triggered interrupt** (`on_plan_risk`/`on_target_assessment`) — gates a plan or a single browser action using signals that already existed: the Agent Gateway's own critic verdict (a `WARN` on an otherwise-allowed plan was previously computed and then silently discarded by `Executor.execute()` — now captured and surfaced) and this session's `dom_diff.assess_target()` pre-execution check (previously dry-run-only, now also reachable at real-execution time). The interrupt-and-wait itself reuses the exact `PendingConfirmation`/`confirm` RPC mechanism `ThreatContainmentBridge` and the Autonomous Healing Engine already established — no new blocking primitive, no new RPC for approval/denial.

**Pre-emptive only, never mid-flight — at the time this was written.** Both interrupt paths gate a plan/action *before* it runs — `on_plan_risk` fires right after `AgentGateway.authorize()`, before any batch executes; `on_target_assessment` fires right before that one action's real dispatch. Neither attempts to cancel something already executing (e.g. a shell command mid-run) — see **Mid-flight Cancellation** below, which closes that specific gap in a later pass.

**Off by default.** `config.narration.enabled` defaults to `False` — spoken interruptions and pausing execution on a risk signal are new user-facing behavior, not a pure restriction, so this gets the same opt-in-first treatment as `gesture_cursor`/`gesture_workflows`/`self_healing`/`risk_gate_enabled`.

**Known scope limits, stated plainly.** Voice output goes through the frontend's browser `speechSynthesis`, not the daemon-side `pilot.system.voice.speak()` used elsewhere in this codebase (e.g. `AutonomousExecutor`'s end-of-job announcement) — those two voice paths remain independent and can, in principle, both be speaking at once if both fire; this pass didn't unify them. `pilot.system.voice.speak()` itself now supersedes any of its own still-in-progress calls (a later fix, mirroring `tts.ts`'s `speechSynthesis.cancel()`-before-every-speak — see `_supersede_current_speech()`/`_current_speech_task` in `voice.py`), so two daemon-side callers (e.g. `executor.py`'s cognitive-stress-gate phrase and `AutonomousExecutor`'s end-of-job announcement) can no longer talk over each other; this closes the daemon-side half of the original gap, not the cross-path one described above. Not verified against a live daemon executing a real risky plan in this pass — the on-trigger wiring is covered by unit/integration tests with fakes (mirroring `test_autonomous_healing.py`'s pattern), and the frontend pieces were verified live in-browser (the TTS utility genuinely invokes `speechSynthesis`, the store loads with the correct default state, the dialog renders/hides correctly), but the full backend-push-to-modal-and-voice round trip needs a real execution to trigger and wasn't exercised end-to-end here.

---

## 🔊 Kyutai Pocket TTS (default daemon-side voice)

**What this is.** `pilot.system.pocket_tts` is now the default engine behind `pilot.system.voice.speak()` (`config.voice.tts_engine`, default `"pocket_tts"`), replacing the previous default of platform-native TTS (Windows SAPI via PowerShell, macOS `say`, Linux `espeak`) as the primary daemon-side voice. It came out of the same research that flagged Kyutai's Moshi above: Moshi itself has since been superseded for production use by a cascaded stack (Unmute, Kyutai STT/TTS, Pocket TTS), and — critically — every piece of that stack *except* Pocket TTS requires a 16GB+ VRAM GPU on Linux/WSL, a poor fit for a general Windows/Mac/Linux desktop app. Pocket TTS (100M params) is the one piece that's genuinely CPU-only and cross-platform, reported by Kyutai at ~6x real-time generation using 2 CPU cores.

**Not the full-duplex capability.** This is a TTS-quality change only — it does **not** provide Moshi's genuine mid-sentence barge-in (simultaneous listen-while-speaking at the token level). `speak_interruptible()`'s existing cancel-and-restart barge-in mechanism (racing playback against the VAD recorder's `wait_for_speech_start()`, see the Continuous VAD-based recording section of `voice.py`) is completely unchanged by this pass — Pocket TTS's `play()` is just as cancellable via `sounddevice.stop()` as the previous `proc.kill()`-based OS-native paths were.

**Free and fully local, by design.** Kyutai's Pocket TTS code is MIT/Apache-2.0 and the model weights are CC-BY-4.0 — no API key, no per-request cost, no cloud inference of any kind. The only network activity is a one-time ~236MB model download from Hugging Face the first time `speak()` is actually called — after that, everything runs offline on the user's own CPU.

**Ships by default, not a manual opt-in.** `pocket-tts` + `sounddevice` (plus `openai-whisper` for the speech-to-text side of voice) live in `pyproject.toml`'s `voice` extras group, which is part of `all`. The real end-user install paths — `tauri-app/src-tauri/src/main.rs`'s `setup_venv_in_background()` (first-run background setup for the desktop app) and `packaging/debian/postinst` (the `.deb`) — both request `pilot-daemon[all]`, so a normal install gets working local voice out of the box, not a config default that silently no-ops until someone separately runs a manual pip install. `config.voice.tts_engine` still lets the user choose Pocket TTS vs. their OS's built-in voice in Settings — the package being present is what makes that choice actually work either way, rather than one option being permanently dead unless a developer intervenes. It remains **not** a hard dependency at the Python-package level: `_speak_impl`'s Pocket TTS branch still catches `ImportError` and any other synthesis/playback failure and falls straight through to OS-native dispatch, so a dev/CI environment that hasn't installed the `voice` extra (this repo's own `requirements-lock.txt`/CI setup, mirroring `openai-whisper`'s existing treatment there) behaves exactly like the pre-Pocket-TTS daemon, with zero network calls ever attempted.

**Known scope limits, stated plainly.** Uses Pocket TTS's one-shot `generate_audio()` API, not its chunked/streaming mode — simpler and consistent with `speak()`'s existing "one atomic, cancellable operation" model, at the cost of the ~200ms first-chunk-latency improvement streaming would give. Only the default English model/voice set is wired up (`tts_voice`, a handful of built-in presets) — the additional languages Pocket TTS has since added are not surfaced in Settings. The cognitive-load-based speech-rate modulation `_speak_impl` already applies to the OS-native paths does not carry over to Pocket TTS, which has no equivalent integer rate knob. Verified end-to-end against the real model on real hardware (not just the mocked/fallback paths automated tests cover): a real first-use download (~42s), `sample_rate` read at runtime as 24000Hz, real CPU inference, and real audio played through actual speakers. **Separately, and pre-existing to this feature**: `pilot-daemon` is not currently published to PyPI at all, so `main.rs`'s `pip install pilot-daemon[all]` does not yet succeed for a real end user until someone publishes it — this affects every feature installed this way, not specifically voice, and remains an open gap outside this repo's automated CI.

---

## 🧠 Cognitive Engine (attention/stress/intent heuristics)

**What this is.** `pilot.cognitive.cognitive_engine.CognitiveEngine` estimates attention/stress/cognitive-load and backs the Attention-Aware UI, Stress-Aware Task Gating, JARVIS intent disambiguation, and User Manual Supervision's cognitive coaching. It replaced an earlier integration with Meta's TRIBE v2 (`tribev2`, weights from Hugging Face's `facebook/tribev2`).

**Why TRIBE v2 was removed.** Both the `tribev2` code (GitHub) and its `facebook/tribev2` model weights (Hugging Face) are licensed CC-BY-NC-4.0 (non-commercial), which is incompatible with a commercial product — confirmed directly against both the GitHub repo's license file and the Hugging Face model card's license tag, not assumed. There is no comparable open, permissively-licensed model in the "predict brain responses to stimuli" niche — that space is exclusively research-only releases — so a heuristic estimator is the practical lightweight/open/free alternative here, not a stopgap pending a better model.

**No functional regression for real installs.** `TribeEngine` already treated the real model as optional (imported in a try/except with a documented heuristic fallback), and every real end-user install path (`main.rs`'s `setup_venv_in_background()`, `packaging/debian/postinst`, this repo's own CI) never had a way to install `tribev2` in the first place — it required a separate manual `pip install git+https://github.com/facebookresearch/tribev2.git` that no shipped install path ever ran. So every real install was already running in heuristic-fallback mode; `CognitiveEngine` simply makes that the sole, permanent, and honestly-named implementation instead of a fallback for a model nobody's install ever actually loaded.

**Signal sources, and a real audit that found several were being silently discarded.** `predict_cognitive_state()` now blends four independent signal streams, each recency-decayed (`exp(-age/tau)`, tau=20s, hard horizon 90s — old samples fade smoothly instead of falling off a hard 30s cliff):
- `record_interaction()` — UI/action event log (frequency, intensity, event-type diversity). Unchanged in shape, but `StressGate.evaluate()`'s call site had a real bug: it fed the engine's own just-computed `stress_level` back in as the next sample's `intensity`, a circular feedback loop rather than an independent signal. Now records the action's own fixed, objective risk tier instead.
- `record_input_dynamics()` — real keystroke/click cadence from `InputSupervisionHook.snapshot()`. This was already computed by `UserSupervisionEngine.tick()` on every tick and then completely discarded; now it reaches `CognitiveEngine` too.
- `record_gaze()` — gaze region/confidence from the frontend's webcam gaze tracking, via `multimodal.fusion.on_gaze_event()`. Previously used only for a small intent-confidence bonus; an off-center gaze now also registers as a mild attention/distraction signal.
- A small auditable keyword table (mirrors `pilot.security.risk_patterns`' "named rule, never persist the source text" contract) applied to whatever `stimulus_description` a caller passes. This parameter existed on the API since the TRIBE v2 days but `_predict_with_heuristics()` never actually read it — every caller building a real stimulus string (OCR snippet + window title in `UserSupervisionEngine`, app/window/VLM description in `screen_vision.py`) was doing that work for nothing. Only two bounded floats are ever derived from the text; the string itself is never stored or logged.

**Confidence now reflects real data richness** (how many signal streams have recent samples) instead of two fixed constants — still capped at 0.6, well below what a real trained model would report, so downstream consumers never mistake this for more certain than it is.

**Known scope limits, stated plainly.** These are frequency/cadence/keyword/threshold heuristics, not a neural cognitive-state model. Intent-affinity scoring uses Jaccard word-overlap (intersection/union) plus a multi-word phrase-containment boost rather than a semantic model — it disambiguates literal wording, not meaning. If a genuinely open, permissively-licensed model in this space becomes available, swapping it in only requires reimplementing `CognitiveEngine`'s method surface (`predict_cognitive_state`/`predict_attention_map`/`predict_intent_affinity`) — every consumer (`AttentionAwareUI`, `StressGate`, `IntentPredictor`, `UserSupervisionEngine`, `screen_vision.py`, `subconscious.py`, `fusion.py`) already treats it as a duck-typed dependency injected at construction time, not tribev2/TRIBE-specific.

---

## 🛑 Mid-flight Cancellation

**What problem this solves.** Both the Live Execution Narrator (above) and the Learned Risk Gate explicitly deferred the same gap: every existing cancellation path — `_handle_abort`'s `cancel_event`, the narrator's pre-emptive interrupt — only ever gates *before* a risky plan/action starts or checks at the next action boundary. None of them could stop something already running (e.g. a shell command mid-execution). This closes that gap for the interactive command path specifically.

**Two mechanisms, because one kill primitive doesn't cover both execution models:**

- **Real task cancellation for subprocess-based actions.** `PilotServer._execute_tracked()` wraps the interactive path's `Executor.execute()` call in a genuine `asyncio.Task` (`self._active_execution_task`). `_handle_abort` now cancels that task directly, and the cancellation propagates all the way down to `platform_detect.py`'s `run_command`, whose existing `except asyncio.CancelledError: proc.kill(); raise` already proves this mechanism works — it's the same pattern `AutonomousExecutor.cancel()` has used for its own jobs all along; this pass just gives the interactive path the same real task handle to cancel.
- **On-demand interrupt for PTY sessions.** `pty_exec` commands run via `PtySession`'s blocking `select()`/`os.read()` loop in a thread-pool thread — `Task.cancel()` on the outer coroutine cannot stop a blocking call already running in a different thread. `PtySession.interrupt()` sets a `threading.Event` that the read loop's existing 100ms-granularity poll checks, making it return early exactly as if it had timed out — landing in `_run_command`'s pre-existing Ctrl+C-and-recover branch rather than duplicating that logic. `PtySessionManager.interrupt_all()` (called unconditionally by `_handle_abort`) interrupts every live session, since the abort path is session-scoped/singular ("stop the current execution"), not correlated to one specific `session_id`.

**Composes with, does not replace, the existing cooperative signal.** `_handle_abort` still sets `cancel_event` first (unchanged boundary-only behavior for the Orchestrator/Executor's own internal batch loop) *and then* cancels the tracked task and interrupts PTY sessions — by the time the resulting `CancelledError` reaches `_handle_execute`, `cancel_event` is already set, so it falls through to the pre-existing "Cancel Token" response path (`{"status": "cancelled", ...}`) rather than needing new response-shaping logic.

**Known scope limits, stated plainly.**
- **Only the main interactive command path is covered** — `PilotServer._handle_execute`'s fresh-plan and resume-from-checkpoint call sites. The other `Executor.execute()` call sites (voice command dispatch, the generic action-command handler, git-conflict-resolution) are single quick actions outside the "Stop button" scope and were left untouched. Workflows/autonomous jobs already have their own working cancel path (`AutonomousExecutor.cancel()`) and were not changed here.
- **Not every `ActionType` is a real cancellable subprocess.** `code_execute`, `download_file`, and other long-running actions are only interruptible to the extent their own underlying implementation is a genuine cancellable subprocess/async call — this pass did not individually audit or guarantee that for every action type.
- **PTY sessions are Unix-only.** Windows has no PTY support in this codebase at all (`PtySessionManager.get_session` raises `RuntimeError` on `win32`); the interrupt mechanism only matters on platforms where `pty_exec` runs in the first place.
- **A narrow interrupt-consumption race.** If `PtySession.interrupt()` is called while no command is actually running, the flag is consumed (cleared) by whatever `_read_until` call happens next — which could, narrowly, cause an unrelated *later* command to bail early instead. This mirrors the same class of un-awaited-cancellation race already present in `AutonomousExecutor.cancel()`'s original form; callers only invoke `interrupt()`/`interrupt_all()` while a command is genuinely known to be in flight, so this wasn't hardened further.

---

## 👁️ User Manual Supervision (opt-in)

**What this is.** `pilot.agents.user_supervision.UserSupervisionEngine` watches the user's OWN independent screen/keyboard/mouse activity — never anything Heliox itself executes, that's the Live Execution Narrator's job above — and can offer a spoken cognitive check-in or a risk warning. This is the single biggest privacy surface in this codebase, built and gated with that weight in mind rather than as a routine feature toggle.

**Two independent, advisory-only trigger sources**, evaluated on one periodic tick (`BackgroundTaskManager`, the same precedent the Autonomous Healing Engine uses):

- **Cognitive coaching** — `pilot.cognitive.cognitive_engine.CognitiveEngine.predict_cognitive_state()` is fed a *real* stimulus for the first time here (an OCR screen snippet plus the active window title), instead of the synthetic activity labels every other call site in this codebase uses (window titles alone, notification metadata, static action-type strings, or the frontend's own client-side mouse-activity labels). A sustained stress/cognitive-load threshold crossing triggers a gentle check-in.
- **Risk-pattern detection** — the OCR snippet and a transient keystroke buffer (see below) are matched against `pilot.security.risk_patterns`' small, explicit, hardcoded regex table — the same "auditable rules, never a learned model" philosophy the Learned Risk Gate's `risk_safety.py` already established. A match triggers a direct warning.

**Advisory only, never a gate.** Unlike the Live Execution Narrator, which pauses a Heliox-issued plan/action *before it runs* via a real blocking confirmation, Heliox has no way to intercept or block the user's own OS-level input — it only observes a copy via the hook described below. Both trigger methods return `None`, not a bool; there is nothing to approve or deny, so the frontend pairs a spoken interjection with a dismiss-only modal, not an approve/deny one.

**Why pattern-matching over an LLM call on raw content.** Correlating raw screen/keystroke content through an LLM — even a local one — would be a strictly bigger leak surface than matching it against a small, explicit, human-readable pattern table entirely in-process. The pattern table lives in `pilot/security/risk_patterns.py`, is short enough to read end to end, and is the only thing standing between "something risky-looking happened" and a warning being shown.

**The privacy contract, stated as code, not just as a design note.** `pilot.system.input_hook.InputSupervisionHook` is the actual boundary:

- Raw keystrokes are buffered *only* in a small, bounded, in-memory deque — purely to be joined into a short-lived local string, pattern-matched, and then immediately discarded, regardless of whether anything matched. That joined string never gets logged, returned, or persisted anywhere — only the matched pattern's *name* (e.g. `"destructive_shell_command"`), or `None`, ever leaves `snapshot()`.
- Mouse clicks are counted, never located — click coordinates are never read, buffered, or stored anywhere in this codebase; only a click-rate number exists.
- The OCR snippet used for both triggers is processed in-memory per tick and never persisted verbatim — only the derived cognitive snapshot and matched-pattern name ever reach a notification or log.

**A tiered opt-in, not one switch.** `config.supervision.enabled` (screen/OCR-based cognitive coaching + risk warnings) and `config.supervision.keyboard_mouse_hook_enabled` (the global keyboard/mouse hook) are deliberately separate flags — a user can turn on the milder capability without ever installing the hook. Both default to `False`. The Settings UI gates the hook toggle further, behind a one-time "I understand" checkbox, since it exceeds every other privacy warning already in this app (including the gesture-cursor-control warning).

**Known scope limits, stated plainly.**
- **UAC/elevation boundary**: a non-elevated hook cannot observe keystrokes delivered to an elevated (Administrator) window — a real blind spot for exactly the kind of destructive command (e.g. an elevated shell) the risk trigger is meant to catch.
- **Antivirus/EDR risk**: a global keyboard hook is a textbook keylogger signature; running this feature carries a real risk of the daemon being flagged or quarantined by antivirus or anti-cheat software on the user's machine.
- **Silent hook death**: Windows silently removes a low-level keyboard/mouse hook whose callback takes too long, with no in-process exception raised. `hook_healthy` (surfaced via the `supervision_status` RPC and the Settings panel) is a best-effort liveness signal, not a guarantee — it can only detect that the listener thread itself died, not that the OS quietly unhooked it.
- **Windows-only verified.** macOS requires an explicit Accessibility/Input Monitoring permission grant outside this codebase's control; Linux/Wayland is likely broken since `pynput`'s hook backend relies on X11/Xlib. Neither was exercised in this pass.
- No per-application context for keystroke or OCR matches — this is flat, content-only pattern matching, false positives included.
- Not verified against a live daemon with the real OS-level hook installed in this pass — the engine's trigger/cooldown logic and the hook's buffer/privacy-boundary behavior are covered by unit tests with fakes (mirroring `test_narrator.py`'s pattern), and the frontend pieces were verified live in-browser (the panel's toggles, the "I understand" gate, and the dismiss-only dialog all render and behave correctly), but the actual `pynput` listener installation and a real risky-keystroke round trip need a real Windows session and weren't exercised end-to-end here.

---

## 📬 Contact

- **Maintainer**: [@VyomKulshrestha](https://github.com/VyomKulshrestha)
- **Private Advisory**: [Submit here](../../security/advisories/new)

---

## 🙌 Acknowledgements

Security researchers who responsibly disclose vulnerabilities will be credited in our release notes — unless you'd prefer to stay anonymous, which is completely fine.

This project participates in **GSSoC 2026** and **NSoC**. All contributors are expected to follow this security policy when handling security-related work.
