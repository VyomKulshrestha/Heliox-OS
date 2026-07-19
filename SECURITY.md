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

## 📬 Contact

- **Maintainer**: [@VyomKulshrestha](https://github.com/VyomKulshrestha)
- **Private Advisory**: [Submit here](../../security/advisories/new)

---

## 🙌 Acknowledgements

Security researchers who responsibly disclose vulnerabilities will be credited in our release notes — unless you'd prefer to stay anonymous, which is completely fine.

This project participates in **GSSoC 2026** and **NSoC**. All contributors are expected to follow this security policy when handling security-related work.
