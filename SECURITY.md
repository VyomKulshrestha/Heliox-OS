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

## 📬 Contact

- **Maintainer**: [@VyomKulshrestha](https://github.com/VyomKulshrestha)
- **Private Advisory**: [Submit here](../../security/advisories/new)

---

## 🙌 Acknowledgements

Security researchers who responsibly disclose vulnerabilities will be credited in our release notes — unless you'd prefer to stay anonymous, which is completely fine.

This project participates in **GSSoC 2026** and **NSoC**. All contributors are expected to follow this security policy when handling security-related work.
