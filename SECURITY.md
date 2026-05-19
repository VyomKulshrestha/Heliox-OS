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

## 📬 Contact

- **Maintainer**: [@VyomKulshrestha](https://github.com/VyomKulshrestha)
- **Private Advisory**: [Submit here](../../security/advisories/new)

---

## 🙌 Acknowledgements

Security researchers who responsibly disclose vulnerabilities will be credited in our release notes — unless you'd prefer to stay anonymous, which is completely fine.

This project participates in **GSSoC 2026** and **NSoC**. All contributors are expected to follow this security policy when handling security-related work.
