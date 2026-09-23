# Heliox OS Daemon

Python backend for the Heliox OS AI System Control Agent. It provides the
append-only experience ledger, durable task loop, temporal context, companion
coordination, hybrid world model, verified online learning, strategy/evolution
harnesses, 21-specialist mesh, Planner/Executor/Verifier, security layers, and
the 157-action system interface.

The interactive companion also has a small offline Jev-inspired scope head.
It adds warning-only signals for selected side-effect actions; it is not a
permission gate or a general-purpose replacement for Jev. See the
[evaluation and limits](../docs/INTELLIGENCE_EVALUATION.md#jev-inspired-local-scope-decision).

The optional neural-intent research path runs acquisition and decoding in the
separately authenticated `pilot-neurod` sidecar. It supports synthetic,
playback, BrainFlow, and local LSL sources, but can emit only signed bounded
intent for dedicated navigation or compiled reversible Tier 0/1 goals. It has
no physical, destructive, arbitrary-command, or approval authority. See
[Neural Intent Research Controls](../docs/NEURAL_INTENT.md).

```bash
pip install "pilot-daemon[neural]"
pilot-neurod --source synthetic --synthetic-frequency 12
pilot-neurod-benchmark brainflow-synthetic --seconds 2
pilot-neurod-benchmark eegbci --subject 1 --runs 6 10 14
```

The benchmark commands need no headset. Their JSON output identifies evidence
as `synthetic` or `recorded_eeg`; neither result is live brain-control evidence.

Typed and spoken requests enter the same observable interaction state machine.
Interactive browser and desktop goals use a bounded observe, act, and verify
loop with fresh screen evidence, target-window re-acquisition, and explicit
no-progress limits. Native application launch is fail-closed and platform
specific: Windows resolves installed-app records and shortcuts, macOS uses
Launch Services, and Linux accepts verified executables or desktop entries.
Issuing a launch command is never treated as proof that the user's goal is
complete.

Bounded read-only health-review requests use the local
`system_health_review` action to collect current CPU, memory, disk, battery,
and process evidence without a model call. Cloud planning supports Gemini,
OpenAI, OpenRouter, Claude, and Meta; OpenRouter accepts automatic routing or
an exact catalog model ID such as DeepSeek V4 Pro. Provider keys remain in the
operating-system credential store.

Planning can also use an existing Codex/ChatGPT or Claude Code subscription
through that provider's official CLI. Heliox delegates login to the CLI and
never reads its OAuth state. The adapter is text-only and cannot execute tools:
Codex runs ephemeral in a sterile read-only directory with user rules ignored;
Claude runs without tools, browser integration, slash commands, or persistent
sessions. Returned plans still pass through the normal policy, approval,
execution, and verification pipeline.

An opt-in benchmark measures this path without executing actions:

```bash
python benchmarks/subscription_planning_suite.py --provider codex
```

The report separates Heliox prompt size from provider input, cached input,
uncached input, output, and latency. Running it consumes the provider account's
subscription allowance.

See the [main README](../README.md) and
[Architecture](../docs/ARCHITECTURE.md) for the full runtime contract.

## SSH Agent (Remote Host Execution)

Heliox includes an optional `SshAgent` for `ssh_command` and `ssh_script`
actions against preconfigured, allowlisted hosts through Paramiko.

### Install

```bash
pip install "pilot-daemon[ssh]"
```

### Configure allowed hosts

Open **Settings > Integrations > SSH**, add a named host alias, hostname, port,
username, private-key PEM, and optional passphrase, then use **Test connection**.
Strict known-host verification is enabled by default. The authenticated daemon
stores only non-secret host metadata in `config.toml`; private keys and
passphrases go directly to Windows Credential Manager, macOS Keychain, or a
Secret-Service-compatible Linux keyring. Raw credential-storage RPCs are not a
supported user setup path.

Keys and passphrases are never logged and are retrieved only when needed. SSH
actions retain the dedicated `ssh_agent` gateway profile, exact host allowlist,
confirmation, irreversibility, audit, and verification behavior.
