---
title: Heliox OS changelog
canonical_url: https://www.helioxos.dev/changelog.md
last_updated: 2026-10-16
---

# Heliox OS changelog

This page is generated from the changelog shipped by the Heliox daemon. It distinguishes published installers from draft and source-only milestones; current limitations remain in the proof center.

## 0.14.0 — Co-Adaptive Assistance and Governed Desktop Agency

Status: **Published**

Released: **2026-10-16**

Consented assistance co-adaptation, accessibility-grounded desktop control, durable and scheduled autonomy, MCP tools, voice-first control while working, and fail-closed settings and approvals

- **Assistance Co-Adaptation** — An opt-in, consented selection study models how a person, the interface and Heliox adapt to each other and predicts which help suits this person today (a larger target, a confirm step, a clarifying question or another input) within rules the person sets. It changes presentation only, never commits an ambiguous selection by guessing, and every published result so far comes from simulated users.
- **Accessibility-Grounded Desktop Control** — Desktop clicks are grounded in the accessibility tree instead of pixels, buttons can be pressed in the background without moving the cursor, browser actions use page snapshots, and the person can take the desktop back at any time while Heliox can hand it over when it needs help. No virtual machine is involved.
- **Durable and Scheduled Autonomy** — Autonomous jobs survive daemon restarts and resume only when the person asks, scheduled runs are limited to an approved list of actions, unreachable goals are reported instead of faked, and each tool call of background work appears as a card in chat.
- **Tools from Your MCP Servers** — Heliox can call tools from the person's own MCP servers listed in mcp_servers.json. Read-only tools run directly, every other tool asks first, and Settings shows each server with a reload button.
- **Voice-First Control While Heliox Works** — Speaking while a task runs does what the person means: corrections, questions and a spoken no reach the running task, background approval dialogs stay open while the person keeps working, and the approval dialog shows what an action will actually run. The world model reviews every autonomous plan.
- **Fail-Closed Settings and Approvals** — Settings change safely while work runs and survive unreadable or hand-edited config files; an action family a gateway profile does not list is read-only; an approval that is never answered expires instead of running; a plan resumed after a restart runs only the ticked actions; and a gesture can approve only a dialog the person can see.

[Release artifacts](https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/v0.14.0)

## 0.13.0 — Verified Autonomy and Runtime Hardening

Status: **Published**

Released: **2026-08-27**

Truthful outcomes, transactional settings, authenticated collaboration, coordinated multimodal control, bounded resources, and measured execution quality

- **Truthful Execution Contracts** — Completed work is revalidated against executor and checkpoint contracts, while failed, rejected, skipped, partial, and unavailable paths remain visible instead of being reported as successful.
- **Transactional Live Settings** — Runtime configuration changes are validated and applied atomically, roll back on failure, and can update supported model limits, devices, calendars, camera, gesture, voice, and mesh controls without leaving split state.
- **Authenticated Peer Collaboration** — Opt-in LAN peers use authenticated setup, bounded timeouts, constrained delegation, and the same permission and startup gates as local collaboration.
- **Coordinated Multimodal Control** — Voice, gesture, gaze, cursor, approval, and staged neural inputs arbitrate shared resources more reliably, including camera ownership, gesture priority, and neural-sidecar diagnostics.
- **Bounded Runtime Resources** — Daemon shutdown now quiesces autonomous jobs, monitors, browser resources, subscription CLIs, input services, and collaboration work; idle local TTS and model workers release heavy memory outside the long-lived daemon.
- **Measured Execution Quality** — The executor records step-budget quality, skips actions only when deterministic postconditions are already satisfied, preserves fast local status paths, and publishes a refreshed reproducible benchmark bundle.

[Release artifacts](https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/v0.13.0)

## 0.12.0 — Governed Intelligence and Handoff

Status: **Published**

Released: **2026-08-16**

Subscription-backed planning, local MCP, secure Air Handoff, verified integrations, and one governed cognitive runtime

- **Existing AI Subscription Providers** — Use an existing Codex or Claude Code login through the provider's official CLI, choose an available model, and inspect bounded plan-usage evidence without copying OAuth credentials into Heliox.
- **Local Heliox MCP** — IDE agents can stage Heliox tasks through a local MCP server while the Heliox daemon retains identity checks, permission gates, approvals, and execution.
- **Secure Air Handoff** — Cast a deliberately selected Heliox view to a paired mobile browser over an ephemeral encrypted session, with explicit start, stop, and gesture controls.
- **Verified Integrations** — Email, calendar, and SSH settings now reach their executable specialists through credential-vault, RPC, policy, and result contracts.
- **Unified Cognitive Runtime** — Companion narration, learned-risk interruption, follow-up suggestions, and autonomous execution now use the live production gateway instead of duplicate runtime state.
- **Evidence-Driven Reliability** — Local health reviews, neural staged-goal dispatch, model timeouts, microphone capture, MCP version reporting, and forensic results now fail more truthfully.

[Release artifacts](https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/v0.12.0)

## 0.11.1 — Adaptive Companion Release

Status: **Published**

Released: **2026-08-12**

The complete v0.11 companion feature set with deterministic Windows packaging

- **Reliable Windows Release Delivery** — The Windows release pipeline now prefetches and checksum-verifies WiX with explicit retries before building MSI and NSIS installers.

[Release artifacts](https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/v0.11.1)

## 0.11.0 — Adaptive Multimodal Companion

Status: **Source Milestone**

Source milestone: **2026-08-12**

Adaptive learning, coordinated multimodal control, expanded specialists, and guarded neural research

- **Continuous Companion Loop** — Always-on voice, spoken suggestions, interruption, and autonomous browser or application control now share one coordinated execution loop.
- **Adaptive Intelligence Stack** — A bounded experience ledger, temporal memory, verified online adaptation, strategy evolution, and an optional JEPA-style predictor advise future plans.
- **Expanded Specialist Mesh** — Twenty-one specialists provide concrete coverage for all 156 declared action types across desktop, browser, developer, integration, and research workflows.
- **Reliable Multimodal Control** — Gaze, 3D hand gestures, cursor control, wake-word audio, and workflow bindings can operate together with temporal false-positive rejection.
- **Guarded Neural Research Pipeline** — Synthetic BrainFlow and recorded EEGBCI paths now exercise calibrated, signed, consent-bounded neural intents without claiming live brain control.
- **Truthful Execution and Delivery** — Application launch, browser targeting, approvals, cancellations, result reporting, first-run Chromium setup, and cross-platform packaging are more deterministic.

This milestone is included in source and has no separate public installer.

## 0.10.1 — Reliable Interactive Sessions

Status: **Source Milestone**

Source milestone: **2026-07-30**

Persistent sessions, faster browser control, and coordinated companion reliability

- **Persistent Private Sessions** — Start isolated chats, reopen prior sessions, and retain adaptive memory without leaking one conversation's transcript into another.
- **Faster Semantic Browser Control** — Common browser tasks use bounded planning, resolve controls by meaning, and serialize shared-page actions to avoid slow or conflicting execution.
- **Coordinated Companion Services** — Narration, learned-risk interruption, voice, and verified follow-up suggestions remain available together while approvals pass through every action gate.
- **Safer, Truthful Task Results** — Cancelled work, cloud-provider failures, destructive file changes, and exact findings now produce bounded, redacted, recoverable results.

This milestone is included in source and has no separate public installer.

## 0.10.0 — Interactive Companion and Reliability

Status: **Source Milestone**

Source milestone: **2026-07-29**

Companion intelligence, learned safety, multimodal control, and reliable execution

- **Interactive Companion** — Opt-in narration, risk interruption, spoken follow-up, and suggestions now share one coordinated conversation loop.
- **Learned Risk World Model** — On-device learned predictions now advise the deterministic safety gate and can add caution or pause risky actions.
- **Unified Camera Intelligence** — Gaze, hand gestures, and cursor control can run together from one camera stream without suppressing enabled inputs.
- **Moderated Plugin Marketplace** — Reviewed packages are verified by manifest and SHA-256 before installation, and merged catalog entries appear without a desktop release.
- **Reliable Approvals and Results** — Approval responses remain live on the active connection, while cancelled, blocked, and failed actions now finish with truthful terminal results.
- **Durable Voice and Gesture Workflows** — Multi-step workflows can be submitted, paused, resumed, and inspected while autonomous healing remains separately permission-gated.

This milestone is included in source and has no separate public installer.

## 0.9.0 — JARVIS Autonomy

Status: **Published**

Released: **2026-07-22**

Background autonomy and local multimodal control

- **Autonomous Background Tasks** — Run permission-gated multi-step tasks while continuing to use the app.
- **Live Execution Narrator** — Optionally hear short progress descriptions and risk interruptions.
- **Local Voice, Gesture, and Gaze Inputs** — Use on-device multimodal controls without sending camera frames.

[Release artifacts](https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/v0.9.0)

## Current evidence

See [evidence and limitations](https://www.helioxos.dev/proof.md) and the [machine-readable capability catalog](https://www.helioxos.dev/capabilities.json).
