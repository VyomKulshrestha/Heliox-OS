# Heliox OS Evidence and Limitations

> This page separates reproducible software evidence, live CI status, developer-run hardware observations, and claims that have not yet been established. It is an evidence index, not a promise that every feature works on every computer.

Evidence snapshot date: **2026-09-25**

Product version: **0.13.0**

## Capability and routing evidence

- **160** declared action types are generated from `daemon/pilot/actions.py`.
- **21** executable specialists register providers for all **160** action types.
- Mesh coverage is **complete**; uncovered action types: **0**.
- **22** action types have a separate observed post-condition verifier.
- **138** action types currently rely on the executor result without an independent post-condition check.
- **6** plugin manifests are represented in the generated catalog.

Source: [machine-readable capability catalog](https://github.com/VyomKulshrestha/Heliox-OS/blob/main/capabilities.json). Platform declarations describe product targets; host tools, credentials, permissions, hardware, and integrations still determine runtime availability.

## Continuous-integration coverage

The committed CI workflow currently defines:

| Gate | Coverage | Live result |
| --- | --- | --- |
| Python | Ruff and Pytest on ubuntu-latest, windows-latest, macos-latest with Python 3.11, 3.12 | [CI workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/ci.yml) |
| Frontend | Prettier, Svelte type checking, dependency audit, static/unit tests, and Vite build | [CI workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/ci.yml) |
| Visual regression | Chromium snapshots on ubuntu-latest, windows-latest, macos-latest | [CI workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/ci.yml) |
| Rust desktop shell | Formatting, Clippy with warnings denied, and tests | [CI workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/ci.yml) |
| Marketplace | Manifest, hash, and moderation validation | [Marketplace workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/marketplace.yml) |
| Installers | Separate gated Windows, macOS, and Linux release jobs | [Release workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/release.yml) |
| Windows signing | SignPath test-policy signing and Authenticode signer-presence verification for the EXE, MSI, and embedded application | [SignPath test workflow](https://github.com/VyomKulshrestha/Heliox-OS/actions/workflows/signpath-test.yml) |

The result is intentionally linked rather than copied as “green”: CI status can change after this file is generated.

## Reproducible software benchmarks

Command:

```text
cd daemon
python ../scripts/generate_benchmark_evidence.py
```

Bundle environment: Windows 11, Python 3.12.6. Source commit: `4ea926d6f2ac`.

### Guarded local request latency

Scope: Ready-daemon guarded CPU usage request: local planning, routing, risk assessment, execution, postcondition verification, and response shaping.

| Metric | Ready-cold | Warm steady state |
| --- | ---: | ---: |
| Median | — | 25.867 ms |
| p95 | — | 27.637 ms |
| p99 | — | 29.640 ms |
| Minimum | — | 24.524 ms |
| Maximum | 58.872 ms | 29.793 ms |

- The harness executes local planning, routing, risk assessment, execution, post-condition verification, and response shaping.
- All 100 measured steady-state iterations made **0 model calls**.
- Ready-cold starts after production-equivalent risk and system probes are initialized; daemon process startup is excluded.

### Local status action suite

| Real guarded action | Iterations | Median | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| CPU usage | 50 | 26.244 ms | 28.282 ms | 29.414 ms |
| Memory usage | 50 | 5.931 ms | 8.473 ms | 9.532 ms |
| Disk usage | 50 | 4.776 ms | 7.942 ms | 8.360 ms |
| Comprehensive system information | 50 | 57.404 ms | 84.405 ms | 85.650 ms |

Each case validates the exact selected action, executes the real read-only host probe, and makes zero model calls.

### Event-loop responsiveness

During a real one-second CPU monitor sample, a 10 ms asyncio heartbeat produced **66 ticks**, with **15.520 ms median**, **16.420 ms p95**, and **16.541 ms maximum** gaps. This measures scheduler availability—not monitor completion speed—and is affected by Windows timer granularity.

### Deterministic intent dispatch

The curated routing regression set passed **59/59 cases** with **0.025 ms median** dispatch latency. It covers bounded positive intents and ambiguous controls that must fall through to model planning. It is not a population-level language-understanding benchmark, and application routing does not prove an application is installed.

### Learned-risk world model

The shipped `risk-mlp-v4-heldout` artifact records **25,200 training** and **5,400 stratified temporal validation samples** across **12 action types**.

| Held-out metric | Learned model | Zero predictor | Improvement |
| --- | ---: | ---: | ---: |
| Disk-delta MAE | 0.0000000330 | 0.0000000718 | 54.0154% |
| Process-delta MAE | 0.0000130251 | 0.0022166655 | 99.4124% |

The audit passed **5/5 direction invariants** and measured **0.025 ms median** inference. These are coarse disk/process predictions. Deterministic safety rules remain authoritative; this is not a general physical-world or user-intent model.

### Subscription-backed planning

A developer-machine run through the official **Codex CLI** passed **3/3 fixed planning cases** with **14.708 s median** provider round-trip latency. The provider reported `codex-cli 0.144.5` with model `provider-default`.

| Fixed case | Planned action types | Latency | Input tokens | Cached input | Output tokens | Destructive actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Health Review | `system_health_review` | 13.188 s | 17,538 | 8,960 | 326 | 0 |
| Semantic Browser | `browser_navigate`, `browser_extract_links`, `browser_click_text`, `browser_page_info` | 14.708 s | 17,545 | 0 | 403 | 0 |
| Evidence First Files | `shell_command` | 15.867 s | 17,235 | 8,960 | 423 | 0 |

- Scope: **side-effect-free planning only; no action was executed**.
- The run used a real subscription-authenticated provider CLI, but it did not execute the plans or prove the actions' runtime outcomes.
- Subscription requests consume the provider plan's allowance; they are not counted as metered API-dollar spend by Heliox.
- Heliox rejected provider tool activity and retained schema validation, deterministic policy, approvals, execution, and verification.
- Claim boundary: One developer-machine subscription planning sample. It does not measure action execution, provider availability for other accounts, universal latency, plan correctness outside the fixed cases, or Claude Code because that CLI was not installed for this capture.

### Historical CPU-path improvement

| Metric | Before | After `f2df192` | Change |
| --- | ---: | ---: | ---: |
| Mean | 117.68 ms | 37.08 ms | 68.5% lower |
| Median | 110.71 ms | 30.59 ms | 72.4% lower |
| Minimum | 108.97 ms | 29.24 ms | — |
| Maximum | 285.49 ms | 194.21 ms | — |

- This historical table explains the original blocking-sample fix; the current distribution tables above supersede it for present performance.
- None of these software benchmarks measures model-provider, network, browser page-load, UI-rendering, microphone, TTS, camera, gaze, gesture, EEG, or human latency/accuracy.
- Local snapshots are reproducibility evidence, not universal performance guarantees.

Raw evidence: [`software-benchmarks-2026-09-25.json`](https://github.com/VyomKulshrestha/Heliox-OS/blob/main/docs/evidence/software-benchmarks-2026-09-25.json), [`subscription-planning-codex-2026-08-16.json`](https://github.com/VyomKulshrestha/Heliox-OS/blob/main/docs/evidence/subscription-planning-codex-2026-08-16.json), [`local-tts-isolation-2026-08-27.json`](https://github.com/VyomKulshrestha/Heliox-OS/blob/main/docs/evidence/local-tts-isolation-2026-08-27.json), and the [historical CPU artifact](https://github.com/VyomKulshrestha/Heliox-OS/blob/main/docs/evidence/react-latency-2026-08-12.json).

## Platform and hardware evidence

| Feature | Automated/software evidence | Physical evidence status | Permitted claim |
| --- | --- | --- | --- |
| Typed plans and action routing | Schema, permission, executor, provider-coverage, and result-contract tests | No special hardware required | Software path is tested; individual host actions still depend on platform adapters and permissions. |
| Browser automation | Unit/integration and visual-browser contracts | Site behavior and browser versions vary | Supported through guarded browser actions; no claim of universal website compatibility. |
| Voice recognition | Configuration, routing, cancellation, and fusion tests | Human microphone accuracy is not a release gate | Hardware test required for the user's microphone, language, noise, and accent. |
| Pocket/Kokoro/OS TTS | Engine, fallback, cancellation, response, worker isolation, and idle-release tests; real Kokoro WAV synthesis was 21.401 s cold and 0.138 s warm with zero Torch/CUDA modules retained by the parent | Audible output was not assessed in this run and is not continuously reproduced in CI | Local TTS and bounded process isolation are implemented; audible quality, device output, and universal latency require separate validation. |
| Camera gesture and cursor control | Explainable hand-quality gates, candidate-specific temporal verification, stale-motion reset, calibration, workflow, false-positive regressions, and offline metric arithmetic | No consented, person-disjoint physical benchmark corpus has been run across cameras, lighting, skin tones, backgrounds, or users | Experimental opt-in input; users must retain the stop controls. |
| Gaze tracking | Model loading, finite geometry, blink/closed-eye, head-turn, eye-agreement, iris-plausibility, temporal-hysteresis, event validation, fusion, and settings tests | Synthetic fixtures validate deterministic behavior, not physical gaze accuracy | Coarse on-device region signal, not eye-tracking-grade measurement. |
| Neural intent | Synthetic BrainFlow, recorded EEG playback, provenance, calibration, decoder, bounded text-authored task staging, neural selection, autonomous dispatch, gateway, and fault tests | No live headset/human validation has established control accuracy | Research pipeline can select a pre-staged goal and launch the normal guarded autonomous path; it does not decode an unstated task and is not proven live brain control or medical use. |
| Subscription model access | Official Codex and Claude CLI adapters, login/status checks, selectable models, prompt-budget controls, quota accounting, and provider-tool rejection tests | Provider availability, plan eligibility, and quotas remain provider-owned | Uses the user's existing provider login without copying OAuth credentials; it is not unlimited and does not bypass provider terms. |
| Snapshots and rollback | Fail-closed policy and backend contract tests | Backend availability and real restoration depend on OS support and privileges | Destructive work is blocked when a required snapshot cannot be created; not every external effect is reversible. |

Neural details and the recorded EEGBCI snapshot are documented in [Neural Intent](https://github.com/VyomKulshrestha/Heliox-OS/blob/main/docs/NEURAL_INTENT.md).

## Known limitations

1. Only 22 of 160 actions currently have an independent post-condition verifier; inspect `verification.independent_postcondition` in the capability catalog.
2. CI validates software contracts but cannot establish camera, microphone, speaker, accessibility-permission, EEG, or human-factors accuracy.
3. Browser pages, third-party APIs, cloud models, and external applications can change independently of Heliox.
4. Local-first operation does not mean every configured path is offline. Cloud model and integration tasks send necessary context to the selected provider.
5. Snapshots cover supported local-system changes. Messages, purchases, remote hosts, pushed Git commits, browser scripts, and other external effects may be irreversible.
6. Learned risk and world-model outputs can add caution or interrupt; deterministic policy remains authoritative.
7. Public installers are not yet production-signed. The SignPath test-policy pipeline is validated, but the production certificate is still pending; operating-system reputation warnings may continue until that certificate is issued and the release workflow is migrated.
8. The subscription planning sample covers one Codex CLI account and three fixed prompts. Claude Code was not installed for that capture, and no provider-backed action was executed by the benchmark.

## Closed regression history

This is not a claim that no defects remain. It records representative failures that materially affected trust or evidence:

| Date | Observed failure | Resolution |
| --- | --- | --- |
| 2026-08-27 | A local neural TTS request retained PyTorch/CUDA libraries in the long-lived daemon after speech completed. | `2666b39` moved Pocket/Kokoro inference into a reusable worker that exits after a 10-second idle window; the raw isolation artifact records real file synthesis and parent-process module evidence. |
| 2026-08-12 | The latency benchmark used an obsolete memory/permission harness contract and leaked worker threads after failure, appearing to hang. | [`f2df192`](https://github.com/VyomKulshrestha/Heliox-OS/commit/f2df192) repaired teardown and reduced blocking CPU sampling latency. |
| 2026-08-13 | Background CPU samples blocked the shared asyncio loop for up to one second. | `bf6ac9c` moved interval sampling to workers; the evidence bundle records concurrent heartbeat responsiveness. |
| 2026-08-13 | Ambiguous tasks such as “run the tests” were misrouted as application launches. | `cae908d` tightened the bounded app fast path; the 59-case dispatch suite now passes all controls. |
| 2026-07-30 | Face-like frames could produce false gesture events. | [`6d4025b`](https://github.com/VyomKulshrestha/Heliox-OS/commit/6d4025b) added false-positive rejection and [`1d810b7`](https://github.com/VyomKulshrestha/Heliox-OS/commit/1d810b7) added temporal verification. Physical validation remains required. |
| 2026-07-30 | An approval could be accepted yet denied by a later cognitive action gate. | [`3e034d4`](https://github.com/VyomKulshrestha/Heliox-OS/commit/3e034d4) carried approval authority across the guarded flow. |
| 2026-07-26 | Marketplace package hashes differed across operating-system line endings. | [`f39648e`](https://github.com/VyomKulshrestha/Heliox-OS/commit/f39648e) normalized verified marketplace hashing. |
| 2026-07-24 | Approval RPC handling could deadlock the active request. | [`a20297d`](https://github.com/VyomKulshrestha/Heliox-OS/commit/a20297d) separated approval handling from the blocked request path. |

For all current failures, use the [live CI history](https://github.com/VyomKulshrestha/Heliox-OS/actions) and [issue tracker](https://github.com/VyomKulshrestha/Heliox-OS/issues).

## Reproduction entry points

```text
# Capability and provider coverage
python scripts/generate_capability_catalog.py --output capabilities.json
python -m pytest daemon/tests/test_capability_catalog.py daemon/tests/test_specialist_expansion.py -q

# Full local software benchmark bundle
python scripts/generate_benchmark_evidence.py

# Subscription-backed planning only; consumes provider-plan allowance and executes no actions
cd daemon
python benchmarks/subscription_planning_suite.py --provider codex --output ../docs/evidence/subscription-planning-codex-YYYY-MM-DD.json

# Individual benchmark entry points (run from daemon)
python benchmarks/react_latency.py --iterations 100 --warmup 10 --json
python benchmarks/local_status_suite.py --iterations 50 --warmup 5 --json
python benchmarks/event_loop_responsiveness.py --json
python benchmarks/intent_dispatch_suite.py --json
python benchmarks/world_model_suite.py --iterations 1000 --json

# Neural no-hardware paths; these do not validate live brain control
pilot-neurod-benchmark brainflow-synthetic --seconds 2
pilot-neurod-benchmark eegbci --subject 1 --runs 6 10 14
```
