"""Generate the public Heliox evidence and limitations page."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/VyomKulshrestha/Heliox-OS"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_benchmark_path() -> Path:
    evidence_dir = REPO_ROOT / "docs" / "evidence"
    candidates = sorted(evidence_dir.glob("software-benchmarks-*.json"))
    if not candidates:
        raise FileNotFoundError("No software benchmark evidence bundle is committed")
    return candidates[-1]


def _latest_subscription_benchmark_path() -> Path:
    evidence_dir = REPO_ROOT / "docs" / "evidence"
    candidates = sorted(evidence_dir.glob("subscription-planning-*.json"))
    if not candidates:
        raise FileNotFoundError("No subscription planning benchmark evidence bundle is committed")
    return candidates[-1]


def _latest_tts_benchmark_path() -> Path:
    evidence_dir = REPO_ROOT / "docs" / "evidence"
    candidates = sorted(evidence_dir.glob("local-tts-isolation-*.json"))
    if not candidates:
        raise FileNotFoundError("No local TTS isolation evidence bundle is committed")
    return candidates[-1]


def _matrix_values(workflow: str, key: str) -> list[str]:
    match = re.search(rf"^\s*{re.escape(key)}:\s*\[([^\]]+)]", workflow, re.MULTILINE)
    if not match:
        raise ValueError(f"Unable to locate {key!r} matrix in CI workflow")
    return [item.strip().strip("'\"") for item in match.group(1).split(",")]


def _percent_reduction(before: float, after: float) -> float:
    return round((before - after) / before * 100, 1)


def _coadaptation_proof(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return ""
    rows = []
    for family, result in entry["metrics"]["by_family"].items():
        for policy, metrics in result["policies"].items():
            rows.append(
                f"| {family} | `{policy}` | {metrics['people']} | "
                f"{metrics['realized_utility_per_trial']['mean']:.3f} | "
                f"{100 * metrics['ua_per_trial']:.2f}% | "
                f"{metrics['acting_under_ambiguity']} | {metrics['constraint_violations']} |"
            )
    passed = sum(audit["passed"] for audit in entry["agency_audit"].values())
    return (
        "### Assistance co-adaptation: simulated users\n\n"
        f"**{entry['claim_boundary']}**. Scope: {entry['scope']}. "
        "F1 represents precise mouse users and F3 represents hand-pointer novices in the simulator.\n\n"
        "| Family | Policy | Simulated people | Utility per trial | Unintended activations | "
        "Acts on ambiguous input | Rule violations |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + f"\n\nAgency audits passed **{passed}/{len(entry['agency_audit'])} policies**. "
        "The adaptive engine does not beat the plain interface for every simulated family. "
        "These tuning families share the model's functional forms; this run does not establish "
        "held-out generalization or an improvement for real people. No model was promoted by this benchmark. "
        "The feature is opt-in and changes presentation only.\n\n"
        f"Protocol SHA-256: `{entry['protocol_sha256']}`. "
        f"[Study, consent and reproduction details]({REPOSITORY_URL}/blob/main/docs/COADAPTATION.md).\n\n"
    )


def build_proof() -> str:
    capabilities = _load_json(REPO_ROOT / "capabilities.json")
    legacy_latency = _load_json(REPO_ROOT / "docs" / "evidence" / "react-latency-2026-08-12.json")
    benchmark_path = _latest_benchmark_path()
    benchmarks = _load_json(benchmark_path)
    subscription_benchmark_path = _latest_subscription_benchmark_path()
    subscription_benchmark = _load_json(subscription_benchmark_path)
    tts_benchmark_path = _latest_tts_benchmark_path()
    tts_benchmark = _load_json(tts_benchmark_path)
    tts_results = tts_benchmark["results"]
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_operating_systems = _matrix_values(ci_workflow, "os")
    python_versions = _matrix_values(ci_workflow, "python-version")

    summary = capabilities["summary"]
    action_total = summary["action_types"]
    independent_total = summary["independent_postcondition_verifiers"]
    executor_only = action_total - independent_total
    before = legacy_latency["before"]
    after = legacy_latency["after"]
    mean_reduction = _percent_reduction(before["mean_ms"], after["mean_ms"])
    median_reduction = _percent_reduction(before["median_ms"], after["median_ms"])
    guarded = benchmarks["guarded_cpu_request"]
    steady = guarded["steady_state"]
    status_scenarios = benchmarks["local_status_suite"]["scenarios"]
    responsiveness = benchmarks["event_loop_responsiveness"]
    intents = benchmarks["intent_dispatch"]
    world_model = benchmarks["learned_risk_world_model"]
    world_validation = world_model["validation"]
    subscription_rows = "\n".join(
        "| {name} | {actions} | {latency:.3f} s | {input_tokens:,} | {cached_tokens:,} | "
        "{output_tokens:,} | {destructive} |".format(
            name=result["name"].replace("_", " ").title(),
            actions=", ".join(f"`{action}`" for action in result["action_types"]),
            latency=result["latency_seconds"],
            input_tokens=result["usage"]["input_tokens"],
            cached_tokens=result["usage"]["cached_input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            destructive=result["destructive_actions"],
        )
        for result in subscription_benchmark["results"]
    )

    return f"""# Heliox OS Evidence and Limitations

> This page separates reproducible software evidence, live CI status, developer-run hardware observations, and claims that have not yet been established. It is an evidence index, not a promise that every feature works on every computer.

Evidence snapshot date: **{benchmarks["captured_on"]}**

Product version: **{capabilities["product"]["version"]}**

## Capability and routing evidence

- **{action_total}** declared action types are generated from `daemon/pilot/actions.py`.
- **{summary["mesh"]["specialists"]}** executable specialists register providers for all **{summary["mesh"]["registered_action_types"]}** action types.
- Mesh coverage is **{"complete" if summary["mesh"]["coverage_complete"] else "incomplete"}**; uncovered action types: **{len(summary["mesh"]["uncovered_action_types"])}**.
- **{independent_total}** action types have a separate observed post-condition verifier.
- **{executor_only}** action types currently rely on the executor result without an independent post-condition check.
- **{summary["plugins"]}** plugin manifests are represented in the generated catalog.

Source: [machine-readable capability catalog]({REPOSITORY_URL}/blob/main/capabilities.json). Platform declarations describe product targets; host tools, credentials, permissions, hardware, and integrations still determine runtime availability.

## Continuous-integration coverage

The committed CI workflow currently defines:

| Gate | Coverage | Live result |
| --- | --- | --- |
| Python | Ruff and Pytest on {", ".join(ci_operating_systems)} with Python {", ".join(python_versions)} | [CI workflow]({REPOSITORY_URL}/actions/workflows/ci.yml) |
| Frontend | Prettier, Svelte type checking, dependency audit, static/unit tests, and Vite build | [CI workflow]({REPOSITORY_URL}/actions/workflows/ci.yml) |
| Visual regression | Chromium snapshots on {", ".join(ci_operating_systems)} | [CI workflow]({REPOSITORY_URL}/actions/workflows/ci.yml) |
| Rust desktop shell | Formatting, Clippy with warnings denied, and tests | [CI workflow]({REPOSITORY_URL}/actions/workflows/ci.yml) |
| Marketplace | Manifest, hash, and moderation validation | [Marketplace workflow]({REPOSITORY_URL}/actions/workflows/marketplace.yml) |
| Installers | Separate gated Windows, macOS, and Linux release jobs | [Release workflow]({REPOSITORY_URL}/actions/workflows/release.yml) |
| Windows signing | SignPath test-policy signing and Authenticode signer-presence verification for the EXE, MSI, and embedded application | [SignPath test workflow]({REPOSITORY_URL}/actions/workflows/signpath-test.yml) |

The result is intentionally linked rather than copied as “green”: CI status can change after this file is generated.

## Reproducible software benchmarks

Command:

```text
cd daemon
python ../scripts/generate_benchmark_evidence.py
```

Bundle environment: {guarded["environment"]["operating_system"]} {guarded["environment"]["operating_system_release"]}, Python {guarded["environment"]["python"]}. Source commit: `{benchmarks["source_commit"][:12]}`.

### Guarded local request latency

Scope: {guarded["scope"]}.

| Metric | Ready-cold | Warm steady state |
| --- | ---: | ---: |
| Median | — | {steady["median_ms"]:.3f} ms |
| p95 | — | {steady["p95_ms"]:.3f} ms |
| p99 | — | {steady["p99_ms"]:.3f} ms |
| Minimum | — | {steady["min_ms"]:.3f} ms |
| Maximum | {guarded["cold_ready_request_ms"]:.3f} ms | {steady["max_ms"]:.3f} ms |

- The harness executes local planning, routing, risk assessment, execution, post-condition verification, and response shaping.
- All {steady["iterations"]} measured steady-state iterations made **{guarded["model_generate_calls"]} model calls**.
- Ready-cold starts after production-equivalent risk and system probes are initialized; daemon process startup is excluded.

### Local status action suite

| Real guarded action | Iterations | Median | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| CPU usage | {status_scenarios["cpu_usage"]["iterations"]} | {status_scenarios["cpu_usage"]["median_ms"]:.3f} ms | {status_scenarios["cpu_usage"]["p95_ms"]:.3f} ms | {status_scenarios["cpu_usage"]["p99_ms"]:.3f} ms |
| Memory usage | {status_scenarios["memory_usage"]["iterations"]} | {status_scenarios["memory_usage"]["median_ms"]:.3f} ms | {status_scenarios["memory_usage"]["p95_ms"]:.3f} ms | {status_scenarios["memory_usage"]["p99_ms"]:.3f} ms |
| Disk usage | {status_scenarios["disk_usage"]["iterations"]} | {status_scenarios["disk_usage"]["median_ms"]:.3f} ms | {status_scenarios["disk_usage"]["p95_ms"]:.3f} ms | {status_scenarios["disk_usage"]["p99_ms"]:.3f} ms |
| Comprehensive system information | {status_scenarios["system_information"]["iterations"]} | {status_scenarios["system_information"]["median_ms"]:.3f} ms | {status_scenarios["system_information"]["p95_ms"]:.3f} ms | {status_scenarios["system_information"]["p99_ms"]:.3f} ms |

Each case validates the exact selected action, executes the real read-only host probe, and makes zero model calls.

### Event-loop responsiveness

During a real one-second CPU monitor sample, a 10 ms asyncio heartbeat produced **{responsiveness["heartbeat_ticks"]} ticks**, with **{responsiveness["heartbeat_median_ms"]:.3f} ms median**, **{responsiveness["heartbeat_p95_ms"]:.3f} ms p95**, and **{responsiveness["heartbeat_max_ms"]:.3f} ms maximum** gaps. This measures scheduler availability—not monitor completion speed—and is affected by Windows timer granularity.

### Deterministic intent dispatch

The curated routing regression set passed **{intents["passed"]}/{intents["case_count"]} cases** with **{intents["latency"]["median_ms"]:.3f} ms median** dispatch latency. It covers bounded positive intents and ambiguous controls that must fall through to model planning. It is not a population-level language-understanding benchmark, and application routing does not prove an application is installed.

### Learned-risk world model

The shipped `{world_model["model_version"]}` artifact records **{world_model["training_samples"]:,} training** and **{world_model["validation_samples"]:,} stratified temporal validation samples** across **{world_model["learned_action_types"]} action types**.

| Held-out metric | Learned model | Zero predictor | Improvement |
| --- | ---: | ---: | ---: |
| Disk-delta MAE | {world_validation["disk_delta_mae"]:.10f} | {world_validation["disk_zero_baseline_mae"]:.10f} | {world_validation["disk_improvement_percent"]:.4f}% |
| Process-delta MAE | {world_validation["process_delta_mae"]:.10f} | {world_validation["process_zero_baseline_mae"]:.10f} | {world_validation["process_improvement_percent"]:.4f}% |

The audit passed **{world_model["direction_checks_passed"]}/{len(world_model["direction_checks"])} direction invariants** and measured **{world_model["inference"]["median_ms"]:.3f} ms median** inference. These are coarse disk/process predictions. Deterministic safety rules remain authoritative; this is not a general physical-world or user-intent model.

{_coadaptation_proof(benchmarks.get("coadaptation_simulated_users"))}### Subscription-backed planning

A developer-machine run through the official **{subscription_benchmark["provider"].title()} CLI** passed **{subscription_benchmark["passed"]}/{subscription_benchmark["case_count"]} fixed planning cases** with **{subscription_benchmark["median_latency_seconds"]:.3f} s median** provider round-trip latency. The provider reported `{subscription_benchmark["provider_cli_version"]}` with model `{subscription_benchmark["model"]}`.

| Fixed case | Planned action types | Latency | Input tokens | Cached input | Output tokens | Destructive actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{subscription_rows}

- Scope: **{subscription_benchmark["scope"]}**.
- The run used a real subscription-authenticated provider CLI, but it did not execute the plans or prove the actions' runtime outcomes.
- Subscription requests consume the provider plan's allowance; they are not counted as metered API-dollar spend by Heliox.
- Heliox rejected provider tool activity and retained schema validation, deterministic policy, approvals, execution, and verification.
- Claim boundary: {subscription_benchmark["claim_boundary"]}

### Historical CPU-path improvement

| Metric | Before | After `f2df192` | Change |
| --- | ---: | ---: | ---: |
| Mean | {before["mean_ms"]:.2f} ms | {after["mean_ms"]:.2f} ms | {mean_reduction:.1f}% lower |
| Median | {before["median_ms"]:.2f} ms | {after["median_ms"]:.2f} ms | {median_reduction:.1f}% lower |
| Minimum | {before["min_ms"]:.2f} ms | {after["min_ms"]:.2f} ms | — |
| Maximum | {before["max_ms"]:.2f} ms | {after["max_ms"]:.2f} ms | — |

- This historical table explains the original blocking-sample fix; the current distribution tables above supersede it for present performance.
- None of these software benchmarks measures model-provider, network, browser page-load, UI-rendering, microphone, TTS, camera, gaze, gesture, EEG, or human latency/accuracy.
- Local snapshots are reproducibility evidence, not universal performance guarantees.

Raw evidence: [`{benchmark_path.name}`]({REPOSITORY_URL}/blob/main/docs/evidence/{benchmark_path.name}), [`{subscription_benchmark_path.name}`]({REPOSITORY_URL}/blob/main/docs/evidence/{subscription_benchmark_path.name}), [`{tts_benchmark_path.name}`]({REPOSITORY_URL}/blob/main/docs/evidence/{tts_benchmark_path.name}), and the [historical CPU artifact]({REPOSITORY_URL}/blob/main/docs/evidence/react-latency-2026-08-12.json).

## Platform and hardware evidence

| Feature | Automated/software evidence | Physical evidence status | Permitted claim |
| --- | --- | --- | --- |
| Typed plans and action routing | Schema, permission, executor, provider-coverage, and result-contract tests | No special hardware required | Software path is tested; individual host actions still depend on platform adapters and permissions. |
| Browser automation | Unit/integration and visual-browser contracts | Site behavior and browser versions vary | Supported through guarded browser actions; no claim of universal website compatibility. |
| Voice recognition | Configuration, routing, cancellation, and fusion tests | Human microphone accuracy is not a release gate | Hardware test required for the user's microphone, language, noise, and accent. |
| Pocket/Kokoro/OS TTS | Engine, fallback, cancellation, response, worker isolation, and idle-release tests; real Kokoro WAV synthesis was {tts_results["cold_synthesis_seconds"]:.3f} s cold and {tts_results["warm_synthesis_seconds"]:.3f} s warm with zero Torch/CUDA modules retained by the parent | Audible output was not assessed in this run and is not continuously reproduced in CI | Local TTS and bounded process isolation are implemented; audible quality, device output, and universal latency require separate validation. |
| Camera gesture and cursor control | Explainable hand-quality gates, candidate-specific temporal verification, stale-motion reset, calibration, workflow, false-positive regressions, and offline metric arithmetic | No consented, person-disjoint physical benchmark corpus has been run across cameras, lighting, skin tones, backgrounds, or users | Experimental opt-in input; users must retain the stop controls. |
| Gaze tracking | Model loading, finite geometry, blink/closed-eye, head-turn, eye-agreement, iris-plausibility, temporal-hysteresis, event validation, fusion, and settings tests | Synthetic fixtures validate deterministic behavior, not physical gaze accuracy | Coarse on-device region signal, not eye-tracking-grade measurement. |
| Neural intent | Synthetic BrainFlow, recorded EEG playback, provenance, calibration, decoder, bounded text-authored task staging, neural selection, autonomous dispatch, gateway, and fault tests | No live headset/human validation has established control accuracy | Research pipeline can select a pre-staged goal and launch the normal guarded autonomous path; it does not decode an unstated task and is not proven live brain control or medical use. |
| Assistance co-adaptation | Contract, policy-property, agency-audit and simulated-user tests | No human study has reported | Experimental, opt-in, presentation-only assistance research; no claim that it improves selection for people until the pre-registered study reports. |
| Subscription model access | Official Codex and Claude CLI adapters, login/status checks, selectable models, prompt-budget controls, quota accounting, and provider-tool rejection tests | Provider availability, plan eligibility, and quotas remain provider-owned | Uses the user's existing provider login without copying OAuth credentials; it is not unlimited and does not bypass provider terms. |
| Snapshots and rollback | Fail-closed policy and backend contract tests | Backend availability and real restoration depend on OS support and privileges | Destructive work is blocked when a required snapshot cannot be created; not every external effect is reversible. |

Neural details and the recorded EEGBCI snapshot are documented in [Neural Intent]({REPOSITORY_URL}/blob/main/docs/NEURAL_INTENT.md).

## Known limitations

1. Only {independent_total} of {action_total} actions currently have an independent post-condition verifier; inspect `verification.independent_postcondition` in the capability catalog.
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
| 2026-08-12 | The latency benchmark used an obsolete memory/permission harness contract and leaked worker threads after failure, appearing to hang. | [`f2df192`]({REPOSITORY_URL}/commit/f2df192) repaired teardown and reduced blocking CPU sampling latency. |
| 2026-08-13 | Background CPU samples blocked the shared asyncio loop for up to one second. | `bf6ac9c` moved interval sampling to workers; the evidence bundle records concurrent heartbeat responsiveness. |
| 2026-08-13 | Ambiguous tasks such as “run the tests” were misrouted as application launches. | `cae908d` tightened the bounded app fast path; the 59-case dispatch suite now passes all controls. |
| 2026-07-30 | Face-like frames could produce false gesture events. | [`6d4025b`]({REPOSITORY_URL}/commit/6d4025b) added false-positive rejection and [`1d810b7`]({REPOSITORY_URL}/commit/1d810b7) added temporal verification. Physical validation remains required. |
| 2026-07-30 | An approval could be accepted yet denied by a later cognitive action gate. | [`3e034d4`]({REPOSITORY_URL}/commit/3e034d4) carried approval authority across the guarded flow. |
| 2026-07-26 | Marketplace package hashes differed across operating-system line endings. | [`f39648e`]({REPOSITORY_URL}/commit/f39648e) normalized verified marketplace hashing. |
| 2026-07-24 | Approval RPC handling could deadlock the active request. | [`a20297d`]({REPOSITORY_URL}/commit/a20297d) separated approval handling from the blocked request path. |

For all current failures, use the [live CI history]({REPOSITORY_URL}/actions) and [issue tracker]({REPOSITORY_URL}/issues).

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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "proof.md",
        help="Destination for the generated Markdown evidence page.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_proof(), encoding="utf-8")
    print(f"Wrote public evidence page to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
