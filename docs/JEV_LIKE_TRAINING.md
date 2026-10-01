# Jev-like decision model: licensed training recipe

Status: **research candidate with a frozen promotion protocol**. A model that
passes every gate may be enabled as an opt-in, advisory `typed_decision`
scorer. It never receives execution authority, and no result here establishes
parity with TypeSafe's proprietary Jev.

## Why the earlier clean-base model failed

After the per-row CC0 and `typesafe.ai` source filters, the
`release-v2-redistributable` Open-Jev config contains only three game/spatial
sources (`painting-geometry`, `snake`, `vizdoom`). The clean-base 2B candidate
therefore never saw authorization, untrusted-content, verification or
recipient decisions. On the authored desktop cases it allowed an injected
credential email and accepted an unverified postcondition (see the
[2026-09-30 status](INTELLIGENCE_STATUS_2026-09-30.md)).

## Methods adopted

| Source | What we use |
| --- | --- |
| [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) recipe and [Open-Jev-2B card](https://huggingface.co/ZefanCai/Open-Jev-2B) | Candidate-conditioned prompts, one scalar head initialized from the Yes-minus-No readout, in-question softmax, soft cross-entropy + 0.1 Brier, LoRA 2e-5 / head 5e-5, temperature fitted only on calibration, group-isolated splits |
| Open-Jev community controls | One-fact counterfactual minimal pairs with flipped labels, kept inside one group |
| StruQ (arXiv 2402.06363), SecAlign (2410.05451), Meta SecAlign (2507.02735) | Instructions inside untrusted content never authorize an action. Benign controls, where the user really asked for the action, prevent over-defense |
| Ordinal soft labels (SORD-style) | Gaussian-smoothed score targets for fitting only (`--score-label-sigma 0.75`), plus the existing ranked-probability loss |
| Per-kind temperature scaling | One calibration temperature per decision kind (`--kind-temperatures`), stored in `head.npz`, verified against the report, and applied by the runtime and offline scorers |

TypeSafe describes Jev's training (RLCD) only at a high level, and no paper or
reward definition is public. Nothing here imitates Jev outputs.

## Licensed data mixture

| Data | License | Use |
| --- | --- | --- |
| Open-Jev `release-v2-redistributable` | CC0-1.0 per row after filtering | General decisions with a source holdout |
| Open-Jev `email-selection-control-v1` | CC0-1.0 per row | Recipient/span selection that treats document text as data |
| Open-Jev `mailroom-control-v1` | CC0-1.0 per row | Email intent with provider, payment and translation counterfactuals |
| [`heliox_desktop_decisions_v1`](../daemon/benchmarks/desktop_decision_corpus.py) | MIT, authored here | 14 desktop-agent decision families |

Excluded: TypeSafe Jev outputs, released Open-Jev adapter weights, and any row
failing the CC0 or restricted-source filter. Only the two control configs
above are allowlisted (`EXTRA_TRAINING_CONFIGS`).

The authored corpus covers scope authorization, untrusted instructions,
postcondition verification, evidence-qualified claims, instruction revisions,
approval gates, ambiguous referents, least privilege, dependency order,
preference precedence, time conversion, evidence strength, action risk, and
budget fit. Its `test` split uses held-out question phrasings, state field
names and entity names. Three families (`approval_gate`, `least_privilege`,
`action_risk`) are never used for fitting or calibration and measure
family-level transfer. These are still authored by the same project, so they
measure template generalization, not independent real-world accuracy.

## Reproduce

The frozen hyperparameters and gates are in
[`jev_like_mixture_protocol_20261002.json`](../daemon/benchmarks/jev_like_mixture_protocol_20261002.json).

```bash
cd daemon
python scripts/train_general_decision.py --backbone qwen3.5-2b --seed 20261008 --split-seed 20261008 --source-holdout-seed 20261008 --train-rows 256 --calibration-rows 48 --test-rows 64 --ood-rows 96 --extra-config email-selection-control-v1:128:24:48 --extra-config mailroom-control-v1:128:24:48 --desktop-corpus --desktop-train-scenarios 40 --kind-temperatures --score-label-sigma 0.75 --ordinal-weight 0.1 --prompt-style full_options_v2 --candidate-microbatch-size 1 --max-length 1024 --output-dir benchmarks/.bench-state/jev_like_2b_20261002 --report benchmarks/.bench-state/jev_like_2b_20261002.json
```

```bash
python scripts/check_jev_like_promotion.py benchmarks/.bench-state/jev_like_2b_20261002.json
```

Reference evaluation against Jev is optional, bounded and billed. Its answers
are never used for fitting or calibration, and one cache serves every local
candidate:

```bash
python scripts/compare_general_jev.py --acknowledge-provider-terms --cases-json benchmarks/desktop_decisions_v1_eval_cases.json --reference-cache benchmarks/.bench-state/desktop_v1_jev_reference.json --output-dir benchmarks/.bench-state/jev_like_2b_20261002 --model-report benchmarks/.bench-state/jev_like_2b_20261002.json --runtime-scorer --report benchmarks/.bench-state/desktop_v1_vs_jev_2b.json
```

## Enable a promoted model

Only after `check_jev_like_promotion.py` reports `promotable_as_advisory: true`,
set the local Heliox config section `typed_decision`:

```json
{
  "enabled": true,
  "scoring_mode": "trained_head",
  "artifact_dir": "<absolute path to the output directory>",
  "report_path": "<absolute path to the training report>"
}
```

The daemon verifies the adapter and head hashes, the pinned backbone revision,
the token limit, the prompt style and the per-kind temperatures before it
loads anything. It needs CUDA, at least 4 GiB free VRAM and 2.5 GiB available
RAM. Responses stay `research_only`, `advisory_only` and
`execution_authority: false`. They are exposed through the local MCP
`score_heliox_typed_decision` tool and never bypass Heliox approvals.

## Results

Pending the frozen training runs.
