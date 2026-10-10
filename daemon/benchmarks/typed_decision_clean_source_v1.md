# Clean-source typed-decision training probe

**Status: research-only; no trained weights are loaded by Heliox.** This is a
Heliox-owned, independently trained candidate-option scorer, not the reference-model clone,
general planner, or desktop executor.

## Data provenance and method

The pinned public decision dataset release is
the pinned public decision corpus, config `release-v2-redistributable`, revision
`c67699e13d0ae25e35b77165a4b6b079bedc8aba`. A provenance review found that
the earlier CC0-only gate still admitted synthetic/programmatic rows with
TypeSafe-hosted source URLs. The shared loader now requires CC0-1.0, excludes
unverified upstream rights, and rejects any nested provenance URL whose host
is `typesafe.ai` or a subdomain. Per split, the new filter excludes:

| Split | Total | Unverified rights | Restricted source host | Remaining CC0 rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 79,116 | 4,206 | 27,858 | 47,052 |
| Calibration | 4,672 | 348 | 2,115 | 2,209 |
| Test | 10,356 | 606 | 3,750 | 6,000 |
| OOD | 15,701 | 600 | 3,900 | 11,201 |

The three Qwen runs use the pinned `Qwen/Qwen3-0.6B` base at revision
`c1899de289a04d12100db370d81485cdf75e47ca` and the shared LoRA decision head.
Each uses 512 train, 96 calibration, 128 test, and 128 OOD rows, with
`split_seed=20261010`. The model seeds are 11, 29, and 47. Dataset and group
selection inputs match across the three runs; the trainer verifies split
groups are disjoint. The frozen Qwen baseline is evaluated on the same rows.
Recomputed sorted `(row_id, group_id)` hashes from the pinned data and split
seed are:

| Split | Row/group SHA-256 |
| --- | --- |
| Train | `1f57563820f6361e5218e1a251a3e30a35a2c343973831df95f521129802a826` |
| Calibration | `e7c6886b35933c728dd55978594259352f6cf61a76d69edbea050d59549539f0` |
| Test | `56cdd5c162e7321dc3c354553268095c8c5bdecb469fc26295b7db783549f923` |
| OOD | `b6423ad2808c4cb290c718d2c725f684739acd7b1d3e72db6fd503f487772229` |

No the hosted reference model API calls, outputs, labels, or weights were used; the
OpenRouter key was not read. TypeSafe's current
[Master Customer Agreement §2.3(b)](https://typesafe.ai/legal/mca) prohibits
using its services or any output for distillation, imitation, or development
of a similar/competing product. The user-provided key and prompts do not change
those provider terms. The safe path here was training from independently
licensed data and the pinned Qwen base. The listwise candidate-scoring design
follows the public-corpus training guide,
not TypeSafe service behavior or outputs.

## Results

| OOD metric | Frozen Qwen baseline | Heliox scorer, 3-seed mean (SD) | Per-seed Heliox |
| --- | ---: | ---: | --- |
| Top-1 accuracy | 27.3% | 53.7% (2.5 pp) | 54.7%, 50.8%, 55.5% |
| Mean NLL | 2.076 raw | 1.044 (0.042), calibrated | 1.073, 1.062, 0.996 |
| Brier score | 0.379 raw | 0.190 (0.013), calibrated | 0.195, 0.201, 0.176 |

The OOD top-1 comparison is valid because temperature calibration does not
change the selected option. The baseline NLL/Brier values are raw while the
Heliox values use a temperature fitted on the 96-row calibration set, so those
proper-scoring-rule numbers are **not a fair baseline comparison**. The
training code has since been corrected to report a separately calibrated
frozen baseline for future runs; these three runs were not repeated for that
calculation.

On the 128 OOD rows, mean per-kind top-1 was 46.5% for choice and 41.7% for
ordinal score; yes/no balanced accuracy was 53.4%. This is weak, especially
for multi-valued ranking and balanced yes/no classification. The frozen
baseline's OOD top-1 was 27.3%; the official OOD mix contains 48
painting-geometry, 32 snake, and 48 ViZDoom decisions. These source families
also appear in training, so this is not source-held-out evaluation. It is a
small, synthetic probe—not Heliox desktop-task evidence.

A separate no-pretrained-backbone byte scorer was trained once on the same
clean filter (8,000 train and 1,000 each calibration/test/OOD). Its OOD top-1
was 52.0%, but OOD NLL was 1.320 versus 1.220 for the uniform predictor; a
shuffled-context control worsened NLL to 1.582. It is rejected as a calibrated
model and remains a diagnostic only.

## Limits and local reproduction

The best evidence is that Qwen fine-tuning changes candidate rankings on this
small clean OOD split, and the effect repeated over three model seeds. It does
not show performance close to the proprietary reference model: no permitted TypeSafe side-by-
side evaluation was made. These runs do not evaluate multi-step task completion,
tool use, screen understanding, voice interruption, workflow reliability, or
real user outcomes; they cannot justify execution authority. Keep these weights
out of Heliox runtime. The deterministic policy, user approvals, and post-action
verification remain authoritative.

All checkpoints and reports are ignored local artifacts; no trained weights
are committed. From `daemon/`, each run can be reproduced with:

```powershell
uv run --offline python scripts/train_general_decision.py `
  --train-rows 512 --calibration-rows 96 --test-rows 128 --ood-rows 128 `
  --seed 11 --split-seed 20261010 `
  --output-dir benchmarks/.bench-state/qwen_general_clean_cc0_seed11_split61010 `
  --report benchmarks/.bench-state/qwen_general_clean_cc0_seed11_split61010/report.json

uv run --offline python scripts/train_general_decision.py `
  --train-rows 512 --calibration-rows 96 --test-rows 128 --ood-rows 128 `
  --seed 29 --split-seed 20261010 `
  --output-dir benchmarks/.bench-state/qwen_general_clean_cc0_seed29_split61010 `
  --report benchmarks/.bench-state/qwen_general_clean_cc0_seed29_split61010/report.json

uv run --offline python scripts/train_general_decision.py `
  --train-rows 512 --calibration-rows 96 --test-rows 128 --ood-rows 128 `
  --seed 47 --split-seed 20261010 `
  --output-dir benchmarks/.bench-state/qwen_general_clean_cc0_seed47_split61010 `
  --report benchmarks/.bench-state/qwen_general_clean_cc0_seed47_split61010/report.json
```

The runs used the RTX 4060 Laptop, peaked at 3.82 GiB allocated VRAM, and took
583–835 seconds each in this local environment. Keep the research artifacts
local until a broader, independently sourced action dataset and real,
explicitly consented Heliox outcomes demonstrate repeatable improvement.
