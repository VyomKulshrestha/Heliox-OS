# Open-Jev scale-1024 ranked-probability-loss ablation

Date: 2026-09-28
Status: research-only candidate; passes the frozen point-estimate gate; not promoted.

## Method

This is a one-variable ablation from the frozen
[`openjev_scale_1024_rps_protocol_v1.json`](openjev_scale_1024_rps_protocol_v1.json).
It warm-starts the pinned Qwen/Qwen3.5-2B model from the released Open-Jev 2B
checkpoint and changes only the generalized ordinal RPS loss weight from 0 to
0.5. Training used only rows whose per-row provenance passed the CC0-1.0 gate;
no TypeSafe Jev calls or outputs were used. The process ran offline and used the
already-cached environment and model.

The selected split limits were 1,024 train and 256 each for calibration, test,
and OOD. After tokenization, 906 / 224 / 227 / 219 rows remained. Temperature
was fit on calibration only. The run resumed from its exact local optimizer
checkpoint after interruption and reused completed baseline evaluations.

## Results

| Split | Rows | Model | Top-1 | NLL | Brier | Score RPS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Test | 227 | Released Open-Jev | 85.90% | 0.36757 | 0.06003 | 0.03794 |
| Test | 227 | RPS candidate | 88.55% | 0.25578 | 0.03948 | 0.02250 |
| Preregistered OOD | 219 | Released Open-Jev | 83.56% | 0.74460 | 0.08524 | 0.08536 |
| Preregistered OOD | 219 | RPS candidate | 84.02% | 0.70364 | 0.07974 | 0.07116 |

OOD per-kind top-1 was 79.71% to 75.36% for Choice (−4.35 pp), 86.87% to
88.89% for Noul (+2.02 pp), and 82.35% to 86.27% for Score (+3.92 pp). The
primary OOD Score RPS fell by 0.01420 (about 16.6%). Every preregistered
point-estimate condition passed: lower Score RPS, no lower overall top-1,
lower overall NLL, and no decision kind more than five points down.

## Limits and disposition

This is a retrospective ablation on the already-used OOD split, not independent
confirmation. The 51 OOD Score rows and the −4.35 pp Choice change are small
and close enough to the per-kind guardrail to warrant a fresh group-disjoint,
pre-registered replication before relying on the result. Passing this gate does
not show parity with proprietary TypeSafe Jev, Heliox task success, desktop
execution quality, or safety. The candidate remains unpromoted and cannot
authorize or execute actions.

## Exploratory option-order smoke

On a new OOD selection (seed 482190), the artifact evaluator scored 110 of 128
rows; 18 over-budget rows were excluded without truncation. Reordering Choice
alternatives with seed 482191 changed the top choice on 0 of 35 Choice rows.
After aligning probabilities by option text, mean L1 distance was 0.001036
(maximum 0.010819). This small post-hoc diagnostic is not a preregistered
benchmark or proof of universal permutation invariance. See
[openjev_scale_1024_rps_permutation_smoke_v1.json](openjev_scale_1024_rps_permutation_smoke_v1.json).

## Reproducibility

- Dataset: ZefanCai/Open-Jev release-v2-redistributable at revision
  c67699e13d0ae25e35b77165a4b6b079bedc8aba.
- Warm-start: Open-Jev 2B at revision
  0c7aa498b1627be8da4acf34c863ff0ee0a92785.
- Seed: 482176; LoRA rank 8; max length 512; ordinal weight 0.5.
- Candidate adapter SHA-256: 5bb2153d689476375b7784884964605411d000f20883b38a9ccd44eae0bfc0a3.
- Candidate head SHA-256: aa2a39aee6e7b48efc50eff3a0f1ac5c4fdf391b534bf8b874a4a8da9ba65676.
- Training checkpoint SHA-256: 5bc03079e326633fbb3770a8410a0036b8714207d45788747d3ecbff34011ee4.
- Full metric and manifest report: [`openjev_scale_1024_rps_v1.json`](openjev_scale_1024_rps_v1.json).
- Peak PyTorch GPU allocation: 3.503 GiB. The model weights and evaluation
  caches are local ignored artifacts, not committed.
