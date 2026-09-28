# Open-Jev scale-1024 local study

Date: 2026-09-28  
Status: research candidate accepted for further evaluation; not enabled for action authority.

## What was tested

This is a local comparison of a Qwen3.5-2B candidate fine-tuned from the pinned
released Open-Jev adapter/head against that released checkpoint. The training
data came from the pinned `ZefanCai/Open-Jev` redistributable dataset, and the
trainer admitted only rows whose per-row provenance passed its CC0-1.0 gate.
Training, model loading, and scoring ran offline from the existing cache; no
TypeSafe Jev API calls or Jev outputs were used.

The experiment used seed `482176`, LoRA rank 8, NF4 double quantization,
`candidate_v1` prompts, a 512-token limit, 1,024 selected train rows, and 256
rows each for calibration, test, and the preregistered OOD split. After
tokenization, the usable counts were 906 / 224 / 227 / 219 respectively.
The candidate's temperature was fit on calibration rows only. Option scoring
was microbatched in pairs; the CUDA parity check matched the serial scores and
direct gradients for Noul, Choice, and Score decisions.

The frozen protocol and thresholds are in
[`openjev_scale_1024_protocol_v1.json`](openjev_scale_1024_protocol_v1.json).
It required OOD top-1 no lower than baseline, lower aggregate NLL, and no
decision kind more than five points below baseline.

## Results

| Split | Rows | Model | Top-1 | NLL | Brier |
| --- | ---: | --- | ---: | ---: | ---: |
| Test | 227 | Released Open-Jev | 85.90% | 0.3676 | 0.0600 |
| Test | 227 | Candidate | 88.55% | 0.2554 | 0.0393 |
| Preregistered OOD | 219 | Released Open-Jev | 83.56% | 0.7446 | 0.0852 |
| Preregistered OOD | 219 | Candidate | 84.02% | 0.7038 | 0.0806 |

The preregistered gate passed on point estimates. That first OOD sample was
small: the top-1 gain was one row, and paired/group-aware bootstrap intervals
crossed zero. I therefore ran a supplemental holdout after freezing the
candidate. It selected one example per previously unused group, excluded all
groups from the training, calibration, test, and first OOD splits, and scored
998 of 1,024 selected groups (26 exceeded the token limit). This supplemental
sample was not used for training or calibration; because it followed review of
the first OOD result, treat it as a post-hoc independent validation, not a
preregistered primary result.

| Fresh group-disjoint OOD | Rows | Released Open-Jev | Candidate | Candidate delta |
| --- | ---: | ---: | ---: | ---: |
| Top-1 accuracy | 998 | 79.26% | 81.36% | +2.10 pp |
| Mean NLL | 998 | 0.5928 | 0.4495 | -0.1433 |
| Mean Brier | 998 | 0.0999 | 0.0727 | -0.0273 |

Paired bootstrap (5,000 resamples; one row per group) gave 95% intervals of
`+0.60 to +3.71 pp` for top-1 accuracy, `-0.1863 to -0.1020` for NLL, and
`-0.0367 to -0.0188` for Brier. The fresh-set gain was concentrated in
Noul decisions and the snake domain; reasoning-control and VizDoom top-1 were
unchanged. The 76-row Score slice lost 3.95 pp top-1 and its Brier score
worsened (0.0828 to 0.0949), although its NLL improved. Workflow-control
sources total only 24 rows, so those numbers are descriptive, not evidence of
desktop-workflow performance.

After adding a generalized ranked-probability metric, I rescored the cached
logits on the original preregistered 219-row OOD split (no model or data
changes). Its 51 Score rows had mean rank error `0.3932` and mean RPS
`0.0854` for released Open-Jev, versus `0.3294` and `0.0703` for the candidate.
This is a retrospective metric on the already-reported split, not a fresh
evaluation and not evidence that the candidate was trained with RPS (its
`ordinal_weight` was zero). It does not negate the separate 76-row post-hoc
Score regression above.

A final local typed-scoring smoke test used a new read-only system-health
question and four runtime-supplied options. The model assigned 0.788 to the
read-only evidence-and-report option; its response explicitly carried
`research_only: true` and `execution_authority: false`. No desktop action ran.
This checks the scoring entry point, not task-success quality.

## Interpretation and limits

The candidate is a measurable improvement over this released Open-Jev
checkpoint on the fresh, group-disjoint public OOD sample. It is not proof of
parity with proprietary TypeSafe Jev, a reproduction of undisclosed Jev
training, a Heliox task-success result, or an authorization/safety signal.
Keep it research-only; do not let its scores approve actions, bypass
confirmation, or replace the deterministic gateway/world-model rules. Improve
and independently test Score decisions and real Heliox workflow cases before
considering any product integration.

## Data and local artifacts

- Open-Jev source: [repository](https://github.com/razorback16/openjev)
- Dataset revision: [`c67699e13d0ae25e35b77165a4b6b079bedc8aba`](https://huggingface.co/datasets/ZefanCai/Open-Jev/tree/c67699e13d0ae25e35b77165a4b6b079bedc8aba)
- Warm-start revision: [`0c7aa498b1627be8da4acf34c863ff0ee0a92785`](https://huggingface.co/ZefanCai/Open-Jev-2B/tree/0c7aa498b1627be8da4acf34c863ff0ee0a92785)
- Fresh selection: seed `482181`; selected IDs SHA-256 `bdbdfc57265b5fc729592ef3cc08bf64c9628fa19e84befe05b229384ae16e2a`; group IDs SHA-256 `dca711b9a65a2ea3c43f72c1a206568eff80ab8ae2e2d258de8476cc8c02d32e`.
- Candidate adapter SHA-256: `c3a271d1ff81b05340485754de1ba0b34e4b8bde02b21e3e32da665895c9e9f7`.
- Candidate head SHA-256: `28449933c9cb0ecbaa28b36697486ba43acbf17f9850ac0e0a076448bbd85a1d`.
- Training checkpoint SHA-256: `68a7ab0c7774a92c5cc27ac0c1a52319ac5318d357003b85cdfa8803f6761ad2`.
- Local model, exact JSON reports, and evaluation logits remain under the ignored
  `daemon/benchmarks/.bench-state/openjev_scale_1024_batched_v1/`; weights are
  intentionally not committed.
- Training plus candidate-side evaluation took 3,278 seconds and peaked at
  3.503 GiB of PyTorch GPU allocation on the RTX 4060 Laptop GPU (8 GiB).
