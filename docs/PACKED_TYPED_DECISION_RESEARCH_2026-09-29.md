# Packed typed-decision research — 2026-09-29

Status: research only. The new packed scorer is not wired into Heliox's runtime, does not grant action authority, and is not evidence of Jev parity.

## Approach

Recent public Jev-inspired work suggests three useful ideas: isolate independent questions while sharing request context; train a shared candidate-scoring head on labeled option sets; and calibrate probabilities on a separate split. Kev reports a Qwen3.5/3.8 model family with typed `choice`, `score`, and `noul` outputs, while its authors also report that their numbers are not a controlled comparison with Jev and that option order can affect answers. The Open-Jev scoring notes warn that raw continuation likelihood is a measure of textual plausibility, not correctness, and recommend labeled-head training when labels exist.

- [Kev research and limitations](https://github.com/jaredpalmer/kev)
- [Open-Jev one-pass scoring design](https://github.com/daseinlabs/open-jev/blob/main/docs/design/one-pass-option-scoring.md)
- [Open-Jev teacher-to-student distillation pipeline](https://github.com/Shalimov04/open-jev)

Heliox's probe uses a pinned, cached Qwen3-0.6B backbone and a shared rank-512 bilinear head. Question tokens cannot read candidates; each candidate can read only the request state, its own question, and itself. Choice candidates are canonicalized, while `noul` and ordinal score choices preserve their meaning/order. The head uses layer normalization, a listwise target loss, and held-out temperature calibration. Training rows come only from the public CC0 subset of the pinned Open-Jev dataset revision; no Jev outputs, user screen captures, or private tasks are used.

## Result

The normalized head was selected on a 512-train / 96-calibration / 128-test / 128-OOD experiment. Rows longer than the 512-token limit were excluded, leaving 122 test and 114 OOD rows. The OOD set holds out source families. Uniform is the within-row uniform-probability baseline.

| Split | Metric | Uniform | Packed head v2 |
| --- | --- | ---: | ---: |
| Test (122) | Top-1 accuracy | 42.6% | 58.2% |
| Test (122) | NLL ↓ | 1.160 | 0.975 |
| Test (122) | Brier ↓ | 0.200 | 0.184 |
| OOD (114) | Top-1 accuracy | 43.9% | 47.4% |
| OOD (114) | NLL ↓ | 1.137 | 1.135 |
| OOD (114) | Brier ↓ | 0.200 | 0.225 |

OOD top-1 by type was 54.3% for choice, 49.1% for Noul, and 34.6% for ordinal score. The Noul baseline was 64.2%, so the trained head regressed on that type; aggregate OOD calibration was not improved (Brier worsened). The overall OOD lift is small and should not hide this failure mode.

The unnormalized v1 head was more overconfident (test NLL 5.323, OOD NLL 7.663). Layer normalization plus a calibration temperature of 6.918 substantially corrected that probability pathology. The final v2 process used a 1.169 GiB CUDA peak and took 46.6 seconds while reusing cached train/calibration features and extracting the remaining test/OOD features; earlier train/calibration extraction attempts are not included in that elapsed time. Its cache is about 6.7 MiB and is ignored under `daemon/benchmarks/.bench-state/`.

## Interpretation and next gate

This is a useful architecture and calibration result, not a Heliox-ready decision model. The test/OOD numbers were inspected during iteration and are now development evidence, not a fresh confirmatory holdout. Do not tune against these rows again or enable this scorer for OS/browser actions. The existing Qwen LoRA general-decision runs remain the stronger local baseline; their separate sample metrics must not be compared as if they shared this split.

Before any promotion, use a newly preregistered source-held-out split and report per-type selective risk, calibration, option-order invariance, shuffled-state controls, and comparison to the current Heliox baseline. Any personalization data must be explicitly consented, locally retained, minimized, and license/rights reviewed. Keep deterministic policy checks, approval prompts, and action allowlists authoritative regardless of model confidence.

## External compact-model lead reviewed 2026-09-29

[openJev-verdict-2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) is a useful architecture lead: it describes a marker-pointer encoder, a soft option-distribution head, a separate correctness head, per-type/cardinality calibration, and a symmetric option-permutation loss ([model](https://github.com/Heman10x-NGU/openJev-verdict-2.0/blob/main/verdict2/model.py), [training losses](https://github.com/Heman10x-NGU/openJev-verdict-2.0/blob/main/verdict2/losses.py)). Its README claims 77.10% top-1 and roughly 20–25 ms per decision, but these are the authors' results on their own benchmark, not Heliox evidence or proof of Jev parity.

The evaluation needs reconciliation before reuse: the README describes 2,000 held-out decisions, while the checked-in [dataset manifest](https://github.com/Heman10x-NGU/openJev-verdict-2.0/blob/main/data/dataset_manifest.json) reports 5,000 rows with a 500-row test split and 211 test-contamination cases. These could describe different evaluation artifacts, but the distinction is not established by the headline claim. No checkpoint or dataset was downloaded. The safe next step is to verify the authors' receipts and licenses, then test portable ideas only on a newly selected, rights-cleared Heliox split; do not import its reported score as a benchmark or train on its data by assumption.

Reproduce with cached dependencies and weights (no download):

```powershell
cd daemon
python scripts/train_packed_typed_decision.py `
  --feature-cache benchmarks/.bench-state/packed_pointer_features_v2.npz `
  --output-dir benchmarks/.bench-state/packed_pointer_v2
```

Focused verification on the implementation: `15 passed` across `test_packed_typed_decision.py` and `test_typed_decision.py`; Ruff lint and format checks passed. The feature cache, trained weights, and generated JSON report remain local and ignored.
