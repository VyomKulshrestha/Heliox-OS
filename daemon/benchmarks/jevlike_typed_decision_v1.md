# Jev-inspired typed-decision probe (local, experimental)

## Method

Open-Jev's public training guide recommends a frozen language-model backbone,
cached features, a small listwise decision head, and a shuffled-context control.
This probe adapts that shape to Heliox's typed decision contract: a frozen local
Qwen3-0.6B backbone with an isolated packed state/question/candidate attention
mask and a trainable bilinear pointer head. Context-shuffle evaluation
re-encodes each question and its options against another same-kind example's
state; swapping only pooled vectors would be invalid because the packed
candidate vectors are contextualized.

Reference: [Open-Jev training and evaluation](https://github.com/daseinlabs/open-jev/blob/main/docs/training.md).

## Run

- Dataset: `ZefanCai/Open-Jev` at revision
  `c67699e13d0ae25e35b77165a4b6b079bedc8aba`,
  restricted by the loader to verified CC0-1.0 rows.
- Split seed: `20260930`; 960 train, 188 calibration, 121 test, and 116 OOD
  examples remained after model-length filtering.
- Backbone precision: BF16; head checkpoint selected at epoch 5; calibrated
  temperature: 2.842.
- Backbone revision: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- GPU: NVIDIA GeForce RTX 4060 Laptop; peak allocated CUDA memory: 1.191 GiB.
- The final run reused the ignored local feature cache. Its 39.7-second runtime
  is not the full first-run time, which also includes feature extraction.
- Outputs (report and weights) are local-only under ignored
  `benchmarks/.bench-state/packed_pointer_v3_bf16_shuffle_control/`; no trained
  artifact is shipped or loaded by Heliox.

## Results

| OOD evaluation, same 115-row subset | Top-1 | Mean NLL | Mean Brier |
| --- | ---: | ---: | ---: |
| Correct state | 61.7% | 0.962 | 0.173 |
| Shuffled state, re-encoded | 46.1% | 1.155 | 0.201 |
| Uniform | 45.2% | 1.118 | 0.209 |

The correct-state score drops 15.7 percentage points after shuffling and its
NLL worsens, which is evidence that the head uses state for at least part of
this benchmark. The effect is not uniform: choice accuracy is 60.0% with the
correct state and 25.7% shuffled (20.0% uniform); score accuracy is 38.5%,
23.1%, and 19.2%, respectively. Noul does not show useful state dependence:
correct-state balanced accuracy is 54.6%, shuffled is 56.8%, and the model
misses 12 of 14 positive cases. Its ordinary 74.1% accuracy is mostly explained
by the negative-heavy label distribution.

I also selected a binary yes threshold on the separate calibration split by
maximizing Noul balanced accuracy (85 calibration rows, 21 positive). The
selected threshold was `0.144`; on OOD it produced 8/14 true positives and
21/41 true negatives (balanced accuracy 54.2%), with 20 false positives. That
does not improve over the default argmax balanced accuracy of 54.7%, so the
threshold is not promoted.

An earlier local Qwen3.5-2B zero-shot next-token experiment did not produce a
reliable all-task improvement. Neither experiment establishes Jev parity,
real-desktop generalization, safety, or execution authority. The scorer remains
opt-in/research-only; production decisions and approvals continue through
Heliox's existing deterministic policy path. Jev/TypeSafe provider outputs were
not used as training targets or evaluation labels.

## Reproduction

With the pinned model and dataset already cached locally, from `daemon/`:

```powershell
python scripts/train_packed_typed_decision.py `
  --output-dir benchmarks/.bench-state/packed_pointer_v3_bf16_shuffle_control `
  --feature-cache benchmarks/.bench-state/packed_pointer_features_v3_bf16.npz `
  --train-rows 1024 --calibration-rows 192 --test-rows 128 --ood-rows 128 `
  --max-length 512 --epochs 8 --patience 2 --seed 20260930 `
  --shuffled-context-control
```

The script is local-only for model loading and refuses to use an incomplete or
mismatched feature cache. The benchmark is diagnostic—not an acceptance gate.
