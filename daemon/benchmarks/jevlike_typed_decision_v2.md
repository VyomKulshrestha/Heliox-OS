# Compact Jev-like option scorer (research-only)

## What the internet research changed

[Open-Jev's training guide](https://github.com/daseinlabs/open-jev/blob/main/docs/training.md)
describes two efficient patterns: score a supplied candidate set in one pass instead of generating a
free-form answer, and train a small listwise head over cached context/option
features. [jevlike](https://github.com/vinnylarouge/jevlike) provides a
permissively licensed byte-level version of the same input/output shape. A
newer independent [ODM Mini model card](https://huggingface.co/samatv256/mini-Jev)
also reports that strong single-step scores do not guarantee reliable
multi-step control: its shadow trajectories expose premature `finish`
decisions. Heliox therefore keeps this scorer out of execution and approval.

This probe applies jevlike's tiny, no-pretrained-backbone approach to the
public Open-Jev rows: learned UTF-8 byte embeddings, ordered context positions,
independent option pooling, and a shared option-to-context attention head. It
uses no task IDs or fixed option vocabulary. The data loader accepts only rows
whose provenance is verified CC0-1.0, and splits remain group-disjoint.

## Three-seed result

Each seed trained on 8,000 rows and used separate 1,000-row calibration, test,
and OOD partitions. Each OOD set contains 268 choice, 532 yes/no (`noul`), and
200 score decisions. Metrics below are averages of the three independent sampled
sets unless a seed is shown. Seed `20260930` is an additional run of the same
protocol and model size; it used the RTX 4060 for 66.2 seconds and peaked at
0.196 GiB of CUDA memory.

| OOD metric | Seed 20260928 | Seed 20260929 | Seed 20260930 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Correct-context top-1 | 60.7% | 58.0% | 57.1% | 58.6% |
| Correct-context mean NLL | 0.950 | 0.934 | 1.040 | 0.975 |
| Uniform expected top-1 | 38.0% | 38.1% | 38.1% | 38.1% |
| Uniform mean NLL | 1.077 | 1.073 | 1.072 | 1.074 |
| Full-context shuffled top-1 | 47.4% | 47.2% | 49.5% | 48.0% |
| Full-context shuffled mean NLL | 1.324 | 1.323 | 1.245 | 1.297 |
| Correct-context 10-bin ECE | 9.2% | 10.4% | 12.0% | 10.5% |

The complete-context control swaps each row's state **and question** with a
different same-kind, different-group example. Across the three seeds, shuffling
reduces top-1 by 10.6 percentage points on average and worsens NLL by 0.323.
That is repeatable evidence that this scorer uses context on this dataset—not
evidence of Heliox desktop generalization.

Aggregate top-1 is partly inflated by the many negative yes/no rows, so the
task-type results matter more:

| OOD task type | Seed 20260928 | Seed 20260929 | Seed 20260930 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Choice top-1 (random-choice expectation) | 42.9% (25.6%) | 33.6% (25.8%) | 35.8% (25.8%) | 37.4% (25.7%) |
| Yes/no balanced accuracy | 68.5% | 60.9% | 63.9% | 64.4% |
| Score top-1 | 44.0% | 49.0% | 35.5% | 42.8% |

Choice accuracy beats random chance on all three seeds, but the 33.6–42.9%
spread, weak score results, and 10.5% mean ECE are not enough to claim reliable
action selection. Do not use these outputs as trusted probabilities or as an
action-selection policy. The uniform top-1 baseline is the expected
`1 / option_count`, not the accuracy of an arbitrary first-option tie break.

## Cost, reproduction, and limits

- Context window: 2,048 bytes; only 5/1,000 OOD contexts were truncated.
- Option window: 128 bytes; none of the evaluated options were truncated.
- RTX 4060 Laptop: 42.1–66.2 seconds per 8k-row run; peak allocated VRAM was
  0.196 GiB. Each compressed local checkpoint is about 0.6 MB.
- Single-decision inference over 128 OOD rows (8 warmups, 120 measured) had
  1.14 ms GPU / 1.12 ms CPU median latency; p95 was 2.34 / 1.81 ms. These
  timings include local text encoding and tensor preparation but exclude the
  daemon/RPC and approval path. CPU and CUDA selected the same option on all
  128 decisions; that is numerical portability, not an accuracy measure.
- Reports and weights remain ignored local artifacts under
  `benchmarks/.bench-state/jevlike_tiny_8k_ctx2048_seed28_final/` and
  `benchmarks/.bench-state/jevlike_tiny_8k_ctx2048_seed29_final/`.
- Dataset: `ZefanCai/Open-Jev`, config `release-v2-redistributable`, revision
  `c67699e13d0ae25e35b77165a4b6b079bedc8aba`. The dataset must already be
  cached locally; the runner does not download data implicitly.

From `daemon/`, with CUDA-enabled PyTorch and that dataset revision cached:

```powershell
python scripts/train_tiny_typed_decision.py `
  --output-dir benchmarks/.bench-state/jevlike_tiny_8k_ctx2048_seed28_final `
  --train-rows 8000 --calibration-rows 1000 --test-rows 1000 --ood-rows 1000 `
  --context-bytes 2048 --option-bytes 128 --seed 20260928

python scripts/train_tiny_typed_decision.py `
  --output-dir benchmarks/.bench-state/jevlike_tiny_8k_ctx2048_seed29_final `
  --train-rows 8000 --calibration-rows 1000 --test-rows 1000 --ood-rows 1000 `
  --context-bytes 2048 --option-bytes 128 --seed 20260929
```

The ECE calculation and uniform top-1 control were corrected in this revision:
earlier local tiny-run reports divided ECE by the row count twice and scored
equal logits as an arbitrary option-0 choice, so those values are invalid and
are intentionally not reused here. Regression tests check analytical ECE and
expected random-choice accuracy; reports include calibration-bin counts,
confidence, and accuracy.

This is a compact research prototype, not a Jev/TypeSafe reproduction or
parity claim. It was trained only on licensed public rows; no TypeSafe/Jev
outputs, weights, or labels were used. It has not been tested on multi-step
Heliox workflows, live desktop state, or real user outcomes, and it is not
loaded by the daemon. The next useful evidence is reviewed, opt-in Heliox
workflow data with app/group-held-out tests, especially early-finish and
counterfactual-action cases. The existing deterministic policy, visible
approval, execution, and verification remain authoritative.
