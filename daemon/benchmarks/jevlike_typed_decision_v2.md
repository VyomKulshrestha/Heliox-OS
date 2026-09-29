# Compact Jev-like option scorer (research-only)

## Provenance correction — 2026-09-30

A later source-URL audit found that the original CC0-only filter admitted
synthetic/programmatic rows whose provenance referenced TypeSafe-hosted docs or
eval pages. Dataset metadata did not identify those labels as TypeSafe model
outputs, and no TypeSafe service, key, output, or checkpoint was used, but the
older results did include provider-origin task references. Treat every metric
below as exploratory, not as the clean-source comparison. The shared loader now
also rejects any row with a `typesafe.ai` source URL; see the clean-source
replication report before using a result. Rows whose upstream license was
unverified were already excluded by the original filter.

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

Further public-repo research found [Kev](https://github.com/jaredpalmer/kev),
an Apache-2.0 typed-decision model. Its documented adaptation recipe warm-starts
a released checkpoint, replays prior examples, and reserves untouched data for
evaluation; its model card states that no Jev outputs were used for training.
The larger [Decider](https://github.com/Mapika/decider) recipe covers about 95
public datasets and 1.47M examples but reports a 5.3-hour GH200 training run,
so it is not a sensible local-laptop reproduction. These projects reinforce
the practical route for Heliox: learn from licensed data and Heliox-owned
outcomes, then test on held-out tasks.

This probe applies jevlike's tiny, no-pretrained-backbone approach to the
public Open-Jev rows: learned UTF-8 byte embeddings, ordered context positions,
independent option pooling, and a shared option-to-context attention head. It
uses no task IDs or fixed option vocabulary. The original data loader accepted
rows whose provenance declared CC0-1.0, and splits remained group-disjoint; it
did not yet exclude provider-origin source URLs. See the provenance correction
above.

## Source-heldout transfer check (2026-09-30)

The old row-heldout `ood` split reused source families across training and
evaluation. A new protocol now dynamically selects a source category shared by
all four frozen splits, removes it from train/calibration/test, and uses it
only for OOD. Source selection and row sampling have separate seeds. The
loader validates metadata in bounded Parquet batches, keeps the existing
sorted-ID/source-kind round-robin sample semantics, retrieves only selected
rows in bounded batches, and hashes the file incrementally. No source names
are included in model prompts.

Three 4k-row leave-one-source-out runs used the same model seed, row seed,
2,048-byte context window, and architecture (width/rank 64/64). Each used
4,096 train, 512 calibration, 512 seen-source test, and 512 held-out OOD rows.
The fold seed selected each available source category in turn:

| Held-out source | OOD top-1 (uniform expected) | Choice top-1 (uniform expected) | Yes/no balanced accuracy | OOD NLL (uniform) | OOD ECE | Shuffled-context NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `painting-geometry-v1` | 28.1% (28.3%) | 0.0% (8.8%) | 50.0% | 1.991 (1.513) | 23.9% | 1.999 |
| `snake-v1` | 54.3% (41.7%) | 23.4% (33.3%) | 52.6% | 1.758 (0.896) | 28.9% | 1.687 |
| `vizdoom-basic-v1` | 37.3% (34.5%) | 19.3% (33.3%) | 54.1% | 1.584 (1.133) | 25.8% | 1.661 |

Seen-source test top-1 was 68.0%, 67.0%, and 61.0%, respectively, but
held-out choice accuracy stayed below its random-choice baseline in all
folds, yes/no balanced accuracy stayed near chance, and OOD NLL was worse
than uniform in every fold. Shuffling context sometimes improved NLL. The
above-chance aggregate top-1 in the snake fold is therefore not evidence of
useful generalization. A separate 8k-train/1k-OOD snake fold also had 32.8%
choice accuracy versus 33.3% chance, 51.8% yes/no balanced accuracy, 1.969
OOD NLL versus 0.896 uniform, and 26.8% ECE.

These source-heldout runs **reject this scratch byte scorer as a generic
decision model**. It remains research-only and is not wired into the daemon.
No TypeSafe Jev service, API key, output, or checkpoint was used; these results
do not establish Jev parity or real-desktop capability. The next meaningful
training signal is Heliox-owned task outcome data, not additional tuning on
source-overlapping synthetic rows.

## Five-seed row-held-out result (not source-generalization)

Each seed trained on 8,000 rows and used separate 1,000-row calibration, test,
and OOD partitions. Each OOD set contains 268 choice, 532 yes/no (`noul`), and
200 score decisions. Metrics below average five independently seeded runs.
The additional seed `20261002` used the RTX 4060 for 66.4 seconds and peaked
at 0.196 GiB of CUDA memory. Seed `20261001` is also the baseline for the paired
ordinal-loss ablation below.

| OOD metric | Seed 20260928 | Seed 20260929 | Seed 20260930 | Seed 20261001 | Seed 20261002 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct-context top-1 | 60.7% | 58.0% | 57.1% | 59.7% | 57.7% | 58.6% |
| Correct-context mean NLL | 0.950 | 0.934 | 1.040 | 0.959 | 0.955 | 0.968 |
| Uniform expected top-1 | 38.0% | 38.1% | 38.1% | 38.3% | 38.2% | 38.2% |
| Uniform mean NLL | 1.077 | 1.073 | 1.072 | 1.066 | 1.071 | 1.072 |
| Full-context shuffled top-1 | 47.4% | 47.2% | 49.5% | 51.0% | 47.3% | 48.5% |
| Full-context shuffled mean NLL | 1.324 | 1.323 | 1.245 | 1.164 | 1.201 | 1.251 |
| Correct-context 10-bin ECE | 9.2% | 10.4% | 12.0% | 10.6% | 10.4% | 10.5% |

The complete-context control swaps each row's state **and question** with a
different same-kind, different-group example. Across the five seeds, shuffling
reduces top-1 by 10.2 percentage points on average and worsens NLL by 0.284.
That is repeatable evidence that this scorer uses context on this dataset—not
evidence of Heliox desktop generalization.

Aggregate top-1 is partly inflated by the many negative yes/no rows, so the
task-type results matter more:

| OOD task type | Seed 20260928 | Seed 20260929 | Seed 20260930 | Seed 20261001 | Seed 20261002 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Choice top-1 (random-choice expectation) | 42.9% (25.6%) | 33.6% (25.8%) | 35.8% (25.8%) | 41.0% (25.8%) | 39.9% (25.8%) | 38.7% (25.8%) |
| Yes/no balanced accuracy | 68.5% | 60.9% | 63.9% | 62.5% | 61.7% | 63.5% |
| Score top-1 | 44.0% | 49.0% | 35.5% | 46.0% | 42.5% | 43.4% |

Choice accuracy beats random chance on all five seeds, but the 33.6–42.9%
spread, weak score results, and 10.5% mean ECE are not enough to claim reliable
action selection. Do not use these outputs as trusted probabilities or as an
action-selection policy. The uniform top-1 baseline is the expected
`1 / option_count`, not the accuracy of an arbitrary first-option tie break.

### Rejected ordinal-loss ablation

On the same seed `20261001` OOD rows (identical row/group hashes), adding a
0.25 ranked-probability-score loss to ordinal score decisions made the result
worse: score RPS changed from `0.2218` to `0.2295`, score top-1 from `46.0%` to
`41.0%`, and all-task OOD NLL from `0.959` to `1.014`. The experimental
objective was not retained. This paired result is evidence against that
setting, not evidence that further loss tuning will close a gap to Jev.

## Follow-up scaling and capacity probes

The 40k-row scaling check used seed `20261002` and kept calibration, test, and
OOD row/group hashes identical to its 8k-row baseline. More training data
improved the sampled test NLL but harmed OOD calibration and NLL, so this
setting is rejected as a general improvement:

| Training rows / head | Test NLL | OOD top-1 | OOD NLL | OOD ECE |
| --- | ---: | ---: | ---: | ---: |
| 8k / width 64, rank 64 | 0.780 | 57.7% | 0.955 | 10.4% |
| 40k / width 64, rank 64 | 0.692 | 57.2% | 1.170 | 16.4% |

Widening the small head from 64 to 128 gave a modest improvement on that same
sample, but capacity selection had already observed those OOD rows. A paired
follow-up therefore fixes the partition seed independently of model
initialization. All six runs used the same 8k train, 1k calibration, 1k test,
and 1k OOD row/group hashes, with `split_seed=20261010`; all groups are
disjoint across those splits.

| Width / rank | Model seeds | OOD top-1 | OOD NLL | OOD ECE | Choice top-1 | Noul balanced accuracy | Score top-1 | Shuffled-context NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 / 64 | 11, 29, 47 | 57.2% | 0.996 | 13.1% | 38.2% | 64.9% | 39.5% | 1.256 |
| 128 / 128 | 11, 29, 47 | 58.1% | 0.949 | 10.2% | 37.9% | 68.1% | 37.7% | 1.211 |

This paired result supports a limited claim: width 128 improved aggregate
OOD NLL, calibration, and yes/no balanced accuracy on this one fixed public
split, while choice and score top-1 did not improve. It is not evidence of
Jev parity, desktop-workflow generalization, or a safe execution policy. The
model remains research-only and is not loaded by Heliox. The training code now
accepts `--split-seed` for reproducible paired comparisons; omitting it keeps
the prior behavior (`split_seed` defaults to `--seed`).

Artifacts and per-run reports remain in ignored local
`benchmarks/.bench-state/jevlike_tiny_*` directories. The runs used only the
revision-pinned, per-row CC0-filtered public dataset described above; no
TypeSafe Jev service, output, weights, or labels were used.

## Cost, reproduction, and limits

- Context window: 2,048 bytes; only 5/1,000 OOD contexts were truncated.
- Option window: 128 bytes; none of the evaluated options were truncated.
- RTX 4060 Laptop: 42.1–66.4 seconds per 8k-row run; peak allocated VRAM was
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

python scripts/train_tiny_typed_decision.py `
  --output-dir benchmarks/.bench-state/jevlike_tiny_8k_ctx2048_seed20261002 `
  --train-rows 8000 --calibration-rows 1000 --test-rows 1000 --ood-rows 1000 `
  --context-bytes 2048 --option-bytes 128 --seed 20261002
```

The ECE calculation and uniform top-1 control were corrected in this revision:
earlier local tiny-run reports divided ECE by the row count twice and scored
equal logits as an arbitrary option-0 choice, so those values are invalid and
are intentionally not reused here. Regression tests check analytical ECE and
expected random-choice accuracy; reports include calibration-bin counts,
confidence, and accuracy.

This is a compact research prototype, not a Jev/TypeSafe reproduction or
parity claim. No TypeSafe Jev service, output, weights, or labels were used.
However, as noted above, these older runs included rows whose task sources
referenced TypeSafe-hosted pages; they are superseded for clean-source claims.
It has not been tested on multi-step
Heliox workflows, live desktop state, or real user outcomes, and it is not
loaded by the daemon. The next useful evidence is reviewed, opt-in Heliox
workflow data with app/group-held-out tests, especially early-finish and
counterfactual-action cases. The existing deterministic policy, visible
approval, execution, and verification remain authoritative.
