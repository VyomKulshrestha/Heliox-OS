# Open-Jev correctness scorer v2 — fresh group-held-out evaluation

Date: 2026-09-29

Status: research-only; no runtime integration or promotion.

## Protocol and reproducibility

The v1 report used rows from the released checkpoint's training split and is
therefore only an in-training diagnostic. Version 2 rejects any fit/evaluation
split listed in the base model's training mixture, reconstructs checksum-pinned
prior exposures, and selects disjoint task groups from a calibration fit split
and an OOD evaluation split. See the frozen
[`opendec_correctness_head_v2.json`](opendec_correctness_head_v2.json).

- Dataset: `ZefanCai/Open-Jev`, release-v2-redistributable, revision
  `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.
- Previously exposed groups excluded before selection: 2,696.
- Correctness fit: 128 selected calibration rows; 121 scored after seven rows
  exceeded the candidate's 512-token limit. Evaluation: 256 OOD rows, all
  scored, on groups disjoint from fit and prior exposure.
- OOD evaluation composition: 245 `snake-v1`, 11 `reasoning-control-v1`;
  131 Choice, 121 Noul, and 4 Score decisions. This is group-held-out but
  heavily source-skewed, so it is not broad agent-task generalization evidence.
- The Heliox model weights were the already-trained 1,024-row RPS candidate;
  this run fit only the post-hoc correctness scorer on the 121 calibration
  examples. No model checkpoint was updated.
- Candidate and released Open-Jev were scored on the exact same 256 test rows.
  No TypeSafe Jev API outputs were used. The direct API comparison harness is
  currently disabled pending verified written authorization.
- The complete machine report is stored locally at
  `benchmarks/.bench-state/openjev_correctness_head_v2/report.json` and is
  intentionally ignored; report SHA-256:
  `4a0126cb3e9aa9453c298c3b4294a05d227b06ec16fed7ba126d45795a9a64ed`.

## Candidate versus released Open-Jev

| Metric on the same 256 OOD decisions | Released Open-Jev | Heliox RPS candidate | Paired bootstrap, candidate − baseline |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 81.64% | 82.81% | +1.17 pp; 95% CI −1.57 to +4.30 pp |
| Mean NLL | 0.4458 | 0.4041 | −0.0417; 95% CI −0.1035 to +0.0248 |
| Mean Brier | 0.07086 | 0.04727 | −0.02359; 95% CI −0.03956 to −0.01085 |
| AURC (lower is better) | 0.05914 | 0.08303 | Candidate is worse by 0.02389 |

The top-1 and NLL intervals include no difference. The lower Brier score is a
positive calibration signal on this narrow sample, but the candidate's worse
selective-risk ranking and source skew prevent a general improvement claim.

The same paired rows are source-skewed in different directions:

| Source | Rows | Released Open-Jev | Heliox candidate | Delta |
| --- | ---: | ---: | ---: | ---: |
| `snake-v1` | 245 | 82.04% | 83.67% | +1.63 pp |
| `reasoning-control-v1` | 11 | 72.73% | 63.64% | −9.09 pp |

The simple two-source macro is 77.38% for Open-Jev and 73.65% for Heliox.
That macro is descriptive—not a stable estimate with only 11 reasoning rows—
but it makes clear that the overall row-weighted lead is not consistent across
the available sources. The next comparison should add untouched task families,
not tune against these already exposed rows.

## Post-hoc correctness scorer

The logistic scorer uses only generic probability features (maximum candidate
probability, top-two margin, normalized entropy, and option count); it has no
task IDs or fixed task vocabulary.

| Held-out correctness estimate | Raw max probability | Learned scorer |
| --- | ---: | ---: |
| AURC | 0.08303 | 0.07857 |
| Brier | 0.10860 | 0.11015 |
| AUROC | 0.8215 | 0.8324 |
| Mean estimated correctness | 83.45% | 86.22% |

Across 2,000 paired bootstraps, learned-minus-raw AURC was −0.00446
(95% interval −0.01091 to −0.00088; learned AURC was lower in 99.75% of
resamples). Brier worsened slightly, and the scorer overestimated correctness
by about 3.4 points on the held-out rows. Keep it offline/shadow-only: it may
rank cases for review, but it is not a calibrated probability, safety gate, or
execution authority.

## What current research changes

Open-Jev's updated public results show why a 2B adapter alone should not be
called Jev-equivalent: on its 231-task public subset, Open-Jev-2B scored
150/231, Open-Jev-9B 179/231, and its independently audited 27B v1.1 scored
197/231 versus Jev's 200/231 (80/111 vs. 81/111 on Hard). The completed 27B
checkpoint used a frozen 328,672-row mixture containing 148,639 training rows;
its public benchmark report explicitly limits the comparison to 231 available
tasks, not the full 534-task benchmark. The recipe uses a shared scalar scorer
over candidate descriptions, group-isolated evaluation, and a separate
calibration split. These are upstream results, not Heliox measurements, and
the protocols are not interchangeable. Reproducing the 27B run required four
H100 GPUs, so it is not a practical local-training target for Heliox's 8 GB
RTX 4060.

See the [audited public JevBench report](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md),
[shared-scoring and training method](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/multidomain-training.md),
[internal evaluation protocol](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/internal-evaluation.md),
and [Open-Jev project history](https://zefan-cai.github.io/open-jev/story/).

For Heliox's world model, [WorldEvolver](https://arxiv.org/abs/2606.30639)
suggests a lower-cost complementary direction: retain observed action/state
transitions, derive reusable rules from prediction mismatches, and pass only
reliable foresight to the planner instead of continually updating large-model
weights. [MemWM](https://arxiv.org/abs/2608.07107) independently emphasizes
retrieved transition rules, state caches, and hard-to-predict facts, evaluated
with structured state-fidelity checks rather than text similarity alone.
[Agentic Abstention](https://arxiv.org/abs/2606.28733) likewise treats
stopping and gathering more evidence as sequential choices. This points to a
more efficient Heliox next step than another broad fine-tune: retrieve only
verified, relevant episodes/rules and abstain from using low-confidence visual
foresight. Heliox already has an append-only experience ledger, verifier-backed
online learning, and an opt-in encrypted visual-transition trainer; the latter
remains shadow-only because its 16x16 observations are not enough to establish
desktop state fidelity. These are design hypotheses, not proof of desktop
performance; Heliox still needs consented, real before/action/after traces and
an untouched task-family test.

## Disposition and next evidence

No candidate, correctness scorer, threshold, approval rule, or world-model
artifact is promoted. Do not train on TypeSafe Jev outputs; the existing
repository guard blocks direct Jev comparisons pending verified written
authorization, and this run respected that guard. Continue training only from
redistributable public data or explicitly consented Heliox observations.

The next useful step is a prospective, source-balanced evaluation using
previously unused task groups, then a separate opt-in local recording run to
collect actual Heliox transitions. Current recorded visual samples are
encrypted and local; they are not automatically fed into the planner or used
for background training. See the [recorded-workflow design](../pilot/workflows/recorded_workflows.py)
and [world-model contract](../pilot/intelligence/world_model.py).

Reproduce the frozen experiment from `daemon/` with cached dataset/model files
and the CUDA-enabled research environment:

```powershell
python scripts/evaluate_general_decision_correctness_head.py --resume
```

The run used an RTX 4060 Laptop GPU, Torch 2.14.0+cu130, Transformers 5.10.2,
PEFT 0.19.1, and Accelerate 1.13.0. The ignored report contains selection
hashes, per-kind metrics, and bootstrap output; raw screen or private user data
was not part of this experiment.
