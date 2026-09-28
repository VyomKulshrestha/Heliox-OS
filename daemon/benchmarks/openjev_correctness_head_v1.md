# Open-Jev correctness scorer — exploratory result

Date: 2026-09-29

Status: research-only; no runtime integration or promotion.

## Why this experiment

Open-Jev is a compact decision scorer: its released 2B artifact is a LoRA
adapter plus a scalar decision head that scores caller-supplied candidate
answers, rather than a general chat model that generates the answer. That is a
more relevant, locally runnable design target for Heliox's typed decisions
than trying to imitate a proprietary general assistant. See the
[Open-Jev model card](https://huggingface.co/ZefanCai/Open-Jev-2B) and
[loader](https://github.com/Zefan-Cai/Open-Jev).

Recent work also suggests two useful but distinct next steps: ASPIRE jointly
trains self-evaluation and selective prediction, while COIN calibrates a
decision threshold using a finite-sample risk bound. This run tested only a
small post-hoc logistic correctness scorer; it did not implement either paper's
method. They are follow-up hypotheses, not claimed results. See
[ASPIRE](https://arxiv.org/abs/2310.11689) and
[COIN](https://arxiv.org/abs/2506.20178).

## Frozen setup

- Protocol: [`openjev_correctness_head_v1.json`](openjev_correctness_head_v1.json).
- Public dataset: Open-Jev release at revision
  `c67699e13d0ae25e35b77165a4b6b079bedc8aba`; 2,440 previously exposed groups
  were excluded before selecting 128 fit and 128 test groups.
- The frozen Heliox RPS adapter and released Open-Jev adapter were evaluated
  against the same held-out test rows. The logistic scorer was fit only on the
  fit partition. Long prompts were excluded rather than truncated: 13 fit and
  17 test rows exceeded the candidate's 512-token limit. This left 115 fit
  rows and 111 paired test rows.
- Evaluation used the cached local model and GPU. No model or dataset download
  was needed for this run. No TypeSafe Jev API calls or outputs were used.
- The scorer is descriptive research. It cannot select, authorize, or execute
  desktop actions.

## Head-to-head result

| Same held-out rows | Released Open-Jev | Heliox RPS candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| Top-1 accuracy (111) | 81.98% | 82.88% | +0.90 pp (one row) |
| Mean NLL | 0.3710 | 0.3240 | -0.0470 |
| Mean Brier | 0.05534 | 0.04563 | -0.00971 |

Paired bootstrap (2,000 resamples, seed 731) gave a 95% percentile interval of
[-3.60, +5.41] percentage points for the top-1 difference, so the accuracy
gain is not established. Mean NLL difference was [-0.1048, -0.0036]; the
Brier interval was [-0.02216, +0.00059], narrowly including no difference.
By decision kind, top-1 was tied for Choice (32/45), tied for Score (19/22),
and 91% versus 93% for Noul (40/44 versus 41/44). These small subsets do not
show broad task or desktop parity.

## Confidence-scoring result and disposition

On the 111 held-out candidate rows, post-hoc logistic scoring moved AURC only
from 0.02910 to 0.02848. The paired 95% interval for learned-minus-raw AURC
was [-0.00410, +0.00217]. Brier worsened from 0.08141 to 0.08697. The
confidence score therefore remains rejected for runtime use; a tiny ranking
change is not calibration evidence.

The direct model comparison is a modest research signal, not Jev parity. It
compares Heliox only with the public Open-Jev release and public benchmark
labels—not TypeSafe Jev, live user intent, successful workflows, or real
desktop task outcomes. No model checkpoint, threshold, confidence score, or
authority was promoted. TypeSafe's current
[terms](https://typesafe.ai/legal/mca) restrict using its services or outputs
to distill/train an imitation or develop/facilitate a similar or competing
product; this work deliberately uses no such outputs.

## Fastest defensible next experiment

1. Keep the Open-Jev-style candidate scorer small and local; do not add a
   general-purpose model call to every action.
2. Collect explicitly opted-in Heliox-owned labels for whether a proposed
   decision was correct and whether the approved workflow achieved its goal;
   keep approval and deterministic safety rules authoritative.
3. On a new group-disjoint split, compare a simple frozen baseline against an
   ASPIRE-inspired joint decision/self-evaluation objective. Tune on fit/dev
   only, then report once on untouched test groups, including per-task results,
   abstention coverage, calibration, latency, and memory.
4. Promote nothing unless fresh outcomes show repeatable benefit without
   regressions in any safety-relevant task family.

## Reproduction

Run from `daemon/` with the already-prepared local model artifacts:

```powershell
python scripts/evaluate_general_decision_correctness_head.py --resume
```

The evaluator writes resumable predictions and `report.json` under the ignored
`benchmarks/.bench-state/openjev_correctness_head_v1/` directory. Model weights,
predictions, and caches are intentionally not committed.
