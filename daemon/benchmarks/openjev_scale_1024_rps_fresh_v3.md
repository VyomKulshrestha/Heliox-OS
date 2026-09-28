# Independent Open-Jev candidate comparison (v3)

Status: research-only. The RPS candidate remains unpromoted and does not control
Heliox actions.

## Result

The frozen Heliox RPS candidate was compared with the released Open-Jev 2B
checkpoint on the same 482 previously unseen, group-disjoint public decisions.
The selection started with 512 licensed test-split rows; 30 candidate prompts
above its 512-token limit were excluded without truncation. The 482 common rows
include 67 ordered Score decisions.

| Decision set | Released Open-Jev | Heliox candidate | Candidate change |
| --- | ---: | ---: | ---: |
| All: top-1 accuracy | 78.84% | 80.71% | +1.87 pp |
| All: mean NLL | 0.5308 | 0.4477 | -0.0831 |
| All: mean Brier | 0.08279 | 0.06498 | -0.01781 |
| Choice: top-1 accuracy | 74.67% | 73.80% | -0.87 pp |
| Noul: top-1 accuracy | 86.02% | 89.25% | +3.23 pp |
| Score: top-1 accuracy | 73.13% | 80.60% | +7.46 pp |
| Score: ranked probability score | 0.06863 | 0.05426 | -0.01437 (20.9% lower) |

On this single held-out sample, the candidate improves aggregate NLL/Brier and
Score ranking quality; Choice top-1 is slightly lower. These are descriptive
benchmark results, not a confidence-tested claim or evidence of desktop task
success.

## Selection and reproducibility

- Dataset: licensed Open-Jev revision
  `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.
- Fresh selection: the held-out `test` split, seed `20261008`, 512 rows.
- The checksum-pinned exclusion manifests cover 1,928 groups from prior
  training, calibration, testing, OOD, and the earlier independent comparison.
- Protocol: [`released_openjev_general_protocol_v4.json`](released_openjev_general_protocol_v4.json)
  and [`openjev_scale_1024_rps_fresh_protocol_v3.json`](openjev_scale_1024_rps_fresh_protocol_v3.json).
- Machine-readable metrics and scored row IDs:
  [`openjev_scale_1024_rps_fresh_v3.json`](openjev_scale_1024_rps_fresh_v3.json).
- The initial independent OOD comparison is recorded in
  [`openjev_scale_1024_rps_fresh_v2.json`](openjev_scale_1024_rps_fresh_v2.json);
  it had only 8 Score rows, so it was not used to claim ordinal improvement.

The model artifacts and resumable logits remain in ignored `.bench-state`
directories. No model or dataset was downloaded during these evaluations; the
existing pinned CUDA environment and cached artifacts were reused.

## Limits and next use

This compares with the public Open-Jev release, not proprietary TypeSafe Jev;
it does not establish Jev parity. No TypeSafe Jev outputs were collected or used
as labels. The candidate was not tuned on this comparison. Treat these rows as
consumed; any further tuning requires a new untouched selection. Product
promotion still requires separate runtime, safety, approval-path, and live
Heliox workflow evidence.
