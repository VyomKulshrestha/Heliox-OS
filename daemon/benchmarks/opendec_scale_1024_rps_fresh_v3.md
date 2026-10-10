# Independent public-corpus candidate comparison (v3)

Status: research-only. The RPS candidate remains unpromoted and does not control
Heliox actions.

## Result

The frozen Heliox RPS candidate was compared with the released public 2B baseline
checkpoint on the same 482 previously unseen, group-disjoint public decisions.
The selection started with 512 licensed test-split rows; 30 candidate prompts
above its 512-token limit were excluded without truncation. The 482 common rows
include 67 ordered Score decisions.

| Decision set | Released public baseline | Heliox candidate | Candidate change |
| --- | ---: | ---: | ---: |
| All: top-1 accuracy | 78.84% | 80.71% | +1.87 pp |
| All: mean NLL | 0.5308 | 0.4477 | -0.0831 |
| All: mean Brier | 0.08279 | 0.06498 | -0.01781 |
| Choice: top-1 accuracy | 74.67% | 73.80% | -0.87 pp |
| Noul: top-1 accuracy | 86.02% | 89.25% | +3.23 pp |
| Score: top-1 accuracy | 73.13% | 80.60% | +7.46 pp |
| Score: ranked probability score | 0.06863 | 0.05426 | -0.01437 (20.9% lower) |

## Exploratory selective-risk readout

After the original comparison had been inspected, the frozen per-row predictions
were also scored with a tie-aware risk/coverage metric. Maximum candidate
probability ranks rows for this analysis; ties use expected risk under random
ordering. This is not a correctness head or calibrated probability of success.

| Selective metric ↓ | Released public baseline | Heliox candidate | Candidate change |
| --- | ---: | ---: | ---: |
| AURC (all 482) | 0.0593 | 0.0476 | -0.0117 |
| Selective risk at 50% coverage | 2.49% | 1.66% | -0.83 pp |
| Selective risk at 80% coverage | 14.25% | 10.10% | -4.15 pp |
| Selective risk at 90% coverage | 16.59% | 14.98% | -1.61 pp |
| Choice AURC (229) | 0.0873 | 0.0840 | -0.0033 |
| Noul AURC (186) | 0.0358 | 0.0223 | -0.0135 |
| Score AURC (67) | 0.0628 | 0.0301 | -0.0327 |

This secondary analysis is exploratory because it reuses the already-consumed
v3 selection. It cannot support threshold selection or promotion. In particular,
the scores are only a ranking heuristic; a learned, validated correctness head
would require representative Heliox outcome labels and a new untouched split.

On this single held-out sample, the candidate improves aggregate NLL/Brier and
Score ranking quality; Choice top-1 is slightly lower, while its secondary
AURC is slightly better. A post-hoc paired bootstrap over the 482 independent
`group_id` clusters estimates a +1.87 pp top-1 change (95% percentile interval
−0.41 to +4.15 pp; 93.3% of replicates positive). Because the interval includes
zero, the measured top-1 lift is not conclusive. This interval is descriptive,
not a promotion gate or evidence of desktop task success. The reproducible
calculation is in
[`opendec_scale_1024_rps_fresh_v3_group_bootstrap_exploratory.json`](opendec_scale_1024_rps_fresh_v3_group_bootstrap_exploratory.json).

## Selection and reproducibility

- Dataset: licensed public-corpus revision
  `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.
- Fresh selection: the held-out `test` split, seed `20261008`, 512 rows.
- The checksum-pinned exclusion manifests cover 1,928 groups from prior
  training, calibration, testing, OOD, and the earlier independent comparison.
- Protocol: [`released_opendec_general_protocol_v4.json`](released_opendec_general_protocol_v4.json)
  and [`opendec_scale_1024_rps_fresh_protocol_v3.json`](opendec_scale_1024_rps_fresh_protocol_v3.json).
- Machine-readable metrics and scored row IDs:
  [`opendec_scale_1024_rps_fresh_v3.json`](opendec_scale_1024_rps_fresh_v3.json).
- The initial independent OOD comparison is recorded in
  [`opendec_scale_1024_rps_fresh_v2.json`](opendec_scale_1024_rps_fresh_v2.json);
  it had only 8 Score rows, so it was not used to claim ordinal improvement.

The model artifacts and resumable logits remain in ignored `.bench-state`
directories. No model or dataset was downloaded during these evaluations; the
existing pinned CUDA environment and cached artifacts were reused.

## Limits and next use

This compares with the public-corpus release, not the proprietary hosted reference model;
it does not establish reference-model parity. No the hosted reference model outputs were collected or used
as labels. The candidate was not tuned on this comparison. Treat these rows as
consumed; any further tuning requires a new untouched selection. Product
promotion still requires separate runtime, safety, approval-path, and live
Heliox workflow evidence.
