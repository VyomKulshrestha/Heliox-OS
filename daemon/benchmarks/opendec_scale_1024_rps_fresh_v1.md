# Exploratory group-filtered public-corpus candidate comparison

Status: exploratory research evidence only; candidate remains unpromoted.

The current RPS candidate was scored on the exact public OOD selection used by
the pinned released-public-corpus comparison. The cached baseline logits were
verified against the row IDs and selection signature before comparing. Review
after the run found that 38 of the 128 selected groups also appear in the
candidate's earlier OOD selection (seed 482176). Thus, despite being disjoint
from the released-baseline protocol's archived groups, this is **not** an
independent candidate holdout. Both models were evaluated on the same 112 rows;
16 of 128 selected rows exceeded the candidate's 512-token input limit and
were omitted without truncation. No training or calibration used these rows,
but aggregate results could still reflect prior candidate selection. See the frozen
[protocol](opendec_scale_1024_rps_fresh_protocol_v1.json) and
[machine-readable metrics](opendec_scale_1024_rps_fresh_v1.json).

## Results

| Metric | Released public baseline | RPS candidate | Candidate − released |
| --- | ---: | ---: | ---: |
| Top-1 accuracy, 112 rows | 79.46% | 82.14% | +2.68 pp |
| Mean NLL | 0.63792 | 0.52609 | −0.11183 |
| Mean Brier score | 0.08602 | 0.06840 | −0.01762 |
| Score RPS, 20 rows | 0.02037 | 0.01940 | −0.00098 |

By decision type, Choice top-1 was 60.53% → 63.16% (38 rows), Noul was
88.89% → 92.59% (54 rows), and Score was 90.00% → 90.00% (20 rows). These
are descriptive point estimates on a small synthetic sample, not a significance
claim. The Score subset is especially small, so its RPS difference is uncertain.

## Interpretation and limits

On this overlapping retrospective slice, the candidate improved all four
aggregate point estimates and did not reduce top-1 in any decision type. These
metrics are descriptive only: they do not establish general improvement or
closeness to the proprietary hosted reference model. A genuinely new group-disjoint
selection—excluding both earlier candidate OOD groups and baseline groups, and
chosen before either model is scored—is still required before promotion.

This remains a public synthetic benchmark only. It says nothing about Heliox
desktop task completion, permissions, or action safety. No the hosted reference model API
outputs were requested, stored, used as labels, or used for tuning. The model
stays outside Heliox's runtime and execution authority.

The implementation follows the documented open public-corpus pattern—candidates
are explicit inputs, a shared scalar head scores each candidate, and scores are
normalized into probabilities—rather than claiming to reproduce the reference model's private
method. See the public-corpus method notes.
