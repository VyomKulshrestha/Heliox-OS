# Released Open-Jev general-decision comparison (v1)

Status: research-only. This evaluates Open-Jev's published checkpoint; it is
not a Heliox-trained model and is not enabled in Heliox.

## Frozen artifacts and protocol

- Open-Jev checkpoint: `ZefanCai/Open-Jev-2B@0c7aa498b1627be8da4acf34c863ff0ee0a92785`.
- Base model: Qwen3.5-2B at `15852e8c16360a2fea060d615a32b45270f8a8fc`, loaded in NF4.
- Adapter SHA-256: `2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f`.
- Scalar-head SHA-256: `3532cd576c58d5ad5bf17c3e9f2df4be8c70e08c07fa6bb7fa673dcd7b401f2a`.
- The protocol pins group-disjoint archived and fresh public OOD selections,
  prompt length, and checkpoint identity in
  [`released_openjev_general_protocol_v1.json`](../daemon/benchmarks/released_openjev_general_protocol_v1.json).
- The 128-row fresh selection was compared against the live Jev model
  `typesafe/jev-1.13-20260917` on the exact same rows. The 128 Jev responses
  cost `$0.003261426` and were used for evaluation only, never training or
  calibration. The selection and comparison rules are pinned in
  [`released_openjev_jev_protocol_v1.json`](../daemon/benchmarks/released_openjev_jev_protocol_v1.json).

The quantized local run is not numerically identical to the published BF16
model. All rows are public synthetic decisions; neither the data nor this
comparison is evidence of desktop transfer, real-user task performance, or
execution authority.

## Paired fresh OOD results

| Decision kind | Rows | Open-Jev / local gold result | Jev gold result | Agreement |
| --- | ---: | ---: | ---: | ---: |
| `noul` (binary) | 70 | 90.0% accuracy | 84.3% accuracy | 80.0% |
| `choice` | 38 | 60.5% accuracy | 42.1% accuracy | 39.5% |
| `score` (ordinal) | 20 | 0.207 mean absolute error | 0.343 mean absolute error | — |

For binary probabilities, mean absolute probability difference was `0.270`.
For ordinal scores, the mean absolute local-to-Jev score difference was
`0.246` levels. Metrics and call metadata are preserved in the ignored local
artifact `daemon/benchmarks/.bench-state/released_openjev_jev_v1/report.json`;
the model logits and row-level checkpoints remain local as well.

The low choice agreement is not uniform across sources:

| Source family | Rows | Open-Jev gold accuracy | Jev gold accuracy | Agreement |
| --- | ---: | ---: | ---: | ---: |
| reasoning-control | 15 | 26.7% | 40.0% | 40.0% |
| Snake | 15 | 73.3% | 60.0% | 53.3% |
| VizDoom basic | 8 | 100.0% | 12.5% | 12.5% |

The VizDoom subset is especially diagnostic: the local model followed the
public labels on all eight choices while Jev did so on one. In contrast, both
models were weak on reasoning-control choices. These small per-source counts
are hypotheses for follow-up, not stable domain rankings; the current evidence
does not support distilling Jev's choices as ground truth.

## Interpretation and next gate

This result does **not** show Jev parity. The local checkpoint has better
accuracy than Jev on this small fresh set, but agrees on only 15/38 choice
decisions. On this selection, lower agreement cannot be read as worse task
performance because Jev itself misses many public choice labels. The result
shows both that the released checkpoint is a useful research baseline and that
matching Jev behavior is a separate objective from matching the dataset's
labels.

The next research step is to inspect disagreement by source and decision
pattern, then preregister a new held-out comparison before any adaptation.
Jev responses remain evaluation-only unless provider terms explicitly permit
training use. No Heliox runtime, authority boundary, or model artifact was
changed by this evaluation.

## Reproduction

From `daemon/`, with the pinned Hugging Face artifacts cached and an
OpenRouter key in the parent `.env` only for the live comparison:

```bash
python -m scripts.evaluate_released_openjev_general --resume
python -m scripts.compare_released_openjev_jev --resume
python -m pytest tests/test_released_openjev_general.py tests/test_compare_released_openjev_jev.py tests/test_general_decision.py -q
ruff check scripts/evaluate_released_openjev_general.py scripts/compare_released_openjev_jev.py tests/test_released_openjev_general.py tests/test_compare_released_openjev_jev.py
```
