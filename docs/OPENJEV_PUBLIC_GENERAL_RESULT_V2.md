# Open-Jev 2B public general-decision baseline (v2)

This is an offline evaluation of the independent Open-Jev project checkpoint,
not a Heliox-trained or Heliox-runtime model. It uses only its pinned public
weights and the independently licensed public decision dataset; it does not
use outputs from a hosted proprietary model.

## Frozen inputs

- Checkpoint: `ZefanCai/Open-Jev-2B@0c7aa498b1627be8da4acf34c863ff0ee0a92785`.
- Base: Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Adapter SHA-256: `2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f`.
- Scalar-head SHA-256: `3532cd576c58d5ad5bf17c3e9f2df4be8c70e08c07fa6bb7fa673dcd7b401f2a`.
- Public dataset revision: `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.
- Both partitions contain 128 rows; the fresh partition excludes archived
  groups and duplicates within itself. Selection rules and the archived
  row-ID hash are frozen in
  [`released_openjev_general_protocol_v2.json`](../daemon/benchmarks/released_openjev_general_protocol_v2.json).
- Inference used Qwen3.5-2B NF4; quantized results need not match the published
  BF16 checkpoint.

## Public-label results

| Decision kind | Archived rows / result | Fresh rows / result |
| --- | --- | --- |
| Binary (`noul`) | 67 / 86.6% accuracy | 70 / 90.0% accuracy |
| Choice | 35 / 71.4% accuracy | 38 / 60.5% accuracy |
| Ordinal score | 26 / 0.139 mean rank error | 20 / 0.207 mean rank error |

The weakest fresh category is choice selection. The result supports continuing
offline work on choice transfer, but does not establish quality on real Heliox
tasks, desktop transfer, latency under an interactive workload, or safe
execution. No weights were trained or promoted by this evaluation; approvals
and deterministic execution gates remain unchanged.

The complete local report and resumable logits are under the ignored path
`daemon/benchmarks/.bench-state/released_openjev_general_public_v2/` and are
not committed.

## Reproduction

From `daemon/`, with the pinned base and adapter already cached:

```bash
python -m scripts.evaluate_released_openjev_general --resume
python -m pytest tests/test_released_openjev_general.py tests/test_general_decision.py -q
ruff check scripts/evaluate_released_openjev_general.py tests/test_released_openjev_general.py
```

If complete `archived_logits.json` and `fresh_logits.json` checkpoints already
exist but the report is missing, add `--summarize-checkpoints` to rebuild the
report without loading the GPU model.
