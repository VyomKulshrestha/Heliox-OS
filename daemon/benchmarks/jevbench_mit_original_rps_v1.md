# JevBench MIT original tier — frozen Heliox RPS comparison

Status: research-only. Neither model output can authorize or execute an action.

## Result

The frozen Heliox RPS adapter and a local copy of the released Open-Jev 2B
adapter were scored on the same 72 MIT-licensed original JevBench tasks using
the same Heliox prompt mapping. Neither model saw expected labels, and no
JevBench rows were used for training or calibration.

| Model / reference | Correct | Accuracy | Mean Brier ↓ | Mean NLL ↓ |
| --- | ---: | ---: | ---: | ---: |
| Heliox 1,024-row RPS adapter | 54/72 | 75.00% | 0.11021 | 0.65854 |
| Local released Open-Jev 2B adapter | 54/72 | 75.00% | 0.11072 | 0.65807 |
| Published Open-Jev 2B | 56/72 | 77.78% | — | — |
| Published Jev | 71/72 | 98.61% | — | — |

The paired Heliox-minus-local-Open-Jev accuracy difference is 0.0 percentage
points; its 95% group-bootstrap interval is −5.56 to +6.94 points over 36 task
groups. This small sample does not show an accuracy advantage for Heliox. The
small Brier difference favors Heliox, while NLL and ordered-score error favor
the baseline; none establishes a general improvement.

| Decision type | Heliox | Local Open-Jev 2B |
| --- | ---: | ---: |
| Choice | 26/36 (72.22%) | 26/36 (72.22%) |
| Noul | 17/24 (70.83%) | 17/24 (70.83%) |
| Ordered score | 11/12 (91.67%) | 11/12 (91.67%) |

The published 56/72 Open-Jev and 71/72 Jev results use their upstream prompt
contract ([upstream report](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md)),
so they are reference points—not a controlled head-to-head against this Heliox
mapping. The separately tracked
[`jevbench_mit_original_heliox_20260928.json`](jevbench_mit_original_heliox_20260928.json)
uses a different native next-token scorer and is not directly comparable either.

## Reproduction and limits

- Dataset: [JevBench original tier](https://github.com/fstandhartinger/jevbench/tree/9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3/datasets/public), pinned at revision `9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3`; SHA-256 `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`.
- The evaluator checks the pinned checksum and MIT provenance for every task. It fetches the small JSONL file in memory, rejects any prompt above the frozen model's token limit without truncation, and writes aggregate metrics only—no task text or per-task predictions.
- Both adapters ran locally on the RTX 4060 Laptop GPU. Each 72-task pass, including model loading, took about 28–31 seconds and peaked at 1.659 GiB of allocated CUDA memory. The model weights were already cached; no model or dataset was written to disk by this run.
- Run with the cached evaluation environment:

  ```powershell
  daemon\benchmarks\.bench-state\openjev-pinned-env\Scripts\python.exe daemon\scripts\evaluate_jevbench_mit_original.py --report daemon\benchmarks\jevbench_mit_original_rps_v1.json
  ```

The machine-readable aggregate result is
[`jevbench_mit_original_rps_v1.json`](jevbench_mit_original_rps_v1.json).
The run is an exploratory cross-dataset check, not Jev parity, broad agent
generalization, screen perception, desktop-task completion, or a safety result.

## Research implications

The [Open Alternative to Jev implementation and its benchmark notes](https://github.com/ikermoel/open-alternative-jev)
report higher results from larger Qwen models on a different typed-decisions
benchmark, while warning that packing several judgments into one prompt can
make neighboring answers influence one another. That suggests model capacity
and validated training data are stronger next levers than packing questions
for speed. The 4B/27B reported runs are not local Heliox measurements, and the
larger checkpoint is outside this laptop's practical 8-GiB training budget.
Heliox therefore keeps its separate-question scoring and the RPS adapter
research-only; no inference mode or checkpoint is promoted from this result.

The next useful quality signal is a fresh, group-disjoint task set plus
user-consented, locally recorded Heliox outcomes. Direct Jev-result distillation
remains disabled until written provider authorization is verified. The opt-in
desktop recorder was not activated for this evaluation.
