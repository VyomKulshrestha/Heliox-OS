# JevBench MIT original tier — frozen Heliox RPS comparison

Status: research-only. Neither model output can authorize or execute an action.

## Result

The frozen Heliox RPS adapter and a local copy of the released Open-Jev 2B
adapter were scored on the same 72 MIT-licensed original JevBench tasks using
the same Heliox prompt mapping. A second, already-trained Heliox attention-head
experiment was also scored on the same tasks with its own feature prompt.
Expected labels were withheld from all models; no JevBench rows were used for
training or calibration.

| Model / reference | Correct | Accuracy | Mean Brier ↓ | Mean NLL ↓ |
| --- | ---: | ---: | ---: | ---: |
| Heliox 1,024-row RPS adapter | 54/72 | 75.00% | 0.11021 | 0.65854 |
| Local released Open-Jev 2B adapter | 54/72 | 75.00% | 0.11072 | 0.65807 |
| Heliox cross-attention v2 (0.6B) | 24/72 | 33.33% | 0.22502 | 1.37748 |
| Published Open-Jev 2B | 56/72 | 77.78% | — | — |
| Published Jev | 71/72 | 98.61% | — | — |

The paired Heliox-minus-local-Open-Jev accuracy difference is 0.0 percentage
points; its 95% group-bootstrap interval is −5.56 to +6.94 points over 36 task
groups. This small sample does not show an accuracy advantage for Heliox. The
small Brier difference favors Heliox, while NLL and ordered-score error favor
the baseline; none establishes a general improvement.

The cross-attention head is near the row-weighted uniform-choice reference on
this transfer set and substantially below both 2B scorers. It was trained on
960 Open-Jev rows, uses independent context/option features rather than the
RPS candidate prompt, and was added after inspecting the 2B result; treat it as
an exploratory architecture check, not a preregistered or directly comparable
model selection result. Its paired gap from the local Open-Jev 2B is −41.67
points (95% group-bootstrap interval −62.50 to −19.44). It should remain
unpromoted.

| Decision type | Heliox | Local Open-Jev 2B |
| --- | ---: | ---: |
| Choice | 26/36 (72.22%) | 26/36 (72.22%) |
| Noul | 17/24 (70.83%) | 17/24 (70.83%) |
| Ordered score | 11/12 (91.67%) | 11/12 (91.67%) |

The published 56/72 Open-Jev and 71/72 Jev results use their upstream prompt
contract ([upstream report](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md)),
so they are reference points—not a controlled head-to-head against this Heliox
mapping. The separately tracked
[`decbench_mit_original_heliox_20260928.json`](decbench_mit_original_heliox_20260928.json)
uses a different native next-token scorer and is not directly comparable either.

## Reproduction and limits

- Dataset: [JevBench original tier](https://github.com/fstandhartinger/jevbench/tree/9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3/datasets/public), pinned at revision `9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3`; SHA-256 `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`.
- The evaluator checks the pinned checksum and MIT provenance for every task. It fetches the small JSONL file in memory, rejects any prompt above the frozen model's token limit without truncation, and writes aggregate metrics only—no task text or per-task predictions.
- Both 2B adapters ran locally on the RTX 4060 Laptop GPU. Each 72-task pass, including model loading, took about 27–29 seconds and peaked at 1.659 GiB of allocated CUDA memory. The model weights were already cached; no model or dataset was written to disk by this run.
- The cross-attention check used the existing 0.6B backbone and head cache, took 3.49 seconds including model loading, and peaked at 1.174 GiB. All input sequences fit its 512-token cap without truncation.
- Run with the cached evaluation environment:

  ```powershell
  daemon\benchmarks\.bench-state\openjev-pinned-env\Scripts\python.exe daemon\scripts\evaluate_decbench_mit_original.py --report daemon\benchmarks\decbench_mit_original_rps_v1.json
  ```

The machine-readable aggregate result is
[`decbench_mit_original_rps_v1.json`](decbench_mit_original_rps_v1.json).
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
The local cross-attention result also gives no reason to replace the separate
2B scorer with the cached 0.6B head. Heliox keeps separate-question scoring;
neither the RPS candidate nor the cross-attention head is promoted from this
small evaluation.

The next useful quality signal is a fresh, group-disjoint task set plus
user-consented, locally recorded Heliox outcomes. Direct Jev-result distillation
remains disabled until written provider authorization is verified. The opt-in
desktop recorder was not activated for this evaluation.
