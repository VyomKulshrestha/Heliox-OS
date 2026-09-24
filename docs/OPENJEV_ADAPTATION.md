# Open-Jev architecture study for Heliox decisions

Status: **research only; no new decision model is enabled in Heliox**.

## What Open-Jev actually trains

[Open-Jev](https://github.com/Zefan-Cai/Open-Jev) is an independent
implementation inspired by TypeSafe's Jev, not the proprietary Jev model or
its training pipeline. Its [model](https://github.com/Zefan-Cai/Open-Jev/blob/main/jev/model.py)
freezes a language-model base, trains a low-rank adapter and one shared scalar
head, scores each candidate from a candidate-conditioned prompt, and normalizes
scores within that question. Its yes/no (“Noul”) path uses one candidate score
against a zero baseline. It initializes the scalar from the base model's
single-token Yes-minus-No output weights. A separate calibration split fits
temperature; grouped test and OOD splits are not training inputs. See the
[training design](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/multidomain-training.md)
and [public dataset card](https://huggingface.co/datasets/ZefanCai/Open-Jev).

This is materially different from Heliox's production scope head, which uses
fixed numeric features. It also requires a capable backbone and diverse,
well-separated supervision; copying the scalar formula alone is not enough.

## Bounded local probes

We first adapted the shared candidate head to a cached, frozen Apache-2.0 NLI
encoder. Its pooled 768-dimensional representation is read from the pinned ONNX
graph; only a numeric scalar head is fitted. This is an architectural ablation,
**not LoRA training or an Open-Jev checkpoint**. All data below are public or
authored synthetic cases. No user screen, task, or private execution trace was
sent to Jev or used in fitting.

| Probe | Train / calibration | Held-out result | Interpretation |
| --- | --- | --- | --- |
| Heliox scope, mixed ScopeJudge and Jev labels | 3,221 expert-labeled public calls + 816 synthetic Jev pairs; grouped by trajectory/request | ScopeJudge test: 0.482 balanced accuracy, 33 false allowances / 785 calls. Fresh Jev suite: 30/60 agreement, three false allowances. | Fails transfer and safety gate. |
| Same scope head with existing deterministic guard | Same weights | Fresh Jev suite: 33/60 agreement, zero false allowances but 27 false warnings. | Guard is not a cure for missing semantic recall. |
| Existing Heliox semantic-v5 baseline | 916 synthetic Jev pairs, developed previously | Fresh Jev suite: 53/60 agreement, zero false allowances. | Better than the candidate-head ablation, but not Jev parity or real-world proof. |
| Open-Jev public multidomain sample | 1,128 train / 332 calibration, source-balanced sample; shared 768-parameter head | 246/444 top-1 test (55.4%); 227/360 OOD (63.1%). Workflow domains 82.5–95% on test, while several game/spatial domains were 10–52.5%. | Small text NLI backbone and 256-token cap do not support broad decisions. |

The fresh 60-case scope suite was committed **after** the scope-head artifact
froze and before the Jev calls. Jev answered 60/60 of these synthetic cases
consistently with the authored labels. This suite is now inspected development
evidence, not an untouched final test for future iterations. The older v4
suite is also no longer a fresh holdout. The exact numeric results and hashes
are in [`openjev_scope_probe_v1.json`](../daemon/benchmarks/openjev_scope_probe_v1.json),
[`openjev_scope_fresh_v1.json`](../daemon/benchmarks/openjev_scope_fresh_v1.json),
and [`openjev_multidomain_probe_v1.json`](../daemon/benchmarks/openjev_multidomain_probe_v1.json).

The public scope validation AUC was 0.861, but its trajectory-held-out test
AUC was 0.502. Training the same head on synthetic Jev pairs only improved the
already-inspected fresh suite to 44/60, but produced eight false allowances.
This points to both domain shift and an inadequate frozen representation. It
does not justify tuning a threshold on the test set or shipping the head.

## Trainable-backbone follow-up

[`train_qwen_scope_noul.py`](../daemon/scripts/train_qwen_scope_noul.py) uses a
pinned [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
base, rank-8 LoRA, Open-Jev-style Noul prompt, Yes-minus-No scalar
initialization, a one-pass soft-label loss, and separate request-group fit,
calibration, and validation partitions. It trained 520 rows on an 8 GB RTX
4060 Laptop GPU, peaking at 1.401 GiB of allocated CUDA memory. Its untouched
request-group validation was 149/164 agreement, with two false allowances.
The prior 60-case Jev suite scored 54/60, zero false allowances, but had been
inspected during earlier model development and is **not** an independent final
test for this adapter. The adapter and scalar head remain local under ignored
`daemon/benchmarks/.bench-state/qwen_scope_v1/`, with frozen file hashes in
[`qwen_scope_probe_v1.json`](../daemon/benchmarks/qwen_scope_probe_v1.json).
The 1.5 GB base model is not bundled. This smaller base and training dataset
cannot establish Jev parity.

The independent 60-case comparison suite in
[`jev_scope_qwen_v1_holdout.py`](../daemon/benchmarks/jev_scope_qwen_v1_holdout.py)
and the adapter hashes were committed before Jev was queried. The weights
themselves are not tracked due to unverified Jev-output redistribution rights.
The frozen evaluator verifies both model-file hashes and does not train or
select a threshold from this suite. Jev agreed with all 60 authored labels. On these
same cases, the adapter agreed with Jev on **51/60 (85%)**, with zero false
allowances and nine false warnings. The existing Heliox semantic-v5 head scored
**45/60 (75%)** with one false allowance and 14 false warnings after its guard.
Thus the trainable adapter improved this narrow synthetic comparison, but did
not match Jev and remains unpromoted. These paired cases are intentionally
simple; zero false allowances here says nothing conclusive about safety in
uncontrolled desktop tasks.

The full results, including every mismatch and artifact hash, are in
[`qwen_scope_fresh_v1.json`](../daemon/benchmarks/qwen_scope_fresh_v1.json),
[`qwen_scope_baseline_v1.json`](../daemon/benchmarks/qwen_scope_baseline_v1.json),
and [`jev_scope_qwen_v1_teacher.json`](../daemon/benchmarks/jev_scope_qwen_v1_teacher.json).

## Licensed multi-domain GPU follow-up

The scope adapter above handles only binary action-scope questions. A separate
research model now trains **one candidate-conditioned scalar head** across
Open-Jev's `noul`, variable-option `choice`, and ordinal `score` records. It
uses the pinned Qwen3-0.6B base, rank-8 LoRA, no prompt labels or source IDs,
and group-disjoint train/calibration/test/OOD partitions. The loader accepts
only records whose per-row provenance is CC0-1.0; it excludes 4,206 training
records with unverified upstream description rights, plus corresponding
calibration/test/OOD records. There are no proprietary Jev outputs in this
experiment. The base model stays frozen; the adapter and one shared head are
trained on the GPU. Candidate gradients are replayed one at a time to keep
memory bounded when option counts vary.

The 512-train / 96-calibration / 128-test / 128-OOD source-and-kind-balanced
probe, with 768-token maximum and no truncation, peaked at **3.653 GiB of
allocated CUDA memory** on the 8 GB RTX 4060 Laptop GPU. Relative to the
untrained candidate head, held-out test top-1 rose from **38.3% to 61.7%**
and OOD top-1 from **27.3% to 57.8%**. Test negative log-likelihood fell
from 2.117 to 1.024; OOD from 2.691 to 0.941 after calibration. The full
[report](../daemon/benchmarks/qwen_general_probe_v1.json) contains per-kind
and per-source results and artifact hashes. Importantly, OOD choice accuracy
is only **34.3%** and OOD score accuracy **38.5%**; these are broad synthetic
decision weaknesses, not Jev parity. The pilot model remains the old,
independent production path. This experiment grants no action authority.

With cached public data and pinned base model, reproduce using:

```bash
cd daemon
python scripts/train_general_decision.py --train-rows 512 --calibration-rows 96 --test-rows 128 --ood-rows 128 --max-length 768
python scripts/check_general_decision_gradient.py
python -m pytest tests/test_general_decision.py tests/test_score_general_decision.py -q
```

The research-only scorer `scripts/score_general_decision.py` takes a local
JSON file with `state`, `kind`, `question`, and `options`. It validates the
request and saved adapter/head hashes, refuses overlong input instead of
truncating it, and returns probabilities without invoking Heliox tools.
An actual saved-adapter reload on the
[read-only action fixture](../daemon/tests/fixtures/general_decision_read_only.json)
incorrectly favored **deleting** the file (40.2%) over reading it (34.8%).
This qualitative failure reinforces the OOD metrics. Neither synthetic
accuracy nor a score constitutes permission to execute.

```bash
cd daemon
python scripts/score_general_decision.py --input tests/fixtures/general_decision_read_only.json
```

A larger 2,048-train / 256-calibration / 256-test / 256-OOD GPU run used a
different seed and sample; its within-run untrained baseline was 30.5% test
and 28.1% OOD top-1, versus **62.5% on both** after training. The test and OOD
samples differ from the 512-row run, so the two trained percentages are not a
controlled head-to-head comparison. The larger run still had 46.8% test
choice accuracy and 43.1% OOD score accuracy. Its saved adapter again favored
**deleting** the file over reading it on the fixture (46.5% versus 44.3%).
On the previously frozen 60-case Heliox scope suite, it agreed with Jev on
only **29/60**, with two false allowances. A separate evaluation-only sample
of 24 licensed public OOD questions cost $0.000609 in Jev API calls: agreement
was 7/11 Noul and 7/8 choice; mean score gap was 1.05 levels across five
score questions. The small samples and constructed criteria do not establish
general Jev parity. Reports: [larger GPU probe](../daemon/benchmarks/qwen_general_probe_v2.json),
[scope transfer](../daemon/benchmarks/qwen_general_scope_transfer_v2.json),
and [public Jev comparison](../daemon/benchmarks/qwen_general_jev_comparison_v2.json).
The adapter remains local and disabled in Heliox.

## Reproduction and release boundary

The offline probes need `onnx`, `onnxruntime`, `pyarrow`, `tokenizers`,
`huggingface_hub`, NumPy, SciPy, and scikit-learn. They use cached model/data
snapshots and do not silently fetch a model during scoring. The Open-Jev data
probe pins dataset revision `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.
The ScopeJudge input is the public `dreadnode/scopejudge` `train.jsonl`; each
report records its SHA-256. The Qwen follow-up requires CUDA plus optional
PyTorch, Transformers, and PEFT packages; they are not pilot runtime
dependencies. Its research
adapter is versioned under the benchmark directory, but no Heliox runtime
imports or enables it.

OpenRouter's [distillation guidance](https://openrouter.ai/docs/cookbook/evaluate-and-optimize/distillation)
requires checking model-provider permission before using outputs for training.
The Jev Decisions API did not establish such permission in the material we
reviewed. The trained adapter is kept local and must not be redistributed or
promoted until TypeSafe/OpenRouter terms for this use are confirmed. The
Apache-2.0 license of Qwen's base does not resolve the Jev-output question.

For offline verification with the pinned base already in the Hugging Face
cache:

```bash
cd daemon
python scripts/evaluate_qwen_scope_frozen.py
python scripts/evaluate_qwen_scope_baseline.py
python -m pytest tests/test_qwen_scope_frozen.py tests/test_train_qwen_scope_noul.py -q
```

No experiment here can override the deterministic gateway, user approvals, or
least-privilege execution boundaries. Promotion requires a fresh, consented
and de-identified desktop workload, grouped held-out validation, calibrated
uncertainty, a strict false-allowance gate, and a runtime latency/resource
check. Public synthetic accuracy or Jev agreement alone is not authorization.
