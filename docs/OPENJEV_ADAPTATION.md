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

The next planned multi-task run adds the
[MIT-licensed ScopeJudge](https://github.com/dreadnode/scopejudge) expert labels
to the public general questions, keeping complete security trajectories
disjoint. The pinned source contains 4,897 labeled calls. The selected scope
sample has 512 balanced training calls, 80 balanced calibration calls, and
608 calls in a full trajectory-held-out test; at the 768-token limit, 492,
61, and 554 respectively were eligible without truncation. This run is
**incomplete**: eager attention stopped after 192 training rows and optimized
attention after 96, both with CUDA launch failures. Windows recorded
`nvlddmkm` event IDs 13 and 153 at the failure times. That correlation does
not establish whether the driver, hardware, power settings, or a kernel is
the root cause. No v3 weights or held-out metrics exist; see the
[attempt record](../daemon/benchmarks/qwen_general_scope_attempt_v3.json).
The training script now saves atomic, signature-checked local checkpoints and
can resume an unfinished run with the same inputs and `--resume`. A small CUDA
round-trip verified trainable parameters and optimizer moments are restored.
It does not repair the GPU fault, so sustained training should wait for a
stable GPU environment. No Windows driver or TDR setting was changed.
Silently lowering the token limit is not an equivalent fix: a 512-token cap
would exclude 278 of the 608 selected ScopeJudge test calls, materially
changing the evaluation population.

## Released Open-Jev checkpoint transfer

Open-Jev has since published an [Apache-2.0 2B adapter and scalar head](https://huggingface.co/ZefanCai/Open-Jev-2B)
for pinned Qwen3.5-2B. This is Open-Jev's trained model, **not** a Heliox
model or TypeSafe's proprietary Jev. The local GPU evaluation pinned the
adapter repository to `0c7aa498b1627be8da4acf34c863ff0ee0a92785`,
the base to `15852e8c16360a2fea060d615a32b45270f8a8fc`, and Open-Jev
loader code to `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`.
The adapter file matched its published SHA-256. No user data or Jev API key
was sent to this localhost-only evaluation.

On the previously inspected 60-case synthetic scope suite, the released
checkpoint scored **45/60 (75%)**, with **zero false allowances and 15 false
warnings** at its saved 0.5 decision threshold. Its separate read-only file
choice fixture favored reading (96.2%) over deleting (2.6%) or uploading
(1.2%). These two results are not interchangeable: a single good action
choice does not rescue weak scope recall. The full [transfer report](../daemon/benchmarks/openjev_2b_scope_transfer_v1.json)
records per-case probabilities, model hashes, token count, and GPU inference
time. The suite was already inspected during earlier development, so this is
transfer evidence rather than a fresh final holdout. It does not justify
replacing Heliox's deterministic approval and scope checks.

To reproduce with the published checkpoint running on localhost port 8791:

```bash
cd daemon
python scripts/evaluate_openjev_checkpoint.py --suite qwen-v1
python -m pytest tests/test_evaluate_openjev_checkpoint.py -q
```

### Public ScopeJudge transfer and frozen-feature adaptation

The same released checkpoint was evaluated on the pinned, trajectory-disjoint
[ScopeJudge](https://github.com/dreadnode/scopejudge) partition. Of 608 selected
test calls, 546 fit the explicit 768-token limit (517 allowed, 29 disallowed);
62 were excluded, never silently truncated. At the checkpoint's 0.5 threshold,
it made **zero false allowances but 507 false warnings**: 39/546 accuracy and
0.510 balanced accuracy. Its 59 eligible calibration calls similarly produced
zero false allowances and 33 false warnings. The [test](../daemon/benchmarks/openjev_2b_scopejudge_v1.json)
and [calibration](../daemon/benchmarks/openjev_2b_scopejudge_calibration_v1.json)
reports preserve every prediction, exclusion, source hash, and GPU inference
time. This uses a different prompt and tokenization path than the unfinished
multi-task training attempt above, so the eligible-row counts are not expected
to match that attempt.

A bounded adaptation froze the entire released backbone and extracted its
2,048-dimensional final hidden state on GPU. The extracted raw scores matched
the served checkpoint probabilities within 3e-8 after the published
temperature was applied. A small, balanced logistic head fitted only the 482
eligible training rows. Its regularization and zero-false-allowance threshold
were selected from 59 eligible calibration rows before test predictions were
computed; 546 eligible test rows came from disjoint security trajectories.
The [head report](../daemon/benchmarks/openjev_2b_scope_head_v1.json) records
the saved local artifact hash, selected hyperparameter, and all test scores.
At 0.5 it attained 0.905 test AUC and 88.5% accuracy, **but made nine false
allowances**. At the calibration-chosen threshold it made **zero false
allowances**, but warned on **285/517 allowed actions** (44.9% allow recall,
0.724 balanced accuracy). Only 29 eligible test calls were disallowed, so
zero observed false allowances is a weak safety estimate. The public suite was
inspected during development; this is not an untouched desktop holdout. The
local head is research-only and is not connected to Heliox execution.

A cross-domain check on the already-inspected 60-case synthetic Jev comparison
suite is more decisive: the frozen head ranked answers with 0.911 AUC but its
ScopeJudge-calibrated threshold made **20 false allowances** and two false
warnings (38/60 agreement). At the default 0.5 threshold it allowed every
case, including all 30 disallowed actions. The
[transfer report](../daemon/benchmarks/openjev_2b_scope_head_transfer_v1.json)
is a counterexample to treating in-domain zero false allowances as a portable
safety property. Recalibrating on these already-inspected cases would be test
leakage, not evidence of a generalized solution.

The GPU extraction and head fit are reproducible with
[`extract_openjev_scope_features.py`](../daemon/scripts/extract_openjev_scope_features.py)
and [`fit_openjev_scope_head.py`](../daemon/scripts/fit_openjev_scope_head.py).
They verify pinned input provenance, keep large feature/weight artifacts under
ignored `.bench-state`, and do not train on or export proprietary Jev outputs.
The synthetic transfer is scored by
[`evaluate_openjev_scope_head_transfer.py`](../daemon/scripts/evaluate_openjev_scope_head_transfer.py)
using the already-frozen scaler, weights, and threshold.

### Cross-domain relevance check (research only)

The [Apache-2.0 TBAC task/tool dataset](https://github.com/outshift-open/tbac-research-datasets)
offers a second, independently curated semantic-matching domain: 1,136
validation and 935 official test pairs. We pin its repository revision and
file hashes in [`tbac_tool_relevance.py`](../daemon/benchmarks/tbac_tool_relevance.py).
Its label means **tool relevance, not permission to execute**. The official
test has disjoint task text and tool names from validation; repeated validation
tasks stay in one fit or calibration group.

The released Open-Jev 2B base and adapter were frozen while GPU features were
extracted with resumable 50-case checkpoints. Windows reported `nvlddmkm`
events alongside two CUDA illegal-instruction failures during validation;
resuming the same pinned job completed all 1,136 validation and 935 test rows.
No driver or production runtime was changed. The mixed logistic head was fit on
482 ScopeJudge train calls and 910 TBAC validation pairs. Its regularization and
zero-calibration-false-positive threshold were selected using only the two
calibration partitions, then frozen before either public test was scored.
[`fit_openjev_multidomain_head.py`](../daemon/scripts/fit_openjev_multidomain_head.py)
and the [frozen fit report](../daemon/benchmarks/openjev_2b_multidomain_head_v1.json)
record the hashes, partitions, threshold, and per-case scores.

| Frozen head at its calibration threshold | ScopeJudge test (546) | TBAC official test (935) | Inspected Heliox suite (60) |
| --- | --- | --- | --- |
| Scope-only head | 0 false positives, 285 false negatives | 401 false relevance positives, 16 negatives | 20 false allowances, 2 warnings |
| Mixed head | 0 false positives, 487 false negatives | 3 false relevance positives, 344 negatives | 1 false allowance, 15 warnings |

The mixed head's remaining Heliox false allowance would replace a source
document even though the user requested a translation **in chat only**. The
released Open-Jev checkpoint at its default threshold made **zero** false
allowances on this same inspected suite, albeit with 15 false warnings; the
mixed head is not an improvement on that safety measure. TBAC relevance and
scope are different targets, and mixing them trades away substantial legitimate
scope recall. The [synthetic transfer report](../daemon/benchmarks/openjev_2b_multidomain_transfer_v1.json)
is not a fresh holdout. This head is **not** connected to Heliox execution,
is not a production policy, and does not establish Jev parity.

A second, hypothesis-driven check kept the two targets separate: a TBAC-only
relevance head was calibrated on TBAC validation, then conjoined with the
already-frozen scope head. Its [report](../daemon/benchmarks/openjev_2b_separate_heads_v1.json)
shows zero false allowances on the inspected Heliox suite, but **22/30**
legitimate actions were rejected. On ScopeJudge test it rejected **all 517**
legitimate calls. The relevance-only head's official TBAC test had one false
positive and 407 false negatives. This gate cannot be shipped either: low false
allowance at the cost of disabling useful execution is not Jev-like behavior.

### Soup fit assessment

[Soup v0.75.0](https://github.com/MakazhanAlpamys/Soup/tree/f0d7696f1da1f86f965309b2d22e5fb48afb4fae)
is an Apache-2.0 fine-tuning toolkit with a resident sequence-classifier LoRA
path and a separate experimental low-VRAM layer-streaming path. Its
[streaming task gate](https://github.com/MakazhanAlpamys/Soup/blob/f0d7696f1da1f86f965309b2d22e5fb48afb4fae/src/soup_cli/config/schema.py#L5798-L5812)
permits SFT and several preference losses, **not classifier training**; its
[classifier trainer](https://github.com/MakazhanAlpamys/Soup/blob/f0d7696f1da1f86f965309b2d22e5fb48afb4fae/src/soup_cli/trainer/classifier.py)
loads a new sequence-classification model rather than Open-Jev's released
candidate-conditioned scalar checkpoint. Thus `soup train` cannot reproduce
this adaptation or solve the earlier CUDA launch failure by selecting
`stream_layers=true`. Soup's source-level Windows/VRAM measurement practices
are useful for future full-backbone experiments, but no Soup dependency was
added to the pilot and its training claims were not treated as Heliox results.
The project's [4 GB throughput remeasurement](https://github.com/MakazhanAlpamys/Soup/issues/361)
is still open after a correctness repair; its pre-repair speed figure should
not be projected onto this GPU or task.

### Independent instruction-review baseline (in progress)

[`evaluate_qwen_instruct_scope.py`](../daemon/scripts/evaluate_qwen_instruct_scope.py)
adds a pinned, zero-shot text-only Qwen3.5-2B baseline for the same synthetic
scope suite and the separately partitioned public ScopeJudge test. It uses a
fixed prompt, strict YES/NO parsing (malformed answers deny), no Jev outputs
for training, no action execution, and identity-checked resumable reports.
The text-only loader follows the official
[Transformers Qwen3.5 guidance](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)
and its [integration test](https://github.com/huggingface/transformers/blob/main/tests/models/qwen3_5/test_modeling_qwen3_5.py)
for `Qwen3_5ForCausalLM`. It is an instruction-following comparison, not a
replacement for a calibrated safety policy or a claim of Jev parity.

Only a three-case smoke run of the earlier full multimodal wrapper completed.
The 60-case run then hit a CUDA error before producing a report. The smaller
text-only path has passed its parsing/checkpoint unit tests but **has not
produced an inference report**: on this Windows host, importing PyTorch
stalled before CUDA allocation in both the research environment and the
system Python. No score is reported for it, and nothing was connected to
Heliox runtime. When the host can load PyTorch again, run the pinned local
checkpoint with `--checkpoint-every 1` and review the complete synthetic and
ScopeJudge reports before considering any further change.

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
