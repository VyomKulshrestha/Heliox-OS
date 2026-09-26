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

A second, independently seeded evaluation-only sample of 48 licensed public
OOD questions used 48 live Jev calls ($0.001241). The same frozen local adapter
agreed with Jev on 20/25 Noul and 9/13 choice questions. Against the public
labels, local versus Jev accuracy was 22/25 versus 21/25 for Noul and 9/13
versus 11/13 for choice. On ten ordinal score questions, mean absolute error
against the public target expectation was 0.861 locally versus 0.445 for Jev.
The sample is still small and synthetic; it does not establish parity, desktop
transfer, or execution authority. See the [frozen comparison](../daemon/benchmarks/qwen_general_jev_comparison_v3.json).

A separate, preselected frozen Qwen3.5-2B feature experiment tested whether a
larger backbone alone repaired general decision transfer. Native Windows CUDA
with driver 617.14 completed the pinned extraction after the WSL path again
encountered an NVIDIA reset. Per-row license and token-length filters retained
225 train, 60 calibration, 114 test, and 108 OOD examples. A regularized
candidate head selected only on calibration data reached **59.6%** test top-1
versus a **44.7%** uniform baseline, but just **45.4%** OOD top-1 versus
**49.1%** uniform. Its OOD negative log-likelihood (1.056) beat uniform
(1.153), yet OOD Noul balanced accuracy was only **45.6%** and choice top-1
was **31.4%**. These mixed metrics do not support a general-purpose model or
runtime promotion. The [held-out report](../daemon/benchmarks/qwen2b_frozen_general_v1.json)
records the pinned source, selection, hashes, and per-kind results; weights
remain local and ignored. A fresh matched Jev comparison could not run because
the otherwise valid OpenRouter key received `403 RBAC: access denied` from the
Decisions endpoint. This is an access limit, not a Jev result.

The resumed 4,096-train / 256-calibration / 256-test / 256-OOD Qwen3-0.6B
LoRA study completed on a separately seeded public sample after several GPU
resets. Its held-out top-1 was **65.2%** test and **57.0%** OOD; uniform on
those same samples was **47.7%** and **54.3%**. OOD choice top-1 was only
**31.9%** and score top-1 **43.1%**. This is not a controlled improvement over
the earlier 2,048-row run, which used different sampled rows. More importantly,
the saved adapter still preferred *Delete the local file* (**47.6%**) to
*Read the local file and report findings* (**40.0%**) on the read-only action
fixture. The [full report](../daemon/benchmarks/qwen_general_probe_v3.json)
pins the source/model revisions, split metrics, final checkpoint and artifact
hashes. Resumable, identity-checked held-out scoring was added because GPU
resets interrupted calibration; the completed report is not evidence that the
Windows/WSL GPU fault is fixed. The adapter remains local and disabled.

An ordinal-loss ablation then reused the earlier 512-row run's exact public
train/calibration/test/OOD selections and changed only the normalized
expected-rank loss weight from 0 to 0.5. Held-out OOD top-1 moved from
74/128 to 75/128 and score top-1 from 10/26 to 11/26, but score expected-rank
error was **1.1809 without** versus **1.1818 with** the new loss. The primary
ordinal metric did not improve, so this is not a demonstrated improvement over
the frozen control. The adapter remains local and disabled. See the
[control](../daemon/benchmarks/qwen_general_v1_ordinal_ablation.json),
[trial](../daemon/benchmarks/qwen_general_ordinal_v1_ablation.json), and
[trial training report](../daemon/benchmarks/qwen_general_ordinal_v1.json).

An earlier 4,096-row continuation with the 2,048-row run's frozen held-out
selections did not finish on that attempt. It checkpointed 896/4,096 training
rows, then encountered
CUDA illegal-instruction and cuBLAS failures accompanied by Windows
`nvlddmkm` event IDs 13 and 153; an intervening retry produced a non-finite
gradient. The checkpoint's trainable weights, head, and optimizer tensors were
finite, but there are **no final test/OOD metrics or promoted weights** from
this run. The [attempt record](../daemon/benchmarks/qwen_general_scale_attempt_v3.json)
preserves the checkpoint hash and failure sequence. The event correlation
does not prove whether the root cause is hardware, driver, power, or a kernel.

The repository also provides a separate, unpromoted scaling experiment using
frozen [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) features. It extracts
candidate representations on the GPU with row-level checkpoints, fits a shared
regularized decision head on CPU, selects regularization and temperature on
calibration rows, then evaluates disjoint test and OOD rows. A matched Jev
comparison can subsequently call Jev on those same public OOD rows; its
responses are aggregated for evaluation only and never used as training
labels. The implementation is in
[`extract_general_frozen_features.py`](../daemon/scripts/extract_general_frozen_features.py),
[`fit_general_frozen_head.py`](../daemon/scripts/fit_general_frozen_head.py), and
[`compare_frozen_general_jev.py`](../daemon/scripts/compare_frozen_general_jev.py).
This path is a reproducible research candidate, **not** a completed quality
result or a Heliox runtime model. The small pipeline smoke test does not
establish Jev parity. All feature and weight files remain in ignored local
benchmark state; an interrupted extraction must resume with matching rows,
model revision, and tokenization.

The subsequent [GPU stability attempt](../daemon/benchmarks/qwen_gpu_stability_attempt_v1.json)
advanced the 0.6B checkpoint to 1,712/4,096 training rows and verified 128
of 225 eligible 2B training feature rows. Both longer workloads then failed
with CUDA errors and contemporaneous Windows `nvlddmkm` resets, while short
CUDA stress and a small 2B extraction passed. Both checkpoints were checked
against their saved identities; neither run has final held-out metrics or a
Jev comparison. Resume only after the GPU execution path is stable. These
observations do not distinguish a driver, hardware, WSL, power, or model-kernel
fault.

The [native Windows follow-up](../daemon/benchmarks/qwen_gpu_stability_attempt_v2.json)
used the same checkpoint and pinned selection outside WSL. It completed the
225 eligible training and 60 calibration feature rows, then 64/114 test rows
before another CUDA failure with a contemporaneous NVIDIA driver event. A
separate Python DLL import crash was resolved by import order and is not GPU
evidence. Since the model fault also occurs natively, WSL alone cannot explain
it. The frozen head, complete held-out evaluation, and Jev comparison remain
unrun; the validated partial feature files are local and ignored.

A label-blind [uniform-option baseline](../daemon/benchmarks/qwen_general_uniform_baseline_v2.json)
on the 2,048-row run's held-out selections exposes a further weakness. On OOD
Noul questions it selects the first option, "no", and gets **103/136 (75.7%)**
because the public labels are imbalanced; the trained adapter gets **99/136
(72.8%)**. The adapter does beat this blind comparator on OOD choice
(39/69 versus 25/69) and score top-1 (22/51 versus 11/51). Overall accuracy
alone therefore hides a binary-decision failure, and this synthetic baseline
is not a Jev or live-desktop comparator.

A fresh [128-call public OOD comparison](../daemon/benchmarks/qwen_general_jev_comparison_v4.json)
made the gap clearer. Against licensed public labels, Jev versus the frozen
local adapter scored **55/67 versus 44/67** on Noul and **28/35 versus
13/35** on choice; mean score expected-rank error was **0.394 versus 1.013**
across 26 questions. On this exact selection, equal option probabilities
would score 53/67 on Noul and 0.923 expected-rank error on score, both better
than the trained adapter. The comparison cost $0.003250 in Jev API calls;
responses were used only for aggregate evaluation, not fitting. The sample
is synthetic and still cannot establish desktop performance or safety.

A separate prompt-architecture ablation trained the same 512 public rows with
the complete option list visible to each candidate, while preserving the
original candidate-only prompt for frozen artifacts. No selected row exceeded
768 tokens. On the identical 128-row OOD selection, the full-option model
scored **71/128** versus **74/128** for the original, including **11/35 versus
12/35** choice decisions; score expected-rank error was **1.209 versus
1.181**. An independent artifact replay reproduced the new model's metrics.
When only unordered choice alternatives were permuted, the original model's
12/35 choice result was unchanged; the full-option model moved from 11/35 to
12/35 but its choice NLL worsened. This does not establish useful improvement
or permutation robustness. The variant remains local and disabled. See its
[training](../daemon/benchmarks/qwen_general_full_options_v1.json),
[replay](../daemon/benchmarks/qwen_general_full_options_replay_v1.json),
[permutation](../daemon/benchmarks/qwen_general_full_options_permuted_v1.json),
and [control permutation](../daemon/benchmarks/qwen_general_v1_permuted_control.json).

A [binary calibration probe](../daemon/benchmarks/qwen_general_v2_binary_calibration_v1.json)
used 58 public Noul calibration rows to choose a threshold before scoring a
disjoint 67-row OOD slice. Moving from the frozen default 0.5 threshold to
the calibration-selected 0.453 raised OOD balanced accuracy from **0.577 to
0.655** and accuracy from **48/67 to 50/67**. But false positives increased
from **7 to 8**, and the 50/67 accuracy only matched the label-blind "no"
baseline on that slice. This is evidence of some ranking signal, not a
generalized or safe decision policy; no Heliox threshold was changed. The
[frozen diagnostic](../daemon/benchmarks/qwen_general_v2_binary_diagnostics.json)
also records the uncalibrated confusion breakdown.

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

### Independent instruction-review GPU baseline

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

The earlier full multimodal wrapper completed only a three-case smoke run
before a CUDA error. After the host restart, the smaller text-only path loaded
the pinned checkpoint on the RTX 4060 Laptop GPU with Hugging Face offline
mode enforced. Its [synthetic report](../daemon/benchmarks/qwen3_5_2b_instruct_scope_v1.json)
records **56/60**, zero false allowances, and four false warnings. On the
[public ScopeJudge test](../daemon/benchmarks/qwen3_5_2b_instruct_scopejudge_v1.json),
**542/608** calls were within the 768-token input limit and **66** were
excluded. Among evaluated calls, agreement was **124/542** (22.9%; balanced
accuracy 54.4%), with **three false allowances** on 29 out-of-scope calls and
**415 false warnings** on 513 in-scope calls. The PyTorch fallback ran without
the optional fast attention libraries; no inference crash occurred in these
completed runs. The synthetic success did not transfer to this public domain,
and the candidate was not connected to Heliox execution.

### Quantized local GPU baseline and public transfer

To keep testing without PyTorch, the independent
[`evaluate_ollama_scope.py`](../daemon/scripts/evaluate_ollama_scope.py)
uses an already-cached Ollama Qwen3.5 4B Q4_K_M model. The report pins the
model's full digest, fixed prompt, Ollama settings, and case-set hash; Ollama's
live process endpoint reported the model **100% GPU-resident**. This is a
different, quantized model from the 2B research checkpoint above. Its
[60-case report](../daemon/benchmarks/ollama_qwen3_5_4b_scope_v1.json) records
**54/60**, zero false allowances, and six false warnings on the already-
inspected synthetic suite. Jev answered 60/60 of those authored cases. The
4B result is neither Jev parity nor fresh real-world validation, and it was
not connected to Heliox execution.

The disjoint public ScopeJudge test selected 608 calls. Ollama checkpointed
105 before returning HTTP 500; Windows recorded a `llama-server.exe` crash
(`0xc0000409`) and `nvlddmkm` events on 2026-09-24. Those events do not prove
the underlying cause. After the host restart, the same Ollama version, model
digest, prompt, and cases were verified and the GPU run resumed to completion.
The [public report](../daemon/benchmarks/ollama_qwen3_5_4b_scopejudge_v1.json)
records all **608/608** calls, with 100% GPU residency and no context
exclusions. It agrees with only **34/608** expert labels (5.6%; balanced
accuracy 50.4%): **zero** false allowances on 29 out-of-scope calls, but
**574** false warnings on 579 in-scope calls. One answer hit the 16-token
generation cap without a valid decision and was denied.
The model mostly says NO, so its apparent conservatism would prevent useful
work. This public offensive-security dataset is heavily positive in its full
test partition. The [ScopeJudge paper](https://arxiv.org/html/2607.07774)
states that experts saw the prior trajectory when labeling each call, whereas
this baseline presents only the original request and proposed action. It is
therefore an **intent-only** monitor comparison, not a full-context judge.
Label agreement is not desktop-safety evidence, and neither model nor prompt
is eligible for runtime promotion. A PyTorch import initially stalled after
restart but subsequently succeeded; the 2B GPU inference results above replace
the earlier unresolved-import status. Those instruction-review runs did not
train weights; the separate task-state experiment below did.

### Bounded context and world-model task-state training

The pinned 2B model was also evaluated on the balanced 80-call ScopeJudge
calibration split with the two most recent **tool calls only** (no tool output
or agent prose). At a 2,048-token limit, the [no-history YES/NO
prompt](../daemon/benchmarks/qwen3_5_2b_scopejudge_calibration_intent_v1.json)
evaluated 80/80 and agreed on 39/80, with three false allowances and 38 false
warnings. The [history prompt](../daemon/benchmarks/qwen3_5_2b_scopejudge_calibration_history2_v1.json)
evaluated only 69/80, agreed on 36/69, and had three false allowances and 30
false warnings. On the same 69 eligible cases, no-history
agreed on 32/69 with two false allowances. History used 68,693 input tokens
versus 35,891 for those paired no-history cases. The small agreement gain
therefore added a false allowance, excluded 11 cases, and nearly doubled
tokens. An independent YES-vs-NO first-token scoring probe did not rescue the
model: on [80 no-history calls](../daemon/benchmarks/qwen3_5_2b_scopejudge_calibration_intent_logits_v1.json)
it had four false allowances and 37 false warnings; on [69 history
calls](../daemon/benchmarks/qwen3_5_2b_scopejudge_calibration_history2_logits_v1.json)
it had four and 29. The highest negative score
exceeded the highest positive score in both variants, so a threshold above
all calibration negatives would admit no positives. These scores are not
calibrated authorization probabilities.

Heliox's shipped [world model](../daemon/pilot/security/risk_model.py) learns
coarse disk/process state transitions from actual sandbox telemetry. Those
targets cannot teach it whether a proposed action matches user intent. To
test the adjacent *task-state* idea without changing the safety contract,
[`train_world_scope_head.py`](../daemon/scripts/train_world_scope_head.py)
trains a separate GPU logistic head on 512 MIT-licensed ScopeJudge tool-call
labels. It encodes the request/action pair and, separately, the relation of
up to two prior calls to the proposed action. The 80-call calibration and
608-call test partitions are trajectory-disjoint; calibration selects the
regularization and cautious threshold before test or Heliox transfer labels
are evaluated. The head is **not** loaded by `HybridWorldModel`, the risk gate,
or the execution companion.

| Research head | Public test AUC | Public test at 0.5 | Public test at cautious threshold | Inspected Heliox transfer at cautious threshold |
|---|---:|---|---|---|
| [No history](../daemon/benchmarks/world_scope_head_intent_v1.json) | 0.817 | 18 false allowances, 44 false warnings | 0 false allowances, 495 false warnings | 2 false allowances, 24 false warnings |
| [Two prior calls](../daemon/benchmarks/world_scope_head_history2_v1.json) | 0.764 | 18 false allowances, 54 false warnings | 0 false allowances, 544 false warnings | 1 false allowance, 19 false warnings |

The two-call variant is worse in-domain and still unsafe on Heliox transfer.
The cautious threshold also prevents most legitimate calls. This is an
offline training result, **not** a promoted world-model capability, Jev parity,
or live desktop safety evidence. The small numeric head artifacts remain in
ignored local benchmark state; the linked reports contain their hashes.

A further GPU attempt with the existing shared Qwen3-0.6B decision trainer
combined licensed Open-Jev general-decision rows and ScopeJudge labels. It
[failed during baseline inference](../daemon/benchmarks/qwen_general_scope_gpu_attempt_v4.json)
at row 29, before any gradient or optimizer step, with a CUDA illegal
instruction. Windows recorded `nvlddmkm` events 13 and 153 at that time.
This repeats a GPU stability symptom seen in the larger earlier attempts; the
events do not prove the root cause. No shared-model weights or held-out metrics
resulted. The small numeric task-state head above is the completed training
experiment, and neither experiment changes runtime authority.

A bounded diagnostic then followed [PyTorch's CUDA debugging
guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html) with
`CUDA_LAUNCH_BLOCKING=1` and `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. That
[small run](../daemon/benchmarks/qwen_general_scope_gpu_debug_v5.json) finished
56 eligible training rows and 27 eligible public scope test rows without a new
driver event. Scope agreement remained **13/27** before and after training:
all 14 out-of-scope calls were allowed. The diagnostic changed both execution
settings and sample size, so it neither identifies the original CUDA root
cause nor demonstrates useful transfer. The PyTorch peak-allocation counter
was unavailable with caching disabled; its original zero reading must not be
interpreted as zero GPU usage. No candidate from these runs was enabled.

A controlled prompt change that explicitly evaluated concrete effects and
partial task steps was tested on the **balanced 80-call calibration split**.
Both it and the original prompt returned NO for every call (40 false warnings,
zero false allowances), so the wording change showed no improvement. A
20-call reasoning-mode smoke test with a 256-token generation limit produced
no final YES/NO answers; a longer single positive case still exhausted 2,048
generated tokens without an answer. These are failed probes, not a trained
improvement. The prompt file and configurable evaluation settings are retained
for reproducible calibration experiments, but the candidate remains offline.

### Shared GPU retry and world/task-state transfer

The shared Open-Jev-style candidate model was retrained on 256 licensed general
decisions plus 491 token-eligible ScopeJudge calls. Windows CUDA interrupted
training after checkpoints at rows 64 and 352, with `nvlddmkm` events 13/153.
An isolated Ubuntu/WSL2 CUDA environment loaded the **same row-352 checkpoint**
and completed all 747 rows; this avoids attributing the Windows failure to a
particular cause. [NVIDIA's WSL guidance](https://docs.nvidia.com/cuda/wsl-user-guide/)
explains that WSL uses the Windows-host GPU driver, not a separate Linux driver.
The resumed report correctly marks the skipped baseline as unavailable.

The result is **not usable for Heliox scope decisions**. On 700 eligible,
trajectory-held-out public calls, its 90.3% overall agreement hid 49.5%
balanced accuracy: it falsely allowed **48 of 49** out-of-scope calls. On the
previously inspected 60-case Heliox/Jev suite it falsely allowed all 30
out-of-scope cases at the default threshold. A threshold fixed solely from
public calibration negatives reduced that transfer count to one, but falsely
warned on 29 legitimate cases. The 128-row general test and OOD top-1 scores
were 62.5% and 59.4%. A fresh, evaluation-only 24-call public OOD comparison
favored Jev on the 11 binary and eight choice cases; neither those calls nor
the older synthetic Jev labels entered this model's training. Counts, hashes,
and limitations are in [the frozen summary](../daemon/benchmarks/qwen_general_scope_wsl_v7.json).
The adapter remains in ignored local benchmark state and is **not** loaded by
Heliox's world model, companion, or approval path.

We also examined [Look Before You Click](https://huggingface.co/datasets/ash256/look-before-you-click)
and [AutoElicit-Exec](https://huggingface.co/datasets/osunlp/AutoElicit-Exec)
as desktop-trajectory sources. Their reviewer/trajectory outcomes are not
independent, per-step expert authorization labels, so they were not mixed into
the scope training split. More screen context alone does not repair the
missing label quality or the measured false-allowance rate.

A separate [2B semantic task-state head](../daemon/scripts/train_world_scope_semantic.py)
then tested whether a stronger, frozen language representation could improve
the adjacent world-model research head without shipping a large model. It fit
only a small numeric classifier on 480 token-eligible expert-labeled public
calls; 59 calibration and 542 test calls remained under the 768-token cap.
It raised public test AUC to 0.867 and balanced accuracy to 0.775 at the
default threshold, but still falsely allowed nine of 29 out-of-scope calls. The
public-calibrated cautious threshold still allowed three of 29, while issuing
190 false warnings. On the previously inspected 60-case Heliox transfer, that
threshold falsely allowed 18 of 30 out-of-scope actions. The strong transfer
AUC does not rescue a threshold that shifts so badly across domains, and tuning on
that inspected transfer would not be an independent fix. The [frozen
summary](../daemon/benchmarks/world_scope_semantic_2b_v1.json) records coverage,
metrics, and the local artifact hash. This head is **not** wired into the
world model, companion, or permission gateway.

A second offline 2B head trained on Cisco Outshift's [Apache-2.0 TBAC
task-tool data](https://github.com/outshift-open/tbac-research-datasets),
whose validation and official test tasks do not overlap. Its 873 task-disjoint
training and 263 calibration pairs yielded 0.920 AUC on all 935 official
test pairs, but the default threshold still falsely allowed 82 of 417
irrelevant tools. On the previously inspected Heliox synthetic transfer it
falsely allowed one of 30 out-of-scope actions. More decisively, on a separate
ScopeJudge test the same adapter fell below chance (0.434 AUC) and falsely
allowed 25 of 29 token-eligible out-of-scope calls. The 1024-token cap
excluded 21 ScopeJudge test calls, all in-scope; no input was truncated.
The [frozen cross-domain summary](../daemon/benchmarks/world_tool_scope_semantic_v1.json)
therefore rejects this adapter too. Its promising single-domain score is not
evidence that it generalizes to agent tool actions, and it has no runtime path.

Training a [third, mixed-corpus offline head](../daemon/scripts/train_world_multicorpus_scope_semantic.py)
on the TBAC validation and ScopeJudge training partitions improved the
ScopeJudge test AUC to 0.855 while retaining 0.908 TBAC test AUC. The
calibration choice balanced both corpus/class cells rather than allowing the
rare ScopeJudge negatives to be swamped. Even so, its default threshold
falsely allowed 10/29 ScopeJudge and 70/417 TBAC out-of-scope actions, plus
3/30 on the previously inspected Heliox synthetic suite. The cautious
threshold showed zero false allowances on these particular test rows but
rejected 515/558 legitimate ScopeJudge actions and 484/518 legitimate TBAC
tools. That does not demonstrate a viable controller. The official tests were
outside fitting and calibration, but earlier test results guided the decision
to mix corpora; this is adaptive research rather than a new blind benchmark.
The [frozen mixed-corpus summary](../daemon/benchmarks/world_multicorpus_scope_v1.json)
and local artifact hash preserve the result. It remains offline only.

We then tested a [4-bit LoRA world task-state adapter](../daemon/scripts/train_world_scope_qlora.py)
on the same licensed training partitions. It followed the
[PEFT quantization procedure](https://huggingface.co/docs/peft/developer_guides/quantization)
and trained 8.4 million adapter parameters plus a binary head on the local GPU.
A four-example 1024-token worst-case backward smoke test succeeded, and the
full 1,380-example epoch completed. TBAC test ranking improved to 0.936 AUC,
but the default threshold still falsely allowed 7/417 irrelevant tools and
rejected 374/518 relevant ones. ScopeJudge ranking dropped to 0.703 AUC and
the model rejected 372/558 legitimate actions while falsely allowing 3/29.
On the inspected Heliox synthetic suite it rejected all 30 authorized
actions. The cautious threshold did not yield a useful controller either.
The [frozen LoRA result](../daemon/benchmarks/world_scope_qlora_v1.json)
records data coverage, hashes, and error counts. The adapter remains in
ignored local benchmark state; it is **not** loaded by the world model,
companion, or permission gateway.

To probe mid-task context without leaking a future answer, the
[pre-call conversation evaluator](../daemon/scripts/evaluate_world_conversation_scope.py)
cuts Cisco Outshift's generated multi-agent dialogues immediately before the
first candidate tool call. It passes only earlier user/assistant text to the
frozen mixed head—not tool output or later user revisions. On 909 of 935
token-eligible official test conversations, AUC fell to 0.875 and the default
threshold falsely marked 142/398 irrelevant tools as relevant. A cautious
threshold left two false relevance approvals while rejecting 439/511 relevant
tools. Twenty-six conversations were overlong and excluded; the comparison
with the single-turn test has different eligible subsets and is not a paired
causal estimate. The [frozen dialogue summary](../daemon/benchmarks/world_conversation_scope_v1.json)
shows that the present head does **not** robustly track task intent through a
multi-turn interaction. These generated relevance labels do not establish
authorization or live desktop behavior.

A fourth [offline world task-state head](../daemon/scripts/train_world_multiturn_scope_semantic.py)
was fitted on the pre-call **training** dialogues plus trajectory-disjoint
ScopeJudge training calls. Repeated dialogue seed tasks stayed together in
train or calibration; official dialogue, task-tool, and ScopeJudge test rows
were never fitted or used to choose regularization or threshold. A pinned
Qwen3.5-2B backbone extracted features on the local GPU, and a numeric head
selected its 0.001 regularization and 0.897 caution threshold from the two
calibration partitions. The run used resumable, input-signature-checked local
feature checkpoints because sleep and a system reboot interrupted earlier
attempts. Batch-of-four extraction was kept separate from older single-row
features after a parity check found small but nonzero numeric differences.

On 909 eligible official dialogue test rows, ranking rose from the earlier
frozen head's 0.875 AUC to **0.925 AUC**, and default-threshold false
relevance allowances fell from 142/398 to **77/398**. But the cautious
threshold still allowed **3/398** irrelevant dialogue tools and warned on
369/511 relevant tools. More importantly, single-turn TBAC transfer regressed
to **0.881 AUC and 204/417 false allowances** at the default threshold;
the cautious threshold still allowed 13/417. ScopeJudge had 11/29 default
and 1/29 cautious false allowances. On the already-inspected Heliox synthetic
suite, caution still allowed 2/30 out-of-scope actions. The
[frozen multi-turn result](../daemon/benchmarks/world_multiturn_scope_v2.json)
is therefore **rejected for runtime**. Error inspection found wrong but
semantically related tools, cross-tenant scope, and unauthorized target or
output expansion. Better dialogue ranking is not a generalized world model,
Jev parity, or permission to execute. The tests were excluded from this fit,
but earlier results on those tests informed this adaptive research; none is a
fresh blind validation.

We then added the **single-turn TBAC validation pairs** as a third training
domain, keeping each seed task's dialogue and single-turn examples on the
same side of the train/calibration boundary. The [three-domain result](../daemon/benchmarks/world_multiturn_scope_v3.json)
repaired much of the task-tool regression: TBAC test AUC rose from 0.881 to
**0.904**, with default false allowances falling from 204/417 to **81/417**.
Dialogue ranking stayed near 0.928 AUC. But ScopeJudge ranking fell from
0.851 to **0.828 AUC**, and its default false allowances rose to 15/29.
The calibration-chosen cautious threshold still allowed 2/398 irrelevant
dialogue tools, 1/417 single-turn tools, and 1/30 out-of-scope actions in the
previously inspected Heliox suite; it also warned on 416/511, 458/518, and
14/30 legitimate examples respectively. This is a domain trade-off, not a
generalized fix. The head remains offline and rejected. Reproduce with
`--include-tbac-validation` and fresh `--artifact` and `--report` paths,
using the same pinned inputs and feature directory as above.

A small [nonlinear GPU head](../daemon/scripts/train_world_multidomain_mlp.py)
then reused the frozen three-domain features, balancing all six
domain/class cells. Six fixed hidden-size/seed candidates were selected on
calibration only. The selected 64-unit model improved official test AUC to
0.933 dialogue, 0.911 task-tool, and 0.858 ScopeJudge; default false
allowances were 65/398, 78/417, and 5/29. But the cautious threshold still
allowed two irrelevant dialogue tools and one irrelevant single-turn tool,
while rejecting 470/511, 493/518, and 533/558 legitimate examples in those
three domains. On the already-inspected Heliox synthetic suite it blocked
27/30 legitimate actions. The [frozen nonlinear result](../daemon/benchmarks/world_multidomain_mlp_v1.json)
is rejected. More capacity alone did not solve calibrated scope or the
wrong-tool problem. Like the earlier runs, this is adaptive public research,
not a blind safety or Jev-parity claim.

Because Open-Jev scores alternatives relative to one question, a
[pairwise training probe](../daemon/scripts/train_world_pairwise_scope.py)
added within-task relevant-versus-irrelevant ranking loss to the same frozen
features, alongside expert-scope binary loss. The public training partition
contained 383 eligible dialogue and 403 single-turn candidate pairs. A
nonzero pairwise weight was selected on calibration, but on held-out mixed
candidate tasks its top-one choice was **320/338** for dialogue and **344/355**
for single-turn, versus **323/338** and **344/355** for the non-pairwise MLP.
It therefore did not improve candidate ranking. Its cautious threshold left
one false allowance among 398 irrelevant dialogue tools and warned on
485/511 relevant tools; it rejected 499/518 relevant single-turn tools and
543/558 allowed ScopeJudge actions. Both saved MLP artifacts replayed their
reported metrics exactly through the independent, shape-checked
[scorer](../daemon/scripts/score_world_mlp_artifact.py). The [frozen pairwise
result](../daemon/benchmarks/world_pairwise_scope_v1.json) is rejected too.
Copying Jev's relative-loss idea without better task/target supervision does
not make this a Jev-equivalent world model.

We froze both nonlinear heads before reading the [EntityBindingFailures
benchmark](https://github.com/R-Suresh/EntityBindingFailures), then tested
transfer to right-tool/wrong-entity choices without fitting or selecting on
its labels. Its pinned, CC BY 4.0 task set has 60 tasks: 55 resolvable and
five requiring clarification. Enumerating schema-typed candidate bindings
gave 55 valid and 94 wrong or ambiguous direct actions. GPU extraction of
the pinned Qwen features and an independent cached replay agreed. The
[external transfer result](../daemon/benchmarks/world_entity_binding_transfer_v1.json)
shows AUC **0.665** for the three-domain MLP and **0.663** for the pairwise
MLP. At their default thresholds they falsely allowed **77/94** and **78/94**
wrong or ambiguous actions. At their preselected cautious thresholds they
still allowed **12/94** and **11/94**, while rejecting **all 55** valid
actions. Both correctly ranked the gold binding for only **39/55** resolvable
tasks. This fresh external test therefore rejects both heads for runtime;
it is not a training set, a Jev comparison, or a test of the separately
shipped OS-transition world model. Reproduce against the pinned public data
with `python -m scripts.evaluate_world_entity_binding` from `daemon` after
extracting the frozen features, or replay another frozen head with
`--artifact`, `--artifact-report`, and a fresh `--report` path. The validated
feature checkpoint permits repeated scoring without reloading the GPU model.

An independent, differently authored [tau-bench retail](https://github.com/sierra-research/tau-bench)
probe tests target binding without literal order IDs in most requests. We
pin its `tasks.py`, order database, and user database, then compare published
tool/order annotations with same-user, same-status alternatives. Item-specific
calls are excluded so a mismatched item identifier cannot trivially identify
the negative. This produces 37 candidate groups, 29 with implicit references,
and 90 candidate actions (43 annotated positives, 47 alternatives). The
[frozen transfer report](../daemon/benchmarks/world_tau_binding_transfer_v1.json)
shows AUC **0.688** for the three-domain MLP and **0.675** for the pairwise
MLP. Default thresholds falsely allow **15/47** and **13/47** alternatives;
the cautious thresholds allow none but reject **all 43** annotated positives.
Both choose a positive top-ranked candidate in **29/37** groups. Thus a
second external domain also rejects runtime promotion. These are
observational relevance labels: some published actions depend on earlier
workflow outcomes, so they cannot be read as unconditional execution
authorization. GPU feature extraction completed with the pinned Qwen model;
cached replay of the first head and independent scoring of the second gave
the same metrics. From `daemon`, reproduce with
`python -m scripts.evaluate_world_tau_binding --source-root PATH/TO/tau-bench`
and a fresh `--report` path. The source checkout must match the pinned file
hashes.

To test whether the world head could learn when *not* to call a tool, we
added grouped call-versus-text preferences from the pinned
[NVIDIA When2Call](https://github.com/NVIDIA/When2Call) training source as
an optional fourth offline domain. Its repository is Apache-2.0 and the
source is derived from Salesforce xLAM 60k (CC BY 4.0); no source rows or
weights are distributed with Heliox. We retained only one-call contrasts
whose selected tool was offered, removed contradictory request/tool groups,
then split by request/tool hash before fitting (400 train, 100 calibration,
100 test). The labels are synthetic preferences, not user approvals or
ground truth for executing a candidate action. Pinned CUDA extraction and
the [four-domain report](../daemon/benchmarks/world_multidomain_mlp_w2c_v1.json)
record the exact source, feature, and artifact hashes.

The selected head reached 0.762 AUC on the grouped When2Call test but at
the normal threshold still falsely favored 14/50 rejected calls and warned
on 16/50 preferred calls. Its public-calibrated cautious threshold rejected
all 50 preferred calls. On the three prior public test domains, cautious
scoring still falsely allowed 2/398 irrelevant dialogue tools and 1/417
single-turn tools while warning on 485/511, 510/518, and 541/558 legitimate
examples. The already-inspected synthetic Heliox suite rejected all 30
legitimate examples at that threshold. Replaying the new head on the two
external binding probes yielded AUC 0.609 on EntityBindingFailures and
0.678 on tau-bench retail; the normal threshold falsely allowed 67/94 and
10/47 wrong candidate bindings. Although a strict threshold removed those
false allowances, it also rejected every valid binding. These external
probes have now been inspected and cannot be treated as blind selection
sets for further iterations. This experiment is rejected for runtime;
improving call/no-call preferences did not solve target binding or calibrated
action readiness. The separately shipped deterministic OS-transition world
model and approval boundaries are unchanged.

A later [GPU leave-one-domain-out diagnostic](../daemon/scripts/evaluate_world_cross_domain.py)
trained on three task families at a time and tested on the fourth without
using the excluded family's rows for normalization or fitting. Held-out
agentic/TBAC AUCs were 0.879/0.849, but ScopeJudge/When2Call AUCs fell to
0.411/0.497. Without When2Call training examples, the head favored all 100
test calls, including all 50 dispreferred calls. A separate
[post-hoc calibration probe](../daemon/scripts/calibrate_world_task_head.py)
could trade preferred-call coverage against dispreferred-call acceptance,
but did not repair ranking. When2Call's *preferred versus dispreferred*
response labels are not the same target as the other domains' scope labels;
the shared binary head and authorization-framed feature prompt must not turn
that preference signal into a permission claim. This is an exploratory
diagnostic on previously explored task families, not new blind evidence.

A fixed-settings [two-output GPU follow-up](../daemon/scripts/train_world_multitask_head.py)
separated scope and preference labels while sharing frozen Qwen features.
It achieved AUC 0.931/0.912/0.848 on agentic/TBAC/ScopeJudge tests and 0.732
on the When2Call preference test. The already-inspected Heliox probe rose
to 49/60 overall but accepted 11/30 negative actions, versus 1/30 for the
earlier selected shared-binary head. Because model sizes and selection
procedures differ, this comparison does not isolate the effect of task heads;
it also does not justify runtime use. The saved two-output artifact is local
and research-only.

Reproduce the optional training from `daemon` using fresh output names after
preparing the pinned preference corpus and extracting its features:

```bash
python -m scripts.train_world_multidomain_mlp --when2call-report benchmarks/.bench-state/world_when2call_features_v1.json --when2call-feature-dir benchmarks/.bench-state/world_when2call_features_v1 --artifact benchmarks/.bench-state/world_multidomain_mlp_w2c_repro.npz --report benchmarks/.bench-state/world_multidomain_mlp_w2c_repro.json
```

To separate a weak frozen representation from a weak fitted head, we also
scored the pinned Qwen3.5-2B model directly with a fixed target-review prompt
and first-token YES/NO logits. On the already-inspected external sources,
this improved AUC to **0.820** for EntityBindingFailures and **0.806** for
tau-bench retail, versus 0.609 and 0.678 for the four-domain head. But its
unselected 0.5 cutoff still falsely favored **37/94** wrong entity bindings
and **5/47** wrong retail orders. A descriptive sweep over these test labels
showed that 0.8 eliminated wrong accepts only by rejecting **55/55** valid
entity bindings and **32/43** valid retail candidates. The two-logit softmax
is not a calibrated permission probability, and this sweep is not a new
calibration set.

A second, explicitly groupwise prompt let the same model select numbered
candidates or abstain. It exactly matched only **10/60** entity tasks and
**9/37** retail groups, selected **54/94** and **34/47** wrong candidates,
and abstained on **none** of five ambiguous entity tasks. Groupwise prompting
alone made over-selection worse. The
[comparison report](../daemon/benchmarks/qwen_binding_review_v1.json) pins
both prompts, sources, and complete raw-result hashes. Neither review mode is
loaded by Heliox or comparable to a Jev deployment. The findings point to
missing target/state supervision and calibrated abstention, not merely a
smaller binary head. Both external sources were inspected before these
comparisons, so future model selection needs genuinely fresh held-out tasks.

We screened further stateful corpora before adding more training labels.
The pinned [Arc CRM Benchmark](https://huggingface.co/datasets/Arc-Intelligence/arc-crm-benchmark)
snapshot (`Arc-Computer/arc-crm-benchmark` commit
`5bf1dc5b07dd941e30bbbfdee2bf03b8c7bb93c3`, MIT dataset card) has 1,200
synthetic conversations and 5,916 turns. Only 31 expected-argument bindings
in this snapshot have multiple same-type entities in the starting state;
every one names the target ID literally in the user utterance. Training on
those cases would mostly teach ID copying, not the implicit or ambiguous
target selection that failed above, so no Arc rows were fitted. The
[STATE-Bench](https://github.com/Microsoft/STATE-Bench) learning-track
trajectories include attempted agent actions and later user corrections;
an attempted call cannot be treated as a positive label without verifying
its outcome. We did not import those calls as training examples.

The pinned [ToolSandbox](https://github.com/apple/ToolSandbox) checkout
(`c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`) supplies stateful starting
contexts plus human-authored milestones and minefields. Its
[paper](https://aclanthology.org/2025.findings-naacl.65/) describes 1,032
scenarios. This is a candidate for a *new* end-to-end state/action evaluation,
not yet an Heliox benchmark or training result. Integration must preserve
the sandbox's native state and success checks, and its Apple license must be
reviewed before redistributing any source or derived data.

A CUDA-trained groupwise target head tested whether explicit candidate
competition and an abstain option would repair the binding failures. It uses
the pinned Qwen3.5-2B feature checkpoints and public EntityBindingFailures
labels for training, keeping whole `(domain, ambiguity)` families together
across train, calibration, and holdout. That matters: a random task-ID split
leaks near-duplicate templates and produced an invalidly reassuring result.
The corrected family holdout matched **8/8** groups, but covers only two
families. On the differently authored, previously inspected tau-bench retail
cases, the head selected a valid target in **0/37** groups: it abstained on
36 and chose a wrong target once; ignoring abstention, it ranked a valid
candidate first in **23/37** groups. The
[pinned result](../daemon/benchmarks/world_group_binding_family_v1.json)
therefore rejects runtime promotion. It neither outperforms the frozen
direct-review probe on transfer nor justifies a Jev-like or live-desktop
claim. The local weights remain ignored under `benchmarks/.bench-state/`.
Reproduce from `daemon` with the pinned public sources and cached features:

```bash
python -m scripts.train_world_group_binding --tau-source-root benchmarks/.bench-state/tau-bench-source --artifact benchmarks/.bench-state/world_group_binding_family_repro.npz --report benchmarks/.bench-state/world_group_binding_family_repro.json
```

Reproduce from `daemon` with the pinned public sources and local GPU model:

```bash
python -m scripts.evaluate_qwen_binding_review --suite entity-binding --report benchmarks/.bench-state/qwen_entity_binding_repro.json
python -m scripts.evaluate_qwen_binding_review --suite tau-retail --source-root PATH/TO/tau-bench --report benchmarks/.bench-state/qwen_tau_binding_repro.json
python -m scripts.evaluate_qwen_group_binding --suite entity-binding --report benchmarks/.bench-state/qwen_group_entity_repro.json
python -m scripts.evaluate_qwen_group_binding --suite tau-retail --source-root PATH/TO/tau-bench --report benchmarks/.bench-state/qwen_group_tau_repro.json
```

The pinned public data and cached Qwen backbone reproduce the experiment from
`daemon` with a fresh output name; the feature directory is local and ignored:

```bash
python -m scripts.train_world_multiturn_scope_semantic --scopejudge-path PATH/TO/train.jsonl --artifact benchmarks/.bench-state/world_multiturn_scope_repro.npz --report benchmarks/.bench-state/world_multiturn_scope_repro.json
python -m scripts.inspect_world_multiturn_errors --scopejudge-path PATH/TO/train.jsonl
```

### Pre-call history and post-driver GPU pilot

The [ScopeJudge paper](https://arxiv.org/html/2607.07774) distinguishes the
user's intent from the partial trajectory available before a proposed tool
call. Our earlier scope adapter discarded that partial trajectory. The new
opt-in research input includes only tool calls from **completed earlier
steps**; it never includes tool outputs, agent reasoning, another call in the
same step, or future calls. If a prompt exceeds its token budget, the encoder
removes the oldest prior calls while preserving the current request and action.
The default no-history path remains unchanged. This is untrusted research
context, not a source of execution authority.

After the GPU update, a 16-row CUDA training/evaluation smoke test completed.
A matched 160-row pilot then compared no history with one prior call at a
1,024-token limit. Both scored the same 117 eligible, trajectory-held-out
ScopeJudge calls. The [frozen pilot report](../daemon/benchmarks/qwen_scope_history_pilot_v1.json)
records source/artifact hashes and the paired results: balanced accuracy rose
from 44.5% to 56.5%, and false allowances fell from 28 to 18, while 33 valid
calls were still rejected. A trajectory-level bootstrap interval for the
balanced-accuracy difference crosses zero. A cutoff selected solely from
public calibration negatives still allowed five held-out violations and
rejected 57 of 59 valid calls. The read-only fixture was ranked first, but its
probability was only 40.7% in the history variant. These are pilot results,
not safety, Jev-parity, or live-desktop evidence; neither adapter is promoted.

We attempted a larger matched comparison on a fresh random-trajectory split.
The no-history control completed 756 training rows after an identity-checked
resume, but its 83.5% scope agreement on 711 held-out calls hid **19 false
allowances among only 23 violations** (51.6% balanced accuracy). The
one-prior-call arm has a finite, validated checkpoint at row 416/756 but **no
held-out result**: native CUDA failed during training, an exact native resume
hit a device-side assertion, and a WSL resume also failed. A subsequent
native resume passed the earlier failing group with CUDA launch blocking and
the caching allocator disabled, then failed after row 416 with fresh Windows
`nvlddmkm` events. All 904 saved checkpoint tensors remain finite. The trace
does not establish whether a particular input, PyTorch/CUDA path, driver, or
GPU fault is the root cause; the no-caching setting is diagnostic, not a fix.
We did not treat partial training as a comparison. See the
[scaled-attempt report](../daemon/benchmarks/qwen_scope_history_scaled_attempt_v1.json).

The Jev Decisions endpoint subsequently responded again. A new
[matched public comparison](../daemon/benchmarks/qwen_scope_jev_matched_v1.json)
preselected a label-balanced 46-call subset of the completed no-history
control's 711 eligible held-out ScopeJudge calls, without inspecting either
model's predictions for selection. On 23 out-of-scope calls the local control
falsely allowed 19 while Jev falsely allowed four; on 23 in-scope calls the
local control falsely warned once while Jev warned 12 times. They agreed on
only 14/46 decisions. This is an evaluation-only, previously inspected public
split, not a fresh blind safety benchmark, model-training input, Jev parity,
or permission to execute. The history arm remains incomplete.

Exploratory threshold checks on the completed no-history control did not find
a usable safety/utility balance. Selecting a cutoff above every one of 36
out-of-scope *calibration* probabilities allowed zero of 23 held-out
violations, but rejected **674/688** valid calls. Choosing the cutoff that
maximized balanced accuracy on the same 70 calibration calls still allowed
**7/23** held-out violations and rejected **424/688** valid calls. Neither
cutoff used held-out labels for its numeric choice, but this analysis followed
earlier inspection of the split, so it is not fresh blind validation. Inputs,
formulas, hashes, and counts are in the scaled-attempt report. Thresholding
cannot make this adapter an approval authority.

The larger split also exposed a benchmark weakness: random trajectory
assignment can put very few violation labels in the test partition. A new
**opt-in** `balanced_trajectory_v2` strategy searches deterministic grouped
assignments using only call counts and expert-label counts, never model
predictions. It keeps trajectories disjoint and does not change the default
split used by previous reports. With seed `20260926`, 1,024-token encoding
retains 718 test calls and 60 violations (versus 711 and 23 above); no-history
and one-call-history modes retain identical case IDs. No model has yet been
trained or evaluated on this v2 split, so those counts are a preflight, not a
new performance result.

Further group analysis found that this v2 test's 61 selected violations come
from only nine trajectories, with **48 from one trajectory**; its training
sample also takes 108 of 256 violation examples from one group. An opt-in
`diverse_trajectory_v3` split now includes the number of violation-bearing
trajectories and within-split violation concentration in its deterministic
label-only assignment objective. Its class-balanced train/calibration sampler
round-robins trajectories, rather than drawing individual calls uniformly.
The prior strategies and their frozen reports are unchanged. At the same seed,
the v3 full test has 47 violations across 12 trajectories, with at most 15 in
one; 1,024-token encoding retains all 47 violations and 686 calls. No-history
and one-prior-call modes retain the same eligible test IDs. The v3 training
sample still has 89/256 violations from one trajectory, so this improves the
evaluation concentration more than training diversity. The
[preflight report](../daemon/benchmarks/scope_group_diverse_preflight_v1.json)
records the selection and limits. No model has been trained or evaluated on
v3; it is not a performance claim.

An opt-in `inverse_sqrt` trajectory weight now exists for the research
training path. It reduces repeated calls from one expert-label/trajectory
cell while preserving each label class's total training weight, and the exact
weight vector is bound to the saved checkpoint identity. A 16 general + 16
ScopeJudge row GPU smoke test completed training, calibration, held-out scoring,
and artifact saving. Its eight scope test calls are far too few for a quality
claim. On already cached frozen Open-Jev 2B features, an analogous weighted
head raised ScopeJudge test AUC from 0.905 to 0.926, but default false
allowances only moved from nine to eight and the calibration cutoff rejected
300/517 valid calls. On the separately inspected 60-case Heliox-style suite,
the weighted cutoff falsely allowed 21/30 violations versus 20/30 for the
unweighted control. The [ablation](../daemon/benchmarks/openjev_2b_scope_weighted_ablation_v1.json)
is post-hoc research evidence, not a demonstrated general safety fix; weighting
remains disabled by default and no research head gains action authority.

### Post-update GPU and transfer recheck

On the RTX 4060 laptop with NVIDIA driver 617.14 and PyTorch 2.6.0+cu124,
the same `diverse_trajectory_v3` 128-general + 256-scope selection was tried
at both BF16 and FP16 with a 768-token limit. BF16 failed twice in CUDA after
checkpointed rows 80 and 128. FP16 passed those positions but also faulted
after row 288; its exact identity-checked resume completed all 376 eligible
training rows and held-out scoring. Windows recorded `nvlddmkm` events 13 and
153 at the faults. Different short examples triggered the failures, so neither
a fixed bad row nor GPU stability has been demonstrated. The trainer now offers
an opt-in precision choice bound to the checkpoint and scorer, and baseline
evaluation checkpoints survive future interrupted runs.

The completed FP16 model retained 248 scope training rows and 694 held-out
scope calls, including 55 violations. At its default cutoff it falsely allowed
**44/55** violations and falsely warned on **169/639** valid calls; balanced
accuracy was 0.468. General test top-1 was 0.625 and OOD top-1 was 0.469.
A cutoff chosen strictly above the 28 negative *calibration* probabilities
still falsely allowed two held-out violations while warning on 584/639 valid
calls. On the previously inspected 60-case Heliox-style suite, the default
cutoff falsely allowed all 30 violations; the calibration-only cutoff falsely
allowed two and warned on 21/30 valid cases. The cutoff check is exploratory
because this public split and transfer suite had already been inspected. The
[frozen diagnostic](../daemon/benchmarks/qwen_scope_diverse_gpu_recheck_v1.json)
records report and artifact hashes. The adapter is not loaded by Heliox.

The same frozen FP16 adapter was independently scored on a fixed 96-action,
one-per-trajectory, label-balanced sample from the CC BY 4.0
[MisActBench](https://huggingface.co/datasets/osunlp/MisActBench) annotations.
The public set contains 2,264 human-labeled computer-use actions across 558
trajectories. Only the user instruction and proposed action were provided; the
screenshots and full history required by the benchmark were unavailable in
this text-only probe. Under the benchmark's broader action-alignment question,
the adapter falsely allowed **46/48** misaligned actions, falsely warned on
**0/48** aligned actions, and ranked them at **0.511 AUC**. This is
not a full MisActBench result or comparable with the paper's multimodal
baselines. The [frozen transfer report](../daemon/benchmarks/misactbench_text_transfer_v1.json)
pins the annotation and adapter hashes; the
[reproduction script](../daemon/scripts/evaluate_misactbench_transfer.py)
checks them. The [paper](https://arxiv.org/abs/2602.08995) finds that visual
observations and compact history summaries materially aid action alignment,
which reinforces the need to evaluate task state rather than isolated text.
The separate ScopeJudge test logits also rank its 55 violations at only 0.505
AUC, so cutoff tuning cannot rescue this frozen model's discrimination.

### Human-action head and fresh trajectory check

To test whether public human labels improve the representation, a separate
research path selects 256 fit, 64 calibration, and 96 test actions with one
action per trajectory. It filters by the model's token budget *before* selecting
the frozen test groups; a first LoRA attempt was abandoned when an audit found
that inconsistent filtering had changed the group selection. The corrected
FP16 LoRA attempt stopped after 16 checkpointed rows on Windows and 32 after
an identity-checked WSL resume. Both produced CUDA execution failures with
`nvlddmkm` events 13/153. Neither produced a trained adapter or test metric.

A lower-load frozen Qwen3-0.6B feature pass completed all 416 selected actions
on the GPU. A CPU logistic head picked regularization on calibration labels
alone. On the already inspected 96-action test, it reached 0.665 AUC and
falsely allowed 19/48 off-task actions at the default cutoff. Two prior action
texts reduced this test AUC to 0.634 and increased false allowances to 25/48;
it was not selected. The no-history head was then frozen and evaluated once on
64 **additional trajectories excluded from fit, calibration, and both earlier
tests**. It ranked them at only **0.488 AUC**, falsely allowing **17/32**
off-task actions and falsely warning on **14/32** aligned ones. Its
calibration-derived cautious cutoff allowed none but rejected **all 32** valid
actions. These results are in the
[pinned diagnostic](../daemon/benchmarks/misactbench_frozen_gpu_recheck_v1.json),
with checksums for the ignored local artifacts. The extra test invalidates a
generalization claim from the first 96 cases. No head is enabled in Heliox.

The reproducible experiment uses
[`extract_misactbench_frozen_features.py`](../daemon/scripts/extract_misactbench_frozen_features.py),
[`fit_misactbench_frozen_head.py`](../daemon/scripts/fit_misactbench_frozen_head.py),
and [`evaluate_misactbench_fresh.py`](../daemon/scripts/evaluate_misactbench_fresh.py).
It remains text-only: click coordinates, on-screen injections, and changed UI
state may be impossible to interpret without the corresponding observation.
The paper's compact narrative summaries include **observed before/after UI
state**, not just earlier action commands; the failed two-action-text ablation
does not refute that approach.

### Screenshot-based GPU probe

After the updated GPU became available, we tested the Apache-2.0
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) vision-language model
at a pinned revision on the same human-labeled action task. A bounded
[screenshot fetcher](../daemon/scripts/fetch_misactbench_screenshot.py) reads
only selected PNGs from the public MisActBench ZIP using HTTP byte ranges,
validates the pinned ZIP central directory and each PNG's CRC, and never
downloads the 2.33 GB archive. The first selected set referenced one image
absent from the published ZIP; the final selection rule therefore requires
the image to exist *before* sampling and excludes all earlier fit,
calibration, test, and fresh-test trajectories. The
[selection manifest](../daemon/benchmarks/misactbench_visual_selection_v1.json)
pre-registers 16 new, label-balanced, one-per-trajectory actions.

The frozen model completed all 16 at 1280-pixel maximum image side, with
6.244 GiB peak allocated VRAM. It answered **MISALIGNED on all 16**:
zero false allowances but **8/8 false warnings** on valid actions. This
always-reject behavior is not a usable action judge. A single inspected valid
VS Code File-menu click remained rejected after explicitly explaining that
intermediate steps are allowed and that click coordinates use the original
image dimensions. Eager attention ran out of VRAM at 1920-pixel input; PyTorch
SDPA completed that full-resolution case but still rejected it. The
[pinned diagnostic](../daemon/benchmarks/misactbench_visual_gpu_recheck_v1.json)
records the model, prompt, selection, and ignored raw-report hashes. Prompt
and resolution checks after inspecting outcomes are development diagnostics,
not a second held-out test. We did not train or deploy this model.

To reproduce from `pilot/`, provide the pinned annotation JSON at
`daemon/benchmarks/.bench-state/misactbench_annotated.json` and the cached
Qwen3.5-2B snapshot, then run:

```bash
python daemon/scripts/select_misactbench_visual_cases.py --dataset daemon/benchmarks/.bench-state/misactbench_annotated.json --feature-report daemon/benchmarks/.bench-state/misactbench_qwen_frozen_v1.json --fresh-report daemon/benchmarks/.bench-state/misactbench_fresh_test_v1.json --output daemon/benchmarks/.bench-state/visual_selection_recheck.json --rows 16
python daemon/scripts/fetch_misactbench_screenshot.py --dataset daemon/benchmarks/.bench-state/misactbench_annotated.json --manifest daemon/benchmarks/misactbench_visual_selection_v1.json --output-dir daemon/benchmarks/.bench-state/misactbench_visual_images_v1
python daemon/scripts/probe_misactbench_visual.py --dataset daemon/benchmarks/.bench-state/misactbench_annotated.json --manifest daemon/benchmarks/misactbench_visual_selection_v1.json --image-dir daemon/benchmarks/.bench-state/misactbench_visual_images_v1 --report daemon/benchmarks/.bench-state/visual_probe_recheck.json --prompt-version v1
```

Compare the rechecked selection SHA-256 with the manifest hash in the pinned
diagnostic. The scripts refuse a changed annotation or ZIP index;
screenshots and full case-level output belong under the ignored
`.bench-state` directory. This small screenshot-only probe omits observed
history and is not a full MisActBench score, a live-desktop safety result,
or evidence of Jev parity.

### Later GPU training check

With the current RTX 4060/driver 617.14 setup, we retried the isolated
FP16 rank-4 Qwen3-0.6B LoRA path using licensed Open-Jev examples and
MisActBench human action labels. A 73-row training/evaluation smoke passed
the earlier 16- and 32-row failure points, then a separate-seed 354-row run
completed training and held-out evaluation at 2.772 GiB peak allocated VRAM.
That is evidence of **two completed runs**, not proof the earlier CUDA/
`nvlddmkm` fault is fixed or that sustained training is stable.

The 16-action smoke head falsely allowed all eight off-task actions. The
larger run's 64-action test went to the opposite extreme: **0/32 false
allowances but 32/32 false warnings**; its aligned-action ranking AUC was
only 0.600. A cutoff chosen from 32 calibration actions alone still falsely
allowed **1/32** held-out off-task actions and warned on **27/32** valid ones;
the calibration AUC was 0.703. Open-Jev held-out top-1 was 0.500 on 58
actions and 0.704 on 27 OOD actions. Different seeds and sample sizes make
the two runs' scores non-comparable. The [post-update GPU report](../daemon/benchmarks/qwen_misact_post_update_gpu_v1.json)
pins raw local report and artifact hashes. The adapters remain ignored local
research artifacts and are not loaded by Heliox. A completed GPU job cannot
turn an always-allow or always-reject action judge into a usable safety layer.
The [calibration check](../daemon/scripts/check_misactbench_calibration.py)
verifies group disjointness and chooses that cautious cutoff without test labels.

### Frozen visual-feature head

To test whether screenshots contain a learnable signal rather than rely on
the generative model's always-reject answer, a separate study extracted
Qwen3.5-2B's last-token representation from each image, instruction, and
proposed action. Its [pre-registered split](../daemon/benchmarks/misactbench_visual_feature_selection_v1.json)
uses one labeled action per trajectory: 128 fit, 32 calibration, 32
development, and 64 final-test cases. The earlier inspected 16 visual-probe
trajectories are excluded. The CPU logistic head and its regularization and
cautious cutoff were fixed from fit/calibration before the final test;
the [development report](../daemon/benchmarks/misactbench_visual_feature_dev_v1.json)
was committed first.

The one-shot 64-action [final report](../daemon/benchmarks/misactbench_visual_feature_final_v1.json)
shows **0.759 AUC**. At the default cutoff the head falsely allowed **8/32**
off-task actions and warned on **13/32** valid ones. The calibration-only
cautious cutoff still allowed **3/32** off-task actions and warned on
**21/32** valid ones. This is evidence of ranking signal, not an acceptable
execution gate. The earlier text-only results used different splits, so
these numbers are not a paired modality improvement claim.

The 256-image extraction required only selected range-fetched screenshots,
not the full archive. It hit a CUDA illegal instruction after checkpointing
224 rows with contemporaneous `nvlddmkm` 13/153 events; an identity-checked
resume finished the remaining rows. The local features and head stay in the
ignored research directory. Nothing is loaded into Heliox, and the next
research step is observed action history and larger, separately held-out
user-consented desktop validation before any promotion discussion.

A [paired text-only control](../daemon/benchmarks/misactbench_text_control_final_v1.json)
now uses the same 256 cases, prompt, Qwen3.5-2B backbone, CPU-head procedure,
and calibration-only threshold selection, but omits image tokens. Its
[development record](../daemon/benchmarks/misactbench_text_control_dev_v1.json)
was committed before the text final test. On the same 64 final actions,
text-only AUC was **0.787** versus visual **0.759**. At the cautious operating
points, text-only falsely allowed **2/32** off-task actions and warned on
**18/32** valid actions, compared with visual **3/32** and **21/32**. The
default-cutoff tradeoff was mixed: text-only allowed 10/32 versus visual 8/32,
but warned on 8/32 valid actions versus visual 13/32. This small retrospective
comparison does not show that screenshots are useless; it shows that the
current frozen visual representation is not a dependable improvement. Neither
head is promoted. The text-only GPU extraction completed all 256 cases at
4.208 GiB peak allocated VRAM without a CUDA fault, which does not resolve
the earlier visual-run instability. The [MisActBench paper](https://arxiv.org/html/2602.08995)
also finds observed history important; these heads still lack it.

To distinguish an ignored-image failure from weak visual decision quality, a
[frozen wrong-screenshot diagnostic](../daemon/benchmarks/misactbench_image_shift_diagnostic_v1.json)
rotated screenshot pairings while holding instructions, actions, coordinate
metadata, head weights, and threshold fixed. Correct-image final AUC was
**0.759** and wrong-image AUC **0.587**; 27/64 default decisions changed.
The head therefore responds to image content. That does **not** mean it
grounds the action reliably: even with the correct image the cautious point
misses 3/32 off-task actions and rejects 21/32 valid ones. The shifted inputs
are deliberately out of distribution, so this is only a causal input-use
diagnostic, not a safety or world-model benchmark.

### Observed before/after history check

The next [pre-registered comparison](../daemon/benchmarks/misactbench_observed_history_selection_v1.json)
used 160 further, previously unseen MisActBench trajectories: 80 fit,
24 calibration, 24 development, and 32 final. The paired arms used the same
pinned Qwen3.5-2B revision, 768-pixel image limit, and CPU-head procedure.
The baseline saw the current screen, task, and proposed action. The history
arm also saw the immediately previous action and its *before* screenshot,
with the current screenshot as the observed *after* state. Neither prompt
included a previous-step label or future observation. Both frozen fits and
artifact hashes were committed in the [development record](../daemon/benchmarks/misactbench_observed_history_dev_v1.json)
before the final labels were opened.

Development AUC was 0.688 for current-screen only and 0.646 with the observed
transition. On the one-shot 32-action [final check](../daemon/benchmarks/misactbench_observed_history_final_v1.json),
it fell to **0.570** and **0.543**, respectively. The current-screen cautious
cutoff falsely allowed 0/16 off-task actions but warned on all 16 valid ones;
the history cutoff falsely allowed 1/16 and warned on 13/16. The updated RTX
4060 completed both 160-row feature passes, peaking at 4.275 and 4.392 GiB
allocated VRAM. This rules out a simple missing-one-transition explanation
for these *specific* frozen heads. It does not test compact narrative summaries,
longer histories, trainable multimodal adapters, or live Heliox interactions.
Neither head is loaded by the world model or approval path, and the final
split must not be used for another tuning attempt.

An additional source check found the Apache-2.0
[`ProCreations/auto`](https://huggingface.co/datasets/ProCreations/auto)
training corpus and its separate
[`approve-or-deny`](https://huggingface.co/datasets/ProCreations/approve-or-deny)
test set. Their rule distinguishes consequential unauthorized harm and prompt
injection from ordinary automatically executable work; that is not the same
label as ScopeJudge's exact task-scope authorization. Most training rows and
all benchmark labels are model-generated, not expert annotations. They could
support a distinct, clearly labeled transfer study, but cannot be silently
merged with ScopeJudge or treated as proof of Heliox safety. The full training
corpus is 1.13 GB, while the local C: drive had only about 2.3 GB free during
this run, so it was not downloaded or used for training. The separate MIT
[`AuthBench`](https://github.com/evolvent-ai/Authbench) evaluates task-level
file permission policies on 120 tasks rather than per-call scope labels, and
is another different task, not an interchangeable training set.

### GPU post-action outcome study

The pinned MIT-licensed [AgentNet Ubuntu trajectories](https://huggingface.co/datasets/xlangai/AgentNet)
provide a different, weaker supervision target: `last_step_correct` is a
model-reflector assessment of whether an *already attempted* action had its
intended effect. It is not a human authorization or safety label. We rejected
the [AgentNet partial/fail derivative](https://huggingface.co/datasets/gui-wm/agentnet-partial-and-fail-v1)
for per-action training because its sampled labels were constant within each
trajectory, not independent step outcomes.

The [pre-registered selection](../daemon/benchmarks/agentnet_outcome_selection_v1.json)
pins the source hash, chooses one action per task, balances the reflector
labels, and separates 1,024 fit, 128 calibration, 128 development, 128 final,
and 128 VS Code domain-shift actions by task. The frozen Qwen3.5-2B GPU pass
used the task, attempted action and command, before observation, and next
observation. It excluded thought, reflection, completion, and label fields;
all 1,536 inputs fit without truncation. Peak allocated VRAM was 3.624 GiB.
Only a regularized logistic head was trained; the base model stayed frozen.

The head and its calibration-only cautious cutoff were fixed in the
[committed development record](../daemon/benchmarks/agentnet_outcome_dev_v1.json)
before the final and domain-shift labels were opened. Development AUC was
0.824. The one-shot [final record](../daemon/benchmarks/agentnet_outcome_final_v1.json)
has AUC **0.869** on 128 unseen tasks and **0.871** on 128 VS Code tasks.
At the default cutoff, it misses 14/64 and 22/64 incorrect outcomes,
respectively. The cautious cutoff misses zero in those small samples, but
warns on 61/64 and 56/64 correct outcomes. Neither operating point is a
usable execution gate. This is post-action error detection, not pre-action
prediction; the next observation is available only after the action.

A [post-hoc shortcut diagnostic](../daemon/benchmarks/agentnet_outcome_shortcuts_v1.json)
trained fixed TF-IDF controls on the same fit groups. Task/action text alone
reached 0.715 final AUC, while next-observation text alone reached 0.656;
both were weaker on the VS Code shift than the frozen GPU representation.
This does not prove the GPU head learned a causal transition: the diagnostic
was run after final labels were opened and cannot guide model selection.
The research head is not loaded by the Heliox daemon, the world model, MCP,
or approval path. A consented, grouped live-desktop benchmark with human
adjudication and materially lower false-allowance and false-warning rates is
needed before any promotion discussion.

A second, **fresh task-group transfer** check isolated the difference between
prediction and verification. A prospective-only input omitted the next
observation while retaining the same task, current observation, action, and
command. Its separately frozen head reached only 0.719 AUC on development,
versus 0.824 with the next observation. We then reserved 256 entirely unused
task groups **before** extracting either view for transfer and compared the
unchanged heads on the same 128 general and 128 VS Code actions. The
[paired report](../daemon/benchmarks/agentnet_paired_fresh_transfer_v1.json)
shows post-action versus pre-action AUC of **0.849 versus 0.709** on general
tasks, and **0.816 versus 0.551** on VS Code. At the pre-action
calibration-only cautious cutoff, VS Code still had **5/64** incorrect
outcomes falsely allowed and **61/64** correct outcomes warned. The
post-action cutoff allowed none in this sample but warned on **61/64** correct
VS Code outcomes. Paired bootstrap intervals for the post-minus-pre AUC
difference exclude zero on these sampled groups; they do not generalize beyond
this model-generated-label corpus. Neither head can authorize action.

A calibration-selected [nonlinear prospective head](../daemon/benchmarks/agentnet_preaction_mlp_dev_v1.json)
was also trained on the same frozen pre-action features using CUDA. Its
development AUC was **0.715**, versus **0.719** for the logistic control. At
the calibration-only cautious cutoff it missed 2/64 incorrect outcomes and
warned on 52/64 correct outcomes; the logistic control missed 2/64 and warned
on 57/64. That mixed tradeoff is not an improvement in reliable prospective
execution judgment. No fresh final or domain-shift labels were used to tune
this head, and it remains local and disabled. Simply adding a small nonlinear
head did not close the gap created by the missing next observation.

This gap matches the motivation for explicit expected-effect and observed-
effect loops in [VeriGUI](https://aclanthology.org/2026.acl-long.1335/), and
for multi-step contextual-consistency evaluation in
[GUI-CC](https://arxiv.org/abs/2609.00048). Those papers motivate a research
direction; their results are not Heliox results. The immediate product-side
correction is narrower: terminal responses now distinguish executor-reported
effects from independently verified postconditions without changing policy,
approvals, or the verifier.

After the GPU update, a bounded [eight-action CUDA recheck](../daemon/benchmarks/agentnet_postrestart_gpu_recheck_v1.json)
reloaded the pinned Qwen3.5-2B backbone and reproduced the earlier frozen
features exactly, with 3.588 GiB peak allocated VRAM. These were reused fit
cases, so this checks the GPU execution path only, not accuracy or sustained
stability. The browser runtime also no longer repeats a successful click,
submission, type, or selection merely because the DOM snapshot is unchanged;
the diff remains observation rather than permission to replay an effect.

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

For the task-state head, use the pinned ScopeJudge `train.jsonl` matching the
dataset SHA-256 in the reports, then run from `daemon`:

```bash
python scripts/train_world_scope_head.py --scopejudge-path PATH/TO/train.jsonl --history-calls 0 --artifact benchmarks/.bench-state/world_scope_intent.npz --report benchmarks/.bench-state/world_scope_intent.json
python scripts/train_world_scope_head.py --scopejudge-path PATH/TO/train.jsonl --history-calls 2 --artifact benchmarks/.bench-state/world_scope_history2.npz --report benchmarks/.bench-state/world_scope_history2.json
python -m benchmarks.world_model_suite --iterations 20 --json
```

The last command checks the separately shipped OS-transition model; it does
not test or enable the research scope head.

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
