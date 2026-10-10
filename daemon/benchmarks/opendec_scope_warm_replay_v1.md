# Public-corpus warm start and replay-policy probe (research only)

This experiment tests a pinned public 2B baseline LoRA/head warm start for Heliox's
task-scope world model. The adapter was fine-tuned locally for one epoch on
1,380 training rows (873 TBAC, 507 ScopeJudge) with NF4 Qwen3.5-2B on GPU.
No reference-model response was used as a training label. The adapter and detailed reports
remain local in ignored `benchmarks/.bench-state/`; this note is not a release
or promotion of model weights.

## Paper-guided question

[Dream-RSI](https://arxiv.org/html/2609.14858v1) improves an exploration
*policy* by replaying outcomes of branches that were actually explored. It
does not fine-tune the underlying decision model or infer outcomes for unseen
branches. Heliox has a guarded strategy-evolution archive, but no suitable
completed, independently verified desktop discovery trees for a faithful
Dream-RSI run. We therefore tested only a shallow, **offline** analogue:
compare the scores of two already-observed candidate actions per authored task,
then optionally abstain when their margin is small. This also follows the
test-time candidate-comparison motivation of the
[Computer-Using World Model](https://arxiv.org/html/2602.17365v1) and the
risk/coverage trade-off of
[selective classification](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
It is neither a full Dream-RSI implementation nor an outcome predictor for
untried desktop actions.

## Frozen comparison results

| Evaluation | Default local | Cautious local | Archived the reference model |
| --- | --- | --- | --- |
| TBAC test, 935 cases | AUC .971; 33 false allowances, 47 false warnings | 2 false allowances, 298 false warnings | not compared on this split |
| ScopeJudge test, 589 eligible cases | AUC .869; 9 false allowances, 38 false warnings | 1 false allowance, 396 false warnings | not compared on this split |
| Heliox authored, 60 cases | AUC .951; 6 false allowances, 1 false warning | 1 false allowance, 13 false warnings | not compared here |
| Random authored, 100 cases | AUC .988; 11 false allowances | 1 false allowance, 8 false warnings | 0 errors |
| Qwen-v1 authored, 60 cases | AUC .952; 6 false allowances | 1 false allowance, 12 false warnings | 0 errors |

The cautious threshold, 0.992893517, was chosen on public calibration data,
not on these authored comparisons. The authored suites have been inspected in
earlier research, so their results are not blind generalization evidence.

For a two-candidate replay policy (one labeled direct action and one labeled
extra action per task), choosing the higher local score gets 49/50 random and
30/30 Qwen-v1 pairs right. Abstaining at an absolute score margin of 0.1 gets
44/44 decided random pairs and 26/26 decided Qwen-v1 pairs right, with 6 and
4 abstentions respectively. The 0.1 margin was examined after seeing the
random-suite scores, so these numbers are exploratory, **not** a calibrated
error guarantee. On the random suite, the reference model ranks 50/50 pairs correctly.

A tiny Dream-RSI-like policy-selection thought experiment exposes transfer
risk. With exploratory utility `+1 correct / -20 incorrect / 0 abstain`, the
Qwen-v1 replay prefers always choosing (30 versus 26), while on random replay
the same policy scores 29 versus 44 for abstaining. Historical replay alone
does not establish that the selected policy improves new tasks.

On the same previously inspected 128 licensed public OOD decisions as the
archived the reference-model comparison, local top-1 choice accuracy is .743 (the reference model .800), and
local yes/no accuracy is .761 (the reference model .821). Score-kind top-1 accuracy is .923,
but it is not a direct reference-model parity measure. Scope fine-tuning did not preserve
general decision parity.

## Reproduction and decision

The training script is `scripts/train_world_scope_qlora.py`; the frozen
comparisons use `scripts/evaluate_world_scope_qlora.py` and
`scripts/evaluate_world_general_qlora.py`. Local report SHA-256 values:

- `opendec_scope_warm_v1.json`: `9e4519c990d1fd65743df0f7cea45af030e593c48327e42a49a3e7e86c6f0e97`
- `opendec_scope_warm_random_v1.json`: `15c1c2afe8983f53065729adcbd23d892cb86f318615fa0cea68a7d903cf2c06`
- `opendec_scope_warm_qwen_v1.json`: `f8ec72f903f9fc1493b5f95fa92a1b47d25aba3d80056d3347cbe19be5c63ac5`
- `opendec_scope_warm_general_v1.json`: `ae9d10a565d6ca02d03b22068482fce1a6caf7886bdd651befca933d4ec6f84e`

**Decision: research-only, no runtime promotion.** False allowances remain on
all held-out scope splits, and paired candidate search cannot make a single
proposed action safe. Keep deterministic scope rules, confirmation, and the
fail-closed gateway as the execution authority. A meaningful Dream-RSI test
requires consented, outcome-verified multi-branch task traces and prospective
replay selection on distinct tasks; desktop-world-model evaluation additionally
requires real before/action/after transitions and live verification.
