# Intelligence evaluation and research map

Heliox treats neural intent, MCP, multi-agent execution, and world-model
prediction as four different control surfaces. A strong result in one is not
evidence for another. The release boundary is deterministic policy, visible
approval, execution, and postcondition verification; learned components may
add caution but cannot grant authority.

## Neural intent

Calibration is accepted only when held-block balanced accuracy beats both the
configured floor and chance by a registered margin, every class meets its
recall floor, and expected calibration error is bounded. Product evaluation
then scores the full commit gate: exact-intent precision/recall/F1, idle
abstention, false commits per idle hour, missed active trials, and commit
latency.

```bash
pilot-neurod-benchmark control-trials operator-labeled-trials.json
```

The manifest records independent active and no-control trials. It does not
turn a recorded-data result into live-human evidence. Candidate comparison
corpora are:

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/), already supported by the no-hardware benchmark;
- [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.benchmark.html), for reproducible cross-dataset evaluation;
- [BNCI2014-001](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html), a motor-imagery candidate; and
- [OpenBMI](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/), a larger open BCI candidate.

Additional datasets are not bundled or downloaded by default. Before adding
one, verify its license, provenance, subject-disjoint split, channel mapping,
and whether it includes the idle/no-control evidence needed to estimate false
activation. Classifier accuracy alone is not a product acceptance gate.

## Jev-inspired local scope decision

[Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev) answers
typed questions with probabilities, not prose or action plans. Heliox used its
`noul` primitive as a *teacher* for one narrow question: whether a proposed
action is within the user's request. This is separate from the hybrid world
model, which predicts action effects. It is not a copy of Jev or an Instinct-
class general agent.

The original [72-case study](../daemon/benchmarks/jev_scope_teacher.json)
trained the shipped, numeric-only v1 head. It scored 12/12 on its small final
synthetic split, but that result did not establish Jev-level generalization.
Its 18 repeated calls on six selected cases are
[retained separately](../daemon/benchmarks/jev_scope_repeat.json).

The subsequent challenge used **634 successful live Jev calls** in total:
616 distinct authored request/action pairs plus those 18 repeats. Total
reported API cost was **$0.009855**. The 616 pairs span the original suite,
[challenge](../daemon/benchmarks/jev_scope_challenge.py),
[augmentation](../daemon/benchmarks/jev_scope_augmentation.py),
[stress](../daemon/benchmarks/jev_scope_stress.py),
[fresh](../daemon/benchmarks/jev_scope_fresh_final.py),
[seeded random](../daemon/benchmarks/jev_scope_random_final.py), and
[third unseen](../daemon/benchmarks/jev_scope_unseen_final.py) suites. Jev
matched the independently authored binary labels on 615/616 distinct pairs;
the one disagreement is retained, not silently relabeled.

| Frozen experimental candidate | New unseen pairs | Agreement with Jev | Too-permissive / false warnings |
| --- | ---: | ---: | ---: |
| v5 on [fresh](../daemon/benchmarks/scope_decision_fresh_metrics.json) | 100 | 92/100 | 4 / 4 |
| v6 on [seeded random](../daemon/benchmarks/scope_decision_random_metrics.json) | 100 | 92/100 | 4 / 4 |
| v7 on [third unseen](../daemon/benchmarks/scope_decision_unseen_metrics.json) | 100 | 91/100 | 5 / 4 |

The v7 [numeric-only candidate](../daemon/benchmarks/scope_decision_candidate.npz)
(SHA-256 `6c7238fa140b02d77d2eb7b2a809402317ffd2e01014bd18932b9a95c78c9b5c`)
was trained on 516 already-inspected pairs. Request-grouped five-fold
development evaluation was 499/516 (96.7%), but the third unseen set fell to
91%. The candidate therefore **was not promoted**. The shipped v1 weights and
runtime path are unchanged. More synthetic tuning against inspected failures
would not make a credible parity claim. This narrow local classifier is not a
Jev replacement, a general agent, or a safety certification.

The shipped local head only adds an advisory `WARN` on a strong disagreement
over a describable side-effect action. It never approves, blocks, bypasses
the full companion review, or changes permissions. Unknown actions abstain.
No real screen data or private user task was sent to Jev, and the OpenRouter
key is absent from the results and weights.

To reproduce the frozen candidate and third unseen score (from `daemon/`):

```bash
python scripts/train_scope_decision_cv.py --include-fresh --include-random
python scripts/evaluate_scope_candidate.py --suite unseen --teacher benchmarks/jev_scope_unseen_teacher.json --output benchmarks/scope_decision_unseen_metrics.json
python -m pytest tests/test_jev_scope_suite.py tests/test_scope_decision.py tests/test_scope_decision_experiment.py tests/test_execution_companion.py -q
```

These commands are offline; repeating live Jev calls is opt-in and paid via
`scripts/evaluate_jev_scope.py --suite unseen --env-file ../.env`. Before any
promotion, collect consented, de-identified real plan/action examples, freeze
environment-disjoint test sets, measure false assurances and false warnings,
and review provider privacy and licensing terms. Jev's probability on a
synthetic authorization question is not itself permission to execute.

## Local MCP

The local stdio bridge follows an explicit asynchronous handle model:
preview, submit, poll, and cancel. `request_id` makes a host retry idempotent;
task IDs remain daemon-owned and cannot be forged into another ownership
scope. Tool schemas bound input and identifier length, and the daemon URL must
be an unambiguous loopback WebSocket.

The [2026-07-28 MCP specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
removed protocol sessions and moved optional long-running Tasks into an
extension. Heliox's explicit task handles fit that direction, but Heliox does
not claim support for the optional Tasks extension until compatible IDE hosts
can exercise it end to end. The current seven-tool contract remains the stable
surface.

## Multi-agent execution

Model-produced decompositions are data, not authority. Heliox validates every
title, description, specialist class, complexity, dependency reference,
subtask count, and dependency graph before execution. Cycles, deadlocks,
unknown dependencies, and unsupported specialists fail closed. A failed
prerequisite skips its dependents; it is never force-scheduled. Independent
branches are parallelized only through the separate bounded API that requires
an explicit independence attestation.

Useful external evaluations include [GAIA](https://arxiv.org/abs/2311.12983),
[AgentBench](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html),
and [OSWorld](https://github.com/xlang-ai/OSWorld). They are research inputs,
not current release scores: web accounts, environment images, side effects,
licenses, and task-specific success verifiers must be isolated before any
subset can enter CI. Anthropic's
[multi-agent architecture report](https://www.anthropic.com/engineering/multi-agent-research-system)
supports orchestrator-worker evaluation, but its results are not Heliox
results.

## Hybrid world model

The shipped OS-impact head predicts coarse disk/process changes, not whether
an arbitrary proposed tool call fulfills a user's task. Its stored temporal
validation covers 5,400 samples across 12 action types; the benchmark checks
the saved evidence and direction invariants, not fresh Jev-style reasoning.

A separate **research-only GPU task-state head** was retrained from frozen
Qwen3.5-2B features on the pinned agentic, TBAC, ScopeJudge, and When2Call
training splits. On the held-out When2Call preference test it reached 70/100,
accepting 14 dispreferred calls and rejecting 16 preferred calls; on the
previously inspected 60-case Heliox probe it reached 46/60. A calibration
threshold that rejected all dispreferred calls also rejected all 50 preferred
When2Call calls. These are *preference* labels, not user-authorization labels;
the original experimental report's "false allowances" field must not be read
as a permission metric.

This head is **not loaded by the daemon** or used as permission evidence. It
does not establish Jev parity or world-model transition fidelity. Reproduce
the experiment from `daemon/` with `scripts/train_world_multidomain_mlp.py`,
the pinned feature checkpoints, and fresh artifact/report paths. The private
local replay report is not a shipped benchmark.

Post-hoc calibration of that *frozen* GPU head was also tested with
[temperature scaling](https://proceedings.mlr.press/v70/guo17a.html) and a
monotone scale-plus-bias map. Both were fit on balanced calibration domains;
test sets were scored only after fitting. On the 100-case When2Call test,
raw/temperature/scale-plus-bias Brier scores were 0.1968/0.1965/0.1980.
At a fixed 0.5 cutoff, dispreferred calls accepted were 14/14/9, while
preferred calls accepted were 34/34/32. Thus the lower negative-call count
is not a free accuracy gain, and calibration did not improve ranking (AUC
0.7616 for all three). At a 0.9 cutoff, the three maps accepted 7/3/3 actions,
respectively.
The [risk-coverage tradeoff](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
must be evaluated before an abstaining predictor could be considered useful.
The GPU head was already selected using these calibration partitions, and
the Heliox probe was previously inspected, so this is exploratory evidence,
not independent validation. No calibrator is shipped or authorized for
runtime use. Reproduce from `daemon/` using
`scripts/calibrate_world_task_head.py` with the saved head, source feature
report/checkpoints, optional When2Call report/checkpoints, and a fresh report
path.

A stricter GPU leave-one-domain-out diagnostic retrained the same small head
four times with fixed architecture, seed, and epochs. Each fold normalized and
trained on only the *other* three domains' training rows, then scored the
excluded domain's test split. Held-out agentic and TBAC AUCs were 0.879 and
0.849, but held-out ScopeJudge and When2Call fell to 0.411 and 0.497.
The When2Call-excluded fold accepted all 100 When2Call test cases at 0.5,
including all 50 dispreferred calls. This is a generalization failure: the
current task-state head needs domain-specific training examples and cannot be
called a transferable Jev-like world model. Reproduce with
`scripts/evaluate_world_cross_domain.py` and fresh report path; the local
feature checkpoints and generated report remain research-only. These domain
families informed earlier hyperparameter exploration, so this diagnostic is
not a pristine independent benchmark either.

A fixed-settings [two-output GPU probe](../daemon/scripts/train_world_multitask_head.py)
then kept authorization-scope and call-preference labels on separate outputs
of one shared nonlinear head. Held-out agentic/TBAC/ScopeJudge scope AUCs were
0.931/0.912/0.848; When2Call preference AUC was 0.732. On the already-inspected
Heliox probe it scored 49/60 but accepted 11/30 negative actions, compared
with 46/60 and 1/30 for the earlier selected shared-binary head. Architecture
and selection settings differ, so this is not a controlled causal comparison;
it is another unpromoted failure to achieve reliable cross-task decisions.
The artifact and report remain local research outputs, not runtime models.

The separate River transition-reliability learner now requires an executed,
non-dry-run action plus a matching independently observed postcondition before
it accepts a training label. Existing ledger verification events lack the
per-action observation field and are not retroactively relabeled from executor
callbacks. This learner remains advisory and needs new validated outcome data.
For session plans, a promoted estimate with enough observations for that
action type and executor source can add RiskGate caution; it cannot lower existing risk,
approve an action, or replace a rule. Executor-callback history remains a
separate, weaker signal and is labeled as such in prediction provenance.

### Research-only tool-response transition probe

To test a learned *outcome* predictor rather than another action-scope label,
`scripts/sample_world_transition_corpus.py` samples revision-checked rows from
the [Patronus world-model corpus](https://huggingface.co/datasets/PatronusAI/world_model_corpus).
`scripts/train_world_transition_lora.py` trains a local Qwen3-0.6B LoRA to
predict the next tool response from the supplied state and action;
`scripts/evaluate_world_transition_generation.py` compares exact held-out
outcomes against the unchanged base. The adapter and sampled rows remain in
ignored local benchmark state, and no runtime world-model path loads them.

The first bounded sample contained 64 source-train and 9 source-eval rows.
Its dataset-server revision was `d4aeb7d73de29bcf6ffceca3e132d734059e0237`;
the local source-rows SHA-256 was
`2f6dbc9889f821ab55ddd1ddc81b038178933ed4fe153f4c6ca9dd0819d2cdb6`.
After no-truncation token filtering, only 33 train and 8 eval rows remained;
all eligible eval rows came from one contiguous `pandora` page. One GPU epoch
changed mean held-out token loss from 2.377 to 0.411. With explicit attention
masks and greedy generation, exact tool-outcome matches changed from 0/8 for
the base to 3/8 for the adapter. This is evidence that the small adapter learned
some synthetic response patterns, **not** that Heliox learned actual OS action
effects: the set is tiny, highly correlated, and mostly not OS/UI work. A
larger resample attempt hit dataset-server 429/502 responses and supplied no
additional validation data. An environment-disjoint corpus of independently
observed Heliox outcomes
and prospective postcondition tests is required before considering any
runtime integration. Lower token loss or a well-formed `<tool_response>` wrapper
cannot authorize an action or establish Jev-level generalization.

### Task-group-disjoint desktop transition probe

The [Computer-Using World Model paper](https://arxiv.org/abs/2602.17365)
motivates predicting action-induced UI changes from state/action/next-state
transitions. Heliox's next research probe uses the MIT-labeled
[world-model-transitions dataset](https://huggingface.co/datasets/ageppert/world-model-transitions),
derived from [OpenCUA AgentNet](https://huggingface.co/datasets/xlangai/AgentNet).
The source contains textual descriptions of the current desktop state,
planned action, and subsequent outcome, not the raw next screenshot. The
sampler checks the Hugging Face source revision and keeps task IDs disjoint
between the sampled training and evaluation rows, since the source's published
row-level validation split can include the same task as training.

At revision `aa6bc2dcef1bcbf922133e621a3c75f502919d8e`, a seeded local
sample yielded 384 training rows from 34 tasks and 256 held-out rows from 123
different tasks (SHA-256
`834dcd9a5626ba8da2fc121993b9e8b542b6060593cbe03d2b6019c45a03a516`).
One RTX 4060 LoRA epoch on pinned Qwen3-0.6B reduced mean per-row held-out
target-token loss from **2.741 to 1.643** at learning rate `1e-4`; no rows were
excluded by the 1536-input/384-target token limits. This is a better-grounded
*text transition* probe than the small tool-response sample above, but still
does not measure actual screenshot
prediction, Heliox task success, calibration, safety, or Jev parity. The adapter
and source text stay in ignored local benchmark storage and are not loaded by
the daemon. Reproduce the sampler, training, and generation diagnostics with
`daemon/scripts/sample_grouped_world_transitions.py`,
`daemon/scripts/train_world_transition_lora.py`, and
`daemon/scripts/evaluate_grouped_world_generation.py`. Reproduce the sampled
source with `--dataset ageppert/world-model-transitions` and
`--expected-revision aa6bc2dcef1bcbf922133e621a3c75f502919d8e`, plus the sampler's default
seed/page settings; use `--max-input-tokens 1536 --max-target-tokens 384` for
training and generation. Compare the source SHA-256 above before interpreting
new results.

The fixed greedy-generation check used one row from each of 12 held-out task
groups. A second one-epoch run at `2e-5` tested whether a lower learning rate
would avoid repetition on the same sample:

| Run | Mean held-out loss | Mean ROUGE-L F1 | Mean token F1 | ROUGE-L wins over base | Hit 384-token cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base, first run | 2.741 | 0.205 | 0.342 | — | — |
| Adapter, `1e-4` | 1.643 | 0.232 | 0.360 | 8/12 | 4/12 |
| Adapter, `2e-5` | 1.781 | 0.220 | 0.354 | 7/12 | 5/12 |

Small differences in base generation metrics were observed across runs (second
run: ROUGE-L 0.206, token F1 0.343). The first adapter's worst ROUGE-L
regression hallucinated and repeated formatting-dialog controls to the token
cap. The lower-rate adapter improved that case's lexical score and stopped at
114 tokens, but still incorrectly claimed the Find and Replace dialog had
closed and repeated control descriptions. The base response also erred, claiming
the document was already formatted. These exploratory comparisons use an
already-inspected evaluation set; they are not a fresh model-selection holdout.
Lexical gains do **not** demonstrate faithful UI effects, and neither adapter
is suitable for live Heliox use.

### Adjacent observed-state probe

[State Transition Pretraining](https://arxiv.org/abs/2607.24112) motivates
learning from consecutive `(state, action, next state)` observations. Its
[AgentNet conversion code](https://github.com/xyliugo/gui-state-transition-pretraining/blob/main/data_preparation/agentnet/convert_forward_task.py)
pairs adjacent screenshots. Heliox's smaller GPU probe instead pairs the
**textual screen observation** at source step `t`, its described action, and
the textual screen observation at step `t+1` from the same task. It does not
download images or perform the paper's visual or inverse-dynamics training.
`daemon/scripts/sample_adjacent_world_observations.py` pins the source
revision, requires consecutive step indices and adjacent source rows, rejects
ambiguous state/action boundaries, and excludes task groups shared by the
sampled training and evaluation partitions. No source action is executed.

At the same `world-model-transitions` revision above, a 12-page/8-page seeded
sample of 100-row pages yielded 914 training pairs from 89 tasks and 68
held-out pairs from 62 different tasks (source SHA-256
`96c7da9a8a14bf5e7d2fe6c9c9163bfceebeec0ddd15ef3c6a292ff7f7ee6971`).
With 1024-input/256-target token limits, 718 training and 56 held-out pairs
from 50 held-out tasks remained. One RTX 4060 LoRA epoch on the same pinned
Qwen3-0.6B reduced mean per-row held-out target-token loss from **2.333 to
1.460**. In a fixed 12-task greedy-generation check, mean ROUGE-L F1 rose
from 0.248 to 0.275 and token F1 from 0.414 to 0.449; 7/12 cases improved
on ROUGE-L and 4/12 adapter responses reached the 256-token cap.
The ignored local evidence is in
`daemon/benchmarks/.bench-state/agentnet_adjacent_world_v1.json`,
`agentnet_adjacent_world_lora_capped_v1.json`, and
`agentnet_adjacent_world_generation_v1.json`. Reproduction uses the pinned
revision above, sampler default seed and page settings, `--max-input-tokens 1024`,
`--max-target-tokens 256`, `--learning-rate 1e-4`, and the evaluator's default
12 cases and seed.

This is not a faithful world model. In the largest regression, opening a
folder produced a fabricated `README.md` and repeated `build.gradle` until
the token cap; the reference showed a `data` folder and `sample.txt`. The
largest lexical win was directionally correct: after a click in an empty
email body, the adapter predicted a positioned cursor while the base model
incorrectly predicted that text had already been typed. This is one audited
example, not evidence of broad task correctness. The first, longer-sequence
GPU attempt terminated with a CUDA launch failure and
coincident NVIDIA driver events, without a model report. The capped run
finished without new driver events. Neither loss nor lexical overlap proves
correct action effects, visual fidelity, cross-environment generalization,
or Jev parity. The adapter remains ignored local research data and is not
loaded by the daemon.

### Action-conditioned latent world-model check

To test whether Heliox's optional latent world-model direction can learn from
the same adjacent observations, `daemon/scripts/train_world_latent_transition.py`
maps current-state and action text into a shared, fixed 512-dimensional hashed
space, trains a residual next-state predictor, and compares it with simply
retaining the current state. The 914 training pairs are split again by task
into 713 fit rows (71 tasks) and 201 validation rows (18 tasks). The original
68-row, 62-task evaluation partition is never used for gradient updates or
checkpoint selection. Unlike the shipped `UiJepaPredictor`, this research
model consumes free-form action text, not Heliox's enumerated action types;
its weights are not compatible with or loaded by the live predictor.

The first CPU control selected epoch 5 by validation target cosine. On the
held-out partition, mean target cosine rose from **0.799** for the unchanged-
state baseline to **0.830**, but next-state retrieval among the 68 observed
candidates fell from **66.2% to 47.1%**. Selecting a checkpoint by validation
retrieval instead chose epoch 55; held-out retrieval remained worse at
**50.0%**, with target cosine falling to **0.680**. This is a failure to
discriminate action effects despite a seemingly better embedding alignment
in the first run. The evaluation partition had already been inspected during
the earlier language-model probe, so these are exploratory diagnostics, not
an untouched promotion result. Reports and weights remain ignored under
`daemon/benchmarks/.bench-state/`.

A later Windows diagnosis traced that import stall to Python 3.12's
`platform._wmi_query` hanging in a local WMI CPU-architecture lookup, before
PyTorch reached CUDA initialization. A process-local workaround that makes
that lookup raise `OSError` allowed the installed CUDA PyTorch to import; no
system-wide monkeypatch was installed. The completed GPU control on the same
text-hash features reached **0.691** held-out target cosine and **52.9%**
retrieval, versus **0.799** and **66.2%** for unchanged state. It still failed
the held-out discrimination baseline. The Python 3.11 CPU-only controls above
remain separate runs.

Reproduce the CPU controls from `daemon` with fresh report and weights paths
(this Windows host used `py -3.11`; use the equivalent Python in another
environment):

```bash
py -3.11 -m scripts.train_world_latent_transition --device cpu --selection-metric cosine --rows benchmarks/.bench-state/agentnet_adjacent_world_v1.jsonl --report benchmarks/.bench-state/world_latent_cosine_repro.json --weights benchmarks/.bench-state/world_latent_cosine_repro.pt
py -3.11 -m scripts.train_world_latent_transition --device cpu --selection-metric retrieval --rows benchmarks/.bench-state/agentnet_adjacent_world_v1.jsonl --report benchmarks/.bench-state/world_latent_retrieval_repro.json --weights benchmarks/.bench-state/world_latent_retrieval_repro.pt
```

Text-hash vectors are not screenshot features or live UI embeddings. Neither
cosine nor retrieval on sampled candidate descriptions establishes physical
or visual prediction fidelity, uncertainty calibration, task success, action
authority, or Jev parity. The result argues against staging this latent model
in Heliox until a fresh, consented observed-state corpus and stronger
environment-disjoint evaluation exist.

### Observed visual transition GPU probe

The [SATraj-OS dataset card](https://huggingface.co/datasets/AI45Research/SATraj-OS)
describes an Apache-2.0 GUI trajectory corpus with ordered screenshot filenames
and alternating screenshot/action messages. It is a **candidate** for a less
ambiguous observed-action probe than TESS; its published first-step example
alone does not establish that every record has a complete post-action
observation, nor does the dataset license resolve privacy or rights questions
for all depicted third-party interfaces. No SATraj-OS data or weights have been
imported or promoted in this repository.

`daemon/scripts/import_aligned_world_trajectories.py` accepts a local
`trajectories.jsonl` and its `images.zip` for an offline, bounded research
import. It requires an observed image, exactly one supported mouse action,
and the immediately following numbered image; rejects missing, unsafe-labeled,
oversized, or malformed records; retains only 16×16 grayscale features; and
keeps identical task descriptions in the same train/validation/evaluation
partition. A source JSONL hash, sampled-image digest, feature hash, and exclusion counts are
recorded. The importer has synthetic contract tests but **has not been run on
the public archive**, which is much larger than this bounded experiment.
An ordered next screenshot is still not proof of a causal effect: app state can
change asynchronously, and the reduced feature representation omits readable
text and task history. If locally acquired under suitable data terms, use fresh
ignored paths and compare correct-action, shuffled-action, and unchanged-screen
baselines before claiming action conditioning:

```bash
python -m scripts.import_aligned_world_trajectories --jsonl /path/to/trajectories.jsonl --images /path/to/images.zip --features benchmarks/.bench-state/aligned_probe.npz --report benchmarks/.bench-state/aligned_probe.json
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --features benchmarks/.bench-state/aligned_probe.npz --source-report benchmarks/.bench-state/aligned_probe.json --report benchmarks/.bench-state/aligned_model.json --weights benchmarks/.bench-state/aligned_model.pt
```

The [GUI-CC offline track](https://github.com/Fu-Fu-Fu-Fu/GUI-CC/blob/main/docs/offline.md)
is a stronger *future full evaluation* target: 500 mobile GUI trajectories and
4,905 transitions, with `initial.png` followed by numbered after-action
screenshots, fixed reference actions, and autoregressive rollout scoring.
Its [dataset card](https://huggingface.co/datasets/minuzero/GUI-CC) distinguishes
the CC BY 4.0-derived offline track from online screenshots restricted to
non-commercial research. A bounded one-step research probe is recorded below;
**the official GUI-CC multi-step benchmark has not been run**. The present
16×16 grayscale predictor cannot return a legible full-resolution next screen
or maintain long-horizon interface state, so its one-step pixel loss is not
comparable to GUI-CC's contextual-consistency score. Promotion would need a
different model/output contract, rights review, and a separately held-out
multi-step benchmark before any live dependency on it.

`daemon/scripts/sample_world_visual_transitions.py` samples adjacent
screen/action rows from the [TESS AgentNet dataset](https://huggingface.co/datasets/TESS-Computer/tess-agentnet),
whose card declares MIT licensing and describes normalized mouse coordinates,
click types, and episode identifiers. It accepts only consecutive steps in the
same episode, checks image URLs against a chosen dataset revision, derives the
same 16×16 grayscale features as Heliox's live screen observer, and retains no
raw screenshots. The row API itself is not revision-pinned, so the recorded
source revision and feature hash are provenance checks, not a guarantee that
future sampling returns identical rows. The ignored local feature file has SHA-256
`221b0bc5b98023224770d84e6de603285789b8898da4d4a759770aeb0074214c`.
The bounded sample has **968 pairs** from **89 episodes**: 707/110/151 rows
and 62/13/14 episode groups in train/validation/evaluation. This is one
public dataset, not environment-disjoint Heliox data. The dataset card does
not explicitly establish the pre/post-action timing of every screenshot, so
adjacency alone cannot prove that the annotated action caused the change.
The upstream [AgentNet release](https://huggingface.co/datasets/xlangai/AgentNet)
describes state–action trajectories, but this probe omits its task instruction,
action history, readable UI detail, and any separate post-action validation.
Those missing conditions make many next screenshots ambiguous from a 16×16
image and click alone; adding model capacity cannot fix missing causal context.

`daemon/scripts/train_world_visual_transition.py` trained a residual linear
predictor, a small residual MLP, and a spatial convolutional predictor on the
RTX 4060. The spatial run required cuDNN disabled locally after its default
CUDA kernel raised a device assertion; that is a machine/runtime workaround,
not evidence about model quality. Checkpoint selection used
validation weighted pixel MSE; evaluation episodes were not used to select
weights. On the held-out episodes, the MLP's changed-pixel MSE improved from
**0.1002** (unchanged-screen baseline) to **0.0719**, and weighted MSE from
**0.1410** to **0.1123**. But overall MSE worsened from **0.0282** to
**0.0315**. More importantly, permuting valid actions and coordinates between
evaluation rows gave **0.1107** weighted MSE, slightly *better* than the
correct-action **0.1123**. The linear run showed the same failure pattern:
**0.1204** correct-action versus **0.1208** shuffled-action weighted MSE, a
negligible difference. The sample is also dominated by left clicks (767/968).
The spatial run reduced weighted MSE to **0.1062**, but shuffled actions
again did slightly better (**0.1049**); even a model with the click location
as a spatial input did not pass the action-conditioning control.
This does **not** establish a learned action-conditioned world model, even
though changing-region predictions improved. All three weight files remain ignored
local research artifacts and are incompatible with the live `UiJepaPredictor`;
none is staged for shadow runtime inference or gating. The live model
still needs action-aligned observations, a compatible action vocabulary,
environment-disjoint transfer, calibrated uncertainty, and consented live
verification before promotion.

After the GPU update, a fresh CUDA run on the same pinned features reproduced
the failure: held-out weighted MSE was 0.10621454 with the correct action and
0.10621420 with a shuffled action. This [bounded recheck](../daemon/benchmarks/world_visual_gpu_updated_check_v1.json)
confirms the updated GPU can train the spatial model, not that the model learned
action effects. It is not a new held-out dataset or a basis for promotion.

### Ordered GUI-CC offline subset, after the GPU update

`daemon/scripts/import_gui_cc_ordered_transitions.py` imports the pinned
GUI-CC offline `initial → tap → after` sequence into the same 16×16 grayscale
feature contract as the live screen observer. It verifies the frozen metadata
hash, exact step order, image bounds, and tap coordinates; it excludes scroll,
navigation, and long press rather than mislabeling them as taps. Raw mobile
screenshots stay in an ignored local folder. Train/validation/evaluation split
by exact casefolded task template, not adjacent frame. The first
[frozen selection](../daemon/benchmarks/gui_cc_ordered_subset_selection_v1.json)
was committed before image inspection or model scoring: 18 templates and
141 accepted taps, split 101/23/17. The later
[62-template expansion](../daemon/benchmarks/gui_cc_all_templates_exploratory_v1.json)
was chosen **after** the first result; it is exploratory, not an independent
confirmation. It has 523 accepted taps across 43/9/10 template groups and
361/75/87 train/validation/evaluation rows. Four otherwise ordered pairs
exceeded the 4 MiB image safety limit. Both imports use the dataset revision
`758a5bcf4093ba4111a2cc882632eb47423130d1` and exclude unsupported actions.

On the first held-out set, CUDA spatial-model weighted MSE was **0.588786**
with the correct tap, **0.588574** with shuffled tap locations, and **0.565484**
for a separately trained, same-architecture state-only baseline. The MLP and
linear action models also failed to meaningfully beat shuffled taps. On the
larger exploratory set, the spatial model scored **0.279375** correct tap,
**0.279181** shuffled tap location, **0.281335** separately trained state-only,
and **0.595941** unchanged screen. Thus more examples improved one-step
prediction but still did **not** show that the model learned to use the action.
The tiny edge over state-only on the expanded set is not a controllability
result when shuffling the action slightly improves error. This illustrates
the statistical-shortcut concern studied in
[action-controllable world models](https://arxiv.org/abs/2608.04653);
that paper does not validate Heliox's model. Full source, weight hashes, and
scores are in the [bounded probe record](../daemon/benchmarks/world_gui_cc_ordered_probe_v1.json).
Weights remain ignored local research artifacts, with **no promotion** to
`UiJepaPredictor`. Mobile GUI pixels also do not establish desktop transfer,
multi-step contextual consistency, safety, or Jev parity. The next necessary
evidence is richer screen/action context plus independent same-state,
different-action outcomes or consented desktop trajectories; adding an
arbitrary action-sensitivity loss to this data would not prove causality.

An exploratory [RGB32 ablation](../daemon/benchmarks/world_gui_cc_rgb32_probe_v1.json)
kept the same 523 rows, task-group split, and source-image hashes but replaced
the gray16 features with 32×32 RGB and used a globally connected residual MLP.
On the already-inspected evaluation rows, weighted MSE was **0.482373** for
correct taps, **0.481927** after shuffling tap locations, and **0.485522** for
the separately trained matched state-only MLP. Richer pixels and global
connectivity still did not establish action use. This ablation was selected
after the gray16 result and is not a new holdout; it also has no runtime model
or authorization effect. The trainer now validates bounded feature shapes so
the same split and controls can compare representations without silently
assuming a 256-value vector.

A further [RGB32 spatial CUDA probe](../daemon/benchmarks/world_gui_cc_rgb32_spatial_probe_v1.json)
on those same, already-inspected rows lowered held-out weighted MSE to
**0.315980** for correct taps, compared with **0.315993** for shuffled tap
locations and **0.317941** for a matched state-only model. This is better
one-step reconstruction, but almost no action-location advantage. The trainer
now reports that advantage and a one-sided rank among 99 action shuffles:
the GUI-CC location result was **0.50**, whereas the verified same-state
controlled-page positive control was **0.01**. These ranks describe the
exploratory probes; prior inspection of the GUI-CC evaluation rows prevents a
fresh-holdout significance claim. Zeroing the action input produces a much
worse score here, but that out-of-distribution ablation does not override the
location-shuffle control. No weights were promoted to the live predictor.

The [RGB32/RGB64 inverse-action check](../daemon/benchmarks/world_gui_cc_inverse_resolution_probe_v1.json)
tests a narrower question: can a small model infer the tap coordinate from the
before and after screens better than from the before screen alone? On the same
already-inspected GUI-CC task split, held-out coordinate MAE was **0.2433**
versus **0.2403** at 32×32 RGB, and **0.2616** versus **0.2362** at 64×64 RGB
(before+after versus separately fitted before-only). A fixed center predictor
scored **0.2510**. All accepted actions are taps, so 100% action-type accuracy
is just the majority baseline. These exploratory results do not show an
incremental transition signal for this flat inverse model; they do not prove
the full-resolution screenshots contain no such signal. More pixels alone are
not a supported route to promoting the current visual predictor.

To reproduce the spatial comparison from `daemon`, use the pinned selection
and fresh ignored output paths:

```bash
python -m scripts.import_gui_cc_ordered_transitions --root benchmarks/.bench-state/gui_cc_all_templates_exploratory_v1 --selection benchmarks/gui_cc_all_templates_exploratory_v1.json --representation rgb32 --features benchmarks/.bench-state/gui_cc_rgb32_repro.npz --report benchmarks/.bench-state/gui_cc_rgb32_repro.json
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --action-permutations 99 --features benchmarks/.bench-state/gui_cc_rgb32_repro.npz --source-report benchmarks/.bench-state/gui_cc_rgb32_repro.json --report benchmarks/.bench-state/gui_cc_rgb32_spatial_repro.json --weights benchmarks/.bench-state/gui_cc_rgb32_spatial_repro.pt
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --train-without-action --features benchmarks/.bench-state/gui_cc_rgb32_repro.npz --source-report benchmarks/.bench-state/gui_cc_rgb32_repro.json --report benchmarks/.bench-state/gui_cc_rgb32_state_only_repro.json --weights benchmarks/.bench-state/gui_cc_rgb32_state_only_repro.pt
```

For the inverse check, import the same selection with `--representation rgb64`
and fresh local output paths, then run
`python -m scripts.probe_world_visual_inverse` on each representation twice:
once normally and once with `--train-without-after`. Both runs use the same
source report and seed; no private or runtime model weights are produced.

Reproduce from `daemon` with the downloaded pinned offline subset and **fresh**
ignored output paths (the importer refuses to overwrite existing artifacts):

```bash
python -m scripts.import_gui_cc_ordered_transitions --root benchmarks/.bench-state/gui_cc_all_templates_exploratory_v1 --selection benchmarks/gui_cc_all_templates_exploratory_v1.json --features benchmarks/.bench-state/gui_cc_repro.npz --report benchmarks/.bench-state/gui_cc_repro.json
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --features benchmarks/.bench-state/gui_cc_repro.npz --source-report benchmarks/.bench-state/gui_cc_repro.json --report benchmarks/.bench-state/gui_cc_action_repro.json --weights benchmarks/.bench-state/gui_cc_action_repro.pt
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --train-without-action --features benchmarks/.bench-state/gui_cc_repro.npz --source-report benchmarks/.bench-state/gui_cc_repro.json --report benchmarks/.bench-state/gui_cc_state_only_repro.json --weights benchmarks/.bench-state/gui_cc_state_only_repro.pt
```

To reproduce from `daemon`, use fresh ignored output names. On affected
Windows Python 3.12 installations, the WMI import workaround described above
is needed before Python can import CUDA PyTorch:

```bash
python -m scripts.sample_world_visual_transitions --pages 24 --page-length 50 --features benchmarks/.bench-state/tess_visual_repro.npz --report benchmarks/.bench-state/tess_visual_repro.json
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --features benchmarks/.bench-state/tess_visual_repro.npz --source-report benchmarks/.bench-state/tess_visual_repro.json --report benchmarks/.bench-state/tess_visual_spatial_repro.json --weights benchmarks/.bench-state/tess_visual_spatial_repro.pt
```

### Controlled action-alignment check

The [state-transition pretraining study](https://arxiv.org/abs/2607.24112)
jointly optimizes forward and inverse GUI dynamics from explicit
`(screen, action, next screen)` tuples. To separate a broken trainer from an
ambiguous public trajectory sample, `daemon/scripts/collect_controlled_ui_transitions.py`
now creates isolated headless pages, executes one click or mouse move per
sample, checks the resulting DOM state, and retains only the same in-memory
16×16 feature used above. It never visits a remote page, touches the user's
desktop, or saves raw screenshots. Four generated layout families are used for
fit, one for validation, and a different one for evaluation; each of two seeds
produced 320/80/80 transitions. The ignored source feature SHA-256 values are
`74adc438341f187845ce58f1393277f015f6e400b2f41cc59916dbdff5a44e4e`
and `823ab8bb94856267049c42873f5d0d1f289b0b04e06502add1662869dd9b2758`.

The same spatial predictor trained on CUDA passed an action-control check on
both held-out layout families. For seed `20260926`, evaluation weighted MSE
was **0.02938** for unchanged screen, **0.00176** with the correct action,
**0.02792** with a fully shuffled action, and **0.02098** when locations were
shuffled only among rows of the same action type. For seed `20260927`, the
same values were **0.02922**, **0.00216**, **0.02781**, and **0.02021**.
This demonstrates that the pipeline can use both action type and location for
simple, causally verified button toggles. It does **not** rescue the failed
public-trajectory result, establish transfer to real desktop applications,
or justify loading these weights into Heliox. The controlled pages have only
two controls and one predictable effect; task context, text, menus, delayed
effects, and competing external events are absent. Neither the paper's joint
forward/inverse dynamics objective nor its larger multimodal backbone has been implemented
here. Both trained artifacts remain ignored local research files.

The collector now also has an opt-in `--paired-counterfactuals` mode. It
resets the same local page, clicks each of its two controls separately, checks
the selected DOM panel, requires identical before features and distinct after
features, then stores the two observations under one group. The trainer can
swap actions **only within such a verified pair**, a stronger counterfactual
control than shuffling across unrelated screens. On two seeds with 160/40/40
train/validation/evaluation rows each, correct-action held-out weighted MSE
was **0.00422** and **0.00437**; giving each row its paired alternative action
raised it to **0.06599** and **0.06581**. Separately trained state-only models
scored **0.01451** and **0.01477**. The [paired evidence record](../daemon/benchmarks/world_controlled_counterfactual_pairs_v1.json)
pins source, report, and weight hashes. This verifies action sensitivity for
simple generated button effects across held-out layout families, **not**
general GUI or desktop-world prediction. The public GUI-CC tests above still
fail their action-shuffle control; no learned weights are promoted.

### Paired transitions in Heliox's own browser UI

The [local UI collector](../daemon/scripts/collect_heliox_ui_transitions.py)
resets a fresh headless Heliox session before each Dashboard or Settings click,
checks the selected tab, and requires exact equality of the two 16×16
pre-action features and distinct outcomes. Decorative canvases and live header
telemetry are hidden only for capture stability; raw screenshots are not
retained. Its current version blocks non-origin HTTP, external WebSockets, and
service workers. The first full 144-row set used fresh browser storage with no
daemon listening but was collected **before** that network restriction was
added. Four viewport families were fitted, one validated, and one held out.

On 24 same-appearance held-out rows, the CUDA MLP's weighted MSE was
**0.57824** with the correct action, **0.58636** for a separately trained
state-only MLP, **0.60770** with the paired alternative action, and **1.65800**
for an unchanged screen. This is a small action signal for two navigation
choices in one app, not broad GUI understanding. A later 16-row, eight-pair
dark-theme holdout used the restricted collector and a frozen light-trained
model: correct-action error was **0.02911**, worse than the **0.00599**
unchanged-screen baseline. This small same-app sample exposes an
appearance-shift failure, not a general performance estimate. An initial
larger run was stopped before artifact creation when C: free space fell below
0.5 GiB; a guarded bounded rerun produced the 16-row holdout. The
[evidence record](../daemon/benchmarks/world_heliox_ui_navigation_probe_v1.json)
pins the local feature, report, and weight hashes. These research weights are
not compatible with or loaded by the live `UiJepaPredictor`; no promotion or
Jev-parity claim follows from this experiment.

The original spatial research model placed click coordinates only in a small
local hotspot. Its shallow convolutional stack could not convey that location
to distant screen regions. A separate `spatial_global` variant also broadcasts
normalized x/y across the feature map. On the same 24 light-theme Heliox eval
rows, correct-action weighted MSE fell from **0.59339** to **0.10659**; swapping
the two actions from the exact same starting screen raised it to **2.35744**.
A frozen model on a second seed of the controlled pages scored **0.00313**
correct versus **0.07256** with the paired alternative action. However, the
Heliox dark-theme transfer remained a severe failure: **2.22222** correct
versus **0.00599** for leaving the screen unchanged. A coordinate-only linear
head, paired-ranking loss, and photometric augmentation were also tried as
development diagnostics and did not repair this failure. The
[global-action evidence record](../daemon/benchmarks/world_visual_global_action_probe_v1.json)
pins the CUDA reports and local weight hashes. This is a narrow, selected
research improvement, not a live model, robust theme transfer, or Jev parity.

To repeat with fresh ignored output paths, start the local Vite UI with
`tauri-app/ui/node_modules/.bin/vite.cmd --host 127.0.0.1 --port 1420`, then
from `daemon` run `python -m scripts.collect_heliox_ui_transitions --help`,
`python -m scripts.train_world_visual_transition --help`, and
`python -m scripts.probe_world_visual_model_transfer --help`. The optional
`--eval-dark` makes the last viewport family a dark-theme holdout. Keep the
browser run bounded and verify free disk space before collecting.

An inverse probe tests whether these 16×16 features retain information about
the executed action. On held-out controlled pages it reached 0.80 action-type
accuracy versus a 0.50 majority baseline; click coordinate mean absolute
error was 0.1298 versus a 0.2334 center baseline. On the public TESS sample
it reached only 0.7616 accuracy versus a 0.7815 majority baseline, with
0.3040 macro recall. This is a controlled representation check, not desktop
action understanding or joint forward/inverse training. Reproduce with
`python -m scripts.probe_world_visual_inverse --help`; keep resulting weights
and reports in ignored `.bench-state` paths.

An opt-in local recorder can collect after/before 16×16 pairs around approved
Heliox mouse actions. Its explicit offline trainer holds decrypted features
in memory, splits by whole recording, evaluates unchanged and shuffled-action
controls, and encrypts its derived weights. Recording is off by default; no
real-desktop dataset or promotion result is claimed yet. See
[Recorded workflows](RECORDED_WORKFLOWS.md).

Reproduce from `daemon/` with fresh ignored paths:

```bash
python -m scripts.collect_controlled_ui_transitions --samples-per-family 80 --seed 20260926 --features benchmarks/.bench-state/controlled_ui_repro.npz --report benchmarks/.bench-state/controlled_ui_repro.json
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --features benchmarks/.bench-state/controlled_ui_repro.npz --source-report benchmarks/.bench-state/controlled_ui_repro.json --report benchmarks/.bench-state/controlled_ui_model_repro.json --weights benchmarks/.bench-state/controlled_ui_model_repro.pt
python -m scripts.collect_controlled_ui_transitions --paired-counterfactuals --samples-per-family 20 --seed 20260926 --features benchmarks/.bench-state/controlled_pairs_repro.npz --report benchmarks/.bench-state/controlled_pairs_repro.json
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --features benchmarks/.bench-state/controlled_pairs_repro.npz --source-report benchmarks/.bench-state/controlled_pairs_repro.json --report benchmarks/.bench-state/controlled_pairs_action_repro.json --weights benchmarks/.bench-state/controlled_pairs_action_repro.pt
python -m scripts.train_world_visual_transition --device cuda --architecture spatial --disable-cudnn --train-without-action --features benchmarks/.bench-state/controlled_pairs_repro.npz --source-report benchmarks/.bench-state/controlled_pairs_repro.json --report benchmarks/.bench-state/controlled_pairs_state_only_repro.json --weights benchmarks/.bench-state/controlled_pairs_state_only_repro.pt
```

The shipped structured predictor remains available without learned weights.
It now identifies the destination (not source or download URL) as the changed
file for copy/download, models moves as separate source-removal and
destination-creation effects, and leaves post-command process count explicitly
unobserved. A successful file-copy callback is independently verified only
when the verifier compares file contents within a 64 MiB read budget. Directory
and larger-file copies can still be reported successful, but their existence
check alone is not a verified outcome or an online-training label.
Browser navigation now produces an online transition label only when the
active page's URL exactly matches the requested URL after execution. For
side-effect-free typing, selection, and unsubmitted form fill, the verifier
requires unique visible fields with matching values. Redirects, submitted
forms, Enter-key actions, unavailable sessions, and explicit non-default tab
targets remain unverified; executor success alone cannot train the learner.
These checks establish only the stated browser state, not downstream web-app
effects or general browser-task success.
An optional UI-JEPA artifact loads only when finite tensor shapes and its exact
ordered `ActionType` vocabulary match the running build. Gating additionally
requires at least 100 training samples, validation error no greater than 0.35,
and an explicit validated-for-gating designation. Otherwise it is unavailable
or shadow-only.

Offline world-model trials report Brier score, calibration error, risk false
negatives, transition-match coverage and mean, high-uncertainty rate, and
median/p95 prediction latency. These metrics deliberately pair a pre-action
prediction with post-action observation; inference speed or a model flag alone
does not establish transition accuracy.

[V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) motivates
action-conditioned latent prediction, while
[VisualWebArena](https://aclanthology.org/2024.acl-long.50/) is a candidate for
web-transition evaluation. Neither is a drop-in Heliox model, and neither is
claimed as shipped training data. Any future artifact must retain dataset and
license provenance, environment-disjoint validation, the exact action
vocabulary, and the frozen gating metrics above.

## Review order

1. Validate data and artifact provenance.
2. Run deterministic contract and parser tests.
3. Run offline, environment- or subject-disjoint evaluation.
4. Keep learned candidates in shadow mode.
5. Perform consented human/hardware or isolated-environment testing.
6. Promote only an exact reviewed artifact; never infer promotion from a
   model's self-reported success.
