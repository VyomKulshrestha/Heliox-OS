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

## Scope-advisory research and current state

The earlier Jev-based scope experiments are historical research, not a
repeatable training recipe or evidence of a current model. Under the current
[TypeSafe Master Customer Agreement, section 2.3(b)](https://typesafe.ai/legal/mca),
using provider outputs to distill, imitate, or develop a similar or competing
service requires written authorization. Until that authorization is verified,
do not make new Jev calls or reuse those outputs for training, tuning,
benchmark comparisons, or product development. The detailed
[historical study](JEV_SCOPE_STUDY.md) is retained for audit and is not a
current product claim.

The runtime no longer loads the historical output-derived scope checkpoint;
the old v1 artifact is incompatible with the current model version, and
`scripts/train_scope_decision.py` is a refusal-only entry point. Existing
deterministic safety rules, user confirmations, and the companion review remain
authoritative. The scope advisory cannot grant permission or replace those
controls.

An independent GPU experiment used only the MIT-licensed
[ScopeJudge dataset](https://huggingface.co/datasets/dreadnode/scopejudge) and
frozen Qwen3.5-2B features. On its 587-row trajectory-disjoint public test
partition, the default threshold had AUC 0.855, 7 false allowances, and 104
false warnings. A threshold selected to produce zero false allowances on the
74-row calibration split still produced 4 false allowances and 197 false
warnings on test. The result is **not promoted**: ScopeJudge covers offensive
security, not everyday desktop tasks, and this partition was already examined
in earlier adaptive experiments. See the
[full hashed result](../daemon/benchmarks/public_scope_head_gpu_v1.json) and
[trainer](../daemon/scripts/train_public_scope_head.py). It is not a blind
desktop-agent acceptance test, live approval evidence, or a permission gate.

## Independent cowork-agent trajectory benchmark

[ActBench](https://github.com/ZJUICSR/ActBench) is a promising separate
evaluation for Heliox's tool-using and multi-agent behavior. Its MIT-licensed
repository describes 300 matched benign/adversarial pairs (600 cases) and
scores utility and realized prohibited effects from execution trajectories,
not final answers alone; see its [paper](https://arxiv.org/abs/2608.09476).
This can test prompt-injection resistance, tool-scope escalation, and whether
multi-step execution preserves the intended task boundary. It is not a direct
desktop-vision, voice, or world-model benchmark, and its trajectory-level
outcomes are not per-action scope labels for training the scope head.

Use only the repository's MIT-covered benchmark code and task fixtures unless
the separate [Hugging Face trajectory dataset](https://huggingface.co/datasets/ZJUICSR/ActBench)
license is independently verified. That dataset's license metadata could not
be verified during this review, so its trajectories are not approved here for
training or redistribution. ActBench has not yet been integrated with or run
against Heliox; a Heliox adapter and isolated mock-service execution are still
needed before reporting benchmark results.

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

A separate [compact option-ranker probe](../daemon/benchmarks/jevlike_typed_decision_v2.md)
uses only verified-CC0 Open-Jev rows and a tiny listwise byte model. Across two
1,000-row OOD samples, full-context shuffling reduced top-1 by 12.1 points,
while choice accuracy was 33.6–42.9% against 25.6–25.8% expected random choice;
10-bin ECE remained 9.2–10.4%. It is not loaded by Heliox and provides no
evidence of live desktop or multi-step transfer. This follows the independent
Open-Jev pattern of ranking supplied choices in one pass; it does not reproduce
Jev's private model or method. A scorer-only microbenchmark measured roughly
1.1 ms median on both CPU and CUDA, excluding Heliox's RPC and approval loop.

An audit found that the earlier row-sampled `ood` partition shared source
categories with training. Those results remain row-held-out diagnostics, not
source-generalization evidence. `scripts/train_general_decision.py` now
defaults to a deterministic 20% source-category holdout: it selects only
license-eligible categories present in every frozen split, excludes those
categories from train/calibration/test, and restricts OOD evaluation to them.
The report binds the holdout seed, category lists, per-split source filters,
and parquet hashes. The loader validates metadata in bounded batches, then
fetches only the seeded, source/kind-balanced row IDs in bounded batches;
runtime prompts still omit source names. Use
`--source-holdout-fraction 0` only to reproduce the older row-held-out
protocol. `--source-holdout-seed` selects source folds independently from
`--split-seed`, which fixes sampled rows within each fold. A source-heldout
score is still a small public-data probe, not Jev
parity or evidence that the daemon can complete real desktop tasks.
Before importing the model or allocating CUDA state, the trainer also checks
available host RAM (2.5 GiB by default) to avoid starting a fine-tune while the
workstation is paging; adjust `--minimum-free-host-gib` only when the machine's
remaining memory is understood.

The optional `--general-group-weighting inverse_sqrt` or `inverse` setting is
an **unproven ablation** for repeated episodes within each source/decision-kind
bucket. It reweights rows by episode frequency and renormalizes each bucket to
its original total loss mass; the default `none` preserves the baseline. A
weighting mode must be compared on the same frozen source-heldout folds before
it is considered an improvement. Repeated rows remain within their existing
group-disjoint split; weighting does not create additional independent
episodes or increase the evidence count.

The first matched 4k leave-one-source-out trial found no useful transfer:
held-out choice accuracy was below its random baseline in all three folds,
yes/no balanced accuracy was 50.0–54.1%, and OOD NLL was worse than uniform
in every fold. The scratch byte scorer is rejected for general decision use
and remains disconnected from runtime. Detailed fold metrics and the 8k
scaling check are in
[`jevlike_typed_decision_v2.md`](../daemon/benchmarks/jevlike_typed_decision_v2.md).

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
effects, and competing external events are absent. These button-toggle
checkpoints do not implement the paper's joint forward/inverse dynamics
objective or its larger multimodal backbone. Both trained artifacts remain
ignored local research files.

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

A subsequent development run added a separate dark-theme fit/validation
archive while keeping the primary light evaluation partition isolated. A
per-example transition-scale loss was tested at powers 0, 0.5, 0.75, and 1;
0.75 was selected after inspecting the earlier same-app diagnostics. With
weights then frozen, two newly captured 20-pair sets gave correct-action
weighted MSE of **0.34786** (light) and **0.00352** (dark), versus unchanged
baselines of **1.65705** and **0.00601**. The paired alternative action was
worse in **20/20** groups for each appearance, at **2.01022** and **0.00414**
respectively. The [mixed-appearance evidence](../daemon/benchmarks/world_visual_multitheme_probe_v1.json)
pins data, model, and transfer-report hashes. The two actions, one app, shared
layout family, and tiny dark error margin make this a bounded development
check—not independent GUI transfer, a calibrated execution gate, or Jev parity.
No learned weights were promoted.

The same global-coordinate architecture was also trained on the existing
pinned public corpora without changing their splits. On TESS AgentNet's 151
evaluation rows, correct-action weighted MSE was **0.10912**, while shuffling
valid actions improved it to **0.10789**. On 87 already-inspected GUI-CC RGB32
rows, correct taps scored **0.31284** versus **0.31626** with locations
shuffled, but the rank among 30 shuffles was only **0.29**. The
[public-corpus check](../daemon/benchmarks/world_visual_public_transfer_v1.json)
pins both local reports and weights. Thus the local paired-page gain does not
establish broad action conditioning: the public data either fail or weakly
support the action-shuffle control, and lack verified same-screen alternatives.

After the GPU update, a new-seed CUDA fit on the same pinned TESS split still
failed the action-shuffle control: correct-action weighted MSE was **0.11325**
versus **0.11105** for shuffled actions on 151 held-out episode rows. This
confirms the training path runs, not that the model understands the action.
The [updated-GPU recheck](../daemon/benchmarks/world_visual_updated_gpu_recheck_v2.json)
pins the ignored local report and weight hashes; no weights were promoted.

The [resolution and fresh-seed MiniWoB comparison](../daemon/benchmarks/world_visual_resolution_transfer_v1.json)
tested 16×16 and 64×64 captures on identical selected click pairs, with
task-family splits and 24 new-seed evaluation pairs. At 64×64, the default
model's correct-action weighted MSE was 0.412518, but swapping in the paired
alternative improved it to 0.411459. The pairwise-trained model narrowly
favored correct actions, 0.553221 versus 0.553224, only a 0.000003 gap.
Resolution improves pixel fit but does not establish transferable action
grounding. All of these GPU weights remain research-only.
An additional [linear geometry feasibility check](../daemon/benchmarks/world_visual_linear_feasibility_v1.json)
scored 0.584081 correct-action weighted MSE versus 0.548813 for simply leaving
the held-out screen unchanged. It is not the live artifact format, but rejects
the shortcut of treating coordinate inputs plus a small linear fit as a
validated world model.

Following the action-transition emphasis in the
[Computer-Using World Model paper](https://arxiv.org/abs/2602.17365) and
[State Transition Pretraining code](https://github.com/xyliugo/gui-state-transition-pretraining),
the [diverse-controls probe](../daemon/benchmarks/world_visual_diverse_controls_v1.json)
expanded MiniWoB collection from native form controls to visible pointer
affordances, adding four task families and 93 verified training pairs. A
CUDA fit lowered held-out pixel error but still scored a wrong paired click
better than the correct one (0.387778 versus 0.389241 weighted MSE). A
pairwise objective reversed that ordering only by about 0.000002 on fresh
seeds, too small for promotion. Inverse localization gained only modestly
from the after-screen, and all sampled actions had the same click type.
This remains small synthetic evidence, not a reproduction of either paper.

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

A separate opt-in `spatial_joint` research model shares a visual encoder
between forward next-screen prediction and inverse observed-click localization.
On the same MiniWoB split and GPU seed, inverse fitting reduced fresh-seed
coordinate error to 0.135 from the 0.208 center baseline, but the forward
model still did not reliably favor the correct click over a paired wrong click
(14/24 fresh pairs in the first split; 17/24 after adding the larger training
split, but the mean margin remained below one-millionth weighted MSE).
It is **not** promoted to runtime.
In the paired holdout, the median action-specific difference spans only 5.5
of 256 features at 16×16, or 67.5 of 4096 at 64×64; whole-frame loss is
dominated by shared screen changes. More inverse fitting or resolution alone
does not repair that representation bottleneck.
The matched controls, source hashes, and limitations are in
[`world_visual_joint_inverse_v1.json`](../daemon/benchmarks/world_visual_joint_inverse_v1.json).
Use `--architecture spatial_joint --inverse-loss-weight 1` with the visual
trainer to reproduce the research mode; weight 0 is the matched forward-only
control. Checkpoint selection uses the forward-side validation objective
(including paired-outcome loss), without the inverse loss.

Because paired outcomes differ on very few pixels, an optional
`--pairwise-focus-weight` research loss fits only the pixels where independently
observed paired outcomes disagree. The candidate and pass gate were
[frozen before new seed-200 collection](../daemon/benchmarks/world_visual_focus_seed200_protocol_v1.json).
On those 44 new-seed pairs, inverse-assisted training preferred the correct
click on 42/44 action-differentiating regions, versus 36/44 for the matched
forward-only model. It still preferred the correct click on only 24/44 whole
frames, and its mean whole-frame margin favored the wrong action. The
[frozen result](../daemon/benchmarks/world_visual_focus_seed200_results_v1.json)
fails the overall gate; no weights are enabled in Heliox. This is a useful
representation diagnostic, not a Jev-equivalent world model.

A further research-only probe, inspired by the latent-displacement inverse
objective in [Delta-JEPA](https://arxiv.org/html/2606.31232v1), predicts a
small next-screen latent and decodes observed click coordinates from the
before/after latent difference. It is a **small local adaptation**, not an
implementation or reproduction of that paper's model. A matched variant adds
an observation-reconstruction loss to discourage latent collapse. The probe
added 182 verified counterfactual training pairs from eight more MiniWoB task
families, kept the original 48-pair task-family eval isolated, and scored
frozen weights on two previously collected seed archives plus the additional
archive's excluded eval family. On the retained primary eval, the unanchored
model preferred the correct action on **23/48** pairs and the reconstruction
variant on **28/48**; both had a **negative mean correct-action advantage**.
The reconstruction variant reached 30/44 on the reused seed-200 archive but
predicted pair separation was only about **0.02%** of observed separation in
latent MSE. The cause was a self-targeting forward loss: the encoder could
shrink the target representation instead of learning the transition. Stopping
gradients through the forward target produced **42/48** on the primary eval,
**22/24** on reused seed 100, **38/44** on reused seed 200, and **21/22** on
the excluded eval family, each with a positive mean latent-MSE advantage.
An untrained, zero-initialized forward head tied all primary-eval pairs; the
improvement was learned, but remains exploratory. The selected checkpoint
was at epoch 1, so it is especially important not to infer robust planning.
These comparisons are not a new preregistered confirmation or task-success
benchmark. Hashes, controls, and transfer results are in
[`world_visual_delta_latent_exploratory_v1.json`](../daemon/benchmarks/world_visual_delta_latent_exploratory_v1.json).
Before collecting new data, the candidate and a minimum 30-pair, 80%-win,
positive-margin unseen-family gate were frozen in
[`world_visual_delta_stopgrad_seed700_protocol_v1.json`](../daemon/benchmarks/world_visual_delta_stopgrad_seed700_protocol_v1.json).
That gate **failed**: three of four requested task families yielded no
verified distinct-outcome pairs, and on the 11 valid `click-button` pairs the
model preferred the correct action only **4/11** times, with a negative mean
margin. Its predicted pair separation was far below the observed separation.
An ensuing training-source audit found that `click-button` was already a fit
task for this frozen model. The prior report's **unseen-family** label for
those 11 pairs is incorrect; the frozen report remains preserved with this
explicit erratum.
The [frozen result](../daemon/benchmarks/world_visual_delta_stopgrad_seed700_results_v1.json)
therefore fails its prospective gate; that collection cannot establish
unseen-family transfer because the other three families had no valid pairs.
An exploratory 64×64 follow-up used the same small objective on a separately
collected public split. On 12 `click-button` pairs, it reached **8/12**
correct-action wins, but the mean margin still favored the wrong action and
predicted pair effects remained far below observed effects. See the
[resolution check](../daemon/benchmarks/world_visual_resolution64_exploratory_v1.json).
Because seeds and sample counts differ from the 16×16 run, this is not a
controlled estimate of resolution's effect; it only shows that this larger
feature grid did not solve the failure in that run.
The public browser-feature archives and weights remain ignored local files;
the paper-inspired model has no live loader. From `daemon/`, use
`python -m scripts.train_world_visual_delta_latent --help` and
`python -m scripts.probe_world_visual_delta_latent_transfer --help` to inspect
the bounded GPU protocol. Hash checks reject mismatched source features or
model weights. The failed transfer evidence blocks runtime promotion; the
current 16×16 representation and one-step click-only supervision do not
establish Jev-like multistep planning or general desktop transfer.

The follow-up [branch coverage and transfer audit](../daemon/benchmarks/world_branch_transfer_seed700_results_v1.json)
expanded valid evaluation coverage to 47 pairs across four families using
painted-control and neutral-action candidates. Its frozen collection gate
still failed because one reset was unstable. The old model won 27/47 pair
rankings with a negative mean margin, far below its frozen transfer gate.
Per-family observed pixel separation shows why the total is misleading:
`click-color` and `click-dialog` supplied 23/23 wins but only about 0.00003 to
0.00004 mean full-frame pair MSE, whereas the visibly divergent `click-test`
family was 0/12 (about 0.217 pair MSE). The already-seen `click-button` family
was 4/12. A matched two-seed 200 ms post-click delay left all four families'
observed pair MSE unchanged, so that experimental collector option was removed.

[Dream-RSI](https://arxiv.org/abs/2609.14858) motivates replaying recorded
exploration to choose better experiments without repeatedly spending online
evaluations; it leaves the underlying coding agent unchanged. Here replay
informed data-coverage and scoring choices, **not** a claim that replay trained
the Heliox world model or that its discovered policy transfers to live desktop
tasks. The next model study needs task-disjoint, materially differentiable
transitions and a fresh frozen evaluation before any promotion discussion.

A separate [observed-outcome ranker study](../daemon/benchmarks/miniwob_reward_ranker_result_v1.md)
fit a paired head on frozen Qwen3-0.6B features from public MiniWoB tasks.
Its corrected single-row CUDA evaluation won **21/33** opposing-reward pairs,
versus **19/33** for randomized first and **33/33** for either simple
goal-text policy. One prelisted evaluation page supplied no eligible pairs.
An attempted padded feature extractor was rejected after substantial
batch-versus-single embedding disagreement; the held-out pages were already
seen before the corrected extraction, so this is exploratory rather than an
independent model win. The coverage and performance gate failed; no ranker
weights were loaded into Heliox.

That targeted [material-change study](../daemon/benchmarks/world_material_branch_results_v1.json)
used three training, one validation, and three evaluation MiniWoB families
selected after a two-seed exploratory signal scan. An action-conditioned
spatial model trained on CUDA ranked the correct branch **24/24** times on
seed 1000 and repeated **24/24** with unchanged weights on predeclared fresh
seed 1100. Its fresh-seed weighted error was 0.200 versus 0.379 for the
matched no-action model and 0.525 for the unchanged screen. All three selected
evaluation families had positive action margins and stable resets.

Visual review then exposed a shortcut: representative pairs contrast the
MiniWoB dark terminal overlay with an active task page. The high score is
therefore evidence of synthetic terminal-state discrimination, **not** a
general transition model. The collector now records termination metadata and
has an opt-in `--nonterminal-only` check. On a small new two-seed nonterminal
probe, only four evaluation pairs remained, the frozen model won **2/4**, and
its weighted error was 0.148 versus 0.00028 for leaving the screen unchanged.
The numeric protocol gates passed, but this external-validity failure blocks
promotion. No research weights were connected to Heliox's live world model.

An exploratory, application-disjoint [GUI-360 Office protocol](../daemon/benchmarks/gui360_office_nonterminal_protocol_v1.json)
then tested nonterminal desktop transitions instead of synthetic terminal
overlays. The [original GUI-360 dataset](https://huggingface.co/datasets/vyokky/GUI-360)
and [processed Office shards](https://huggingface.co/datasets/cua-lite/GUI-360)
provided 105 Excel training clicks, 111 Word validation clicks, and 123
PowerPoint evaluation clicks. The importer requires one aligned computer click,
a following screenshot, and a later action; it rejects mixed application
splits, duplicate trajectory IDs, and identical pre-action screens across
splits. The source images remain in the ignored local benchmark directory.
This is an observational, one-shard-per-application probe, not a controlled
counterfactual dataset or a representative live Heliox test.

On the held-out PowerPoint rows, the action-conditioned spatial model's
weighted error was **0.06429** versus **0.08183** for leaving the screen
unchanged. But a matched model trained without actions scored **0.06377**,
and shuffling click locations gave the action model **0.06447** with a 0.23
one-sided shuffle rank. It therefore failed the action-grounding control;
the apparent screen-prediction gain cannot be attributed to learning the
effect of the click. The [frozen result summary](../daemon/benchmarks/gui360_office_probe_v1.json)
records source, feature, report, and weight hashes. No weights were promoted.
The [Computer-Using World Model paper](https://arxiv.org/html/2602.17365v1)
motivates separating semantic action effects from visual realization, but
that approach has not been implemented or validated in Heliox.

To reproduce from `daemon/` with the optional `pyarrow` package, download
only the three protocol-listed files from the pinned `cua-lite/GUI-360`
revision into an ignored local directory,
verify their hashes, then use fresh ignored output paths:

```bash
python -m scripts.import_gui360_office_transitions --root benchmarks/.bench-state/gui360_office_v1 --protocol benchmarks/gui360_office_nonterminal_protocol_v1.json --features benchmarks/.bench-state/gui360_office_v1_features.npz --report benchmarks/.bench-state/gui360_office_v1_import.json
python -m scripts.train_world_visual_transition --features benchmarks/.bench-state/gui360_office_v1_features.npz --source-report benchmarks/.bench-state/gui360_office_v1_import.json --report benchmarks/.bench-state/gui360_office_v1_candidate_cudnn.json --weights benchmarks/.bench-state/gui360_office_v1_candidate_cudnn.pt --architecture spatial_global --epochs 30 --seed 20260927 --action-permutations 99 --device cuda
python -m scripts.train_world_visual_transition --features benchmarks/.bench-state/gui360_office_v1_features.npz --source-report benchmarks/.bench-state/gui360_office_v1_import.json --report benchmarks/.bench-state/gui360_office_v1_state_only.json --weights benchmarks/.bench-state/gui360_office_v1_state_only.pt --architecture spatial_global --train-without-action --epochs 30 --seed 20260927 --action-permutations 1 --device cuda
```

An opt-in local recorder can collect before/after 16×16 pairs around approved
Heliox mouse actions and consented manual clicks. Manual capture keeps only a
rolling in-memory pre-feature while both manual and visual options are enabled;
stale or interrupted pairs are excluded. Its explicit offline trainer holds
decrypted features
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
For a multi-action plan, RiskGate now carries each predicted UI and browser
state into the next prediction rather than reusing the first observed screen.
An optional UI-JEPA predictor keeps its next latent in memory for that rollout;
the serialized trace omits the latent and labels later steps as predicted,
not observed. Without loaded visual weights, the structured predictor drops
the stale observed latent after the first step instead of pretending it is a
new frame. This is speculative planning only: execution, post-action
observation, deterministic risk, and approval remain separate. The research
visual weights discussed above are not promoted into gating. An explicit
export from the opt-in encrypted workflow store can produce a
`clipped-residual-v1` artifact that the daemon loads from its data directory;
that export always remains shadow-only and is never marked validated for
gating. Legacy and exported artifacts load only when tensor shapes, geometry
version, and the exact ordered `ActionType` vocabulary match the running build.
Gating additionally requires at least 100 training samples, validation error
no greater than 0.35, and an explicit validated-for-gating designation. It
abstains on unresolved `(0,0)` click targets, missing or out-of-bounds screen
coordinates, and relative coordinates. Only explicit pointer actions can
receive learned gating evidence; non-pointer visual predictions remain
shadow-only. No reviewed geometry artifact is staged.

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

## User-rated workflow outcome candidate

The explicit feedback on a saved workflow is also available to an offline,
research-only outcome probe. It uses only coarse action-family counts,
placeholder/incomplete indicators, and the user's `achieved` / `partial` /
`not_achieved` rating; goals, names, action parameters, screens, and executor
status are excluded. Identical action-type recipes stay in one evaluation group.
The encrypted candidate has no planner, approval, or execution integration.

Run it locally only after enough consented recordings and ratings exist:

```powershell
uv run --project daemon python -m scripts.train_recorded_workflow_outcome
```

The trainer requires at least 24 rated runs across 8 independent recipe groups,
uses a group-disjoint holdout, encrypts model weights with the recording
store's OS-keyring key, and writes an aggregate report under the Heliox data
directory. It refuses to lower these sample gates or overwrite existing
artifacts. Every observed outcome class must occur in both train and test
groups; reports include per-class recall and test coverage so partial-class
holdouts cannot inflate balanced accuracy. Tests use synthetic fixtures only;
they are not user training evidence. This is not Jev imitation, parity, or
action authorization.
