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

The shipped structured predictor remains available without learned weights.
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
