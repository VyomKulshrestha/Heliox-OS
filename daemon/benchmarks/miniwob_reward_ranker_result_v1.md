# One-step outcome ranker: failed transfer and coverage gate

This is an offline study of **public synthetic MiniWoB pages**, not a Heliox
runtime model or evidence of reference-model parity. The task-page split, seed, candidate
regularization values, and per-page success criterion in the
[protocol](miniwob_reward_ranker_protocol_v1.json) were chosen before
collection. The feature-extraction batch setting was corrected **after** an
initial held-out score was seen (details below). The corrected score is thus
an exploratory re-evaluation, not a pristine blind holdout.

The pinned MIT MiniWoB source produced 116 train, 60 validation, and 76
evaluation same-prestate action pairs. Only pairs with two observed, opposing
terminal rewards were usable as outcome labels: **64/43/33** respectively.
The visual feature archive SHA-256 is
`346efba11a300ce4510317b3abf028b905e7edf486d3ee93d97839d70c4a3ff9`;
the opt-in public-page text archive SHA-256 is
`6f2d87176d18a3cb287819a3d098ab048d4eb6970af8265c050a3346fa78081b`.
Neither archive contains user desktop content or reference-model outputs.

A frozen local Qwen3-0.6B model extracted each visible-state/candidate-action
feature separately on CUDA (504 rows, at most 1,015 prompt tokens, 1.587 GiB
peak PyTorch allocation). No post-action state or reward enters its prompt.
The feature SHA-256 is
`f676338db1b6ed1ee199e120460f16004eb3cfcc4591ed2d9e1394a99448d8b5`.
A no-intercept logistic ranker learned from paired feature differences;
validation selected `C=0.1` from the prelisted candidates. Research-only
weight SHA-256:
`1a7d18aecfbd2d4fc9d9c43181f7a48dc51bd70525f2a99735d6ff580c8c501f`.
The source feature, text, embedding, and weight files remain ignored local
artifacts; the tracked protocol SHA-256 at scoring was
`b7eff1a3b219b78a76f7abe5d7f42458e76041105e98c882196db3764b59d416`.

| Split | Learned ranker | Randomized first | Goal text match | Goal + control kind |
| --- | ---: | ---: | ---: | ---: |
| Validation | 32/43 | 22/43 | 37/43 | 42/43 |
| Evaluation | **21/33** | 19/33 | **33/33** | **33/33** |

On `click-test-transfer`, the ranker won 20/23 against 23/23 for either
goal-text policy. On `click-tab`, it won only 1/10 against 10/10. The third
prelisted evaluation page, `click-shades`, supplied **zero** opposing terminal
reward pairs, failing the coverage floor. Therefore the preregistered gate
fails both coverage and performance. The ranker is **not promoted**.

An attempted padded GPU batch extractor was rejected. Its first held-out
score was 18/33, but batch-versus-single embeddings on the same two prompts
differed by up to 1.28125 and 0.9375 (mean absolute differences 0.0945 and
0.0862). Those differences are not float16 rounding noise, so the batched
score is not comparable to the original feature path. The final extractor
uses one row at a time and now checkpoints source-bound features to recover
from intermittent CUDA faults. The observed cause of the batch discrepancy
is unresolved; batching is not an enabled option.

This study tests an observed one-step reward selector, **not** the textual
effect-then-visual-realization architecture in the
[Computer-Using World Model paper](https://arxiv.org/abs/2602.17365).
It cannot answer unseen counterfactuals, show long-horizon task completion,
establish real-desktop transfer, or authorize actions. The perfect goal-text
baseline on eligible evaluation pairs also makes these particular pages too
easy to support a reference-style capability claim. A stronger future protocol needs
diverse task goals and environment-disjoint, causally paired actions with
independent success verification.

From `daemon/`, reproduce with fresh ignored output paths:

```bash
python -m scripts.collect_miniwob_branches --train-tasks click-button,click-link,click-color,click-shape --validation-tasks click-test-2,click-widget --eval-tasks click-test-transfer,click-shades,click-tab --samples-per-task 30 --seed-start 7200 --max-candidates 12 --features benchmarks/.bench-state/miniwob_reward_ranker_v1.npz --report benchmarks/.bench-state/miniwob_reward_ranker_v1.json --semantic-archive benchmarks/.bench-state/miniwob_reward_ranker_v1.jsonl
python -m scripts.extract_miniwob_semantic_features --semantic-archive benchmarks/.bench-state/miniwob_reward_ranker_v1.jsonl --visual-features benchmarks/.bench-state/miniwob_reward_ranker_v1.npz --source-report benchmarks/.bench-state/miniwob_reward_ranker_v1.json --features benchmarks/.bench-state/miniwob_reward_ranker_qwen_single_v2.npz --report benchmarks/.bench-state/miniwob_reward_ranker_qwen_single_v2.json --max-length 1024 --checkpoint benchmarks/.bench-state/miniwob_reward_ranker_qwen_single_v2.checkpoint.npz
python -m scripts.evaluate_miniwob_reward_ranker --archive benchmarks/.bench-state/miniwob_reward_ranker_v1.jsonl --features benchmarks/.bench-state/miniwob_reward_ranker_qwen_single_v2.npz --source-report benchmarks/.bench-state/miniwob_reward_ranker_v1.json --feature-report benchmarks/.bench-state/miniwob_reward_ranker_qwen_single_v2.json --protocol benchmarks/miniwob_reward_ranker_protocol_v1.json --report benchmarks/.bench-state/miniwob_reward_ranker_result_single_v2.json --weights benchmarks/.bench-state/miniwob_reward_ranker_candidate_single_v2.npz
```
