# Exact branch replay: research-only result

The [Dream-RSI paper](https://arxiv.org/abs/2609.14858) improves an
exploration policy by replaying branches and outcomes that were actually
observed. This probe applies that **replay principle**, not the paper's full
recursive, long-horizon algorithm: each independently reset public MiniWoB
state has two recorded click outcomes. Policies choose one without executing
it again. Missing rewards and non-contrasting pairs are excluded rather than
predicted. This is not a desktop trial or reference-model comparison.

| Frozen run | Eligible validation | Eligible eval | Fixed-first eval | Selected eval | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| [v1, seed 6500](dream_replay_protocol_v1.json) | 12 | 7 | 3/7 | 5/7 | Failed coverage |
| [v1, seed 6600](dream_replay_protocol_v1.json) | 29 | 23 | 10/23 | 11/23 | Failed improvement |
| [v2, seed 6800](dream_replay_protocol_v2.json) | 22 | 40 | 24/40 | 30/40 | Passed narrow replay gate |

The v2 winner, `goal_grounded`, scored 22/22 on validation and 30/40 on
task-disjoint evaluation. Its improvement over fixed-first came from
`click-link` (10/10 versus 4/10); it tied on `click-button`, `click-color`,
and `click-shape`. Critically, the older label-only `goal_match` policy also
scored **30/40** on evaluation. Generic DOM control type resolved one widget
validation ambiguity but provided **no incremental held-out gain** here.
The new feature is therefore not validated as a transferable improvement.

The v2 archive has 78 evaluation reset groups, but only 40 have two observed,
opposite terminal rewards. Across all splits, 36 groups lacked reward contrast
and 40 groups lacked two valid rewards. `click-color` stayed at 5/10 and
`click-shape` at 13/18 for both fixed-first and the selected policy. Those
visual-target failures cannot be solved by matching labels or DOM kind alone.
No model weights were trained or promoted. Exact replay itself uses recorded
outcomes and does not require GPU inference; the separate CUDA visual-model
experiments remain failed and unpromoted.
The collector selects the first two actions with distinct 16×16 next-screen
features, so these are not randomly sampled desktop actions. The 40 eligible
evaluation pairs come from only four task pages, and the six-win lift is
concentrated on one page; a row-level confidence interval would overstate
independent evidence.

Local generated archives and detailed JSON reports are ignored research
artifacts under `benchmarks/.bench-state/`. Provenance SHA-256 values:

- Seed 6500 semantic archive: `585fa72be8565c1f1845ab49fe91e139ccb8ec02ad8d0f7b9edd765a12ed0062`
- Seed 6600 semantic archive: `add0a99bd7ab5401fc5b7f71a355d346b52d7c6dbb7d34a2c2d3a902be864bef`
- Seed 6800 semantic archive: `6f1df60048785b11ba0f6d3d25fb97292856bd45681f690a90636ae383263ed8`

The next model test needs grounded visual target identity and outcomes for
multi-step goals, with new task-family and real opt-in desktop validation.
Historical replay cannot answer what an unobserved action would have done,
and no synthetic gate can replace the user approval or Agent Gateway checks.
