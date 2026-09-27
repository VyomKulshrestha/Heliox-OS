# Nonterminal MiniWoB branch probe: frozen new-seed retest

This is **research-only synthetic browser evidence**, not a Heliox runtime model or a claim of Jev parity. The pinned source is [Farama MiniWoB++](https://github.com/Farama-Foundation/MiniWoB-plusplus) at `33c3b4ddef8c6eb67c57a29663d844b1eda7e614` (MIT). The collector independently resets each branch, requires identical measured prestates, excludes terminal overlays, and now requires distinct visible task text/control states as well as distinct 16×16 image features. The first scout had admitted focus-only screenshot differences; that failure motivated the general collector filter.

## Selection and coverage

- [Seed-3000 protocol](miniwob_semantic_branch_seed3000_protocol_v1.json) was committed before its collection. It had 40 train, 40 validation, and 20 eval pairs. However, both `click-tab` and `click-collapsible` produced zero qualifying eval pairs; all 20 came from `click-checkboxes-transfer`. This is a coverage failure for broad UI transfer.
- The originally registered `spatial_global` action model was selected at epoch 2. On seed-3000 eval, weighted MSE was `0.000416409` versus `0.000419331` for its matched state-only control, but the correct action won **0/20** paired comparisons. It failed the registered action-grounding criterion. Train and validation correct-action wins were only `8/40` and `7/40`, respectively.
- A subsequent exploratory pairwise-focused run at weight `1.0` improved action sensitivity but worsened eval weighted MSE to `0.005604439`. A smaller, post-hoc weight `0.001` yielded `0.000217595` and `20/20` paired wins on that already-inspected eval page. Those seed-3000 numbers are **not confirmatory**.
- The `0.001` checkpoint and state-only control were then frozen by hash in the [seed-3200 transfer protocol](miniwob_semantic_branch_seed3200_transfer_protocol_v1.json) before new data collection. No fitting or checkpoint selection used seed-3200 observations. Both eval pages yielded 20 nonterminal, semantic-changing pairs; no unstable resets were accepted.

## Frozen seed-3200 result

Weighted MSE (lower is better) and paired correct-action wins:

| Eval page | Frozen action model | Frozen state-only | Paired alternative action | Correct-action wins |
| --- | ---: | ---: | ---: | ---: |
| `click-checkboxes-large` | 0.000231600 | 0.000418373 | 0.000307306 | 20/20 |
| `click-checkboxes-transfer` | 0.000216879 | 0.000419105 | 0.000293830 | 20/20 |

The frozen candidate meets the **narrow** registered criterion on these two checkbox variants. Aggregate weighted MSE is `0.000224240` versus `0.000418739` state-only and `0.000446907` unchanged-screen; correct click wins `40/40` paired comparisons. The feature archive SHA-256 is `ac2f631dbf8dd0bce9b2b62154cb30c38ceb70b205e24049a719152094496719`. Checkpoint and report hashes are in the seed-3200 protocol; full local reports and weights remain under ignored `benchmarks/.bench-state/`.

This is **same-family checkbox transfer**, not general UI understanding: one eval page was already used in exploration, and the other is a related page variant. Visual changes cover only about 1.56% of pixels on the seed-3200 eval set. It does not establish semantic goal completion, multistep planning, desktop transfer, safety, or behavior comparable to Jev. The checkpoint is not loadable by the live `UiJepaPredictor` and was **not promoted**. The next gate is a genuinely different non-checkbox interaction family with qualifying branch coverage, followed by a separately consented real-desktop evaluation; no private screen capture was used here.

Reproduce collection with `python -m scripts.collect_miniwob_branches` using the exact task lists, seed range, and flags in the two protocol JSON files. Score a frozen checkpoint with `python -m scripts.probe_world_visual_model_transfer --model-report ... --weights ... --target-features ... --target-report ... --report ... --device cuda`. The local reports preserve source-asset hashes, feature hashes, checkpoint hashes, per-page errors, and paired diagnostics.
