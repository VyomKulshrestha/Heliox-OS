# Frozen semantic-action cross-site probe: failed gate

The [protocol](infiniteweb_semantic_transfer_protocol_v1.json) fixed the two previously untouched evaluation sites and all model choices before collection. The pinned MIT-licensed [InfiniteWeb dataset](https://huggingface.co/datasets/WhiteFlamesCN/InfiniteWeb-Dataset) yielded 23 train, 6 validation, and 10 evaluation pairs across site-disjoint partitions. The 16×16 visual archive SHA-256 is `0c442a696c5585b40c635df95c90513572d19337eebe0881ae06be677a033552`. The opt-in, local JSONL contained bounded visible text and clicked-control labels from these **public generated sites only**; its SHA-256 is recorded in the ignored collection report. No user desktop or Jev API output was captured.

The frozen local Qwen3-0.6B encoder extracted 78 aligned feature rows on CUDA (maximum 711 prompt tokens, 1.502 GiB peak model allocation; feature SHA-256 `6af7dec37fe17a69427773107ec80b796355c90c5b41ddd645aae791a1ef6a28`). The [GPU ridge probe](../scripts/probe_world_semantic_ridge.py) projected semantic features to 16 dimensions using **training rows only**, selected lambda from `[0.1, 1, 10]` on validation separately for each model, and scored the untouched sites once. All three selected lambda `10`. Semantic validation weighted MSE was `0.174406`, below coordinate-only `0.191726`, but that gain did not transfer.

| Held-out site | Model | Correct-action wins | Correct weighted MSE | Swapped-action MSE |
| --- | --- | ---: | ---: | ---: |
| Butoh dance workshop (4 pairs) | State-only | 0/4 ties | 0.159438 | 0.159438 |
|  | Coordinates | 1/4 | 0.157473 | 0.157284 |
|  | Coordinates + frozen semantics | 2/4 | 0.171022 | 0.170854 |
| Independent music label (6 pairs) | State-only | 0/6 ties | 0.761996 | 0.761996 |
|  | Coordinates | 6/6 | 0.755610 | 0.760548 |
|  | Coordinates + frozen semantics | 6/6 | 0.779275 | 0.784867 |

The semantic model misses the preregistered 75% paired-win floor on the Butoh site and is worse than both baselines in absolute error on **both** sites. Its pooled weighted MSE is `0.535974` versus coordinate-only `0.516355` and state-only `0.520973`. The Qwen features can alter branch ordering, but they did not yield a transferable next-screen predictor here. The validation-selected lambda hitting the largest tested value and the large cross-site error shift warrant a new, separately frozen experiment; they do not justify retroactively changing this gate.

These are selected one-step synthetic browser transitions. No task-success, terminal-state, physical-world, or real-desktop generalization claim follows, and no research weights are loaded into Heliox. The user-approved rolling 16×16 desktop recorder remains off by default and was not activated.
