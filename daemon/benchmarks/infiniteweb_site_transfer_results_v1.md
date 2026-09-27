# InfiniteWeb site-disjoint visual transfer: failed gate

The [protocol](infiniteweb_site_transfer_protocol_v1.json) fixed eight previously uninspected sites and the model settings before collection. The collector fetched the pinned MIT-licensed [InfiniteWeb dataset](https://huggingface.co/datasets/WhiteFlamesCN/InfiniteWeb-Dataset) revision `84e294fc432abca78b89388ed3c98ca778177a1f`, blocked external browser requests, and kept raw HTML/screenshots in memory. The synthetic sites are separate from the user's desktop. The feature archive SHA-256 is `db0fa7ae0d3d04a0543f0b978f0579dd648bd104da9c0e359a5a5d92453392df` (46 train, 12 validation, 18 evaluation rows; two independently reset actions per group). The 23/6/9 group split met all preregistered coverage floors; each evaluation site had at least four groups. Neither reward nor task termination was measured.

CUDA fitted the preregistered spatial-global action model and matched state-only baseline on 16×16 grayscale features. Their weights remain in ignored local research state (SHA-256 `534574bdf349d28db67ecade574f86bd97f2300d1085d86f376b8458278e0d32` and `b093004f06010d26e0fdf89a3416c559fe43672ee9378ae879b91faa0f5c2279`). The models selected validation epochs 5 and 58 respectively. The frozen evaluation was:

| Held-out site | Groups | Correct-action wins | Action weighted MSE | Wrong-action MSE | State-only MSE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fitness coaching | 5 | 3/5 (60%) | 0.007339440 | 0.007340426 | 0.023586188 | Fail: paired wins |
| Ice cream shop | 4 | 4/4 (100%) | 0.140006957 | 0.140169281 | 0.133752732 | Fail: state-only better |

Pooled action weighted MSE was `0.066302781` versus `0.072549097` state-only, but the protocol required **both** sites to pass. The action-versus-swapped margin was also very small. A prior frozen MiniWoB checkbox checkpoint scored only 3/5 paired wins on a separate exploratory InfiniteWeb shopping site and was slightly worse than its state-only counterpart (`0.007024746` versus `0.007012899` weighted MSE). These results argue against a broadly transferable 16×16 coordinate-only world model.

The next hypothesis is that link/control meaning is missing from both the downsampled image and the coordinate-only action. Any semantic-action model must be tested prospectively on new sites, with comparable state-only and action-swap controls. The released Heliox runtime remains rule-gated and unchanged; synthetic site navigation does not establish Jev parity, autonomous execution quality, or real-desktop reliability.
