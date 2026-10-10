# Open-Jev joint decision probe — rejected checkpoint

This is a local GPU research result, not a Heliox runtime model or proof of Jev parity. The setup was frozen in [`opendec_joint_256_protocol_v1.json`](opendec_joint_256_protocol_v1.json) before observing results. Training used the pinned Open-Jev-2B adapter/head on a Qwen3.5-2B NF4 backbone, 128 licensed public decision rows and 128 MIT ScopeJudge action-label rows. No Jev API answer was a training label. The RTX 4060 Laptop GPU run peaked at 3.503 GiB allocated by PyTorch.

| Check | Baseline | Trained | Interpretation |
| --- | ---: | ---: | --- |
| Public test, overall | 55/64 | 53/64 | Regression |
| Public test, choice | 19/26 | 18/26 | Regression |
| Public OOD, overall | 58/64 | 57/64 | Regression |
| Public OOD, choice | 15/18 | 15/18 | No improvement |
| ScopeJudge test, false allowances | 0/63 | 21/63 | Unacceptable default threshold |
| ScopeJudge test, false warnings | 31/63 | 2/63 | Better recall came with unsafe approvals |

On the archived 128-row Jev OOD selection, the trained adapter got choice 26/35 versus Jev 28/35, and yes/no 56/67 versus Jev 55/67. This comparison is diagnostic because the selection was already inspected. On the separately frozen public OOD seed 20261001, it got choice 22/35, yes/no 63/67, and score 22/26. The large gap between OOD selections is itself evidence against a general Jev-like claim.

Public ScopeJudge calibration negatives selected a conservative allow threshold of 0.887428. It reduced false allowances to 1 on the ScopeJudge test but raised false warnings to 16. On the previously inspected 60-case authored Heliox suite, the default threshold produced 5 false allowances; the conservative threshold produced 0 false allowances but 26 false warnings. These finite-sample observations are not safety guarantees under distribution shift. Deterministic authorization and human approval remain mandatory.

The next iteration tests a lower ScopeJudge loss contribution under the frozen [`opendec_weighted_replay_protocol_v1.json`](opendec_weighted_replay_protocol_v1.json). This follows [Dream-RSI's](https://arxiv.org/abs/2609.14858) useful *meta-exploration* idea—reuse recorded evaluations to select experiments—without pretending the paper directly trains a Jev-like model. [Selective classification under distribution shift](https://arxiv.org/abs/2405.05160) motivates reporting abstention/coverage alongside error, and [group-shift robustness](https://arxiv.org/abs/1911.08731) motivates per-source results rather than aggregate accuracy alone. No checkpoint is promoted automatically.
