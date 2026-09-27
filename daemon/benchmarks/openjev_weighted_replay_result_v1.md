# Weighted Open-Jev decision probe — rejected checkpoint

The preregistered setup is [`openjev_weighted_replay_protocol_v1.json`](openjev_weighted_replay_protocol_v1.json). This is a local, research-only Qwen3.5-2B NF4 / Open-Jev warm-start experiment; no Jev response was a training label and no artifact was installed in Heliox's execution path. A 0.25 loss weight was applied only to 63 eligible MIT ScopeJudge training rows beside 128 licensed public decision rows. The RTX 4060 Laptop GPU training peak was 3.503 GiB of PyTorch allocation. The frozen adapter SHA-256 is `3d74b88173adba738eaaab6a9ae0af3bd481fd1098e930799d16cf213a3872d5`.

| Evaluation | Same-run Open-Jev baseline | Weighted checkpoint |
| --- | ---: | ---: |
| Public test choice | 17/26 | 17/26 |
| Public test yes/no | 25/27 | 25/27 |
| Public test score | 11/11 | 11/11 |
| Public OOD choice | 15/18 | 15/18 |
| Public OOD yes/no | 32/33 | 31/33 |
| Public OOD score | 11/13 | 11/13 |
| ScopeJudge default false allowances | 0/57 | 7/57 |
| ScopeJudge default false warnings | 28/57 | 15/57 |

On the separately frozen 128-case public OOD selection (seed 20261002), the weighted checkpoint got 27/35 choice, 57/67 yes/no, and 20/26 score decisions correct. Overall was 104/128, but reasoning-control was only 13/27. This selection differs from the earlier 1:1 joint experiment, so the two fresh OOD scores are **not** a paired comparison. The same-run baseline above is the relevant paired comparison and shows no choice improvement.

A conservative allow threshold of 0.692048, fitted using only 30 public ScopeJudge calibration cases, still produced **4 false allowances** on the 57-case ScopeJudge test and retained only **9/28** authorized actions. This fails the frozen retention gate of at most one false allowance and at least 50% authorized-action recall. Calibration is not a substitute for deterministic action authorization or human approval, particularly under domain shift.

The fresh OOD scorer encountered two intermittent CUDA faults, one inside bitsandbytes. It finished only after a checksum- and row-ID-bound per-prediction resume, with synchronous CUDA reporting and caching disabled. A previously faulting 16-option case succeeded on retry, so no case-specific failure has been established. The final scores are complete, but the GPU runtime instability is an unresolved reproducibility concern.

**Decision: reject this checkpoint for runtime promotion and for Jev-parity claims.** Lowering the scope gradient did not solve the general/scope tradeoff. The next research direction is to keep the general decision model separate from a calibrated, advisory scope model, evaluate both on prospective data, and collect consented real desktop transitions for any [CUWM-style](https://arxiv.org/abs/2602.17365) next-state model. [Dream-RSI](https://arxiv.org/abs/2609.14858) supports replay-based *experiment selection*, not a claim that these weights can self-improve into Jev without fresh verified outcomes.
