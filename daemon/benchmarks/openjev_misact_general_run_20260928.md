# Open-Jev general-decision and MisActBench scope study

**Status: research-only; no candidate is promoted for Heliox action execution.**

These local GPU experiments adapted the pinned Open-Jev/Qwen3.5-2B decision scorer with license-verified public Open-Jev rows and text-only, human-labeled MisActBench actions (CC-BY-4.0). No Jev service outputs were collected or used. The learned component is a shared yes/no, choice, and score scorer—not a general-purpose planner, generative assistant, or Jev-equivalent model.

All four runs used offline Hugging Face caches and the RTX 4060 (3.503 GiB peak VRAM). Generated checkpoints, adapters, logits, and JSON reports remain in ignored `benchmarks/.bench-state/`; no model weights or datasets are committed.

## Results

Each accuracy/NLL pair is baseline → trained on that run's own general-decision split. Scope counts use 128 balanced, trajectory-disjoint MisActBench test actions (64 in-scope, 64 out-of-scope) at the default 0.5 threshold. AUC is for in-scope ranking on those same held-out actions; it does not establish a safe execution threshold.

| Seed; scope train; weight | General test accuracy / NLL | General OOD accuracy / NLL | Scope false allowances; false warnings (baseline → trained) | Scope AUC (baseline → trained) |
| --- | ---: | ---: | ---: | ---: |
| 20260927; 256; 1× | 84.1% / 0.414 → 85.8% / 0.310 | 87.9% / 0.425 → 87.9% / 0.350 | 2 / 59 → 24 / 26 | 0.5112 → 0.6721 |
| 20260928; 256; 2× | 87.0% / 0.372 → 90.4% / 0.239 | 78.9% / 0.940 → 80.7% / 0.643 | 5 / 57 → 36 / 8 | 0.5308 → 0.6694 |
| 20260930; 320; 1× | 88.1% / 0.336 → 90.7% / 0.233 | 82.9% / 0.993 → 85.6% / 0.661 | 4 / 59 → 25 / 25 | 0.5591 → 0.6985 |
| 20261001; 512; 1× | 88.8% / 0.287 → 92.2% / 0.229 | 84.4% / 0.730 → 83.5% / 0.571 | 2 / 56 → 27 / 17 | 0.6106 → 0.7068 |

The 512-row run used 64 separate calibration actions (32 per label), with temperature 2.515. Thresholds below were selected on calibration only, then evaluated without adjustment on the held-out test set:

| Calibration false-allowance budget | Threshold | Test false allowances | Test false warnings | In-scope recall |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9270 | 0 | 64 | 0.0% |
| 1 | 0.8676 | 3 | 58 | 9.4% |
| 2 | 0.8291 | 8 | 51 | 20.3% |
| 4 | 0.8188 | 9 | 49 | 23.4% |

For comparison, the 320-row run's zero-false-allowance calibration threshold (0.9201) yielded 0 test false allowances, 63 false warnings, and 1.6% in-scope recall; a one-allowance calibration budget yielded 0 test false allowances, 56 warnings, and 12.5% recall. These small calibration samples do not reliably select an operating point that balances useful recall with few false allowances.

### Released Open-Jev matched OOD check

The frozen 20261001 Heliox candidate was also scored against the released Open-Jev v3 checkpoint on the protocol v2 fresh OOD selection. Both models used the same 112 rows that fit the Heliox candidate's 512-token context limit; 16 longer yes/no rows were excluded from both rather than truncated. This is a candidate-vs-public-release comparison, not a comparison with proprietary Jev.

| Shared rows | Released Open-Jev v3 | Heliox 20261001 candidate |
| --- | ---: | ---: |
| Top-1 accuracy | 79.5% | 78.6% |
| Mean NLL | 0.638 | 0.509 |
| Mean Brier score | 0.0860 | 0.0792 |

By decision type, top-1 accuracy was 60.5% vs 57.9% for choice, 88.9% vs 88.9% for yes/no, and 90.0% vs 90.0% for score. The candidate is close on this shared subset and improves aggregate probability calibration, but does not improve overall top-1 accuracy; the mixed per-kind NLL results also argue against promotion.

The comparison was rerun on the local GPU with the exact training-pinned stack (`torch 2.8.0+cu128`, `transformers 5.10.2`, `peft 0.19.1`, `accelerate 1.13.0`). All aggregate and per-kind metrics reproduced exactly, with no PEFT configuration warning. That resolves the earlier environment mismatch for these reported scores; the choice-accuracy gap and lack of proprietary Jev comparison remain. The run used cached model weights and released Open-Jev logits, computed Heliox logits in memory, and wrote no new benchmark artifacts.

Paired top-1 outcomes were: both correct on 87 rows, both wrong on 22, Open-Jev-only wrong on 1, and Heliox-only wrong on 2. The exact two-sided McNemar test on the three discordant rows gives p=1.0; this small sample does not establish a meaningful accuracy difference or parity.

## Decision and limitation

General test accuracy/NLL improved on these small public splits, and the fourth run improved held-out scope ranking AUC to 0.7068. However, its default-threshold false allowances rose from 2 to 27; calibrated thresholds either rejected every in-scope test action or admitted 3–9 out-of-scope actions while recalling only 9.4–23.4% of in-scope actions. None passes the action-scope gate, so keep learned scores advisory/research-only and retain deterministic policy, approval, and refusal checks as the authority. These results do not demonstrate safe desktop control, multi-agent completion, Jarvis behavior, or parity with proprietary Jev.

The MisActBench splitter allocates test groups first, then disjoint calibration groups, then disjoint training groups. Test and calibration use one action per trajectory; training can use up to two actions per trajectory while preserving balanced labels and trajectory isolation across splits. This made it possible to increase the training set to 512 without leakage.

### Artifact integrity

Run `20261001` local-only artifact hashes:

- Training checkpoint: `4e786c6a2d325d511aacd98076b9fff34908e772811053956807766cd25acfc0`
- Adapter: `66fdf066b140c14a86678e562cb44b84c3b32ca62f13723ac6abaf057792360a`
- Decision head: `3224b0aa2682b6fdaa342a0f5a23cf837c5b80ff36a4758e172b6d523c141b1f`

This run took 2,885.5 seconds and used 3.503 GiB peak CUDA memory. Its adapter and all predictions remain untracked, local-only benchmark artifacts.
