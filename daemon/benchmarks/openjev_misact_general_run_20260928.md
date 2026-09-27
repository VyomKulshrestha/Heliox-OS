# Open-Jev general-decision and MisActBench scope study

**Status: research-only; no candidate is promoted for Heliox action execution.**

These local GPU experiments adapted the pinned Open-Jev/Qwen3.5-2B decision scorer with license-verified public Open-Jev rows and text-only, human-labeled MisActBench actions (CC-BY-4.0). No Jev service outputs were collected or used. The learned component is a shared yes/no, choice, and score scorer—not a general-purpose planner, generative assistant, or Jev-equivalent model.

All three runs used offline Hugging Face caches and the RTX 4060 (3.503 GiB peak VRAM). Generated checkpoints, adapters, logits, and JSON reports remain in ignored `benchmarks/.bench-state/`; no model weights or datasets are committed.

## Results

Each accuracy/NLL pair is baseline → trained on that run's own general-decision split. Scope counts use 128 balanced, trajectory-disjoint MisActBench test actions (64 in-scope, 64 out-of-scope) at the default 0.5 threshold.

| Seed; scope train; weight | General test accuracy / NLL | General OOD accuracy / NLL | Scope false allowances; false warnings (baseline → trained) |
| --- | ---: | ---: | ---: |
| 20260927; 256; 1× | 84.1% / 0.414 → 85.8% / 0.310 | 87.9% / 0.425 → 87.9% / 0.350 | 2 / 59 → 24 / 26 |
| 20260928; 256; 2× | 87.0% / 0.372 → 90.4% / 0.239 | 78.9% / 0.940 → 80.7% / 0.643 | 5 / 57 → 36 / 8 |
| 20260930; 320; 1× | 88.1% / 0.336 → 90.7% / 0.233 | 82.9% / 0.993 → 85.6% / 0.661 | 4 / 59 → 25 / 25 |

The larger 320-row run's calibration-only threshold that produced zero false allowances on its calibration split was 0.9201. On its held-out scope test it yielded 0 false allowances, but 63 false warnings and only 1.6% in-scope recall. The other runs showed the same unusable tradeoff: at the default threshold they falsely allowed too many actions; at conservative calibrated thresholds they rejected almost all in-scope actions.

## Decision and limitation

General decision accuracy/NLL improved on these small public splits, but all three learned candidates substantially worsened out-of-scope action allowances. None passes the action-scope gate, so keep learned scores advisory/research-only and retain deterministic policy, approval, and refusal checks as the authority. These results do not demonstrate safe desktop control, multi-agent completion, Jarvis behavior, or parity with proprietary Jev.

The MisActBench splitter now allocates test groups first, then disjoint calibration groups, then disjoint training groups. This avoids failures caused by the prior random 75/25 pre-partition while preserving balanced labels and trajectory isolation.

### Artifact integrity

Run `20260930` local-only artifact hashes:

- Training checkpoint: `b4fb7a6fe22b201ab7d364ab62d3cc37baf1292e2804cb9a91c7e7e7862c32dc`
- Adapter: `2310f4ab136a4e9d64c3c140cd5e3652eadf609fdb9d32c7e1f649637a3e6dc2`
- Decision head: `393516fdf090a18f24aefa8312b29c0b7da9eb1914f3de7c90c5b74b573aca40`
