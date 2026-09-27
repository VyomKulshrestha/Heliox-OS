# Frozen non-checkbox page-variant check: coverage gate failed

The [seed-3400 protocol](miniwob_semantic_branch_seed3400_noncheckbox_protocol_v1.json) was committed before collection and reused the frozen checkpoints described in the [seed-3200 results](miniwob_semantic_branch_seed3200_results_v1.md). Neither model was fitted to these observations. The pinned MiniWoB++ source, exact-reset rule, nonterminal-only rule, and visible-semantic-difference rule were unchanged. Feature archive SHA-256: `701f5ca8f8cae0ccce2278938d5ffd4998906927c5de2b74091b5d49bbd1a0a8`.

| Eval page | Qualifying pairs | Correct-action wins | Action weighted MSE | Frozen state-only | Unchanged screen |
| --- | ---: | ---: | ---: | ---: | ---: |
| `click-collapsible-2-nodelay` | 12 | 12/12 | 0.055937482 | 0.061902665 | 0.062744940 |
| `click-tab-2-medium` | 0 | not measurable | not measurable | not measurable | not measurable |

The collapsible variant meets its per-page criterion: paired alternative action error was `0.056597491`, worse than the correct action. The tab variant fails the preregistered minimum of eight pairs, so **the overall non-checkbox gate fails**. It was not replaced after observation. The much larger collapsible error than the checkbox result also warns against comparing raw errors across page families.

These are 16×16 synthetic one-step browser features, not a Heliox desktop world model. Further work requires an interaction corpus that can actually supply diverse, stable, nonterminal action branches; a separately frozen evaluation must then beat state-only and paired-action controls across genuinely different task families. No runtime model promotion or private screen collection occurred.
