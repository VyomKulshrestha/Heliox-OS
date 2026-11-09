# Memory and context in Heliox

Every request Heliox plans goes to a model with a limited context window. What fills that window decides whether
Heliox remembers that the person has two cats, that their manager changed last week, or that the report they set
up an hour ago is for the Porto warehouse. This page describes what goes in, how it is chosen, how it is measured,
and how to wipe it.

## What reaches the model

For each planned request the planner builds one prompt. In order:

1. **The system prompt**, about 4,100 tokens: what Heliox is and how to plan.
2. **The last turns of this conversation**: 6 in the default lightweight mode, 10 in full mode.
3. **Memory**, in a budget of a third of the context budget (about 1,300 tokens in lightweight mode):
   - **What the person told Heliox about themselves**, always, up to 400 tokens: their name, facts ("I work at a
     bakery"), the people they mentioned, preferences and answer style. These are few, explicit and wanted on
     every request, so they are not left to a search that could miss "pets" for "cats".
   - **Earlier in this conversation**, in a long session: from the turns before the recent ones, the first two
     (where a session usually sets its subject) and the ones that share the most words with the request, in order,
     shortened (`pilot/memory/session_recall.py`).
   - **Long-term memory**: working memory for the task, then facts and past episodes ranked by relevance and age,
     with near-duplicates dropped (`pilot/memory/assembler.py`), then semantic search over older conversations when
     the vector index is available.
4. **The request** and what is on screen.

Every reply, not only ones that ran actions, is kept as an episode, so "the sales summary we drafted for Lisbon"
can be found next week even where the vector index is not installed.

## Facts that change

The profile keeps some things with one current value. "I live in Pune now" replaces "I live in Mumbai"; "my new
manager is Rahul" replaces Priya; the old value is retracted from long-term memory, with its history, rather than
left to come back. Home, workplace and the people you can have only one of (manager, partner, landlord, doctor and
similar) work this way. Everything else is kept side by side: two friends, many facts.

Until 9 November 2026 every profile entry in a list shared one label in long-term memory, so each new one quietly
superseded the one before: after three "remember that…" facts, only the newest reached the planner.

## The model's real context window

The prompt above is about 5,200 tokens. Older Ollama releases default to a 2,048 or 4,096 token window and cut the
start of a longer prompt, which is the system prompt. Heliox now asks Ollama for a window that fits each prompt plus
2,048 tokens to answer, rounded up to a power of two and never more than the model's own limit (read once from
`/api/show`). On the development laptop, Ollama 0.32 with qwen3.5:4b read the whole prompt either way; the change
matters on older releases and smaller defaults.

## Measuring it: Heliox Memory Bench

```bash
cd daemon
python -m pilot.memory.bench
```

The bench replays a few weeks of scripted use through Heliox's real memory path: "remember that…" through the
profile, every planned turn through `MemoryStore.record`, the temporal memory underneath. Then it asks probe
questions through the real planner, with a stand-in model that records the prompt instead of answering. A probe
passes when every expected phrase is in the prompt and no forbidden one is. Forbidden phrases are facts that have
since changed or been forgotten: bringing them back is worse than forgetting.

| Probe | What it checks |
| --- | --- |
| sister, pets, workplace | facts told once, weeks and sessions ago |
| home-now | a fact that changed: Pune in, Mumbai out |
| manager | a person who changed: Rahul in, Priya out |
| style | a preference applied to a request that shares none of its words |
| forgotten | something the person asked Heliox to forget stays forgotten |
| same-day, mid-session | details from eleven and nine turns back in a long session |
| last-week | what Heliox answered in an earlier session |

Results on 9 November 2026 ([raw](evidence/memory-bench-2026-11-09.json)): **10 of 10 probes, nothing stale**,
median prompt 5,233 tokens. On the first eight probes, before the changes above, Heliox passed 3: the profile facts
replaced each other, the long session lost everything past six turns, and last week's answer was not kept anywhere
the planner looked without the vector index.

The bench runs in CI. It shows that the right text is in the prompt; it does not show that a given model uses it
well, and its sessions are scripted, not recorded from real use.

## Wiping it

Settings → Your data lists **Conversation history** (every request and answer kept for context, and their semantic
search copies) and **Long-term memory** (facts and episodes, with their evidence), with counts, and clears either.
Evidence in long-term memory is append-only, so no code path can quietly rewrite why Heliox believes something; a
wipe the person asks for is the one exception, for that one transaction. "About you" (the profile) is listed and
cleared separately.

## Not done yet

Part two of this work: a consolidation pass that merges duplicate facts and lets stale episodes fade; links between
people, projects and the facts about them, so related memories come together; and a tuning loop that proposes
changes to the retrieval settings and keeps only those that improve the bench without breaking a probe.
