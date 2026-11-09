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
     every request, so they are not left to a search that could miss "pets" for "cats". When there are more than
     fit, the lines that share words with the request go first, then the most recent. Their copies in long-term
     memory are then left out of the search below, so the same fact never takes room twice.
   - **Earlier in this conversation**, in a long session: from the turns before the recent ones, the first two
     (where a session usually sets its subject) and the ones that share the most words with the request, in order,
     shortened (`pilot/memory/session_recall.py`).
   - **Long-term memory**: working memory for the task, then facts and past episodes ranked by relevance and age,
     with near-duplicates dropped (`pilot/memory/assembler.py`), then semantic search over older conversations when
     the vector index is available. A relation in the request is followed to the person it names
     (`pilot/memory/links.py`): "what did my manager say about extra staff" also searches for "Rahul", so it finds
     "Rahul approved hiring two more bakers" although the request shares none of those words.
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
| via-manager | a memory found through who the manager is, with no shared words |

Results on 9 November 2026 ([raw](evidence/memory-bench-2026-11-09.json)): **11 of 11 probes, nothing stale**,
median prompt 5,092 tokens. On the first eight probes, before the changes above, Heliox passed 3: the profile facts
replaced each other, the long session lost everything past six turns, and last week's answer was not kept anywhere
the planner looked without the vector index.

`--pressure 1` adds 24 more profile facts and eight weeks of ordinary turns around the script; `--pressure 2`
doubles both. `--budget` sets the context budget (memory gets a third, at least 500 tokens). At pressure 2, all 11
probes pass at budgets of 1,500, 2,400 and 4,000 tokens. Two things made that true. First, the profile's copies in
long-term memory used to fill the search with what the profile block already showed, which pushed last week's
answer and the manager's decision out (9 of 11 at 1,500 before that fix). Second, the link from "manager" to
Rahul: without it, the via-manager probe fails at 1,500 tokens.

The bench runs in CI. It shows that the right text is in the prompt; it does not show that a given model uses it
well, and its sessions are scripted, not recorded from real use.

## Wiping it

Settings → Your data lists **Conversation history** (every request and answer kept for context, and their semantic
search copies) and **Long-term memory** (facts and episodes, with their evidence), with counts, and clears either.
Evidence in long-term memory is append-only, so no code path can quietly rewrite why Heliox believes something; a
wipe the person asks for is the one exception, for that one transaction. "About you" (the profile) is listed and
cleared separately.

## Upkeep while idle

Every six hours `pilot/memory/consolidate.py` deletes working memory whose time is up, merges episodes that say
nearly the same thing (nine in ten words shared, same outcome), keeping the newest and raising its importance a
little for each repeat, and lets answer-only episodes older than 90 days go. Their request and answer stay in
conversation history, and episodes of actions that ran are kept. Facts are never touched by upkeep; they change only
through what the person says.

## A tuning loop with a gate

`python -m pilot.memory.tune` is the self-improvement loop in its safe form. It tries combinations of five retrieval
settings: the profile's room, whether over-full profiles are ranked by the request, the share for earlier turns, how
many opening turns are always kept, and the relevance-versus-variety weight. Each runs on the bench at pressure 2
and three budgets. It recommends a combination only if it wins on held-out probes it never optimised for, does no
worse on the rest, breaks no probe that passes today and brings back nothing stale. It never applies anything: the
recommendation is printed and written to JSON for a person to review and change in code.

The first run, on 9 November 2026 ([raw](evidence/memory-tune-2026-11-09.json)), tried 16 combinations in 44
seconds. Every one tied the current settings at 33 of 33 probe runs, so none was recommended. That says the current
settings are not fragile on this bench. It also says the bench is too easy to choose between them, so a harder
bench (more probes, real recorded sessions) comes before tuning can find anything. A setting that starves the profile
is refused by the gate, which a test checks.

## Not done yet

The bench is scripted. It needs probes drawn from real, consented sessions, and a check of answers, not only
prompts, with a local model. People are linked through the profile's relations only; projects and places are not
yet linked.
