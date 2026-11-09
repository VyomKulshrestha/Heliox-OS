# Heliox Live Bench

An always-on companion is only useful if it speaks up at the right moment. Right about the slip but early, and it
cuts into a sentence the person was still writing. Right but late, and the email is already sent. Heliox Live
Bench measures that timing, alongside whether the catch was right, so a change to the live layer can be compared
with numbers instead of impressions.

It runs the real `LiveCompanion` from `pilot.live` (the same detectors, policy and personal timing model the app
uses) against scripted sessions on a simulated clock. Nothing is mocked except the person.

```bash
cd daemon
python -m pilot.live.bench                                  # every persona, both policies
python -m pilot.live.bench --persona thinker --policy personal
python -m pilot.live.bench --scenarios my_pack.json --json results.json
```

A full run takes about four seconds on a laptop CPU and is deterministic: the typing jitter is seeded, so the same
code gives the same numbers on any machine.

## What it scores

| Measure | Meaning |
| --- | --- |
| caught | Expected corrections that were delivered, out of all expected |
| missed | Expected corrections that never came, or came after the person pressed Enter |
| false | Deliveries the scenario did not expect, including a slip the person was already fixing themselves |
| cut-offs | Deliveries while the person was typing, pausing mid-sentence to think, or talking |
| precision | caught ÷ deliveries |
| recall | caught ÷ expected |
| wait s | Median seconds from the moment the person really stopped to a well-timed catch (cut-offs excluded) |

## The simulated person

Each scenario is typed by three personas with different rhythms:

| Persona | Typing | Pause mid-sentence | Pause after a finished thought |
| --- | --- | --- | --- |
| quick | 8 characters/s | 0.8 s | 5 s |
| steady | 5.5 characters/s | 1.5 s | 6 s |
| thinker | 4 characters/s | 2.6 s | 7 s |

Every pause varies by up to 25% either way. Before the scored scenarios each persona types 24 plain sentences
with a pause in the middle of each. That warm-up is when the personal timing model learns their rhythm, and it is
not scored.

The person reacts like a person. A card that arrives at a real break is read and acted on 1.2 s later: they apply
the fix, which makes the card withdraw itself as fixed. A card that cut in is ignored while they finish, then
waved away. Those two outcomes are what the timing model learns from, so the bench exercises the feedback loop
as well as the first guess.

## Results, 9 November 2026

30 scenarios in two packs. `core.json` has 14 with slips (15 expected catches: weekdays, sums, a wrong name, a
doubled word, an attachment, a force push, a `DELETE` without `WHERE`, a pasted API key, an error with a fix, two
spoken slips) and 11 decoys (a correct date, correct sums, code, safe commands, grammar that looks like a doubled
word, a half-typed word, and two slips the person fixes themselves). `stuck.json` has 3 moments of being stuck
(rewriting the same sentence three times, copying between two windows by hand, the same error three times) and 2
look-alikes (one cut while drafting, checking another window twice). Level: Balanced. Raw output with every
delivery: [`docs/evidence/live-bench-2026-11-09.json`](evidence/live-bench-2026-11-09.json).

| Policy | Persona | Caught | False | Cut-offs | Precision | Recall | Wait s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed rules | quick | 19/19 | 0 | 0 | 1.00 | 1.00 | 1.33 |
| fixed rules | steady | 19/19 | 0 | 8 | 1.00 | 1.00 | 1.13 |
| fixed rules | thinker | 19/19 | 2 | 11 | 0.90 | 1.00 | 1.02 |
| personal timing | quick | 19/19 | 0 | 0 | 1.00 | 1.00 | 1.51 |
| personal timing | steady | 19/19 | 0 | 0 | 1.00 | 1.00 | 1.57 |
| personal timing | thinker | 19/19 | 0 | 1 | 1.00 | 1.00 | 1.51 |

Fixed rules wait 1.2 s after typing stops. That is fine for someone who pauses for under a second, and it
interrupts anyone who stops longer to find the next word: 8 of 19 deliveries cut in on the steady typist and 11 of
21 on the thinker, and the thinker was told about two slips they were in the middle of fixing. Personal timing
learned a mid-sentence wait of 1.6 s, 1.75 s and 3.0 s for the three personas, kept every catch, and cut in once.
The low wait for fixed rules on the slower personas is not a win: most of their catches arrived as cut-offs, which
the wait column excludes.

Building the bench also found two engine bugs, both fixed alongside it: a slip first seen as the last thing typed
kept its "may still be typing" flag after the person typed on, which held it back for 3 s; and corrections were
read every 0.6 s but delivered on the 200 ms clock, so a card could land just after typing resumed. The engine now
refreshes each correction on every read and looks at the text once more right before speaking up.

## What it does not show

These are scripted sessions and simulated people, not recordings of real ones. The personas are three points on
a range, the pauses are drawn from a simple distribution, and the reactions are rules. The bench shows how the
policy behaves against those assumptions; it does not show how often real people feel interrupted. Real typing
is burstier, people pause to read or to switch windows, and the focused-field reader in the app sees text
through UI Automation, which some apps do not expose.

## Adding scenarios

Scenario packs are JSON files in `daemon/pilot/live/bench_scenarios/`, or anywhere with `--scenarios`. Each pack
is a list of scenarios:

```json
{
  "name": "email: weekday that doesn't match the date, mid-sentence",
  "app": "OUTLOOK.EXE",
  "title": "RE: Q4 review - Message (HTML)",
  "today": "2026-11-09",
  "people": {"manager": "Priya"},
  "start_text": "",
  "expect": ["weekday_date"],
  "steps": [
    {"type": "Hi Sam, can we move the review to Tuesday, 14 November at 3pm so that"},
    {"pause": "think"},
    {"type": " the whole team can join?"},
    {"pause": "done"}
  ]
}
```

| Step | Effect |
| --- | --- |
| `{"type": "text"}` | Typed at the persona's speed |
| `{"paste": "text"}` | Appears at once |
| `{"replace": ["old", "new"]}` | The person edits their own text: `old` is deleted, then `new` is typed (0.6 s) |
| `{"pause": "think"}` / `{"pause": "done"}` | The persona's mid-sentence or finished-thought pause |
| `{"pause": 1.5, "label": "ready"}` | A fixed pause; `label` is `think` (a cut-off if interrupted), `done`, `ready` or `idle` |
| `{"send": true}` | Enter: the field is cleared, and anything delivered later is too late |
| `{"say": "text", "seconds": 2.5}` | The person talks to Heliox; at the end the sentence is checked like a spoken command |
| `{"delete": 30}` | Backspacing 30 characters, at twice the persona's typing speed |
| `{"switch": "app\|title", "text": "..."}` | Another window comes to the front; `text` is what its field holds the first time |

`expect` lists the correction kinds that should be delivered (see `KINDS` in `pilot/live/detectors.py`); an
empty list makes the scenario a decoy, where any delivery is a false alarm. `today` defaults to 9 November 2026
and `people` to none. Scenarios must not contain real names, addresses, keys or other personal data; a key
should look like a key without being one.

A new detector, or a new kind of slip, should come with scenarios that catch it and decoys that look like it
and should not. `tests/test_live_timing.py` runs the bench in CI and fails when personal timing drops a catch,
raises a false alarm, or cuts in more than twice across all three personas.
