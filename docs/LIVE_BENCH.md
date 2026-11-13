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
| cut-offs | Moments Live spoke up while the person was typing, pausing mid-sentence to think, or talking. Two cards arriving together are one interruption and count once |
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

Every pause varies by up to 25% either way, drawn from a seed per persona and scenario, so adding a scenario never
changes the pauses in another one. Before the scored scenarios each persona types 24 plain sentences
with a pause in the middle of each. That warm-up is when the personal timing model learns their rhythm, and it is
not scored.

The person reacts like a person. A card that arrives at a real break is read and acted on 1.2 s later: they apply
the fix, which makes the card withdraw itself as fixed. A card that cut in is ignored while they finish, then
waved away. Those two outcomes are what the timing model learns from, so the bench exercises the feedback loop
as well as the first guess.

## Results, 9 November 2026

34 scenarios in three packs. `core.json` has 14 with slips (15 expected catches: weekdays, sums, a wrong name, a
doubled word, an attachment, a force push, a `DELETE` without `WHERE`, a pasted API key, an error with a fix, two
spoken slips) and 11 decoys (a correct date, correct sums, code, safe commands, grammar that looks like a doubled
word, a half-typed word, and two slips the person fixes themselves). `stuck.json` has 3 moments of being stuck
(rewriting the same sentence three times, copying between two windows by hand, the same error three times) and 2
look-alikes (one cut while drafting, checking another window twice). `commitments.json` has 2 promises with a
deadline (one in a chat message that is sent before the pause) and 2 look-alikes (a question about someone else's
deadline, a vague promise next to a past one). Level: Balanced. Raw output with every
delivery: [`docs/evidence/live-bench-2026-11-09.json`](evidence/live-bench-2026-11-09.json).

| Policy | Persona | Caught | False | Cut-offs | Precision | Recall | Wait s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed rules | quick | 21/21 | 0 | 0 | 1.00 | 1.00 | 1.53 |
| fixed rules | steady | 21/21 | 1 | 5 | 0.95 | 1.00 | 1.35 |
| fixed rules | thinker | 21/21 | 2 | 10 | 0.91 | 1.00 | 1.12 |
| personal timing | quick | 21/21 | 0 | 0 | 1.00 | 1.00 | 1.48 |
| personal timing | steady | 21/21 | 0 | 1 | 1.00 | 1.00 | 1.48 |
| personal timing | thinker | 21/21 | 0 | 1 | 1.00 | 1.00 | 1.12 |

Fixed rules wait 1.2 s after typing stops. That is fine for someone who pauses for under a second, and it
interrupts anyone who stops longer to find the next word: 5 times for the steady typist and 10 for the thinker, who
was also told about two slips they were in the middle of fixing. Personal timing learned a mid-sentence wait of
1.35 s, 2.06 s and 2.86 s for the three personas, kept every catch, raised no false alarm, and cut in twice in all,
both times on a pause near the long end of that persona's range.
The low wait for fixed rules on the slower personas is not a win: most of their catches arrived as cut-offs, which
the wait column excludes.

Building the bench also found two engine bugs, both fixed alongside it: a slip first seen as the last thing typed
kept its "may still be typing" flag after the person typed on, which held it back for 3 s; and corrections were
read every 0.6 s but delivered on the 200 ms clock, so a card could land just after typing resumed. The engine now
refreshes each correction on every read and looks at the text once more right before speaking up.

## Update, 10 November 2026: reading is not writing

With Live on in the Claude desktop app, a person got cards about text they had not written: a doubled word
from the page, a "you've rewritten this" offer quoting a page button, and an urgent "that looks like a
password or key" card with no evidence, repeated every minute, from a screen OCR match. Three decoys in
`reading.json` cover this: a chat page with a doubled word that has focus while it streams, an email reply that
quotes someone else's doubled word, and key-like text on a web page that screen OCR matches three times.

Before the fix, the first two gave 3 false alarms for every persona and policy (the bench could not script the
OCR notice then). After it, all three give none, and the 34 earlier scenarios score exactly as in the table
above. The fix: the focused-field reader passes text on only from writable fields and terminals, text already in
a field when focus arrives is not counted as the person's slip, and OCR risk matches no longer become cards.

## Update, 12 November 2026: a patient first week

Before Live has learned a person's pauses, it used the level's wait (1.2 s at "balanced") everywhere, and the
rules row above is that first week: 16 cut-offs for the thinker on the 105-scenario bench (precision 0.96), 5 for the
steady typist. Now a pause in the middle of a sentence has to last 2.5 times the level's wait (3.0 s) until the
person's own pauses say otherwise, and that patience fades as they come in (half gone after 5 mid-sentence pauses,
then blended with the learned wait as before). At a sentence's end the level's wait still applies, a slip that is the
last thing typed keeps its own longer wait (stretched from the level's, not from the patient one), and urgent
catches (a risky command) keep their 0.6 s. With learning off the patient wait stays.

| Policy | Persona | Cut-offs before | Cut-offs after | Precision before | Precision after | Median wait before | Median wait after |
|---|---|---|---|---|---|---|---|
| rules (nothing learned) | quick | 0 | 0 | 1.00 | 1.00 | 1.62 s | 1.51 s |
| rules (nothing learned) | steady | 5 | 0 | 1.00 | 1.00 | 1.39 s | 1.61 s |
| rules (nothing learned) | thinker | 16 | 0 | 0.96 | 1.00 | 1.20 s | 1.24 s |
| personal timing | quick | 0 | 0 | 1.00 | 1.00 | 1.42 s | 1.42 s |
| personal timing | steady | 1 | 1 | 1.00 | 1.00 | 1.37 s | 1.37 s |
| personal timing | thinker | 2 | 2 | 1.00 | 1.00 | 1.18 s | 1.18 s |

Recall is 55 of 56 in every row, before and after (the one miss is the known recogniser case). The cost falls on a
fast typist who stops mid-sentence in their first days: the card comes after 3.0 s instead of 1.2 s, then 1.38 s after
20 such pauses and 1.23 s after 60 (the quick persona's own rhythm). At a sentence's end nothing changes.

## Update, 13 November 2026: does Monday change Thursday?

`python -m pilot.live.learning_week` runs three made-up people through four days with one Live engine kept across
them (relevance, snoozes, "not now" days and the muted kinds carry over; a day apart on the clock), typing the same
four slips each day: two weekdays that do not match their dates, a wrong sum, a doubled word. On Monday and Tuesday
each says "not now" to the first weekday card; on Tuesday "don't mention these" to the doubled word. On Wednesday
one says "thanks" to a weekday card, one uses its fix, one ignores it.

| Day | Weekday slips | Wrong sum | Doubled word |
|---|---|---|---|
| Monday | 1 of 2, spoken (the second, minutes after "not now", held back) | shown | shown |
| Tuesday | 1 of 2, spoken (held back again) | shown | shown, then "don't mention these" |
| Wednesday | 2 of 2, both shown, not spoken | spoken | none |
| Thursday | after "thanks" or the fix: spoken again; ignored: shown, not spoken | shown or spoken | none |

The first version of this bench showed that one "not now" changed nothing the next day, which is right, but also
that a person who said "not now" on two days still heard the kind spoken. Now a kind told "not now" on two different
days within a week is shown and not spoken until a week after the last one; "thanks" for it, using its fix, or undoing
the "not now" speaks it again at once; "don't mention these" is unchanged. This is the one mechanism for "not now":
its days are kept with its snoozes (`live_snoozed.json`), and "not now" no longer also lowers the kind's relevance,
which now moves only for thanks, fixes, mutes and cards left alone. Relevance could not carry the rule: it does not
know days, so two "not now"s a minute apart would count as two days, and nothing would let it lapse after a week.
`tests/test_live_learning_week.py` asserts the table.

## When it landed (13 November 2026)

`--when` adds, for every delivery, when it landed on the person's own clock (`bench.moment_of`; measuring only,
nothing in it changes what Live does):
- seconds from the slip being complete (typed: the evidence first whole in its field; said: the end of the sentence);
- what the person was doing then: mid-word, between words, a mid-sentence pause, a pause, speaking, Heliox
  answering, gone to another window, or the text already sent or deleted;
- how long they had paused, and how soon they went on typing or talking;
- for a spoken delivery, whether its words ran into the person's next sentence (at 2.5 words a second).

It prints the distributions, not only the median. Most scenarios end with the slip and a pause, so a card can never
land as the person goes on; `--keep-typing` adds, to every typed scenario, a short pause and one more sentence
with no slip in it. The measurement covers Live's own waits only: the time a voice takes to start speaking
(synthesis, about a second on a CPU) is not in the simulated clock.

## Update, 13 November 2026: the voice yields, the card comes first, and a sent slip is still told

The bench now plays each spoken remark on its own clock (`speak` in `bench.run`): it starts a second after it is
asked (synthesis), lasts its words at 2.5 a second, and stops at the end of a word when Live cuts it. It measures how
long the voice was still going once the person typed or talked again. Three changes, measured with `--when` and
`--keep-typing`, each before and after:

1. **The voice yields to typing as it yields to speech.** A spoken remark about typed text says the card's title
   alone ("25 November is a Wednesday.", at most 2 s in the bench); the rest stays on the card. The first keystroke
   after it is asked cuts it (cancelled, the speech coordinator stops it), the card stays up, and it is not said
   again. With the person typing on after each scenario: the voice was still going when they typed again in 56 of
   104 spoken remarks before (median 2.4 s, p90 4.6 s, up to 6.4 s), and in 2 of 104 after, both ending as they
   started (0.0 s). Voice length: median 3.2 s before, 0.6 s after (98 of 104 cut by the person going on, or
   fixing the slip after reading the card).
2. **The card and the voice have separate clocks.** A slip that is the last thing typed gets its card at the
   ordinary pause without week one's patience, silently (`policy.SHOW_FIRST`); the voice waits the longer pause
   (×2.5, at least 3 s) and speaks only if the card is still unanswered, the person has not typed since and is
   still in that field (`LiveCompanion._voice_later`, which emits `live_interjection_spoken`). A slip ending in a
   digit keeps the card waiting too: a number may still be being typed.
3. **After the text has gone.** A weekday that does not match its date or a wrong sum (`engine.SENT_KINDS`) that
   was whole in a message when it was sent is raised once as a card saying so ("You sent Thursday 25 November; 25
   November is a Wednesday"), never spoken, never for a typo or a doubled word, and a card already up for it stays
   up. A catch about a field the person has left is a card only, dropped at 90 s as before.

New scenarios (`bench_scenarios/sent-and-left.json`, 3 × 6 policy and persona runs): a weekday slip sent 2 s after
it was typed, a wrong sum sent at once, and a slip left behind in another window. Before: the first two missed in
all 12 runs, the third caught and spoken in all 6. After: all 18 caught, none spoken; the first as a card before
the send in 4 of 6 runs (the personal steady and thinker, whose own pause is longer than 2 s, get the "You sent"
card after it), the second as the "You sent" card 1.8 to 2.2 s after the slip, the third as a card only.

The 105 earlier scenarios: catches, misses, false alarms and cut-offs unchanged in every policy and persona, with
and without `--keep-typing`; nothing lands mid-word. The whole bench (108 scenarios): 58/59 caught in each run, no
false alarms.

## New kinds (13 November 2026)

Each new kind has a pack of its own (`bench_scenarios/watch-<kind>.json`, run after the older packs so that what they teach the timing model is unchanged, "today" is Friday 13 November 2026, each
scenario in a field of its own), a switch in Settings > Live like the others, and a line in Settings' answers. Every
pack runs with every policy and persona. False alarms are counted on the whole bench, on the conversation bench's 975
not-addressed sentences typed into a chat, and on this repository's last 2,127 commit messages typed into a note and
into an email. A kind with a false alarm anywhere there would be a card only, never spoken.

- **A date that has already passed, written as if it were ahead** (`past_date`). Rule: a day and month, up to 45
  days back and in this year, in a sentence that looks ahead ("see you", "will", "is due", "by", "next") and has no
  word that looks back ("was", "met", "last", "since", "already", "said"), and nothing about every year
  ("birthday", "next year"). Not in code, terminals, minutes, logs or changelogs; not in quotes; not in what is
  said. Reads: the field's text and the date. "You sent" applies, because a wrong date sent still matters. Pack:
  11 that must fire, 22 that must not; 11/11 caught and 0 false alarms in all 6 runs. False alarms elsewhere: 0 on
  the bench, 0 of 975 sentences, 0 of 2,127 commit messages in each window. Spoken when left alone, like the
  others.
- **A name one letter off someone you told Heliox about** (`near_name`). Rule: where a name addresses someone (the
  start of the text or a line before a comma; after "hi", "hello", "dear", "thanks" and the like; after "@" or
  "cc"), a word one letter off the first name, five letters or more, of exactly one person the person told Heliox
  about ("my manager is Priya"): two neighbouring letters swapped ("Priay"), or one inner letter doubled, added or
  dropped ("Danniel", "Prya"). Never one letter swapped for another (Maria and Mario both exist), never the first
  letter, never an added or dropped last letter (Daniela, Hanna and Michaels are other people or words), and
  nothing when two known people could be meant, when the window (recipient or subject) spells it that way, or when
  a surname after it is not the known person's ("Hi Priay Kumar" to a known Priya Shah). Jan and Jon, Maria and
  Mario, Anna and Anne never fire: one letter for another, and names under five letters, are left alone. Not in code,
  terminals or quotes. Reads: the field's text and the names in the person's profile (UserProfile's people, one
  name per relation), compared on this computer and nothing more. The card asks, never corrects: "Did you mean
  Priya?", "You wrote Priay; your manager is Priya.", with "If Priay is someone else, ignore this." and the name
  as the fix. A card only, never spoken at any level (`policy.CARD_ONLY_KINDS`): a different person's correct
  name must never be corrected aloud. No "You sent" card, because a misspelt greeting once sent cannot be mended. Pack: 11 that must fire, 24 that must not; 11/11
  caught by the steady and thinker personas, 10/11 by the quick one in the pack on its own ("Prriya," typed as the
  very first word of a new field is already there at Live's first look, so it counts as text that was there
  before), 11/11 in all six runs of the whole bench; 0 false alarms in the pack. False alarms elsewhere: 0 on the
  bench, 0 of 975 sentences, 0 of 2,128 commit messages in each window.
- **The same sentence twice** (`pasted_twice`). Rule: a sentence of eight words or more that is in the field twice,
  the same apart from case, spacing and the closing mark. Not quoted lines of an earlier message, list items or
  headings (a squash commit repeats its title as a bullet on purpose: that was the one false alarm found, in 2 of
  2,129 commit messages, and it is gone), not code, terminals or what is read off the screen, and not when the field
  already held both copies. Reads: the field's text only. A low-urgency card, never spoken; no "You sent" card.
  Pack: 11 that must fire, 20 that must not; 11/11 caught and 0 false alarms in all six runs. False alarms
  elsewhere: 0 on the bench, 0 of 975 sentences, 0 of 2,129 commit messages in each window.

What was weighed and left out:
- **"Attached" with nothing attached.** None of the three platform readers can tell whether a message has an
  attachment: they read the focused field's text, and the attachment list is another part of the mail window
  that UI Automation, the macOS Accessibility API and AT-SPI expose differently in each mail app, when at all.
  The existing `missing_attachment` card ("You mention an attachment") stays as it is.
- **A time that collides with the calendar, or contradicts a reminder.** Already the grounded check
  (`personal_fact`), against the calendar, to-dos and reminders.
- **A wrong percentage or unit sum.** Already `arithmetic` ("20% of 150 is 35").
- **A reply that stops mid-sentence at send.** Too often meant (a quick "ok", "and you?"), and it can only be told
  after it is sent.

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
  "editable": true,
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
| `{"notice": "credential_exposure", "source": "ocr"}` | Supervision reports a risk rule match, from `ocr` or `keystroke` |

`expect` lists the correction kinds that should be delivered (see `KINDS` in `pilot/live/detectors.py`); an
empty list makes the scenario a decoy, where any delivery is a false alarm. `today` defaults to 9 November 2026
and `people` to none. `start_text` is what the field held before the person touched it, so slips in it are not
the person's. `editable: false` makes the focused element page text the person reads (a web page, a chat
transcript): the engine is handed the text but must not check it. Scenarios must not contain real names, addresses, keys or other personal data; a key
should look like a key without being one.

A new detector, or a new kind of slip, should come with scenarios that catch it and decoys that look like it
and should not. `tests/test_live_timing.py` runs the bench in CI and fails when personal timing drops a catch,
raises a false alarm, or cuts in more than twice across all three personas.
