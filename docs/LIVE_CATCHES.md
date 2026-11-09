# Catch packs for Heliox Live

Heliox Live notices slips as people type: a weekday that doesn't match its date, a risky command before Enter.
A catch pack adds new things for it to notice, as a JSON file anyone can write and share: a command your team
keeps getting wrong, an idiom people mishear, a phrase that means trouble in your field.

A pack is data, never code. Live reads everything the person types, so nothing in a pack can run. Each rule is a
regular expression with a message, and it has to prove itself before Heliox uses it.

## Where packs go

- `daemon/pilot/live/catches/` ships with Heliox: `writing-idioms` (10 misheard phrases such as "could of" and
  "sneak peak") and `infra-safety` (5 commands that destroy infrastructure, such as `terraform destroy` without
  `-target`).
- `live_catches/` in Heliox's data folder holds your own. Drop a `.json` file there; Heliox picks it up within
  30 seconds and re-checks it whenever the file changes.

Every pack's result, with the reason for every rule it held back, is in `live_status` under `catches`. Turning off
"Catches from catch packs" in Settings → Live companion mutes all of them.

## A rule

```json
{
  "name": "writing-idioms",
  "title": "Misheard idioms and phrases",
  "author": "Heliox",
  "license": "Apache-2.0",
  "rules": [
    {
      "id": "sneak-peak",
      "where": ["prose"],
      "pattern": "\\bsneak peak(s?)\\b",
      "flags": "i",
      "title": "\"sneak peek\"",
      "message": "A quick look is a \"peek\"; a \"peak\" is a summit.",
      "fix": "sneak peek{1}",
      "urgency": "low",
      "positives": ["Here is a sneak peak of the new design.", "Want a sneak peak?", "Two sneak peaks this week."],
      "decoys": ["Here is a sneak peek of the new design.", "Traffic hit its peak at noon.", "We had to sneak past the peak hours."]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `name`, `id` | Lower-case letters, digits and dashes. A pack name must be unique among installed packs |
| `where` | Any of `prose`, `mail` (an email being written, also prose), `terminal`, `code`, `any` |
| `pattern`, `flags` | A Python regular expression; `"flags": "i"` ignores case |
| `title`, `message`, `fix` | What the card says. `{0}` is the whole match, `{1}`… its groups, `{name}` a named group. `fix` is optional |
| `urgency` | `low` (a card), `normal` (may be spoken at a pause) or `urgent` (terminal rules only: lands at the first short pause, even in a focus session) |
| `positives`, `decoys` | At least three sentences the rule must catch and three look-alikes it must leave alone |

Pick decoys that are close calls, not unrelated text. A rule that catches "sneak peak" should be tested against
"peak hours", since a careless pattern would fire on that too.

## The precision gate

A wrong interruption costs more than a missed one, so a rule is used only if it passes every check:

1. It compiles, is at most 300 characters, and has no nested repetition such as `(a+)+`, which can take forever
   on some text.
2. It catches every one of its positives and none of its decoys, tested in an app of each place it applies to.
3. It fires on none of the ordinary lines in `daemon/pilot/live/catch_corpus.txt` that match where it applies:
   1,800 sentences from this project's documentation, 1,200 lines of its source code and 153 terminal commands.
   A rule that fires there would interrupt people writing perfectly ordinary text.
4. It reads that whole corpus in under 200 ms.

A rule that fails is held back with the reason, and the rest of its pack still loads. Check a pack before you share
it:

```bash
cd daemon
python -m pilot.live.catches check my_pack.json
```

```
writing-idioms: 9/10 rules admitted
  ok   modal-of (9.32 ms on the ordinary text)
  HELD one-in-the-same (2.88 ms on the ordinary text)
       fires on its own look-alike: 'Put one in the same folder as the other.'
```

That output is real: the first draft of the shipped `one-in-the-same` rule was held back by its own look-alike,
and now matches only after "are", "is", "were" and similar.

## What the gate does not prove

Passing means the rule is right on its own examples and quiet on this project's ordinary text. It does not mean it
is right everywhere: the corpus is one project's English, and a rule can still misfire on text unlike it. Corrections
from packs show the pack's name, can be muted as a group, and count toward the same relevance learning as every other
catch, so a rule people keep waving away is heard from less.
