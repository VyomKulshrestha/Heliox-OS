"""An hour of speech that is not addressed to Heliox, as text (pure, deterministic).

Four kinds, mixed the way a working day mixes them:
  meeting   several people talking to each other, with requests that sound like commands ("can you open the
            deck", "stop sharing", "yes, approve it") because people say those to each other;
  tutorial  a video narrator giving instructions to its viewers ("open Visual Studio Code, then PowerShell");
  reading   text read aloud (public-domain passages and plain prose);
  misfire   phrases that set Heliox off before ("PowerShell", "Where we store home", "Visual Studio Code",
            "Thanks for watching") and near-misses of the wake phrase said to someone else ("Hey Felix").
Sentences that name Heliox in the third person ("Heliox is the app I use") are a separate, labelled kind,
``mention``: they are not addressed to it, but they contain the wake word, so they are tallied apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

VOICES = ("am_michael", "bm_lewis", "af_bella")
NAMES = ("Priya", "Sam", "Daniel", "Mei", "Omar", "Lucia", "Tom", "Aisha", "Ken", "Nora")
THINGS = ("the deck", "the budget sheet", "the launch notes", "the contract", "the roadmap", "the invoice",
          "the test report", "the design file", "the release checklist", "the hiring plan")  # fmt: skip
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next week", "the twelfth")
APPS = ("PowerShell", "Visual Studio Code", "the terminal", "Chrome", "Excel", "Slack", "Outlook", "Figma",
        "the file explorer", "Notepad")  # fmt: skip
TIMES = ("nine", "ten thirty", "eleven", "two", "four", "half past three", "noon")

MEETING = (
    "Can you open {thing} so everyone can see it?",
    "{name}, could you stop sharing your screen for a second?",
    "Let's move the review of {thing} to {day}.",
    "Yes, approve it, we talked about this last week.",
    "No, don't send it yet, I want {name} to read it first.",
    "Go back to the previous slide, the one with the numbers.",
    "I'll send {name} {thing} by {day}.",
    "Remind me to call {name} after this.",
    "Open {app} and show us the error you got.",
    "Wait, stop, I think we skipped a step.",
    "Can somebody set a timer for five minutes?",
    "Delete the old version, it only confuses people.",
    "What's on the agenda for {day}?",
    "I think {thing} is due on {day} at {time}.",
    "{name}, can you take notes today?",
    "Undo that last change, it broke the layout.",
    "Okay, that makes sense. Let's go with the second option.",
    "Sorry, you were on mute. Could you say that again?",
    "Let's take this offline and pick it up on {day}.",
    "Find the email from {name} about {thing}.",
    "Turn it up a little, I can't hear the video.",
    "Close {app}, it's eating all the memory.",
    "Who owns {thing} now that {name} has moved teams?",
    "I'll put a meeting in for {day} at {time}.",
    "Does anyone mind if we start without {name}?",
    "Search for last quarter's numbers and paste them in.",
)
TUTORIAL = (
    "First, open {app} from the start menu.",
    "Now click on the extensions icon on the left side.",
    "Type PowerShell in the search box and press enter.",
    "In Visual Studio Code, open the command palette with control shift P.",
    "Next, create a new folder and call it projects.",
    "Where we store home directories depends on the system you use.",
    "Run the install command again and wait for it to finish.",
    "If you see an error here, restart {app} and try again.",
    "Delete the cache folder, then reload the window.",
    "Set the font size to fourteen, it's much easier to read.",
    "Remember to save your work before you close the editor.",
    "Open the settings and turn off automatic updates.",
    "Copy this line and paste it into your terminal.",
    "That's it for today, thanks for watching, and see you in the next video.",
    "Don't forget to like and subscribe if this helped you.",
    "Now we move the file into the shared drive.",
    "Select all the text and make it bold.",
    "Hit the stop button to end the recording.",
)
MISFIRE = (
    "PowerShell.",
    "Where we store home.",
    "Visual Studio Code.",
    "Thanks for watching.",
    "Thank you.",
    "Hey Felix, are you coming to lunch?",
    "Hello, Alex, good to see you.",
    "Hey, hello there.",
    "Helix is a nice name for a project.",
    "Hey, he left an hour ago.",
    "Hey, hold on a second.",
    "Okay, go ahead.",
    "Stop it, that tickles.",
    "No, no, no, the other one.",
    "Yes. Yes. Definitely yes.",
)
MENTION = (
    "I've been using Heliox for a month and it's fine.",
    "The review said Heliox is a desktop assistant for people who type a lot.",
    "My brother asked whether Heliox works on a Mac.",
)
READING = (
    "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in "
    "liberty, and dedicated to the proposition that all men are created equal.",
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so "
    "dedicated, can long endure.",
    "We are met on a great battlefield of that war. We have come to dedicate a portion of that field.",
    "It is altogether fitting and proper that we should do this.",
    "When in the course of human events it becomes necessary for one people to dissolve the political bands "
    "which have connected them with another, a decent respect to the opinions of mankind requires that they "
    "should declare the causes which impel them to the separation.",
    "The rain had stopped by the time the train reached the coast, and the platform was empty except for a "
    "porter and his cart.",
    "She opened the window, let the cold air in, and read the letter a second time.",
    "The recipe says to bake the bread for forty minutes, but my oven runs hot, so I take it out at thirty five.",
    "Every spring the river floods the lower fields, and every summer the farmers plant them again.",
    "The museum is closed on Mondays, so we went to the market instead and bought far too much fruit.",
    "He kept the old radio on the kitchen shelf, where it played the news every morning at seven.",
    "The committee will publish its findings at the end of the month, after the last hearing.",
    "Water boils at a lower temperature at altitude, which is why cooking times change in the mountains.",
    "The library extended its opening hours during exams, and the reading room was full until midnight.",
)


@dataclass(frozen=True)
class Line:
    kind: str
    voice: str
    speed: float
    text: str


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        name=rng.choice(NAMES), thing=rng.choice(THINGS), day=rng.choice(DAYS), app=rng.choice(APPS),
        time=rng.choice(TIMES),
    )  # fmt: skip


def pool(size: int = 160, seed: int = 20261112) -> list[Line]:
    """The unique sentences, each with one voice: rendered once, then reused at different speeds and levels."""
    rng = random.Random(seed)
    out: list[Line] = []
    seen: set[str] = set()
    fixed = [("reading", t) for t in READING] + [("misfire", t) for t in MISFIRE] + [("mention", t) for t in MENTION]
    fixed += [("tutorial", t) for t in TUTORIAL if "{" not in t]
    for kind, text in fixed:
        out.append(Line(kind, VOICES[len(out) % len(VOICES)], 1.0, text))
        seen.add(text)
    templates = [("meeting", t) for t in MEETING] + [("tutorial", t) for t in TUTORIAL if "{" in t]
    while len(out) < size:
        kind, template = rng.choice(templates)
        text = _fill(template, rng)
        if text not in seen:
            seen.add(text)
            out.append(Line(kind, VOICES[len(out) % len(VOICES)], 1.0, text))
    return out


def lines(count: int = 975, seed: int = 20261112, size: int = 160) -> list[Line]:
    """``count`` lines drawn from the pool, the way a day mixes them: meetings in runs of turns between two
    voices, tutorial stretches, passages read aloud, and the odd misfire phrase. ``speed`` here is the playback
    speed applied when a block is assembled (pitch kept), so one rendered sentence serves several speakers."""
    rng = random.Random(seed)
    sentences = pool(size, seed)
    by_kind: dict[str, list[Line]] = {}
    for line in sentences:
        by_kind.setdefault(line.kind, []).append(line)
    out: list[Line] = []
    weights = (("meeting", 0.5), ("tutorial", 0.2), ("reading", 0.17), ("misfire", 0.1), ("mention", 0.03))
    while len(out) < count:
        kind = rng.choices([k for k, _ in weights], [w for _, w in weights])[0]
        speed = rng.choice((0.9, 1.0, 1.0, 1.1))
        run = {"meeting": (3, 8), "tutorial": (2, 5), "reading": (2, 4)}.get(kind, (1, 1))
        if kind == "reading":
            start = rng.randrange(len(by_kind["reading"]))
            picks = [by_kind["reading"][(start + i) % len(by_kind["reading"])] for i in range(rng.randint(*run))]
        else:
            picks = [rng.choice(by_kind[kind]) for _ in range(rng.randint(*run))]
        out += [Line(p.kind, p.voice, speed, p.text) for p in picks]
    return out[:count]


def blocks(all_lines: list[Line], per_block: int = 75) -> list[list[Line]]:
    """Cut the lines into blocks of about five minutes, each rendered as one file."""
    return [all_lines[i : i + per_block] for i in range(0, len(all_lines), per_block)]
