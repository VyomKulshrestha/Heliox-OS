# The spoken turn: how Heliox answers when you talk to it

This note covers the fast conversational layer in `daemon/pilot/conversation/`: what a spoken turn did before it,
where the time went, and what the layer does now. Every measured number says where it comes from, in the sentence
that carries it:
- *owner recording*: the owner's own voice, with the microphone and the number of sentences;
- *synthetic voice*: text-to-speech voices from the Conversation Bench;
- *bench mix*: recordings mixed with synthetic noise or babble at a set level;
- *transcript replay*: decisions run over cached recogniser words, no audio;
- *text*: typed or written input, no audio at all;
- *from the code*: what the code does, not a measurement;
- *PortAudio's reported figure*: what the audio library says about a device, not measured;
- *under load*: measured while the machine was busy; a time marked so is not a result.
Settings (windows, limits, thresholds) are the code's own values. Almost everything here was heard from one voice,
the owner's; what that leaves unshown, and the recording that would show it, is in the last section.

## 1. The path of a spoken turn before this layer

1. **Listening and end of turn** (`pilot/system/voice.py`). The listener runs voice-activity detection on the
   microphone, a streaming recogniser for live captions, and the Smart Turn model to decide that the person has
   finished. On the owner's rehearsal recordings played through the real listener (`voice_probe.py`; owner
   recording, a handful of sentences, one voice, machine load not recorded), the end of the turn was declared 0.45
   to 0.65 s after the last word.
2. **The accurate pass.** Parakeet (or Whisper) re-reads the utterance. Measured on the same recordings (same
   source and caveats): Parakeet 0.23 to 0.78 s; when the wake phrase needed Whisper's confirmation, 2.2 to 2.6 s.
   From the last word to the dispatch of the command: 1.3 s for a sentence Parakeet settled, 2.8 to 3.3 s when
   Whisper ran as well. A timing set on a quiet machine is still owed (below).
3. **Dispatch** (`PilotServer._voice_command_dispatch`). In order: barge-in stops Heliox's voice; an approval
   attempt is refused ("I can't approve that by voice"); a denial is applied; `_control_reply` tries nineteen
   handlers Heliox answers in code (help, live replies, profile, goals, the day, to-dos, reminders, undo,
   routines, shortcuts and others); stop and soft-stop words; corrections to a running task. These take tens of
   milliseconds (from the code's own logging), and a local answer was heard 0.2 s after dispatch plus synthesis
   (0.3 to 0.6 s to first audio in the probe; same source and caveats).
4. **Everything else went to the planner.** "How are you", "what do you think about this error", "explain what
   you just did" and "open Notepad" all took the same road: interaction state, `_handle_execute`, the configured
   cloud model writing a JSON plan, then the reviewers (world model, safety review, Gnomon), the executor and the
   verifier. A stock acknowledgement chosen from five phrases ("I'm on it.") was scheduled 2.5 s after dispatch,
   and the result was spoken when the planner returned.

What decided the road was the order of those handlers: anything no handler claimed was a task. Nothing told a
question from a request.

### Where the waits were

From the daemon log of the November rehearsals (owner recording, live; the planner road's times are mostly the
model's), dispatch to the end of the turn on the planner road:

| Utterance (as recognised) | What it was | Dispatch to done |
|---|---|---|
| "good afternoon" | a greeting | 27.1 s |
| "Open" | a fragment | 48.8 s |
| "Let's try and take a look." | a remark | 19.2 s |
| "Where we store home" | misheard | 11.0 s (cut off by the person) |
| "Google, Google, Google." | noise | 61.0 s |

For conversation, the first sound was the stock phrase at 2.5 s plus synthesis, and the answer came 10 to 60 s
later. Noise and fragments became plans. The planner model itself (Claude Opus 5.5 on the owner's machine)
returns a plan in 10 to 15 s before any review (daemon log).

## 2. The design

### Modules

| Module | Job |
|---|---|
| `router.py` | Decides in code what an addressed utterance is: control, local, conversation, action or unclear. No model; about 0.02 ms per call (a code timing). |
| `style.py` | The fast model's instructions; cleaning model text for the ear (no markdown, paths, links, symbols, stock openers); cutting streamed text into sentences; reading the model's hand-over line. |
| `context.py` | The short-term conversation (in memory only) and the context block the model gets each turn, within about 350 tokens. |
| `fast_model.py` | One streamed model per provider kind, all counted in the same budget tracker as the planner. |
| `handover.py` | The hand-over to the planner: the structured intent, the choice of target, the spoken acknowledgement, progress and the approval pointer. |
| `layer.py` | The turn itself: `take`, `interrupt`, `clear`, `prewarm`, and the result the server needs. |
| `adapters.py` | The thin glue to the server, the focused-app lookup and Live, each answering "not available" when missing. |
| `provisional.py` | Provisional answers: the model starts on the listener's words before the turn is confirmed, everything held until the final words match. |

### The turn

```
routed --(conversation / unclear)--> answering --> speaking sentence 1..n --> done
   |                                     |  \--(the model writes an ACT line)--> handed over
   |                                     \--(no model / error)--> one sentence saying why --> done
   |--(action)--> handed over --> acknowledgement, progress, approval pointer --> planner result --> finished
   \--(control word no handler took)--> "Okay." --> done
any state --(the person talks, or a new utterance arrives)--> cut
```

The router runs after every existing handler, so control words, local answers and approvals behave exactly as
before. What reaches it is what used to go straight to the planner:

* **Action** (an imperative with a real object: "open Notepad", "set a timer for ten minutes", "how much disk
  space do I have"): straight to the planner, no model call. The person hears a sentence built in code from the
  parsed verb and object, at once.
* **Conversation** (questions, remarks, opinions, explanations, follow-ups, and requests whose object is a
  pronoun when a conversation is open, such as "open it"): the fast model, streamed.
* **Unclear** (a bare verb, the wake phrase alone, the recogniser repeating one word, "open it" with nothing to
  refer to): the fast model, which says what it heard and asks. It never acts on a guess.

The table in `tests/test_conversation_router.py` has 120 utterances, 51 of them the owner's own recognised speech
(text: the recognisers' words for one voice; **Fakes only** in the sense that no audio is involved). None of the misheard ones reaches the planner as an action; how
other voices are misheard is not yet shown.

### Streaming into speech and cutting off

The model's text is fed to a sentence splitter as it arrives. Each finished sentence is cleaned and handed to the
existing speech path (`_speak_companion_text`: the speech coordinator, the barge-in watcher and the Kokoro
worker) at once, while the model keeps writing. A long first sentence is cut at a comma after 70 characters so
sound starts sooner. Before each sentence the layer checks that the person is not talking.

The turn stops when the sink reports the sentence was interrupted or cancelled, when the person is talking before
the next sentence, or when a new utterance reaches dispatch (`interrupt()` is the second line of the dispatch).
Stopping closes the model stream, so generation and its cost stop, and drops sentences not yet spoken. The
conversation keeps only what was said: finished sentences plus the interrupted one, marked as cut off, and the
model is told on the next turn that the rest was not heard.

If the first sentence is not ready 2 s after the turn reached the layer, a short holding phrase is said ("One
moment.", never the same one twice running). A cloud model rarely needs it; a local model loading into memory does.

**The voice's first sound comes from the first clause.** Whatever hands it text, the local voice
(`local_tts_worker.split_for_speech`) cuts a first sentence of more than 11 words at its first natural pause within
nine words (after a comma, semicolon, colon or dash; else before "and", "but", "which" and the like from the fourth
word on) and synthesises that head first. The rest follows in pieces of at most twice the words of the piece before,
so each is made while the one before plays. A sentence with no such pause is made whole, as before. Kokoro pads every
synthesis with 0.32 to 0.40 s of silence before its first word and 0.45 to 0.55 s after its last; each piece now keeps
30 ms before its first sound, and a piece that ends mid-sentence keeps 0.25 s after it, so the seam (with the output
line's hand-over of about 0.1 s) is about the 0.32 to 0.47 s Kokoro itself pauses at a comma inside one synthesis.
The system voices (`say`, espeak-ng, Windows' speech) already speak as they go and are unchanged.
Measured 12 November on the owner's machine (Kokoro on the CPU with 4 threads, the runner's core count; 15 to 40 %
load from other work, so read these as about, not exact): Kokoro makes 0.38 to 0.45 s of work per second of speech,
whatever the sentence length. Four replies of 15 to 20 words: the whole first sentence took 2.2 to 2.6 s to make,
the head 1.1 to 2.0 s, and every later piece was ready before the one before had finished playing. So a CPU-only
person keeps Kokoro as the default: it runs at under half real time, so speaking clause by clause never waits
for it. What it costs them is the head's synthesis, about 1 s, against about 0.1 s on a GPU.

**A shorter head was measured and rejected (13 November).** The next idea was to cut a one-to-three-word opener
("Okay,", "Sure,", "Got it,") off as its own piece, so the first sound comes sooner. Measured with Kokoro alone on
the CPU with 4 threads, under a scratch root, with the processor 67 to 83 % busy from other work (one run each, so
read these as about):

| Reply | whole first sentence made in | opener made in | next piece made in | opener plays for | silence after the opener | pause at that comma inside one synthesis |
|---|---|---|---|---|---|---|
| "Okay, your files..." | 2.59 s | 1.29 s | 2.36 s | 0.80 s | 1.94 s | 0.12 s |
| "Sure, I'll remind..." | 2.52 s | 1.31 s | 2.28 s | 0.74 s | 1.92 s | 0.14 s |
| "Got it, opening..." | 2.98 s | 1.26 s | 2.67 s | 0.81 s | 2.24 s | 0.08 s |
| "Okay, your next meeting... at two." | 4.45 s | 1.33 s | 4.05 s | 0.80 s | 3.63 s | 0.43 s |

The opener sounds about 1.3 s after the text arrives instead of 2.5 to 3 s, but the next piece is nowhere near ready
when it ends, so the person hears "Okay," and then 1.9 to 3.6 s of silence. Cutting the rest shorter did not help,
because Kokoro takes about 1.3 s for any piece, however short. On a graphics card the head is already about 0.1 s,
so there is no machine the opener helps. Do not try it again without new evidence about synthesis speed.

**Heads said often are kept as audio (`local_tts_worker.HeadCache`).** When the first piece of a reply is exactly a
piece of six words or fewer that this Heliox has already spoken twice with the same engine, voice, speed and trim
settings, the kept audio plays at once and nothing is synthesised for it. The key is the exact text and those
settings, so a reply that only starts like a kept head ("Okay, done then." against a kept "Okay, done.") is made in
full: the cache never adds, drops or changes a word. It holds at most 40 entries (the least recently used go first,
about 3 MB in all), lives in `~/.cache/heliox/voice_heads` on Windows, macOS and Linux (removed with Heliox's other
caches on uninstall), and drops every entry made another way when the voice or its settings change. It is filled only
from audio already made for a reply. Of the 129 distinct sentences the conversation and personal layers write
themselves, 62 start with a fixed piece of six words or fewer that it can keep ("Okay.", "Done.", "Working on it.",
"Next track.", "Still checking the plan.", "Sorry, I didn't catch that."), and 45 of those are the whole reply. Once
said twice, those play at once instead of after about 1.3 s of synthesis on a CPU. It does nothing for the model's
own answers unless one starts with exactly such a piece.

**No cheap seconds are left in the listener or the voice on a processor.** Measured on the owner's machine: the
end of the turn is decided 400 ms after the last word, and ending it 200 ms sooner costs false starts (section 6);
Kokoro on the CPU makes even a one-word piece in about 1.3 s and a first sentence in 2.2 to 4.5 s, by length and load; the
clause cut already makes every later piece ready in time, and a shorter head only moves the wait behind the first
word. The remaining time is the model's first word and the synthesis speed itself, so that is where the next person
should look: a faster first word from the model, a faster voice model or runtime, or the graphics card, not new cut
points.

**Whisper on an NVIDIA card.** Whisper's runtime needs NVIDIA's CUDA 12 cuBLAS and cuDNN libraries, which only the
"Voice on the graphics card" part installs (docs/FIRST_RUN.md). Measured on the owner's laptop on 12 November with
those two libraries added to the development environment (an RTX 4060 Laptop GPU; Whisper base; the libraries'
folders added the way the daemon adds them):
- On the card: 1.6 s to load, then 0.10 s a pass for 2, 4 and 6.5 s of speech alike.
- On the processor: about 0.5 to 1 s a pass (about 1 s for 6.5 s of speech at a third of the CPU busy).
- So a person with a working card has the Whisper wake check and the Whisper final pass in about 0.1 s instead of
  0.5 to 1 s. With Parakeet as the final pass (it stays on the processor: its GPU build is not on PyPI), only the
  wake check gets faster.
- Temperature: this card idles at 79 to 80 °C, and the GPU arbiter's limit is 83 °C. Whisper base on it ended at
  80 °C; Whisper small was not measured on it, because the card reached 83 °C before its turn. On a laptop like this
  one the arbiter will often keep Whisper on the processor, and that is the intended result.

### Starting the answer before the turn is confirmed (`conversation.provisional_answers`)

A cloud model's first word takes about 1.0 s (Claude Haiku 5.5: median 1.04 to 1.18 s over three paid runs). Before
this, that second began only after the endpoint and the accurate recognition pass had decided. With a provisional
answer it runs while they decide.

**How it works:**
- When the person has probably stopped (200 ms into a pause), the listener hands the layer the words it has and a
  turn id (`PilotServer.provisional_voice_turn`). The layer starts the fast model on those words.
- It holds everything: nothing is spoken, shown, kept in the conversation, handed over or offered, and no action or
  approval can come from it.
- The final words arrive as the normal dispatch under the same turn id. If they are the same words, the turn takes
  over the running answer and speaks and keeps it as its own.
- "The same words" means equal after folding case, dropping punctuation, removing "um", "uh", "er", "erm", "hmm" and
  "mm", and reading numbers in words as digits ("twenty two" is 22, "nineteen eighty four" is 1984, "one two" is
  never 3). Nothing looser: answering a question the person did not ask is worse than answering a second later.
- Otherwise the call is closed, so the reply stops costing output tokens, and the turn asks the model as before.
- If the person carries on, says something else (a stop, a new request), or no final words come within 8 s, the
  provisional answer is dropped. Nothing was said from it.

**When it may start:**
- Only for an utterance already known to be addressed to Heliox before its end: the microphone button, or the wake
  phrase on the same proof the final addressing rule accepts.
- Never for a follow-up without the wake phrase, a continuation, a doubtful follow-up, a cut-off, speech in a call,
  or anything the addressing rule has not taken. The listener's replay gate counts how many of the 981 unaddressed
  corpus lines would have started one; it must be zero, or the feature ships off.
- Only on the spoken fast-model road. Requests to act, control words, unclear words, answers given in code (the time,
  sums) and turns during a running task or a waiting approval never start early.

**Money.** A dropped start costs about what an answer costs: the question is billed, and only the reply's few dozen
tokens are saved. In the paid runs, 36 Haiku calls cost $0.00494, about $0.00014 a call. The limits:
- at most 6 dropped starts a minute;
- dropped starts may cost at most $0.05 an hour, whatever the model, by the budget tracker's own estimate;
- once the session has made ten calls, dropped starts never cost more than a tenth of what its other conversation
  calls have cost;
- past a money limit, none start until the hour turns;
- off for the rest of the session once the month's spend reaches 90% of `budget_monthly_limit_usd`.

Every provisional call is recorded by the budget tracker like any other fast-model call, with
`agent_id = "conversation.provisional"`. On Haiku, a turn every 15 s with every start dropped would add about
$0.03 an hour; the 6-a-minute limit holds the most it can waste to about $0.05 an hour.

**The setting.**
- `"off"`: never.
- `"on"`: for every model, local ones included.
- `"auto"` (the default): off with a model on this computer, which would share the CPU with the accurate
  recognition pass. With a cloud model, on only when the measured adoption rate on the recorded corpus is at least
  80%. The adoption rate is how often the streaming words at the last pause equal the final words after the
  normalising above.
- Measured adoption rate: none that buys time, so "auto" is off and `provisional.MEASURED_ADOPTION_RATE` stays
  `None`. What was measured (12 November, the replay corpus's 981 unaddressed lines, the owner's 13 addressed
  recordings and 35 synthetic "Hey Heliox ..." clips; the listener's side is not built):

  | Words the listener could hand over at the last pause | Same as the final words | Head start before the final dispatch | Unaddressed lines that would start one |
  |---|---|---|---|
  | The streaming recogniser it has (zipformer), as it stands | 0 of 4 | about 0.7 s on the runner | 0 of 981 |
  | The same, flushed with 200 ms of silence | 1 of 10 | about 0.7 s | 0 of 981 |
  | The speculative Parakeet pass | 19 of 21 (90 %) | none in the median case, at most about 0.2 s: the final dispatch waits for this same pass | 0 of 981 |
  | Nemotron Speech Streaming EN 0.6B (sherpa-onnx's int8 export, 560 ms chunks), flushed | 17 of 20 (85 %; 19 of 20 counting "tomorrow" and "to morrow" as equal); call-mode headset 0 of 5 | median -0.05 s (-0.28 to +0.21): flushing its last chunk costs about what the Parakeet pass does | 0 of 981; and 0 of 981 dispatched with its words acted on |

  "Unaddressed lines that would start one" counts the predicate a provisional call must pass: the early opening read
  holds the wake phrase, `early_wake_agrees` with the words, the addressing rule's wake check on the merged words
  says "wake" (without it, 3 of 981 passed: "Helix is a nice name for a project", read as "Heliox is a nice
  name."), at least three words, and not `sounds_unfinished`. The predicate needs the opening read, so an utterance
  shorter than 1.6 s never starts one.
  Nemotron, on this laptop's CPU (4 threads, the turn detector beside it; the machine was 92 to 98 % busy with other
  work, so the times are pessimistic): it decodes at 0.53 of real time (0.64 at the 90th percentile), the flush at
  the candidate end takes 0.34 s (0.54 s), it holds about 0.7 GB of memory, and its files take 0.63 GB. It ran on
  half of four cores whenever anyone spoke, in or out of a conversation.
  **Why it is parked:** none of these gives words that are both early and right. The ones that are right come no
  sooner than the final dispatch, and the early ones are wrong. Even a few tenths of a second would be small beside
  the 2.1 to 2.9 s to a model's first sound on the runner.
  **What would reopen it:** a streaming recogniser whose words at the last pause match the final words in at least
  80 % of addressed lines (call-mode headsets included) and arrive at least 0.3 s before the final dispatch, at a
  cost on a 4-core CPU the voice and the turn detector can share. Measured the same way, against the same gate.

**What has been tested.** **Fakes only**: `tests/test_conversation_provisional.py`, on the real server and the real
streaming client over a stubbed transport, driven by a fake listener through the interface the real one will call.
The listener's side is not built yet. The tests cover:
- an adopted answer: one request, and the first sound earlier by the overlap;
- changed words: a new request, and the provisional answer never said;
- carrying on, words not taken, a stop, and no final words: nothing said or kept;
- every way of being addressed except the wake phrase and the button: no request at all;
- the roads that never start early;
- the budget, per-minute and money limits, and the setting.

### The spoken style

The system prompt (in `style.py`) asks for one to three short sentences, the answer first, no markdown or lists
or paths, no stock openers, honesty when it does not know, and a question when it misheard. It tells the model it
can only talk: the planner acts. It knows the hard lines: voice cannot approve (the person presses Approve on the
screen), and nothing is handed over on a guess. Post-processing removes what slips through, and a sentence that
claims an action ("I've opened...") is dropped, since the fast model cannot have done it.

### Context

Each turn the model gets, in this order, dropping from the end when the block passes about 1,400 characters:

1. the time and day;
2. what Heliox is doing: a running task and its phase, an approval waiting (with the reminder that voice cannot
   approve), the last task result;
3. the focused app (see the table below);
4. profile lines from `UserProfile.context_lines`, chosen for the words (see "What the model is sent about the
   person" below), and the person's answer length when they set one;

plus the last six turns of this conversation as real message turns. The turns live in memory only, are cleared by
the existing "new conversation" call and lapse after ten quiet minutes. Speech that was not addressed to Heliox
never reaches this layer, so it is never stored as conversation or sent to a model.

### Sharing the floor with Heliox Live

Live (slips, risks, offers) and the conversation used to start speaking on their own checks, and Live's correction
channel outranks an answer, so a Live remark could cut a reply mid-sentence. `pilot/live/floor.py` holds one rule,
measured on the Live Bench's `converse` steps (text; spoken remarks over a reply: 3 per persona before, 0 after,
recall and false alarms unchanged):

1. Nobody starts while the person is talking.
2. A conversation turn holds the floor from dispatch until it is answered (`PilotServer._voice_turn`), plus 3 s.
   Live shows cards at once (an urgent risk is never held) but does not speak; a non-urgent typed catch waits and
   is said after.
3. A catch about the words that opened the turn may be spoken first, only before the turn has said anything; once
   the reply has started it becomes a card.
4. So Live never asks to speak during a reply, and a reply that starts during a Live remark waits behind it.

What Live says aloud is recorded in `ConversationMemory` as a Heliox turn (cut off if talked over). The floor rule is
**Fakes only**: measured on the Live Bench's simulated `converse` steps, not yet in a real voice session.

**Answering Live.** For 6 s after Live finishes a spoken remark (`LIVE_REPLY_SECONDS`, a setting), with no
conversation open, the listener takes Live's own closed list of replies without the wake phrase
(`pilot.live.engine.is_live_reply`: "why?", "why did you say that", "thanks", "not now", "don't mention these",
"fix it", "never mind", stop words) and nothing else. A reply taken this way opens no conversation, and only one is
taken per remark; Live speaking again opens the next. The 6 s are counted on the capture clock, from the end of
Live's voice to the start of the reply, so a recognition backlog does not close them. A call, a wake from sleep and
anything that clears the follow-up window close them too. A reply begun over Live's voice is not one: a word or two
said over it is a cut-off ("stop"), as before, and a longer sentence over it is not taken. Every other sentence is
not taken and opens nothing (`tests/test_live_reply_window.py`, with the real `Addressing` and listener; text). Of
the 981 utterances not addressed to Heliox in the synthetic hour (transcript replay, both recognisers' words), none
is one of these replies, so the room speech measured so far could not be taken in the window; a real room has not
been recorded yet. The dispatch is marked `voice_turn_addressed` "live_reply". **Fakes only**: the real listener and
`Addressing` fed recognised text, and the transcript replay; no recorded voice session yet (the conversation bench's
audio-level run is pending).

### What leaves the machine

`conversation.share_screen_text` is `auto` by default.

| Setting | Model on this machine (Ollama) | Cloud model (Claude, Gemini, OpenAI-compatible) |
|---|---|---|
| `auto` | app name, the end of the focused text field | app name |
| `on` | app name, the end of the focused text field | app name, the end of the focused text field |
| `off` | app name | app name |

* The app name is the display name ("VS Code", "Windows Terminal"), never the window title.
* The text is the last 400 characters of the field Heliox Live read (Live must be on; never a password field,
  never text read off the screen by OCR), left out whole when it contains anything that looks like a key or
  password (Live's own patterns).
* For Heliox's own window, password managers and apps excluded from the timeline, nothing is sent, in every row.
* What reaches a model in these rows is **Fakes only**: unit tests of the context builder; no request body sent to
  a real provider has been captured and checked yet.
* The person's words, the earlier turns of this conversation, the time and the task state go to whichever model
  is configured; the profile lines go as chosen for the words ("Personal sessions" below). With a local model none of
  it leaves the machine.

Two more context sources, added with Heliox Live and the personal layer (`pilot/live/conversation_link.py`,
`pilot/personal/context.py`):

| What | Model on this machine (Ollama) | Cloud model, `share_screen_text` `auto` or `off` | Cloud model, `on` |
|---|---|---|---|
| A Live remark about what the person *said* to Heliox (a wrong weekday, a promise), as a Heliox turn | as said | as said (it quotes addressed speech the model already gets) | as said |
| A Live remark or card about what the person *typed* (it quotes screen text), as a Heliox turn or the "Live raised this" line | as said, with the evidence | only the kind and the app ("pointed out a weekday that doesn't match its date in Notepad"); the words stay here | as said, with the evidence |
| Up to five profile, goal and habit lines that share words with this utterance, plus the answer style and preferences (one line, at most 360 characters, picked by word overlap in code) | yes | yes, like the profile lines above | yes |
| Offers ("You usually open Outlook about now"), as a Heliox turn | yes | yes (app names only, never window titles) | yes |
| Answers to the getting-to-know questions (name, main use, how to talk) | never; parsed in code | never | never |
| Speech heard near the microphone but not said to Heliox (Live does not check it: no card) | never | never | never |
| A note said in passing ("my flight is on Friday"), kept until the day after its date or for a week | yes, as a profile line | yes, as a profile line | yes |
| What those answers stored (the name, "I mainly use the computer for ...", the answer style, a speak-up or stay-quiet preference) | yes, as profile lines with later turns | the name and the answer style with every turn; the rest only when the words bear on it | the same as `auto` |
| An answer Heliox gave in code earlier in the conversation (the person's words always go) | the answer, whatever it was | only sums, units, dates, the time and the spoken help; every other code answer (the day, the timeline, to-dos, reminders, notes, the profile, goals, memory, workspaces, suggestions, Live, settings, and any handler added later) is sent as "(answered on this computer)" | the same as `auto` |

"Why did you say that", "thanks", "not now", "don't mention these", "fix it", "remember that ...", "forget ...",
"what did I tell you about ..." and the answers to an offer are handled in code before the layer and never reach a
model. Offers and the relevant-lines pick are computed on the machine with no model call.

Never kept, by any route that stores something about the person (the profile's own phrases, "remember that ...",
notes said in passing, the getting-to-know answers): health, money, other people's private details, and anything
that names or looks like a secret (`pilot.personal.never_kept`, one rule applied in `UserProfile.add`). Heliox says
it did not keep it and why. Anything like that stored by an earlier version is left out whenever the profile is read
and removed from it and from long-term memory the next time a profile phrase is said; the next "what do you know about
me" says once how many things were dropped and of which kind, without repeating them. **Fakes only**
(`tests/test_never_kept.py`, and the "not learned: things never kept" personal session).

A correction replaces the old fact: "no, call me Sam", "actually, my name is Sam", "not Alex, Sam", and "it's Sam"
said within ten minutes of giving a name all rename, and the greeting, the context lines and "what's my name" use the
new name. "Stay quiet unless I ask" (also "be quiet", "don't interrupt me", "only speak when I ask"), said any day,
is a profile preference: offers stop, suggestions stop, and Live holds every catch except the urgent kinds (a risky
command), the same hold a focus session gives; "you can speak up again" undoes it. Recall answers questions in a
person's own words about what they told Heliox ("when do I stop work?", "where do I live?", "who is my manager?");
a question about something never told goes on to the planner or the model, and so does one about a note said in
passing, which the model sees in the context. Getting to know stores the same quiet preference and leaves the
Settings for offers and Live's level as they were. "I stop work at six" is kept with its
time, and from then until 05:00 offers and suggestions wait like quiet hours, with one sentence the first time each
evening saying why (`pilot.personal.wiring.quiet_now`). **Fakes only** (`tests/test_personal_rules.py`, and the
"corrected: the name", "told once: being interrupted" and "told once: when work ends" personal sessions).

### Handing over to the planner

`take` returns `done=False` with a `HandoverIntent`: the verb, the object, the person's words, the request
written out on its own, where it came from (`router` or `model`) and the model's own acknowledgement if it wrote
one. `choose_target` returns the target: `QUICK` when a one-action plan on the quick road's allow-list was built
for the intent (next section), otherwise `FULL_PLANNER`, with the plan, its reviewers and the approval rules
unchanged.

When the fast model decides an action is needed it writes one line,
`ACT {"verb": ..., "object": ..., "request": ..., "say": ..., "action": ...}`, which is never read aloud; `action`
is optional, one action in the planner's own schema, used only on the quick road's terms below. The planner's request is
always the person's own words. The written-out request ("Open launch-notes.md in Notepad" for "open it") and
the last few turns of the conversation go to it separately, as context under their own heading, at most six
turns and 1,200 characters, with what Live read from the screen left out unless `share_screen_text` allows it
(`pilot/conversation/planner_context.py`). The reviewers (Gnomon, the safety review, the double-check) judge the
plan against the person's words and their own earlier lines, never Heliox's lines or the model's reading. The
person hears the `say` sentence. A malformed line is treated as unclear, and a reply that declines or asks what
was meant hands nothing over, even with a promise or an ACT line in it.

While the planner works, `keep_company` speaks: the acknowledgement at once; if planning passes 2 s, one sentence
from the interaction state ("I'm working out the steps.", "I'm checking the plan for risks."); then, as long as the
work goes on, a new line every 9 s: what changed, which step it is on, or a "still" line ("Still planning; this one
takes a while."), never the same sentence twice running and never while the person is talking; the first long
plan from a model on this computer says once that it is slow the first time (**Fakes only**: a 40 s planner on an
injected clock, `tests/test_conversation_company.py`); and, when an approval is needed, "This one needs your
approval. It's on the screen; press Approve if you want it to go ahead." The server cancels it when the planner
returns and speaks the result as before; the layer records the result so the next question can refer to it. A
sentence the listener split at a pause and joined is acknowledged once.

**Run for real** for the hand-over decision after some talk: `benchmarks/conversation_eval/handover_context.py`,
nine scripted conversations on Claude Haiku 5.5, text, two runs on 12 November ($0.0026 and $0.0038). The second
run broke none of the cases' must-not rules; four of nine were handed over, carrying 9 of 14 earlier facts. Five
paid planner calls (Claude Opus 5.5, about $0.11) checked two of those cases: with the context the planner
listed Downloads before moving only the 2026 invoices, and still asked before deleting. Everything else in this
section is **Fakes only** (unit tests).

### The quick road

A plain spoken request ("open Notepad", "mute", "next song", "how much disk space do I have") does not need a
model to write a plan. `pilot/conversation/quick.py` builds a one-action `ActionPlan` for it, and the server runs
that plan through the same `_handle_execute_inner` as a planned one, without the planner call.

What may go this way (`quick.ALLOWED`), and nothing else:

| Request | Action | Built how |
|---|---|---|
| open an app by name | `open_application`, no arguments | router parse |
| open a web address; search the web | `open_url` (http or https only; the search page is `conversation.search_url`, Google's by default) | router parse |
| open Desktop, Documents, Downloads, Pictures, Music, Videos or home | `open_application` of the system file manager with that one folder | router parse, Windows and Linux |
| switch to an app | `window_focus` by title | router parse |
| mute, unmute, set the volume, volume up or down, play/pause, next, previous | `volume_mute`, `volume_set`, `keyboard_press` with a media key | router parse |
| battery, disk, memory, processor | `battery_info`, `disk_usage`, `memory_usage`, `cpu_usage` | router parse |
| the same, said with a pronoun or a paraphrase | one of the above | the fast model's `action`, checked the same way |

Timers, reminders and to-dos are already answered in code before this layer is asked.

**Why this is safe.**

1. *Only actions the tiers never ask about.* Every allowed type is tier 0 or 1 in `pilot/actions.py`, so
   `PermissionChecker` never asks approval for it, and none is irreversible. `quick.admissible` checks this for each
   action, plus: no root, no rollback, no chaining; `open_application` takes no arguments and never names a program
   that runs commands (shells, interpreters, script hosts, installers, terminals, the file manager); the one
   exception is the file manager showing one of the person's known folders, built in code and never accepted
   from a model; `open_url` only `http(s)`; `keyboard_press` only media keys, never text or shortcuts; a volume level
   from 0 to 100. Nothing that writes, deletes, moves, sends, installs, runs a shell command or types into another
   app can be expressed.
2. *The same pipeline.* The plan enters `_handle_execute_inner` as the first attempt's plan. Everything after the
   planner call is unchanged: the world model's risk assessment, the permission check, the executor (with
   `ActionValidator` and the protected-path rules), the verifier, the Activity log, the timeline, plan history,
   experience records and the undo journal. Only the model reviewers (the double-check, Gnomon) are skipped; they
   can only add an approval, and the quick road never proceeds where an approval would be needed (next point).
3. *It falls back, never forward.* If the permission check, the world model or the critic would ask for approval,
   the server drops the quick plan before any approval is created and plans the request normally, with every
   reviewer. If the action fails, the existing retry replans with the planner, or reports the exact error.
4. *Voice still cannot approve.* The quick road never creates an approval, so there is nothing to approve.
5. *It cannot be injected.* The server accepts the quick plan only as an `ActionPlan` object, which no client
   message can contain (JSON arrives as a dict and is ignored), and drops it when a voice shortcut rewrote the
   words.
6. *The planner is untouched.* The everyday planner eval exercises `Planner.plan`, which this does not change.

What the person hears: the sentence at once ("Opening Notepad.", "Searching the web for flights to Lisbon."), then
the result in a sentence ("Notepad is open.", "The battery is at 84 percent, discharging.", "You have 197 gigabytes
free on drive C."); a failure says what failed ("That didn't work: Application 'x' was not found.").

**Per system.**

| | Windows | macOS | Linux |
|---|---|---|---|
| Open an app | App Paths, Start menu, Store apps (`applications.py`) | `open -a` (Launch Services) | `PATH`, then `gtk-launch` with a desktop entry |
| Open a known folder | Explorer | to the planner (`open -a` cannot show a folder) | `xdg-open` |
| Web address, web search | `start` | `open` | `xdg-open` |
| Switch to an app | PowerShell window activation | Accessibility, then `osascript` activate | `wmctrl` (X11; most Wayland desktops refuse) |
| Volume | Windows audio API | `osascript` | `pactl`/`amixer` |
| Media keys | `pyautogui` | `pyautogui` (needs Accessibility) | `pyautogui` on X11; not under Wayland |
| Battery, disk, memory, processor | `psutil` | `psutil` | `psutil` |

The decisions (which plan for which words, on which system) are pure and tested on all three (**Fakes only**).
Running the actions is **Run for real** on Windows only (the owner's machine: Notepad opened, battery and disk
read); on macOS and Linux it is **Not tested** by this layer (the platform agent's checks cover launching apps).

### Typed chat

A message typed in the app goes through the same layer: `_handle_execute_inner` calls `take(text,
source="typed")` right after the local answers, for the sources "interactive" (the chat box), "text" (queued
tasks) and "chat" only; voice, MCP, autonomous jobs, phone requests, schedules and replays never reach it (a test
pins the set), with a sink that sends the reply to the chat view as
`token_stream` notifications while it is written (the view already shows those as they arrive). The final text
is the task result, rendered as markdown as before. A typed reply keeps its markdown (lists, code) and is not
cleaned for the ear; it uses a prompt for a chat window, has no holding phrase and no sentence limit. Requests
take the same hand-over and the same quick road, with the same rules; a typed request whose plan needs approval
opens the approval dialog exactly as before. Messages with attached files, revisions and expanded shortcuts keep
the planner road. While the reply streams, the chat view renders it as markdown as it grows (an open code
block is closed and a half-typed bold or code marker is held back until it closes), so it does not jump when it
ends. **Fakes only**: unit tests and an end-to-end test against the real server with a stubbed model; no typed
conversation with a real model has been recorded.

### When the model fails

| What happens | What the person hears | Then |
|---|---|---|
| No network (DNS, refused, reset) | "I can't reach the language model right now..." once, then a short line | 30 s without asking the provider |
| The provider accepts the connection but sends nothing for 4.5 s | the holding phrase at 2 s, at 4.5 s "The language model is slow right now, so I stopped waiting. Ask me again." | the next turn asks again; only late answers in a row rest it (5 s after two, 15 s after three, then 30 s), and an answer resets the count |
| A local model loading into memory | "Loading the local model, one moment." (once a session) | waits up to 90 s for the first word |
| The local server is not running, or the model is not pulled | "...the local one isn't running" / "...isn't installed", with where to fix it | 30 s |
| The provider account is out of credit (Anthropic 400 "credit balance", OpenAI 429 `insufficient_quota`, a 402) | "Your Anthropic account is out of credit. Add credit there, or choose another model in Settings, under Model." | 10 minutes without asking |
| Rate limited (429) | "The language model is busy right now. Try me again in a moment." | the provider's `Retry-After`, else 20 s |
| Heliox's own monthly budget is used up | "This month's model budget is used up..." | until the budget allows |
| The connection drops part-way through a reply | what was said, then "I lost the connection there." | as for no network |
| A Wi-Fi sign-in page (or any proxy's own page) answers instead of the provider: 200, not an event stream or JSON | "I'm getting a web page instead of the language model. The network may want you to sign in first." | 30 s without asking, as for no network |
| A proper stream that carries no text at all | "The language model sent back nothing that time. Ask me again?" | the next turn asks again |

Every row is a test with the provider's own stream format stubbed (**Fakes only**: `tests/test_conversation_models.py`,
`tests/test_conversation_every_person.py`, `tests/test_conversation_portal.py`).

Three rows also ran for real, through `tests/platform_real/network_faults_check.py`. Each check used real sockets in
front of the real client, on the ubuntu-24.04, macos-latest and windows-latest CI runners (run 43223177292 on 0fcfc1d,
all three passing). The servers were local, not a real portal or a real outage. **Run for real** for:

- **No network**: a name that does not resolve gave the "can't reach" line.
- **Slow first word**: the holding phrase came at 2.0 s, and the stopped-waiting line at 4.5 s, within its 4.5 s bound.
  Nothing else was said.
- **Sign-in page**: its line came within 0.0 s (0.2 s on Windows).

The out-of-credit and rate-limit replies are written from the providers' documented errors, not captured from a real
account.

### Claude over plain HTTP

Claude is reached like every other cloud provider: the Messages API with `stream: true`, read as server-sent
events with `httpx`, recorded in the budget the same way (input tokens from `message_start`, output from
`message_delta`, the abandoned-call estimate when cut off before either). The connection is opened ahead with the
free token-counting call when the person starts talking, at most every 45 s.

The Anthropic SDK was tried and dropped. **Run for real**, measured on 12 November, three calls each, alternating, both connections
warmed (network time on a home connection; machine load not recorded): SDK first token median 0.88 s (0.72 to
1.09 s), plain HTTP median 0.77 s (0.75 to 0.77 s); six calls cost $0.0005. Importing the SDK took 22 to 36 s on
the owner's machine (load not recorded; about three minutes without compiled bytecode), which a turn would have waited
for.

### The model for each kind of person

The fast model is a setting per provider (`conversation.claude_model`, `gemini_model`, `openai_model`,
`openrouter_model`, `meta_model`, `ollama_model`). Empty means the provider's fast tier where one is known for
certain, otherwise the model the person already configured.

| Person | Fast model | Streamed | First sound for a question (expected) | Told when missing |
|---|---|---|---|---|
| Claude key | `claude-haiku-5-5`, thinking off, effort low, over plain HTTP | yes | see section 3 | key rejected, budget used up, offline: one sentence each |
| Another cloud (Gemini, OpenAI, OpenRouter, Meta) | the configured model (Gemini Flash with thinking off) | yes, server-sent events | about one round trip plus synthesis | the same |
| Local only (Ollama) | the configured local model | yes, from the local server | seconds while the model loads; the holding phrase covers it | "the local one isn't running" or "isn't installed", with where to fix it |
| No model at all | none | n/a | instant, in code | once per session: "That needs a language model, and none is set up. You can add a local model or a cloud key in Settings, under Model." (24 words); afterwards one short line |

In every case control words, local answers and approvals work as before, and a request to act still reaches the
planner with a specific acknowledgement. Every call is checked against and recorded in the budget tracker the
planner uses, with the same provider names and `usage_source` values; a stream cut off by the person records the
usage it had reported, or the existing abandoned-call estimate when it had reported none.

No provider SDK is needed: every cloud provider is streamed over `httpx`, which Heliox already depends on.

**Run for real** for the Claude key only (the evals in section 3, Claude Haiku 5.5, text). **Fakes only** for
Gemini, OpenAI, OpenRouter and Meta: the eval's recorded replies replayed through each provider's own stream
format; none of them has answered a real turn, so "about one round trip" is an expectation, not a measurement.
**Not tested** with a real local model: the Ollama eval is owed until the machine is quiet enough to run it.
**Fakes only** for no model at all (the session suite).

### Hooks in the server

The hooks, all in or next to `_voice_command_dispatch` (from the code): `interrupt()` at the top of dispatch;
every addressed turn offered to Heliox Live (`live_hold`) after approvals and denials and before any road, in
voice and typed chat alike; "again", "shorter" and "say that again" answered from the last turn;
`take()` before the planner road (a turn the layer finished broadcasts `voice_result` and returns); the
person's words as the planner input, with the conversation passed as context; `keep_company()` in place of the
stock acknowledgement; `finish()` after the result; one sentence when the person drops a plan that is waiting
for approval ("Okay, I've dropped that: ... Nothing ran."); `clear()` and Live's `drop_held()` in the
new-conversation handler; `withdraw_voice_turn` set on the listener for a sentence split at a pause; and
`prewarm()` in `_prewarm_voice`, so the fast model is ready when listening starts. For the quick road: the dispatch passes the quick plan with the planner call and speaks the
layer's result sentence; inside `_handle_execute_inner` four small hunks take the plan, use it on the first
attempt, skip the model reviewers for it, and drop it for the planner when anything would ask for approval. The
layer is active only once the server's model stack is up.

### Platforms

Nothing in the package depends on the operating system. The focused app comes from Live's observation when Live
is on, otherwise from `screen_vision._get_active_window` (Windows, macOS and Linux backends; "Unknown" is treated
as not available). The focused field's text exists only where Live's reader can read the field: Windows (UI
Automation), macOS (Accessibility, once granted) and Linux (AT-SPI); which apps, and what has been run for real on
each, is in [PLATFORM_SUPPORT.md](PLATFORM_SUPPORT.md). When either is
missing the model is told nothing about the screen. All of this is exercised with fakes in the tests, which run on
the three CI systems.

## 3. Measured

**Run for real**: ten conversational turns (text: questions, a remark, a follow-up, a misheard line and a pronoun
to resolve), Claude Haiku 5.5 through the Anthropic SDK (since replaced by plain HTTP, which measured slightly faster) on a home
connection, the real layer and context, 12 November 2026. The sink was silent and took as long as speech would;
the machine had too little free memory to load Kokoro for the run, so the model times below are under load.

| Stage | Before | Now |
|---|---|---|
| Last word to dispatch (owner recording, rehearsal probe, load not recorded) | 1.3 s (Parakeet), 2.8 to 3.3 s (with Whisper) | unchanged; not this layer |
| Dispatch to first token (text, under load) | 10 to 15 s (a plan) | median 0.82 s, worst 1.61 s (first turn) |
| Dispatch to first sentence handed to speech (text, under load) | stock phrase at 2.5 s; the answer 10 to 60 s | median 1.07 s, worst 1.66 s |
| First sentence to first audio (Kokoro, owner's probe runs, load not recorded) | 0.15 to 0.63 s | the same path |
| A request handed over: dispatch to its acknowledgement | stock phrase at 2.5 s | no model call; the sentence goes to speech at once (tested with fakes) |

Adding those stages, which were measured separately and partly under load, a model answer's first sound would come
about 1.2 to 1.7 s after dispatch and 2.5 to 3 s after the last word. That sum is an estimate, not a measurement:
no recording yet times the last word to Heliox's first audio end to end on a quiet machine. The scorecard asks for
1.5 s from the last word; the gap, if the estimate holds, is mostly the end-of-turn and recognition stages.

The ten turns used 8,600 input and 400 output tokens, $0.0011, recorded in the budget tracker.

The quick road, dispatch to action started: **Fakes only**, 0.02 s median over ten allow-listed requests with a
stubbed executor (text; 2.2 s for the first in a fresh process, which loads the world model's weights); **Run for
real** on Windows only, 0.024 s median over three requests (Notepad opened; battery and disk read; load not
recorded), against 10 to 15 s for a planned step before.

### The conversation eval

`python -m benchmarks.conversation_eval` (in `daemon/`) runs 41 turns in 10 scripted conversations through the
real layer and the configured fast model (`--provider` and `--model` pick another), with a silent sink. Every
check is decided in code: length (one to three spoken sentences), nothing unspeakable (markdown, paths, links,
symbols), no stock opener and never the same opening twice running, speaking to the person rather than about them,
a pronoun resolved from the turn before, the screen described only from what the context holds, "I don't know"
when the context cannot answer, a misheard line answered with a question and never an action, voice never
approving, and the hand-over decision on a labelled set (talk stays talk; a request produces a hand-over; the
quick road only where expected, never for a dangerous request). It prints the pass rate per check, the hand-over
precision and recall, every failure quoted, the cost, and the first-sentence time, which it only quotes when the
machine is not busy. About $0.004 a run on Claude Haiku 5.5.

First run (12 November): every check passed except one hand-over. "Put on some music" was answered in words, and
the words were the model thinking aloud ("The person asked to put on music, which is ambiguous... so I'll ask one
short question."). Fixed in the prompt (speak to the person, never about them, never narrate; use their name
rarely), in the post-processing (such sentences are dropped like a claimed action) and in the router ("put on X" is
a request). A second run lost the network part-way and showed that the short offline line was the same on every
turn; it now alternates. Third run: 41 turns, hand-over precision 1.0 and recall 1.0, every check at 100% except
one repeated opening ("I can't..." on two honest don't-know answers in a row), $0.0038. The three runs cost $0.009
in all. First-sentence times are owed: the machine was at 76 to 78% CPU with about 1 GB free. These runs are
**Run for real** on Claude Haiku 5.5 only, text. `tests/test_conversation_eval_more.py` replays the recorded
replies through the OpenAI-compatible, Gemini and Claude stream formats: **Fakes only**, so it shows the parsing,
not what those models would say.

### When the fast model and the planner disagree

Two cases are handled by rule, so the person is never left with a promise and nothing done, and nothing is done
that was only asked about:

* **A spoken promise without an ACT line** ("I'll open the calculator for you."): the promise is not said, and the
  person's own words go to the full planner, with its reviews and approvals, as for any request. If the words were
  misheard or only a question, Heliox asks instead ("Do you want me to ...?").
* **An ACT line for a question that only asks how, what or why** ("How do I open the registry editor?"): nothing is
  handed over. Heliox asks "Do you want me to open the registry editor?" and keeps the request for a minute. A
  "yes" (or "sure", "go ahead", "please do") in that minute hands it to the planner through the normal road: it is
  the person's request, not an approval. With a plan waiting for approval, the server refuses a spoken "yes"
  before this layer is asked, exactly as before, so the two can never be confused. Anything else said closes the
  offer. A request phrased as a question ("Could you open the calculator?") is not held back.
* **A reply that declines or asks what was meant** ("I can't tell which files you mean, so I won't delete anything
  yet."): nothing is handed over, even with a promise or an ACT line after the words, and the ACT line is never
  spoken.

**Fakes only** for the rules (unit tests). The promise and decline cases also ran in the handover eval in section 2
(**Run for real**, Claude Haiku 5.5, text).

### Spoken length, numbers and Live's remarks

A spoken reply stops before a sentence that would take it past 25 words (about 10 seconds; the model is asked for
at most 20, and the session suite holds replies to the same 25); the first
sentence is always said whole, a clause cut at a comma is never the last thing said, and the stream is closed so
the rest is not generated. Typed replies keep their length and their markdown. Units glued to numbers ("16GB"),
percentages, ISO dates and file names that carry one ("backup-2026-11-12.zip" becomes "the backup from the twelfth
of November") are turned into words a voice says well. A remark Heliox Live made on its own is kept in the
conversation as Live's (`add_heliox(..., source="live")`), so "why?" is answered from its evidence rather than as
Heliox's own mistake, and the model is told how its last reply began so it starts the next one differently.

Since 12 November every road to the voice also says amounts, times, ordinals and percentages as a person does
("4 200" and "4,200" as "four thousand two hundred", "3pm" as "three p m", "19th" as "nineteenth", "50%" as
"fifty percent"; versions, addresses and plain years are left alone): **Fakes only**, unit tests of the
conversion; no recording of Heliox's voice saying them has been checked.

**Run for real** on the further checks (text: 53 turns on Claude Haiku 5.5), before these fixes and after:

| Check | Before | After |
|---|---|---|
| Live follow-up answered from its evidence | 0 of 1 | 1 of 1 |
| Numbers said well | 34 of 35 | 36 of 36 |
| At most about 12 seconds of speech | 32 of 35 | 36 of 36 |
| Hand-over (precision, recall) | 0.944, 1.0 | 1.0, 1.0 |
| Quick road where expected | 7 of 7 | 7 of 7 |
| A fresh opening | 21 of 22 | 21 to 23 of 23 (varies between runs) |

### Turns the listener doubted

The listener tells the layer how each turn reached it (`voice_turn_addressed`: "wake", "button", "follow_up" or
"doubtful_follow_up"). After the wake phrase, the button or a confident follow-up, words that make no sense are
answered with a question, as before. A doubtful follow-up is probably speech that was not meant for Heliox: it is
never sent to a model, never acted on and never kept as conversation; Heliox says one short "Sorry?" at most once
per conversation, then tells the listener the turn was not understood (`note_not_understood()`), which closes the
follow-up window instead of extending it. Control words still work. **Fakes only**: unit tests and the session
suite; the Conversation Bench's audio-level re-run is the measurement that would show it.

### "What can I say?"

Answered in two spoken sentences with three examples that work for this person right now: an app their system
has, a reminder, and a question when a model is set up ("why did you say that?" when Live is on), then "Say 'more'
for others." "More" within two minutes gives the next three. The command palette opens at the same time with the
full list. Every example that needs no model is answered in code (a test checks each). The two sentences are 23 words on
Windows and macOS and 24 on Linux (from the code). **Fakes only**.

### Whole sessions

**Where a session writes.** The session driver and the other conversation evals (`__main__`, `more`,
`handover_context`, `interrupt_timing`) call `pilot.real_folders.use_scratch_folders()` before anything imports
`pilot.config`. Every Heliox folder is then under a scratch root: the folder guard's test root under pytest, or a
new one, removed at exit, when run as a script. Each session gets its own folder inside that root. The driver
records where every settings save went, and the personal pack's settings session ("stop suggesting things", "stop
interrupting me", "be my live companion") fails if one lands outside the root.

On 12 November, before this, a session server's settings save ("stop interrupting me") overwrote the person's real
`config.toml` and `restrictions.toml`. A paid run now copies the person's `[model]` settings
(`apply_owner_model_settings`) and `[conversation]` settings (`benchmarks/conversation_eval/owner.py`) onto a config
of its own under the root. It records what it spent with `real_budget_tracker`, the one intended write to the
person's own folders.

`python -m benchmarks.conversation_eval.sessions` runs 21 sessions of 6 to 8 turns (`sessions.json`: a morning, a
task corrected and stopped, "shorter" and "again", an action repeated, a slip Live asks about, a vague destructive
request, an approval voice must not grant, a note said in passing, a misheard turn, someone else talking, a call,
the wake phrase needed again, a sentence split at a pause, and more). The real server, the listener's real
addressing rule on an injected clock, the conversation layer, the quick road, the planner road with its approvals,
Heliox Live's real checks, and the person's stores in a temporary folder all run. Each turn states what must be
said, what must run, what must be asked for approval and what must never happen, and across a session nothing is
said twice running, nothing over about 25 words unless asked for detail, nothing over the person, and nothing out
of turn. Every session runs for a Claude key, an OpenAI-compatible endpoint, a local model and no model.

**Fakes only**: the model streams are stubbed in each provider's own format with scripted replies, and the planner
and the tools are stubs. On that, every session passes for every kind of person (12 November).

**Run for real**, once (12 November, text, Claude Haiku 5.5 for the Claude road only, the budget tracker's own
figure $0.00114 over 7 model calls): 16 of 21 sessions passed the same rules. The run mostly shows what happens
when the model is slow. At least once a first word did not come within 4.5 s, the person heard "The language model isn't
answering right now ...", and the model was then left alone for 30 s. The suite plays sessions back to back with
one model, so every model turn for the next half minute heard the short fallback instead of an answer. Where the
model did answer, the replies kept to the rules ("Tomorrow is Friday the thirteenth of November." after "and
tomorrow?"). One rule in the pack was written for the stub's two-sentence reply: the real one-sentence answer
made "shorter" say "That was the short version." The run says nothing yet about how often the first word is late
on a quiet machine: that needs a timed run on a quiet machine, which is owed.

A late first word no longer costs half a minute. One late answer costs its own turn; the next turn asks the model
again, and only late answers in a row rest it (**Fakes only**: `tests/test_conversation_slow.py`).

**Run for real**, a second time (12 November, a quiet machine at 9% CPU and 3.6 GB free, text, Claude Haiku 5.5,
the Claude road only, a fresh model per session as a person's sessions are minutes apart; the budget tracker's
figure $0.00479 over 35 calls): 18 of 21 sessions passed. The first text came 0.72 to 2.36 s after each request
(median 1.18 s; the slowest was the first call of the run), so no turn reached the 4.5 s deadline. What failed is
what the model said:
- Three replies ran 26 to 30 words (a careful "I won't delete anything; which files do you mean?", a clarifying
  question, an offer to note an amount down). The layer then allowed about 30 words; both are now 25, and the model is asked for 20.
- "And times ten?" after "what's two plus two?" was asked about, because an answer Heliox gave in code was not part
  of the conversation the model saw. It is now: an answer given in code is kept as a conversation turn (never for a
  turn the listener doubted). **Fakes only** for the fix; not yet re-run on the real model.

**Run for real**, a third time, after those fixes (12 November, a quiet machine at 21% CPU and 4.1 GB free, text,
Claude Haiku 5.5, the Claude road only, a fresh model per session; the budget tracker's figure $0.00494 over 36
calls): 20 of 21 sessions passed. The first text came 0.63 to 2.31 s after each request (median 1.04 s). Every reply
kept to 25 words, and "And times ten?" after "what's two plus two?" was answered from the sum. The one failure was
the pack's wording: asked "how do I open Notepad?", the model offered in its own words ("I can open Notepad for you.
Want me to go ahead?") instead of the layer's "Do you want me to ...?", and the "yes" that followed opened it; the
pack now accepts either.

### Personal sessions

`benchmarks/conversation_eval/personal_sessions.json`, run by the session suite
(`python -m benchmarks.conversation_eval.sessions --pack personal`, `tests/test_conversation_personal.py`).

Each session runs across a restart of the session server: a new server over the same folder, so what is on disk
stays and the conversation does not. Dispatch goes through `PilotServer._voice_turn`, as the listener calls it, with
the personal context the daemon gives the layer. Besides what is said, the checks read:
- the request body the fast model was sent (`sent`, `not_sent`; `expect_cloud` for a model in the cloud only);
- what the planner was given (`planned_with`);
- everything on disk after the turn (`stored`, `not_stored`).

**What the model is sent about the person.**
- *Before this round:* the first eight profile lines went with every turn to every model, whatever the turn was
  about. "What's the capital of Peru?" carried the person's manager, their home and their marathon training, and
  "my salary is 90k" once stored.
- *Now, with a cloud model,* only what the answer needs:
  - the name and the answer style, always, since they shape every reply;
  - stated preferences;
  - a person or a fact only when the words bear on it: a name, "my manager", a shared word
    (`pilot.personal.context.relevant_lines`), and the home when the words are about home or nearby.
- *With a model on this computer,* every line (the first eight).
- *For every model:* a line in the never-kept categories is never sent, even when something stored it (health,
  money, someone else's private details, anything that names or looks like a secret). The personal context line no
  longer repeats a profile line.

**What else changed:**
- **Answer length.** The person's answer length now changes the spoken reply as well as the context. "Keep it
  short" caps it at 15 words; "I like detail" allows 60, where the old fixed 25-word cap cut a detailed answer
  after its first sentence. The model is told the same in a context line.
- **People in a hand-over.** "Message my manager" hands over with "Their manager is Priya." in the planner's
  context. Before, the session planner was never called: "message" was not a request to act, so the turn went to
  the model, or with no model to "Not without the language model".

Pass table on the stubbed models (15 sessions, each for a Claude key, an OpenAI-compatible endpoint, a local model
only and no model): 11, 11, 11 and 10 pass. The rest fail on other workstreams' code, listed in the test's KNOWN:

| Session | What fails | Whose code |
|---|---|---|
| told once: being interrupted | "stay quiet unless I ask" is understood only inside getting-to-know; on its own it goes to the model, which said "Okay, I'll stay quiet until you ask me something." (real run) and nothing changed | `pilot.personal.getting_to_know` |
| corrected: the name | "No, call me Sam" is not a name request (the leading "No,"); the real model answered "Got it, Sam." and the next morning was "Good morning, Alex." | `pilot.workflows.user_profile` |
| forgotten: a fact, the forget said as a follow-up | "Forget that I'm a founder." right after an answer is not taken as a follow-up; with the wake phrase it works | `pilot.system.addressing` |
| not learned: things never kept | "remember that my salary is 90k" and a dentist appointment are stored and read back; they are never sent to a model | `pilot.workflows.user_profile`, `pilot.personal.memory_talk` |
| told once: when work ends (no model) | "when do I stop work?" is not a recall phrase, so with no model nothing answers it | `pilot.personal.memory_talk` |

Not shown by any session: a stated end of the working day ("I stop work at six") is stored, but nothing reads it
when deciding whether to speak up or make an offer in the evening. The quiet hours, Live's timing and offers do not
read it.

**Run for real**, once (12 November, Claude Haiku 5.5, a fresh model per session, budget tracker on; machine load not
recorded): 22 calls, $0.00294, first text a median 0.97 s. The same 11 of 15 passed, failing on the same four rows
above. What the model did with what it was sent:
- the name was used once ("I don't know your local restaurants, Sam");
- "keep it short" gave 11 words;
- "I like detail" gave 58 words over three sentences;
- "what does my manager like for lunch" was answered about Priya;
- "when do I stop work" gave "at six";
- "I live in Pune" was sent for "where should I eat near home" but not used.
- Twice the model said it would do what nothing had saved ("Got it, Sam.", "I'll stay quiet until you ask me
  something.").

**A model never claims to have kept or changed something.**
- The system prompt now says the model cannot save anything or change how Heliox behaves. When asked to, and
  nothing it is told shows it was done, it says plainly that it couldn't, and how: "remember that ...", "call me
  ...", or Settings.
- So that it never denies what Heliox did do, a turn Heliox answered in code shows a cloud model what was done there,
  in the change record's own few words: "(answered on this computer; done there: remembered that)". These words
  hold none of the person's data. A local model sees the answer itself.
- **Run for real** a second time (Claude Haiku 5.5, a fresh model per session; 24 calls, $0.00372; the machine was at
  91% CPU, so the timings are not quoted). The prompt line alone did not stop either claim: "No, call me Sam." still
  got "Got it, I'll call you Sam from now on.", and "stay quiet unless I ask" still got "Okay, I'll stay quiet until
  you ask me something." Asked "Did you get that right?" right after "call me Sam", the model answered "Yes, I'll
  call you Sam from now on. Heliox saved that when you asked." That one is right, and was right because of the "done
  there" line. Nothing else got worse: the same 11 sessions passed, plus the new one.
- Now the turn itself says it. Words asking to keep something or behave differently ("call me ...", "remember ...",
  "stay quiet", "from now on ...", "keep it short") reach the model only when Heliox's own code did not take them.
  When they do, the context says "Heliox saved and changed nothing for these words" and how the person can.
  **Fakes only**: it has not run on the real model yet.

## 4. Not in this pass

* The quick road covers single actions only; anything with two steps goes to the planner.
* The existing Claude provider (`models/cloud.py`) still calls the API over plain HTTP, non-streamed.

## 5. When Heliox takes speech as meant for it

The listener (`pilot/system/voice.py`) asks one rule about every utterance (`pilot/system/addressing.py`). The line
that does not move: nothing said near the microphone that is not addressed to Heliox is acted on, stored as
conversation, or sent to a cloud model.

**Opening a conversation.** Outside an open conversation, only two things open one: the wake phrase at the very
start of an utterance, or the microphone button (or hotkey). A sentence is never taken as addressed because it
sounds like a command; "Open the deck" in a meeting is for the meeting.

**The wake phrase counts only when it is said to Heliox.** It must open the utterance, optionally after "Hey",
"Hi", "Okay" or "Hello", and be followed by a pause or by words said to Heliox: "Hey Heliox, open my mail",
"Heliox, what time is it?", "Hey Heliox." It does not count when Heliox is talked about ("Heliox is the app I use",
"Heliox's settings", "I asked Heliox"), for a web address ("Heliox.com"), or for words that only resemble it
("Helix", "Hello, Alex", "Helium"). Recognisers write the name in other ways ("Heliocs", "Helios"); such a spelling
counts only when the second recogniser heard the same words after it, so a misheard name needs two opinions while
the exact name needs one. When the wake check's own words misread what follows the name ("Hey Heliox and do that."
for "Hey Heliox, undo that.", which by the rule is talk about Heliox), the words acted on (the name from Whisper, the
rest from Parakeet) are judged by the same rule, so a wake-phrased request is taken in any state but a call. The
merged transcript holds the wake phrase once ("Hey Heliox, Hey Heliocs, undo that"
became an undo with the phrase twice).

**A follow-up.** After Heliox finishes speaking in an open conversation, the next utterance needs no wake phrase
only if all of these hold:
- it starts within 8 seconds of Heliox's last word (a setting: people are thought to reply within about a second,
  not yet measured on real replies; on the synthetic hour's transcript replay the old 30 s window let 229 room
  sentences in as follow-ups);
- it did not begin while Heliox was still talking, unless it is a control word ("Stop", "Wait");
- it is plausibly meant for Heliox: a control word, a reply to the question Heliox just asked, or a direct request
  or question (imperative, or second person), or, within 4 seconds of Heliox's answer (a setting, not yet measured
  on real follow-ups), a short elliptical follow-up
  on it: a repeat or shorter phrase the conversation layer answers in code ("again", "say that again", "shorter",
  "in short") or a continuation of at most six words starting with "and", "or", "what about", "how about" or
  "what if" ("And tomorrow?", "What about the second one?");
- it is not third-person talk, narration ("So then we went...", "I'll send Priya the deck") or a monologue of more
  than 8 seconds of voice.
A running task the person started, or an approval waiting, keeps the conversation open for control words and
replies without the time limit.

**No cascade.** A follow-up that might be for Heliox but is not clearly so is dispatched as a doubtful follow-up:
the conversation layer says one short "Sorry?" at most per conversation, sends it to no model, stores nothing, and
calls `note_not_understood()`, which closes the window. Only a confidently addressed utterance keeps the
conversation open. After 20 seconds without one the conversation is closed and the wake phrase is needed again.
During a call the conversation is closed and nothing is taken.

**Other languages.** The wake phrase and the microphone button work in any language. A follow-up without the
wake phrase is taken only when the rule can read it, and it reads English (imperatives, question starts, narration),
so in another language a follow-up is more often answered with "Sorry?" or not taken. One part needs no words: a
short answer (at most 4 seconds of voice) that starts within 2 seconds of a question Heliox asked, and did not begin
over Heliox's voice, is a reply in any language ("Ja, die blaue Datei.", "हाँ, नीली वाली फ़ाइल।", "はい、青いほうです。"),
unless it is a line Whisper writes from noise ("Thanks for watching.", one word over and over): right after Heliox's
question is when noise or Heliox's own voice through open speakers is most likely to be read as words. This is
tested on written sentences only (text); no one has yet spoken to Heliox in another language.
Control words ("Stop", "Wait") are English only; talking over Heliox for a moment stops it in any language (from
the code: the cut-off is judged on the length of the voice, not its words; shown on English recordings only).

**Overheard speech** goes only to Heliox Live, marked not addressed, which does not check it: no card, nothing kept.
It is never stored as conversation and never sent to a cloud model. **Fakes only**: unit tests with recognised text.

**A slip in something said to Heliox is asked about before it is acted on** (`pilot.live.held`). Every addressed
turn, spoken or typed, is offered to `LiveCompanion.hold` before any road takes it (`conversation.adapters.offer_to_live`,
from the voice dispatch and the typed path), and a new conversation drops what Live was waiting on. **Fakes only**:
text through the real dispatch and `live_hold` (the four slips of the conversation bench's slips.json held, the
must-not-fire cases silent; in the conversation session suite also "never mind" and an unclear answer); the bench's
audio-level run is pending. A weekday that does not match its date
("put the review in for Monday the nineteenth", when the nineteenth is a Thursday), a date that does not exist (the
thirty-first of November), or an amount that differs from one the person stated earlier in the same conversation
("the invoice for Acme is 2,400", then "email the Acme invoice for 4,200"; or "email it for 4,200", where "it" or
"that" is taken as the last thing an amount was stated about, only when the two look like the same number gone
wrong, and only for this check) holds the turn: Heliox asks one short
question ("The nineteenth is a Thursday. Monday the sixteenth, or Thursday the nineteenth?") and the action waits.
The answer picks the version that is acted on; "never mind" drops it; anything else is asked once more and then
dropped. A hold lapses after 60 seconds, and earlier turns count for 10 minutes. A change said on purpose ("change
it to 4,200", "actually 4,200") and a statement (not a request) are not held. The card comes with the question; Live
never says it a second time.

**What the person sees.** The voice bar shows when a conversation is open and when it closes. When a direct
request or question in the second person was not taken because no conversation was open, it says once: "Say 'Hey
Heliox' first."

**Settings.** `voice.continuous_conversation_enabled`, which took sentences that looked like commands as addressed
without the wake phrase, is no longer read and defaults to off; its text says that it made Heliox act on what other
people say. `voice.follow_up_window_seconds` is 8. Heliox speaking (a reminder, a Live card) opens no reply window
unless someone opened the conversation; the replies Live's own remarks get are in "Answering Live" above.

**Hooks.** Each dispatch sets `pilot.system.voice.voice_turn_addressed` ("wake", "button", "follow_up",
"doubtful_follow_up", "continuation", "live_reply") beside `voice_turn_ended_at`; the conversation layer calls
`ContinuousVoiceListener.note_not_understood()` after a "Sorry?", and Live's speech calls
`ContinuousVoiceListener.note_live_spoke()` once a remark has been heard.

Measured three ways, all on synthetic speech for the part that must not be taken:
- Transcript replay (2026-11-12, both recognisers' words cached through the listener's own calls): of the 981
  utterances not addressed to Heliox in the synthetic hour, the old behaviour dispatched 397 and this rule 0, with
  nothing spoken; with the elliptical follow-ups added it is still 0 (none of the 981 opens a window, and none
  matches the elliptical phrases at all).
- Audio level, through the real listener (synthetic voice): 67 minutes and 1,130 utterances not addressed to
  Heliox; 0 dispatched, 0 spoken, 0 stored.
- The owner's wake sentences (owner recording): desk microphone 4 of 4, Bluetooth headset 3 of 5, nine sentences,
  most of them "Hey Heliox, what can I say?" (in the two misses neither recogniser heard the wake phrase). The four
  "Wait, stop." cut-offs over Heliox stop it (owner recording, desk microphone, four clips).
No real room (a meeting, a television, a colleague talking) has been recorded yet, and no real follow-up
conversation; the line that does not move is shown on synthetic speech, not yet on a real room.

**A new person** (from the code, 2026-11-12). Nothing assumes the owner's voice, accent or microphone except the
numbers each part was tuned on, listed here so they can be revisited:
- *Wake phrase.* Whisper is prompted with "Hey Heliox" and its proper names, so it usually writes the name exactly
  (owner recording: on the owner's nine wake sentences; other voices and accents not yet shown), and the exact name
  needs one recogniser. A spelling close to it ("Heliocs", "Helios", "Hilliox", anything matching
  `h[ei]l{1,2}[iy]` plus up to five letters) counts only when Parakeet heard the same next word. Any other spelling
  ("Hey Lux", "Dear Yorks") never wakes Heliox. A voice the recogniser writes differently every time is not helped
  by anything below; the microphone button always works.
- *Learning from near-misses* (`voice_calibration.py`, on by default): a near-miss (within two letters of "hey
  heliox") followed within two utterances by an exact hit counts once; five make it the person's. A person whose
  "Heliox" is never written exactly never reaches it. It now ignores punctuation ("Hey Heliocs, open" was three
  letters away because of the comma).
- *Teaching the name* (Settings, Voice, "Teach Heliox how you say its name"): the person says "Hey Heliox" four
  times; a spelling the wake check wrote at least twice is kept as theirs, at most five, listed there and removable.
  Refused: spellings made only of ordinary words (Whisper's own 24,000 whole-word vocabulary plus a list of names
  and near-words such as "Alex", "Helix", "Hello"), shorter than four letters, or more than four letters from
  "heliox". Of the 5,868 opening phrases (one to three words) of the 981 utterances not addressed to Heliox, as
  either recogniser wrote them (transcript replay, synthetic voice), the guard refuses every one, so none could
  ever be taught. Only the spellings are stored, on this computer. No one but the owner has tried the step.
- *Whisper size.* The settings default is what the recommended download brings for the computer's language: "base"
  for English (with Parakeet), "small" otherwise. If the configured size is not on the computer, the listener uses
  one that is, and downloading Whisper base switches the setting when the configured size is absent (before, a new
  person who took the recommended set had no wake check at all).
- *End of turn.* Smart Turn ends the turn at 0.85 after 400 ms of quiet (set on the owner's headset, owner
  recording, one voice: pauses inside a sentence scored 0.71 to 0.83), holding up to 2.5 s otherwise. Same for
  everyone; how it suits other voices is not yet shown.
- *Speech level.* The VAD starts at 0.02 RMS and then follows the room: 2.5 times the 20th percentile of recent
  levels outside speech, kept between a quarter and four times the start. From the code, a quiet voice on a quiet
  microphone should be heard once the room is learned (about half a second of frames); not yet shown on a quiet
  voice. The voice bar warns when the voice is below -48 dBFS (a setting; the owner's microphones sat at -35 to
  -44 dBFS), clips, or comes through a narrow-band headset.

**Not done: whose voice.** Nothing checks that a follow-up comes from the person who said the wake phrase. The
cheapest evidence is the speech level of the follow-up against the wake utterance that opened the conversation: the
same person at the same distance from a desk microphone should be within a few decibels, a television or someone
across the room usually not. None of the installed recognisers gives a speaker embedding, so anything stronger
needs a download. Measuring the level rule needs recordings that do not exist yet: two people taking turns at one
desk microphone (one addressing Heliox, the other talking to them), and a television or video playing in the room
while the owner talks to Heliox, each on a desk microphone and a headset.

**Talking over Heliox (barge-in), where the time goes.** On the owner's four recorded "Wait, stop." cuts (owner
recording, desk microphone, four clips), fed through the listener's recorder and `BargeInGate` on the audio clock
with the 2 s before each cut as Heliox's speech. That microphone is noise-gated: it gave digital silence while
Heliox talked through the speakers, so these clips show the case where the microphone does not hear Heliox.

| part | open speakers, before | open speakers, mic does not hear Heliox (now) | headset | ours to shorten |
|---|---|---|---|---|
| first sound to "the person is speaking" (VAD, two 50 ms frames) | 0.08-0.09 s | same | same | yes, at the cost of clicks |
| speaking to the gate firing (hold) | 0.45 s | 0.25 s | 0.25 s | yes |
| gate polling | up to 20 ms | same | same | yes |
| fade-out | 8 ms | same | same | yes |
| audio already in the output device | not measured (silent sink) | | | partly: see below |
| first sound to stop decided | 0.53-0.54 s | 0.33-0.34 s | 0.33-0.34 s | |

The open-speaker hold (0.4 s) is there because the microphone hears Heliox. When it does not, the gate now uses
the headset's 0.25 s: the echo comparison still applies, and a microphone that hears Heliox at -50 dBFS or louder
keeps the 0.4 s (from the code; no recording of a microphone that hears Heliox exists yet). A cough of about 0.15 s
does not stop Heliox in the unit tests (generated tones, not a recorded cough). The output stream is opened with
PortAudio's default host API, which on Windows is MME, whose low output latency is 90 ms (WASAPI 3 ms, DirectSound
120 ms on the same Realtek speakers; PortAudio's reported figures, not measured): up to that much of Heliox's voice
may already be in the device when the stop is decided. Moving the output to WASAPI on Windows is ours to do but
needs an audible test before it ships. macOS (Core Audio) and Linux (PulseAudio or PipeWire through ALSA) report
their own low latencies; not measured here.

**A headset in call mode.** On the owner's wake sentences (owner recording: five through one Bluetooth headset in
call mode, four on the desk microphone, mostly the same sentence), word error rate with both recognisers merged as
the listener does: 0.42 through the headset, 0.04 on the desk microphone. Whisper small instead of base (0.47), a
telephone-band filter (0.57) or louder input (0.40) did not help; for Whisper alone a round trip through 8 kHz did
(0.74 to 0.49), on four sentences, too few to ship. The voice bar's warning for this microphone says how much worse
it is ("four to seven words in ten", from these nine sentences on one headset and one voice) and offers the two
fixes: the computer's own microphone for input, or the headset's stereo profile outside calls.

The wake phrase on such a headset gets a second look. When the input is in call mode (the device's name says
hands-free, or the microphone check judged it narrowband; never from one utterance alone, since a short "Wait,
stop." on the desk microphone looks narrowband too) and the Whisper wake check heard no wake phrase, Whisper reads
the utterance once more through a 16 to 8 to 16 kHz round trip, and that reading counts only if it holds one. On the
bench replay (12 November) it changed two outcomes: the owner's "Hey Heliox, remind me that the launch call is
Monday..." on the headset, which nothing heard at 16 kHz, is now taken ("Here Heliox, invite me to..."), and the
synthetic "Helix is a nice name for a project.", read as "Heliox is a nice name...", is still refused as talk about
Heliox. Unaddressed lines dispatched: 0 of 981; addressed lines taken: 12 of 13 (11 before). The cost on a call-mode
headset is one more Whisper base pass on every utterance without a wake phrase, about 1 s for a 6.5 s utterance on
this laptop's CPU at about a third of its load (the GPU could not be measured: its CUDA libraries do not load here).
An utterance the first look took is never delayed by it.
A known limit that stays: the owner's "Hey Heliox, what can I say?" on the same headset, heard as "Here is what can
I say?" by Parakeet and "Here we go to work the nice video." by Whisper at either bandwidth. No recogniser wrote
anything like the name, and "Here is ..." is ordinary speech, so no rule can take it, and the wake teach step refuses
ordinary words. The advice is the voice bar's own: use the computer's microphone, or the headset outside a call.

**When the listener ended the turn too early.** No word list can hold a turn at a comma (section 6), so the
listener recovers from an early end the way a person who started answering too soon does. If the person speaks
again within 3 s of the end of their own turn (as the listener declared it) and Heliox, if it is talking, began
after that end and at most 2 s before (it is the one who came in too early):
- Heliox yields its voice at once (`BargeInWatcher`, `voice_status` "interrupted"). That is not a stop: nothing is
  cancelled because of it, and the short cut-off rule does not read the words as "stop".
- The new words are joined to the command they continue and dispatched as one sentence, marked as a continuation:
  `voice_turn_addressed` is "continuation", `voice_turn_continues` and `voice_turn_continues_id` name the earlier
  dispatch, `voice_turn_tail` holds the new words alone, and every dispatch carries its number in `voice_turn_id`.
- A stop word ("Stop", "Wait") is never joined, a new "Hey Heliox, ..." is judged as a new request, and when Heliox
  has been talking for longer the cut-off rule applies as before.
The listener asks the server to withdraw the first half's dispatch (`withdraw_voice_turn(turn_id, joined)`,
`ContinuousVoiceListener.withdraw_turn`): "withdrawn" means the server stopped it and dispatched the joined sentence
in its place, with one acknowledgement when it leads to the same action; "too_late" (something already ran, and
Heliox says what) or "unknown" means the listener dispatches the new words alone as a new turn. The server provides
the call (`withdraw_voice_turn`, set on the listener where it is created); a listener without it does the same as
for "unknown": it yields, never reads the second half as "Stop", and dispatches the second half as its own turn.
Measured on the bench's four pause scenarios with the server's call (synthetic voice, one sentence, pauses of 0.6 to
2.0 s inside "Hey Heliox, remind me to | call the bank tomorrow morning", cloud model stub, 12 November): none is
taken as "Stop" (before, the 1.5 and 2.0 s pauses cancelled the request), and nothing is said over the person
(Heliox's first sound came after the second half's last word each time).
- 0.6 s: one action, the joined request, and one acknowledgement.
- 1.0 and 1.5 s: one action, the joined request, but the acknowledgement twice: the first half's "I'm on it" began,
  was cut when the first half was withdrawn, and the joined request's was said again.
- 2.0 s: two actions. The second half was decided 6.1 s after the first half was dispatched (its recognition took
  1.9 s), past the server's 6 s limit for withdrawing, then counted on processing time, so the withdrawal answered
  "unknown" and the second half ran as its own turn. The listener now passes the gap on the capture clock (from
  the first half's last word to the start of the second half: 2.6 s here) to `withdraw_voice_turn(turn_id, joined,
  gap)`, and the server counts its limit on that.
- Measured again on main 008f1a0 with both sides of `gap` (12 November, the bench's wired runner, same audio, run
  once, timing-free): 0.6, 1.5 and 2.0 s were withdrawn and joined, one action each. 1.0 s became two "Stop"s:
  Heliox was still speaking from the scenario before, "Hey Heliox, remind me too" said over it was short enough
  to be the cut-off rule's "stop", and "call the bank tomorrow morning" said over its "Okay, stopped." was too.
  "Hey Heliox" over Heliox is now the start of a request (below), and words that stop mid-sentence are not
  dispatched, so none of the four splits any more.
So the single right action is shown for pauses up to 2.0 s, and the acknowledgement is said once only when nothing
of the first half's was spoken.

**Words that stop mid-sentence are not dispatched.** When the recognised words end on a comma, a dash or an ellipsis,
or on a word no request ends on ("to", "the", "and", "for", "my"; "remind me too" as a recogniser writes "remind me
to"), the listener does not dispatch them: it waits up to 2.5 s (`UNFINISHED_WAIT_SECONDS`, from when the words are
ready; speech that began while they were being recognised counts) for the rest and dispatches the sentence once,
whole. Nothing is acknowledged or answered for half a sentence, so there is nothing to withdraw. No rest: the words
go as they were. A stop word or a new "Hey Heliox, ..." in place of the rest is taken alone. Words that are
already whole never wait, whatever mark ends them: a control word, a short decision or a safe fast phrase of up
to three words ("Stop,", "No,", "Cancel that,"), anything of one or two words, and any reply to Heliox's question
or a Live remark. The cost is a finished request that ends on such a word ("Tell me what this is about."), which
waits the full 2.5 s; none of the corpus's 994 transcripts or the 92 of the owner's own recordings ends that way. A wait for speech to
begin no longer cuts off speech that began in time (a long request after "Hey Heliox" alone was lost at the eighth
second). After the server joined a split sentence, that joined turn ("v1+") is the one a third piece continues;
before, the third piece of "..., but my oven runs hot, | so I take it out at 35" was judged alone and overheard.
"Hey Heliox" said over Heliox is the start of a request, never the short cut-off "stop": its voice has already
stopped for it, and the request after it is heard as a command.
Re-run on 12 November (main 8b21d6f with the bench's wired runner, synthetic voices, stub planner, timing-free; the
same scenarios on 0db01d9 before the change for comparison):
- The four pause scenarios: one dispatch each, of the whole sentence, and nothing said during the pause (before: the
  half "remind me too" was dispatched, "Okay, setting that reminder." began in the pause, and the server withdrew
  it once the rest was recognised and ran the joined sentence).
- 23 sentences said to Heliox with a comma pause: 22 reach the server as one sentence, 10 joined before any
  dispatch and 12 by withdrawal (there the first half's words ended in "." or "?"). Before: 20, 17 by withdrawal
  and 3 never split, with a third piece dropped in one. The one left has a second half 10 s long, decided more than
  6 s after the first half was dispatched: "unknown" until the server counts its limit on the gap. None is taken as "Stop" (two were, when
  "Hey Heliox" was said over Heliox).

**A first half that sounds finished.** "Remind me to call the bank." | "tomorrow morning" reaches the server as two
dispatches, and the first used to be answered before the rest arrived. In the bench's re-run (scratchpad
bench_r5_split_cases.txt, under load) a reply was chosen for the first half in all 15 withdrawn lines, and 0.07 to
0.29 s of it was heard in 3. The rules are read from the listener, never from the words. The listener reports its
last turn, whether it is hearing or recognising speech that began within `CONTINUATION_SECONDS` (3 s) of that turn's
end, and any continuation it has decided (`pilot.conversation.turns.settle`).

- **Nothing changes while the person is audibly going on.** That covers every road: answering, Live's question,
  planning, the executor's door, and the changes Heliox makes in its own code (to-dos, notes, settings). A
  continuation withdraws the first half; speech that is not a continuation lets it go on. When the person is
  silent, this costs nothing. In the bench's lines the second half began 0.4 to 0.55 s after the first half's last
  word, before the first half had been recognised. Denials and approvals of a waiting plan never wait.
- **In silence a change runs at once.** The one exception is a plan holding an action the risk classification calls
  destructive or not undoable: `requires_confirmation` (it would need an approval if the learner had not waived it)
  or `is_irreversible`. That plan waits out the window, up to 3 s after the turn's end. An approval dialog covers
  the cases where one opens. Opening and reading things never wait.
- **A change already made is too late.** If a reversible change ran on a first half and the person then continues
  after a silence, the withdrawal answers "too late". Heliox says what it did, the rest is a new turn, and where the
  undo journal keeps what it needs the sentence ends "Say undo to take it back." Changes made in Heliox's own code
  (a to-do) have no undo, so none is offered.
- **A withdrawn first half leaves no trace.**
  - The conversation goes back to what it was before the first half, so neither its words nor its cut-off reply stay.
  - What Heliox last said goes back too, so "say that again" never repeats the cut reply.
  - Live drops the first half from what it compares the next request against.
  - A request it showed on screen is taken back (`voice_withdrawn`), and its interaction ends as interrupted.
- **The joined sentence is answered as one sentence.** The model and the planner see one sentence, never a first
  half and then a correction of it.

Time from the end of the sentence to the action, on the session server with a fake listener. The first half
reaches the server 0.5 s after the end. "Going on" means speech from 0.4 to 1.9 s after the end, recognised until
2.4 s, and not a continuation. Medians of 3; waits are real time; the machine was at 57 to 76% CPU with 0.7 to
1.7 GB free, so read these as about.

| Command | Before | Round 16 as first sent | Now |
|---|---|---|---|
| Quick road, volume: silence / going on | 0.53 / 0.52 s | 3.05 / 3.05 s | 0.52 / 2.45 s |
| In code, a to-do: silence / going on | 0.51 / 0.51 s | 0.51 / 2.49 s | 0.51 / 2.44 s |
| Planner, rename: silence / going on | 0.60 / 0.59 s | 3.05 / 3.02 s | 0.61 / 2.53 s |
| Planner, delete (destructive, approval waived): silence / going on | 0.52 / 0.52 s | 3.01 / 3.03 s | 3.02 / 3.02 s |

**Fakes only**: `tests/test_conversation_half_sentence.py`, on the real server with a fake listener and a stubbed
planner, executor and model; `session.voice.test.ts` for the chat. Not yet re-run on the bench.

**When the voice does not work.** A system voice that fails (`say`, espeak-ng or Windows' speech returning an
error) or is missing makes the speech outcome "no_voice", never "spoken" (from the code; tested with fake system
programs on each system's path). After it no follow-up window and no Live reply window open, since nothing was
heard; the reply is shown in the voice bar as text, with one line about what to check said the first time only,
until a sentence is spoken again; `speak_text` returns the failure. The 23 corpus lines cut at
a pause (synthetic voice; pauses of 0.40 to 0.72 s) all resume well inside the window. Not yet shown on the owner's
own long requests.

**Times are when the sound arrived.** On a CPU-only laptop the recognisers can fall seconds behind the microphone.
Every captured block therefore carries the time its last sample arrived, and every decision that compares times
uses it: the start of speech, the end of the turn and of the last word, the follow-up window and the conversation's
20 s close (the rule judges an utterance at the moment it began), quick replies, elliptical follow-ups,
continuations and the cut-off. Barge-in reads voice activity worked out on the capture side, not the recorder's
flag, which lags a backlog. With processing time instead, a continuation said 1.4 s after the first half was refused
behind a 7 s backlog; it is now decided the same at 0, 7 and 25 s of backlog (tests/test_voice_capture_clock.py:
generated tones, scripted words, an injected clock). On the bench's 0.6 s pause scenario (synthetic voice, under
load, with a 7.05 s recognition backlog) the second half is now taken as its own turn, where before it was lost.
Where the time comes from (from the code and PortAudio's callback contract; not measured per device, and not yet
run on macOS or Linux):
- PortAudio (sounddevice: MME, DirectSound and WASAPI on Windows, Core Audio on macOS, ALSA on Linux) gives each
  input buffer's ADC time in the callback (`inputBufferAdcTime`, on the stream's clock), moved onto
  `time.monotonic()` with the callback's `currentTime`. A host API that reports 0 there (some ALSA plugins,
  including PulseAudio's and PipeWire's ALSA devices) falls back to the sample count.
- Windows' native WASAPI capture (soundcard, used first on Windows) reports no times: the stream's own sample count
  from its first block is the clock, put right if it drifts more than half a second from the wall (a device restart
  or dropped buffers).
- The Conversation Bench's file source uses the sample count, or a capture time it passes in.

**Another voice in the room.** With a second voice behind the person, the wake phrase was taken 8 of 8 times at
15 dB and above, 4 of 8 at 10 dB and 2 of 8 at 5 dB (bench mix: the owner's four desk wake sentences with
synthetic two-voice babble, two offsets each). The other voice's words replace the name; Parakeet never writes
"Heliox", so the wake rests on Whisper alone. Reading only the first 1.0 or 1.6 s of the utterance does not help
(bench mix, harsher: at 10 dB, 5 and 5 of 16 against 6 of 16 for the whole utterance), and teaching the name
cannot: what is lost is the name itself, not a spelling.
In plain words for an open office (worked out from the levels above by the inverse-square law, not measured in a
room): Heliox should hear its name reliably when other conversations are clearly quieter than you at the
microphone, roughly you within half a metre of it and other people three metres or more away. With someone
talking at the next desk it would miss about half the time; a headset microphone (USB or wired, not a Bluetooth
headset in call mode) or the microphone button is the advice there.

## 6. Ending the turn early (`early_turn_threshold`): measured and left off

The recorder can ask Smart Turn as soon as 200 ms of quiet follow a word and end the turn there when the model is
at least `early_turn_threshold` sure the sentence is finished; otherwise it waits for the usual 400 ms check. The
code stays in and the setting stays at 1.0 (off): on longer sentences it cuts them in two.

Measured 2026-11-12 with `benchmarks/conversation_bench/turn_split.py`: the listener's own recorder fed one 50 ms
frame at a time on the audio's own clock (the next frame only after the recorder has taken the last one and any
Smart Turn call on it), so the numbers do not depend on the speed of the machine. "Split" means one sentence became
two or more utterances.

| `early_turn_threshold` | Owner recording (desk and headset, 18 short sentences): split | ended at 200 ms | Synthetic voice, addressed sentences (49): split | Synthetic voice, corpus lines not addressed (159): split |
|---|---|---|---|---|
| 1.0 (off) | 0 | 0 | 0 | 23 |
| 0.99 | 0 | 0 | 0 | 24 |
| 0.97 | 0 | 3 | 5 | 56 |
| 0.95 | 0 | 12 | 10 | 63 |

The owner's sentences are short ("Hey Heliox, what can I say?") and showed no split at any setting, which is why
they cannot decide this. At 0.97 "Hey Heliox, remind me to call Priya at four" and "Hey Heliox, what did I do this
morning" were cut in two; 0.99 saves nothing measurable (one early end in 67 sentences).

Even with it off, 23 of the 159 longer corpus lines were cut at a pause, mostly at a comma, among them sentences shaped like
requests ("Close Chrome, it's eating all the memory.", "No, don't send it yet, I want Priya to read it first.",
"Run the install command again and wait for it to finish."). These are synthetic voices, which pause at commas;
whether people pause as long has not been measured, but a long request with a thinking pause can be cut today.

**Holding when the words so far are unfinished** (measured, not adopted). The streaming recogniser writes no
punctuation, so "the words end at a comma" cannot be read from its partial words. The hold that can be read was
measured the same way, by lowering Smart Turn's answer when the partial words end on a word that cannot end a
sentence, on the 49 synthetic addressed sentences and 159 corpus lines (synthetic voice):

| hold | addressed split | corpus lines split | finished sentences ended later | by |
|---|---|---|---|---|
| none (today) | 0 | 23 | 0 of 182 | - |
| last word is an article, preposition, conjunction or possessive ("to", "and", "the", "my") | 0 | 15 | 35 of 182 (4 of 46 addressed) | 600 ms each |
| wider list, with pronouns and modals ("it", "you", "can") | 0 | 14 | 49 of 173 | mean 179 ms, up to 2.1 s |
| after 2 s of voice, end only at 0.95 | 0 | 20 | 24 of 177 | mean 132 ms, up to 2.1 s |

The narrow list fixes 8 of the 23 cuts and costs 600 ms (one more Smart Turn check) on one long finished sentence in
five, never on a sentence under 2 s. The cost comes from the streaming recogniser's words lagging the audio: at the
400 ms check its last word is often one before the real last word.

Measured next on the recogniser's flushed words (the whole utterance so far decoded to the end at the moment Smart
Turn would end the turn, so no lag), the same narrow list fixes none of the 23 cuts and delays nothing: 0 addressed
split, 23 corpus lines split, 0 of 181 finished sentences later. The 8 "fixes" above came from the lag itself, holding
at the wrong word. The cuts are at commas after a complete clause ("Close Chrome, | it's eating all the memory."),
where the words so far are a finished sentence; no word list can see that the person is going on. Not adopted. The
flush's computing cost per check was measured only under load, so it is not quoted.

**Re-measured 13 November with both repairs in the loop, and left off.** The replay and turn-split set as it is now
(78 clips, owner and synthetic, 16 of them 3 s or longer) was fed the same way on the audio's own clock, with
Parakeet as the final pass, and every split was checked against the two things that can repair it: the
unfinished-words wait (`sounds_unfinished`) and the server joining a second utterance that follows within
`CONTINUATION_SECONDS` (3 s) by withdrawing the first turn.

| `early_turn_threshold` | clips split | of the 16 long clips | repaired by the unfinished-words wait | repaired by the server join | not repaired | turns ended early | quiet before the end saved per turn |
|---|---|---|---|---|---|---|---|
| 1.0 (off) | 0 | 0 | - | - | - | 0 | baseline: median 400 ms of quiet |
| 0.99 | 0 | 0 | - | - | - | 0 | none |
| 0.97 | 9 | 6 | 1 | 8 | 0 | 31 of 67 | median 0 ms, mean 93 ms, at most 200 ms |
| 0.95 | 16 | 10 | 1 | 15 | 0 | 40 of 60 | median 200 ms, mean 133 ms, at most 200 ms |

Most splits fall after the wake word ("A heliox. | Email the Acme invoice for four thousand...", "Hey Hilliogs. |
What did I do this morning?": 5 of 9 at 0.97, 12 of 16 at 0.95). One is a real correction ("Sorry, I meant the other
folder. | The one on the desktop.") and one is cut inside a date ("The invoice is due on the thirty-first o | Of
November.", which the unfinished-words wait catches). Every second half began straight after the first, so the join
repairs all of them and no request runs twice. What it cannot repair is what the person hears: the first half is a
complete turn, so Heliox starts answering it ("Yes?", or an answer to "Sorry, I meant the other folder."), then the
second half arrives, the turn is withdrawn and the voice stops. At 0.95 that is an audible false start on 16 of 78
turns, to save 200 ms on two turns in three. Making a wake-word-only first half always wait would remove 12 of the
16 and still leave false starts on real corrections. So the setting stays at 1.0.

What would reopen it: a way to end the turn early that never starts speaking on a first half. For example, the
answer to an early end could be prepared but held silent until the usual 400 ms check confirms the turn, so a split
costs nothing audible and a confirmed turn still gains the head start. That would need its own measurement of how
often the held answer is thrown away.

## 7. What is not yet shown, and the recording that would show it

Almost everything above was heard from one voice, the owner's, on one desk microphone and one Bluetooth headset,
or from synthetic voices. These recordings would turn the claims into measurements. The Conversation Bench's
recorder (`python -m benchmarks.conversation_bench.record --mic <desk|headset|...>`) takes them per microphone; the
bench agent is adding the follow-up and open-speaker sets to it, and their set names go here once that is out.

**A real follow-up conversation** (the owner, desk microphone and headset): "Hey Heliox, ..." and then replies the
way they come, quick and slow, short and long, with elliptical follow-ups ("and tomorrow?", "again", "shorter"),
answers to Heliox's questions, a long request with a thinking pause, and a sentence said while Heliox is still
answering. It would show:
- how soon people reply, which the 8 s follow-up window, the 4 s elliptical window and the 2 s quick-reply window
  assume;
- what share of real follow-ups the rule takes, and what it refuses;
- whether a long request with a thinking pause is cut, and whether the continuation recovers it (section 5, and
  section 6's "a long request with a thinking pause can be cut today", shown only on synthetic voices);
- the last word to Heliox's first audio, end to end, on a quiet machine with the speakers on (section 3's 2.5 to 3 s
  is an estimate from separate stages, partly under load), along with section 1's end-of-turn and recognition times.

**A real room with other voices and a television** (the owner's microphones): a meeting or call on speaker, a video
or television, a colleague talking to the owner, the owner talking about Heliox, and the owner saying the wake
phrase in the middle of it. It would show:
- that nothing not said to Heliox is acted on, stored or sent to a model in a real room (section 5's line that does
  not move is shown on synthetic speech, 1,130 utterances at audio level and 981 in the transcript replay);
- wake recall with real background speech (section 5's babble numbers are a bench mix);
- whether the speech level of a follow-up, against the wake utterance that opened the conversation, tells the
  person from the television ("Not done: whose voice").

**Open speakers with coughs** (the owner's desk microphone without a noise gate, and the laptop's own microphone,
Heliox speaking through the speakers): coughs, a door, typing and the owner's "Wait, stop." during Heliox's speech,
with a loopback of what the speakers played. It would show:
- whether a cough or a door stops Heliox (shown only on generated tones);
- the 0.4 s hold where the microphone hears Heliox, which no recording has yet;
- the time from the stop being decided to the last sample at the speaker per host API (MME's 90 ms is PortAudio's
  reported figure).

**Other people.**
- *Other speakers and accents*, on a desk microphone and a headset each: wake recall (the owner's nine sentences
  are the only evidence, mostly one sentence), whether Whisper writes "Heliox" exactly, the teach step, the end of
  turn tuned on the owner's headset, a quiet voice, how other voices are misheard by the router (section 2), and
  the headset's error rate (section 5's "four to seven words in ten" comes from one headset and one voice).
- *Other languages*, at least two: quick replies after Heliox's question and short cut-offs over its voice, tested
  only on written sentences and English recordings.
- *An office at measured distances*: a colleague talking at 1, 2 and 4 m while the owner says the wake phrase at
  30 to 50 cm, on a desk microphone and a wired headset (section 5's "half a metre and three metres" is worked out,
  not measured).

**Not recordings, but still owed.**
- A timing set on a quiet machine (the waiter is armed): every timing in sections 1 and 3 marked "load not
  recorded" or "under load".
- The capture clock on macOS and Linux (CI's first run, then click tracks with known onsets on each system).
- The joined single action for a split sentence, once the server's `withdraw_voice_turn` is wired: the four pause
  scenarios, right action once and one acknowledgement.
- The quick road on macOS and Linux with real actions (section 2: tested for real on Windows only).
