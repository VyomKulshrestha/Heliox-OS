# The spoken turn: how Heliox answers when you talk to it

This note covers the fast conversational layer in `daemon/pilot/conversation/`: what a spoken turn did before it,
where the time went, and what the layer does now. Numbers come from the owner's recorded speech and from the
daemon log unless a line says otherwise.

## 1. The path of a spoken turn before this layer

1. **Listening and end of turn** (`pilot/system/voice.py`). The listener runs voice-activity detection on the
   microphone, a streaming recogniser for live captions, and the Smart Turn model to decide that the person has
   finished. On the owner's recordings played through the real listener (`voice_probe.py`), the end of the turn
   was declared 0.45 to 0.65 s after the last word.
2. **The accurate pass.** Parakeet (or Whisper) re-reads the utterance. Measured on the same recordings: Parakeet
   0.23 to 0.78 s; when the wake phrase needed Whisper's confirmation, 2.2 to 2.6 s. From the last word to the
   dispatch of the command: 1.3 s for a sentence Parakeet settled, 2.8 to 3.3 s when Whisper ran as well.
3. **Dispatch** (`PilotServer._voice_command_dispatch`). In order: barge-in stops Heliox's voice; an approval
   attempt is refused ("I can't approve that by voice"); a denial is applied; `_control_reply` tries sixteen
   handlers Heliox answers in code (help, live replies, profile, goals, the day, to-dos, reminders, undo,
   routines, shortcuts and others); stop and soft-stop words; corrections to a running task. These take tens of
   milliseconds, and a local answer was heard 0.2 s after dispatch plus synthesis (0.3 to 0.6 s to first audio in
   the probe).
4. **Everything else went to the planner.** "How are you", "what do you think about this error", "explain what
   you just did" and "open Notepad" all took the same road: interaction state, `_handle_execute`, the configured
   cloud model writing a JSON plan, then the reviewers (world model, safety review, Gnomon), the executor and the
   verifier. A stock acknowledgement chosen from five phrases ("I'm on it.") was scheduled 2.5 s after dispatch,
   and the result was spoken when the planner returned.

What decided the road was the order of those handlers: anything no handler claimed was a task. Nothing told a
question from a request.

### Where the waits were

From the daemon log of the November rehearsals, dispatch to the end of the turn on the planner road:

| Utterance (as recognised) | What it was | Dispatch to done |
|---|---|---|
| "good afternoon" | a greeting | 27.1 s |
| "Open" | a fragment | 48.8 s |
| "Let's try and take a look." | a remark | 19.2 s |
| "Where we store home" | misheard | 11.0 s (cut off by the person) |
| "Google, Google, Google." | noise | 61.0 s |

For conversation, the first sound was the stock phrase at 2.5 s plus synthesis, and the answer came 10 to 60 s
later. Noise and fragments became plans. The planner model itself (Claude Opus 5.5 on the owner's machine)
returns a plan in 10 to 15 s before any review.

## 2. The design

### Modules

| Module | Job |
|---|---|
| `router.py` | Decides in code what an addressed utterance is: control, local, conversation, action or unclear. No model; about 0.02 ms per call. |
| `style.py` | The fast model's instructions; cleaning model text for the ear (no markdown, paths, links, symbols, stock openers); cutting streamed text into sentences; reading the model's hand-over line. |
| `context.py` | The short-term conversation (in memory only) and the context block the model gets each turn, within about 350 tokens. |
| `fast_model.py` | One streamed model per provider kind, all counted in the same budget tracker as the planner. |
| `handover.py` | The hand-over to the planner: the structured intent, the choice of target, the spoken acknowledgement, progress and the approval pointer. |
| `layer.py` | The turn itself: `take`, `interrupt`, `clear`, `prewarm`, and the result the server needs. |
| `adapters.py` | The thin glue to the server, the focused-app lookup and Live, each answering "not available" when missing. |

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

The table in `tests/test_conversation_router.py` has 110 utterances, 51 of them the owner's own recognised speech.
None of the misheard ones reaches the planner as an action.

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
4. up to eight profile lines from `UserProfile.context_lines` (name, preferences, things they asked to keep);

plus the last six turns of this conversation as real message turns. The turns live in memory only, are cleared by
the existing "new conversation" call and lapse after ten quiet minutes. Speech that was not addressed to Heliox
never reaches this layer, so it is never stored as conversation or sent to a model.

### Sharing the floor with Heliox Live

Live (slips, risks, offers) and the conversation used to start speaking on their own checks, and Live's correction
channel outranks an answer, so a Live remark could cut a reply mid-sentence. `pilot/live/floor.py` holds one rule,
measured on the Live Bench's `converse` steps (spoken remarks over a reply: 3 per persona before, 0 after, recall
and false alarms unchanged):

1. Nobody starts while the person is talking.
2. A conversation turn holds the floor from dispatch until it is answered (`PilotServer._voice_turn`), plus 3 s.
   Live shows cards at once (an urgent risk is never held) but does not speak; a non-urgent typed catch waits and
   is said after.
3. A catch about the words that opened the turn may be spoken first, only before the turn has said anything; once
   the reply has started it becomes a card.
4. So Live never asks to speak during a reply, and a reply that starts during a Live remark waits behind it.

What Live says aloud is recorded in `ConversationMemory` as a Heliox turn (cut off if talked over), and the
listener's existing follow-up window (opened after any spoken Heliox sentence) lets "why?" or "not now" follow it
without the wake phrase.

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
* The person's words, the earlier turns of this conversation, the time, the task state and the profile lines go
  to whichever model is configured. With a local model none of it leaves the machine.

Two more context sources, added with Heliox Live and the personal layer (`pilot/live/conversation_link.py`,
`pilot/personal/context.py`):

| What | Model on this machine (Ollama) | Cloud model, `share_screen_text` `auto` or `off` | Cloud model, `on` |
|---|---|---|---|
| A Live remark about what the person *said* to Heliox (a wrong weekday, a promise), as a Heliox turn | as said | as said (it quotes addressed speech the model already gets) | as said |
| A Live remark or card about what the person *typed* (it quotes screen text), as a Heliox turn or the "Live raised this" line | as said, with the evidence | only the kind and the app ("pointed out a weekday that doesn't match its date in Notepad"); the words stay here | as said, with the evidence |
| Up to five profile, goal and habit lines that share words with this utterance, plus the answer style and preferences (one line, at most 360 characters, picked by word overlap in code) | yes | yes, like the profile lines above | yes |
| Offers ("You usually open Outlook about now"), as a Heliox turn | yes | yes (app names only, never window titles) | yes |
| Answers to the getting-to-know questions (name, main use, how to talk) | never; parsed in code | never | never |
| Speech heard near the microphone but not said to Heliox (Live may show a silent card for 90 s) | never | never | never |
| A note said in passing ("my flight is on Friday"), kept until the day after its date or for a week | yes, as a profile line | yes, as a profile line | yes |
| What those answers stored (the name, "I mainly use the computer for ...", the answer style, a speak-up or stay-quiet preference) | yes, as profile lines with later turns | yes, as profile lines with later turns | yes |

"Why did you say that", "thanks", "not now", "don't mention these", "fix it", "remember that ...", "forget ...",
"what did I tell you about ..." and the answers to an offer are handled in code before the layer and never reach a
model. Offers and the relevant-lines pick are computed on the machine with no model call.

### Handing over to the planner

`take` returns `done=False` with a `HandoverIntent`: the verb, the object, the person's words, the request
written out on its own, where it came from (`router` or `model`) and the model's own acknowledgement if it wrote
one. `choose_target` returns the target: `QUICK` when a one-action plan on the quick road's allow-list was built
for the intent (next section), otherwise `FULL_PLANNER`, with the plan, its reviewers and the approval rules
unchanged.

When the fast model decides an action is needed it writes one line,
`ACT {"verb": ..., "object": ..., "request": ..., "say": ..., "action": ...}`, which is never read aloud; `action`
is optional, one action in the planner's own schema, used only on the quick road's terms below. The planner gets the
written-out request (so "open it" arrives as "Open launch-notes.md in Notepad"); the person hears the `say`
sentence. A malformed line is treated as unclear.

While the planner works, `keep_company` speaks: the acknowledgement at once; if planning passes 2 s, one sentence
from the interaction state ("I'm working out the steps.", "I'm checking the plan for risks."); another only when
the state moves on and at least 6 s later, three at most; and, when an approval is needed, "This one needs your
approval. It's on the screen; press Approve if you want it to go ahead." The server cancels it when the planner
returns and speaks the result as before; the layer records the result so the next question can refer to it.

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

The decisions (which plan for which words, on which system) are pure and tested on all three; running the actions
was tested for real on Windows only.

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
ends.

### When the model fails

| What happens | What the person hears | Then |
|---|---|---|
| No network (DNS, refused, reset) | "I can't reach the language model right now..." once, then a short line | 30 s without asking the provider |
| The provider accepts the connection but sends nothing for 4.5 s | the holding phrase at 2 s, at 4.5 s "The language model isn't answering right now..." | 30 s without asking |
| A local model loading into memory | "Loading the local model, one moment." (once a session) | waits up to 90 s for the first word |
| The local server is not running, or the model is not pulled | "...the local one isn't running" / "...isn't installed", with where to fix it | 30 s |
| The provider account is out of credit (Anthropic 400 "credit balance", OpenAI 429 `insufficient_quota`, a 402) | "Your Anthropic account is out of credit. Add credit there, or choose another model in Settings, under Model." | 10 minutes without asking |
| Rate limited (429) | "The language model is busy right now. Try me again in a moment." | the provider's `Retry-After`, else 20 s |
| Heliox's own monthly budget is used up | "This month's model budget is used up..." | until the budget allows |
| The connection drops part-way through a reply | what was said, then "I lost the connection there." | as for no network |

### Claude over plain HTTP

Claude is reached like every other cloud provider: the Messages API with `stream: true`, read as server-sent
events with `httpx`, recorded in the budget the same way (input tokens from `message_start`, output from
`message_delta`, the abandoned-call estimate when cut off before either). The connection is opened ahead with the
free token-counting call when the person starts talking, at most every 45 s.

The Anthropic SDK was tried and dropped. Measured on 12 November, three calls each, alternating, both connections
warmed: SDK first token median 0.88 s (0.72 to 1.09 s), plain HTTP median 0.77 s (0.75 to 0.77 s); six calls cost
$0.0005. Importing the SDK took 22 to 36 s on the owner's machine (about three minutes without compiled
bytecode), which a turn would have waited for.

### The model for each kind of person

The fast model is a setting per provider (`conversation.claude_model`, `gemini_model`, `openai_model`,
`openrouter_model`, `meta_model`, `ollama_model`). Empty means the provider's fast tier where one is known for
certain, otherwise the model the person already configured.

| Person | Fast model | Streamed | First sound for a question (expected) | Told when missing |
|---|---|---|---|---|
| Claude key | `claude-haiku-5-5`, thinking off, effort low, over plain HTTP | yes | see section 3 | key rejected, budget used up, offline: one sentence each |
| Another cloud (Gemini, OpenAI, OpenRouter, Meta) | the configured model (Gemini Flash with thinking off) | yes, server-sent events | about one round trip plus synthesis | the same |
| Local only (Ollama) | the configured local model | yes, from the local server | seconds while the model loads; the holding phrase covers it | "the local one isn't running" or "isn't installed", with where to fix it |
| No model at all | none | n/a | instant, in code | once per session: conversation needs a model, and where to set one up (a local one keeps everything on the computer); afterwards one short line |

In every case control words, local answers and approvals work as before, and a request to act still reaches the
planner with a specific acknowledgement. Every call is checked against and recorded in the budget tracker the
planner uses, with the same provider names and `usage_source` values; a stream cut off by the person records the
usage it had reported, or the existing abandoned-call estimate when it had reported none.

No provider SDK is needed: every cloud provider is streamed over `httpx`, which Heliox already depends on.

### Hooks in the server

Seven small hooks, all in or next to `_voice_command_dispatch`: `interrupt()` at the top of dispatch; `take()` just
before the planner road (a turn the layer finished broadcasts `voice_result` and returns); the planner input
taken from the hand-over; the stock acknowledgement replaced by `keep_company()`; `finish()` after the result;
`clear()` in the new-conversation handler; and `prewarm()` in `_prewarm_voice`, so the fast model is ready when
listening starts. For the quick road: the dispatch passes the quick plan with the planner call and speaks the
layer's result sentence; inside `_handle_execute_inner` four small hunks take the plan, use it on the first
attempt, skip the model reviewers for it, and drop it for the planner when anything would ask for approval. The
layer is active only once the server's model stack is up.

### Platforms

Nothing in the package depends on the operating system. The focused app comes from Live's observation when Live
is on, otherwise from `screen_vision._get_active_window` (Windows, macOS and Linux backends; "Unknown" is treated
as not available). The focused field's text exists only where Live's reader runs (Windows today). When either is
missing the model is told nothing about the screen. All of this is exercised with fakes in the tests, which run on
the three CI systems.

## 3. Measured

Ten conversational turns (questions, a remark, a follow-up, a misheard line and a pronoun to resolve), Claude Haiku
5.5 through the Anthropic SDK (since replaced by plain HTTP, which measured slightly faster) on a home connection,
the real layer and context, 12 November 2026. The sink was silent and
took as long as speech would; the machine had too little free memory to load Kokoro for the run.

| Stage | Before | Now |
|---|---|---|
| Last word to dispatch (owner's recordings, real listener) | 1.3 s (Parakeet), 2.8 to 3.3 s (with Whisper) | unchanged; not this layer |
| Dispatch to first token | 10 to 15 s (a plan) | median 0.82 s, worst 1.61 s (first turn) |
| Dispatch to first sentence handed to speech | stock phrase at 2.5 s; the answer 10 to 60 s | median 1.07 s, worst 1.66 s |
| First sentence to first audio (Kokoro, owner's probe runs) | 0.15 to 0.63 s | the same path |
| A request handed over: dispatch to its acknowledgement | stock phrase at 2.5 s | no model call; the sentence goes to speech at once (tested with fakes) |

So for a model answer, the first sound now comes about 1.2 to 1.7 s after dispatch, and about 2.5 to 3 s after the
last word once the recogniser's own time is added. The scorecard asks for 1.5 s from the last word; the remaining
gap is mostly the end-of-turn and recognition stages, outside this layer.

The ten turns used 8,600 input and 400 output tokens, $0.0011, recorded in the budget tracker.

The quick road, dispatch to action started: 0.02 s median over ten allow-listed requests with a stubbed executor
(2.2 s for the first in a fresh process, which loads the world model's weights), and 0.024 s median over three
real ones on Windows (Notepad opened; battery and disk read), against 10 to 15 s for a planned step before.

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
in all. First-sentence times are owed: the machine was at 76 to 78% CPU with about 1 GB free.

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

### Spoken length, numbers and Live's remarks

A spoken reply stops before a sentence that would take it past about 30 words (about 12 seconds); the first
sentence is always said whole, a clause cut at a comma is never the last thing said, and the stream is closed so
the rest is not generated. Typed replies keep their length and their markdown. Units glued to numbers ("16GB"),
percentages, ISO dates and file names that carry one ("backup-2026-11-12.zip" becomes "the backup from the twelfth
of November") are turned into words a voice says well. A remark Heliox Live made on its own is kept in the
conversation as Live's (`add_heliox(..., source="live")`), so "why?" is answered from its evidence rather than as
Heliox's own mistake, and the model is told how its last reply began so it starts the next one differently.

On the further checks (53 turns on Claude Haiku 5.5), before these fixes and after:

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
follow-up window instead of extending it. Control words still work.

### "What can I say?"

Answered in two spoken sentences with three examples that work for this person right now: an app their system
has, a reminder, and a question when a model is set up ("why did you say that?" when Live is on), then "Say 'more'
for others." "More" within two minutes gives the next three. The command palette opens at the same time with the
full list. Every example that needs no model is answered in code (a test checks each).

## 4. Not in this pass

* The quick road covers single actions only; anything with two steps goes to the planner.
* The planner does not see the conversation turns held here; it gets the written-out request instead.
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
the exact name needs one. The merged transcript holds the wake phrase once ("Hey Heliox, Hey Heliocs, undo that"
became an undo with the phrase twice).

**A follow-up.** After Heliox finishes speaking in an open conversation, the next utterance needs no wake phrase
only if all of these hold:
- it starts within 8 seconds of Heliox's last word (people reply within about a second; 8 s leaves room to think,
  where the old 30 s window let a room in);
- it did not begin while Heliox was still talking, unless it is a control word ("Stop", "Wait");
- it is plausibly meant for Heliox: a control word, a reply to the question Heliox just asked, or a direct request
  or question (imperative, or second person), or, within 4 seconds of Heliox's answer, a short elliptical follow-up
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
question is when noise or Heliox's own voice through open speakers is most likely to be read as words.
Control words ("Stop", "Wait") are English only; talking over Heliox for a moment stops it in any language.

**Overheard speech** goes only to Heliox Live, marked not addressed, which gives silent cards at most. It is never
stored as conversation and never sent to a cloud model.

**What the person sees.** The voice bar shows when a conversation is open and when it closes. When a direct
request or question in the second person was not taken because no conversation was open, it says once: "Say 'Hey
Heliox' first."

**Settings.** `voice.continuous_conversation_enabled`, which took sentences that looked like commands as addressed
without the wake phrase, is no longer read and defaults to off; its text says that it made Heliox act on what other
people say. `voice.follow_up_window_seconds` is 8. Heliox speaking (a reminder, a Live card) opens no reply window
unless someone opened the conversation.

**Hooks.** Each dispatch sets `pilot.system.voice.voice_turn_addressed` ("wake", "button", "follow_up",
"doubtful_follow_up") beside `voice_turn_ended_at`; the conversation layer calls
`ContinuousVoiceListener.note_not_understood()` after a "Sorry?".

Measured on the Conversation Bench's transcript replay (2026-11-12, both recognisers' words cached through the
listener's own calls): of the 981 utterances not addressed to Heliox (all 62 minutes), the old behaviour
dispatched 397 and this rule 0, with nothing spoken; with the elliptical follow-ups added it is still 0 (none of the
981 opens a window, and none matches the elliptical phrases at all). The owner's wake sentences were taken as before: desk
microphone 4 of 4, Bluetooth headset 3 of 5 (in the two misses neither recogniser heard the wake phrase). The four
"Wait, stop." cut-offs over Heliox stop it. Still to measure: the whole hour and the control words at audio level,
and a real follow-up recording.

**Not done: whose voice.** Nothing checks that a follow-up comes from the person who said the wake phrase. The
cheapest evidence is the speech level of the follow-up against the wake utterance that opened the conversation: the
same person at the same distance from a desk microphone should be within a few decibels, a television or someone
across the room usually not. None of the installed recognisers gives a speaker embedding, so anything stronger
needs a download. Measuring the level rule needs recordings that do not exist yet: two people taking turns at one
desk microphone (one addressing Heliox, the other talking to them), and a television or video playing in the room
while the owner talks to Heliox, each on a desk microphone and a headset.

## 6. Ending the turn early (`early_turn_threshold`): measured and left off

The recorder can ask Smart Turn as soon as 200 ms of quiet follow a word and end the turn there when the model is
at least `early_turn_threshold` sure the sentence is finished; otherwise it waits for the usual 400 ms check. The
code stays in and the setting stays at 1.0 (off): on longer sentences it cuts them in two.

Measured 2026-11-12 with `benchmarks/conversation_bench/turn_split.py`: the listener's own recorder fed one 50 ms
frame at a time on the audio's own clock (the next frame only after the recorder has taken the last one and any
Smart Turn call on it), so the numbers do not depend on the speed of the machine. "Split" means one sentence became
two or more utterances.

| `early_turn_threshold` | Owner's recordings (18 short sentences): split | ended at 200 ms | Synthetic addressed sentences (49): split | Corpus lines, not addressed (159): split |
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
sentence, on the 49 synthetic addressed sentences and 159 corpus lines:

| hold | addressed split | corpus lines split | finished sentences ended later | by |
|---|---|---|---|---|
| none (today) | 0 | 23 | 0 of 182 | - |
| last word is an article, preposition, conjunction or possessive ("to", "and", "the", "my") | 0 | 15 | 35 of 182 (4 of 46 addressed) | 600 ms each |
| wider list, with pronouns and modals ("it", "you", "can") | 0 | 14 | 49 of 173 | mean 179 ms, up to 2.1 s |
| after 2 s of voice, end only at 0.95 | 0 | 20 | 24 of 177 | mean 132 ms, up to 2.1 s |

The narrow list fixes 8 of the 23 cuts and costs 600 ms (one more Smart Turn check) on one long finished sentence in
five, never on a sentence under 2 s. The cost comes from the streaming recogniser's words lagging the audio: at the
400 ms check its last word is often one before the real last word. Not adopted; the next thing to measure is the same
hold on the recogniser's flushed words at the check.
