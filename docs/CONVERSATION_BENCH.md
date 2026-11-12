# Conversation Bench

The Conversation Bench turns the program brief's scorecard into one repeatable command: it plays recorded speech
through Heliox's real listener and voice dispatch and prints, line by line, how far Heliox is from each target.
Before it, every claim about the spoken side came from one-off probes.

```
cd daemon
python -m benchmarks.conversation_bench run --audio <audio folder>             # the baseline profile
python -m benchmarks.conversation_bench run --audio <audio folder> --profile full
python -m benchmarks.conversation_bench soak --minutes 120 --audio <audio folder>
python -m benchmarks.conversation_bench score <results folder>                 # re-score saved timelines
python -m benchmarks.conversation_bench list --audio <audio folder>
```

Code, scenario files and a short README live in `daemon/benchmarks/conversation_bench/`. Its own logic is tested
by `daemon/tests/test_conversation_bench.py`, which needs no audio, speech or model package and runs in CI on
Windows, macOS and Linux.

## What runs and what is replaced

Real: the continuous listener (voice activity, Smart Turn, streaming recognition, the Parakeet or Whisper final
pass, the wake phrase, follow-up windows, barge-in), the server's voice dispatch (approvals, denials, stops,
undo, corrections, local answers), Heliox Live (cards from speech) and the speech coordinator with the local voice
(Kokoro synthesises every reply; the audio is consumed in real time and never played).

Replaced:

- The microphone: a file pump feeds 16 kHz blocks at real-time pace. The real microphone is never opened.
- The planner: a stub that waits as long as the chosen kind of model would and returns "All set." (or, with no
  model, the "no provider" error). No cloud model is called. Plans are not executed.
- The data folder: a fresh, empty one per run under the results folder. A copy of a person's data would carry
  undo history and scheduled runs that act on real files.
- The speaker: silent. Output-device latency and the device's own cut-off time on barge-in are therefore not in
  the numbers, and neither is echo of Heliox's voice into an open microphone.

The listener is wired as the server wires its own: the bench reads the server's listener start-up from its source
and sets the same attributes (turns enter through the server's turn entry point; a sentence split at a pause can
be withdrawn and joined). Scenarios are independent: one server runs them all, and between scenarios the bench
makes the new-conversation call, as a person starting over would, which drops what Live holds, clears the
conversation layer and closes the follow-up windows.

## Audio

Scenario files hold only text and expectations. Audio is referenced as a path inside a private folder
(`--audio`, else `CONVERSATION_BENCH_AUDIO`, else `~/heliox-conversation-bench-audio`), for example
`real/desk/take_2241.wav#24.0-26.5` (a span in seconds). References cannot be absolute or leave that folder. The
folder is never committed: the real recordings are a person's voice.

- `real/`: recordings of the owner. Each scenario names the microphone (`bluetooth-headset-call-mode` for the
  first rehearsals, `ptron-usb-desk` for the later ones).
- `synthetic/<voice>/<speed>/<name>.wav`: the local Kokoro voice, rendered by
  `python -m benchmarks.conversation_bench.synth --audio <folder>`. Synthetic numbers are always reported apart
  from real-voice numbers and never mixed: on 10 November every number measured only on synthetic speech was wrong
  for the owner's voice.
- `synthetic/not_addressed/block_NN.wav` (+ `.json`): the hour of speech not addressed to Heliox (meetings,
  tutorial narration, read-aloud text and phrases that set it off before, such as "PowerShell", "Where we store
  home" and "Visual Studio Code"), generated from `corpus.py` with three voices.
- `synthetic/babble.wav`: two voices talking over each other, for the open-office conditions.

To use your own voice, record sentences with any recorder, put the WAV files under `real/`, and add scenarios
with the sentence you said as `text`.

## Conditions: one recording for many people

Every real sentence is also heard under derived conditions, all labelled derived in the scorecard:
`bt-call` (a Bluetooth headset in call mode: 8 kHz, nothing above 3.4 kHz, 8-bit companding; matched to the
owner's own headset recordings, which are 50 to 65 dB down above 3.4 kHz), `room-20db`, `fan-10db` (a laptop
microphone by its fan), `voice-10db` and `voice-5db` (another person talking), `quiet-12db`, `loud+10db`,
`slow-0.9` and `fast-1.1` (pitch kept). The full profile also plays the hour of not-addressed speech under
`room-20db`, `fan-10db` and `voice-10db`.

## Kinds of model

Control and turn-taking scenarios run once per kind of model (`models.py`): a cloud model, a local model and no
model. Delays are approximate, from public third-party latency measurements and llama.cpp-class local inference,
not measured here: cloud talk model 0.6 s to first token and 150 tokens/s, cloud planner 1.5 s and 50 tokens/s;
local talk model 1.5 s and 25 tokens/s, local planner 2.5 s and 12 tokens/s; none answers at once that no model is
set up. Pass conditions by kind:

- Control words (stop, wait, no, a refused yes, a correction, undo) and local answers do not depend on the model:
  the same result and the same limits under all three.
- A request that needs a plan: cloud and local are held to the limits for model answers. With no model, Heliox
  must say promptly and honestly that it needs a model for that and how to set one up, and claim nothing.
- A remark or question for the fast spoken layer: with no model, a short honest answer about what is missing.

## Scenario sets

| set | what it checks | lines |
|---|---|---|
| understanding | every real sentence of the owner's that survives, plus synthetic coverage in three voices and speeds | 2.1a, 2.1b, 2.1c, 1.3, 1.1a, 3.1 |
| turn_taking | reply start, never starting over the person, barge-in stop time for short and long interruptions, mid-sentence pauses of 0.6 to 2 s | 1.1, 1.2, 1.3 |
| not_addressed | an hour of speech for someone else: nothing may be dispatched, spoken or stored | 2.2 |
| control | stop, wait, no on a pending approval, yes and "approve it" refused, a correction during a task, undo | 4.2 |
| conversation_quality | questions, remarks, follow-ups, explanations and a request to act, sent as text to the fast spoken layer through `pilot.conversation.bench_reply` (listed as pending when that function is absent); first sentence within 1.5 s (cloud), 2.5 s (local), 0.8 s (none) | Q |

`run` also runs the Live Bench (precision and recall) and the Memory Bench and adds their numbers (`--no-benches`
skips them). The everyday planner eval needs a model and stays opt-in (`docs/PLANNER_EVERYDAY_EVAL.md`).

## Output

The scorecard prints each line per voice, condition and kind of model, with its number, the target and pass or
FAIL. The results folder (default `~/heliox-conversation-bench-results/<time>`, outside the repository because
timelines quote what was said) holds `scorecard.txt`, `scorecard.json` (with the worst cases), `platform.json`
and one timeline per scenario run under `timelines/<set>/`. A timeline lists every moment on one clock: the
person's first and last word, voice activity, each transcript and its recogniser, wake or follow-up, dispatch,
planner call, results, cards, when Heliox's voice started, finished or was cut, and approval state.

## The soak

`soak --minutes 120` runs mixed activity (requests, barge-ins, control words, not-addressed speech) on one
listener and samples memory, handles or file descriptors, threads and asyncio tasks of the process and its voice
worker every ten seconds, counts voice-worker restarts, missed and duplicated dispatches, and at the end checks
that Heliox still hears and still speaks. Idle time between activities is compressed to two seconds; the speech
itself cannot be, because endpointing and the recognisers run on the audio's clock (`--pace` above 1 feeds audio
faster and records how far the feed fell behind; timings are then not real).

## Running it on each system

The runner is plain Python: paths are built with `pathlib`, WAV files are read with the standard library and
numpy (no OS audio library), processes are watched with `psutil` when it is installed, and nothing opens an audio
device. What each system needs for a full run is what the daemon's voice needs:

- All systems: the daemon's virtual environment with the voice extras (`sherpa-onnx`, `onnxruntime`, the
  streaming model, Smart Turn, `faster-whisper`; Parakeet files if `final_pass = "parakeet"`), and for synthetic
  audio, `kokoro` with its weights already downloaded (the bench never downloads).
- Windows: `daemon\.venv\Scripts\python.exe -m benchmarks.conversation_bench run --audio C:\path\to\audio`.
- macOS (Apple Silicon or Intel) and Linux: `daemon/.venv/bin/python -m benchmarks.conversation_bench run --audio
  ~/path/to/audio`. No microphone or screen-recording permission is needed, because no device is opened.
- CI: only the unit tests run (`pytest tests/test_conversation_bench.py`); the full bench needs the speech models.

The bench forces the CPU (`CUDA_VISIBLE_DEVICES=""`) so numbers are comparable between machines and a failing GPU
cannot take a run down. Today it has only run on the owner's Windows laptop; on macOS and Linux only its unit
tests have run.

## Fast replay of addressing (line 2.2 and the owner's clips)

The audio-level hour takes about 90 minutes. For iterating on who Heliox thinks is talking to it, the replay runs
the same utterances at transcript level in seconds:

```
python -m benchmarks.conversation_bench.replay build --audio <folder> [--redo]   # once: recognise and cache
python -m benchmarks.conversation_bench.replay run --audio <folder> --decide pilot.system.addressing:decide
```

`build` recognises every corpus line of the not-addressed blocks and every real clip through the listener's own
calls (the Whisper wake check with its English setting and vocabulary prompt, and the Parakeet final pass on the
same normalised samples) and caches the words, voiced seconds and speech level in `replay_cache.jsonl` in the
audio folder. `run` feeds them in order to a decision function, simulates Heliox's reply after each dispatch (so
follow-up windows open as they would), prints the counts per label and writes `replay_table.tsv`: what was said,
both recognisers' words, the decision and its rule, and which earlier turn opened the window it fell in. Its
utterances are corpus lines; the listener's endpointing sometimes joins two, so the audio-level run stays final.

## Re-baseline with the conversation layer up

`python -m benchmarks.conversation_bench.rebaseline --audio <folder>` runs the real-voice, control and turn-taking
scenarios with the server's model stack marked up and a stub fast model per kind of model (`stack.py`), labels
real-voice rows by microphone (desk, headset), samples the load before the run and before every scenario
(`machine.py`), and prints every timing line measured on a busy machine as owed.

## Hooks the product should expose

The driver reaches into private attributes today; each of these would make a number exact and survive refactors:

- `ContinuousVoiceListener`: an `on_transcript(text, engine, endpoint_at, asr_started_at, asr_done_at)`
  callback for every utterance turned into words, addressed or not (the bench polls `transcripts_received`).
- The recorder: a supported audio source (`recorder.feed(block)` with `capture_backend = "file"`) instead of
  setting `_soundcard_recorder` and calling `_publish_audio_block`.
- The speech coordinator: `speech_started` and `speech_ended(reason: done|cut|replaced)` notifications with
  monotonic times taken at the output device, so stop time on barge-in includes the device and is not read
  from a cancelled coroutine.
- The voice worker: a `voice_status` broadcast when the local voice is loading, loaded (with seconds taken) and
  when it falls back to the system voice, and a `load_seconds` stats field.
- The listener: a `busy` property (an utterance is being recognised or acted on), so a run knows when a turn is
  over without waiting on timers.
- `BargeInGate`: its hold times as settings, and the gate decision logged with the level it saw.

## Baseline, 12 November 2026

Owner's Windows laptop, CPU only, planner stubbed, silent output. The machine was shared with other agents and
ran at 75 to 100 percent CPU for the whole run. Every timing measured then is owed and not quoted here; the
raw values stay in the results folder only as evidence of order between stages. Real-voice understanding was run as recorded and under five derived
conditions (bt-call, fan-10db, voice-10db, quiet-12db, fast-1.1). Full numbers per condition and model are in the
results folder's `scorecard.txt`.

| line | real voice, as recorded | synthetic | target | stage responsible (from the timelines) |
|---|---|---|---|---|
| 1.1a local answer starts | owed (laptop, real voice) | runner: 5.00 s (synthetic voice, GitHub runner (4 cores), CPU only); laptop owed | 0.8 s | stage order seen in the timelines: endpoint, then the final pass (the longest stage on this run), then synthesis; the answer itself is immediate |
| 1.1b/c model answer starts | owed (laptop, real voice) | runner: 3.22 s median, 3.55 s worst (synthetic voice, GitHub runner (4 cores), CPU only); laptop owed | 1.5 s / 2.5 s | in this run the spoken acknowledgement was scheduled 2.5 s after dispatch (no longer so when the conversation layer is active) |
| 1.1d longest silence | owed (laptop, real voice) | runner: 6.24 s cloud, 30.20 s local (synthetic voice, GitHub runner (4 cores), CPU only); laptop owed | 2.5 s | nothing is said while the final pass runs |
| 1.2a talks over the person | 0 | 0 | 0 | |
| 1.2b voice stops after barge-in | owed (laptop, real voice) | runner: 0.34 s (synthetic voice, GitHub runner (4 cores), CPU only); laptop owed | 0.4 s | `BargeInGate` waits 0.4 s of speech with open speakers (0.25 s with headphones) by design, which alone uses the whole budget |
| 1.2c interruption heard | 12/12 | 10/10 | all | |
| 1.3 sentence split at a pause | 0/14 | 4/4 | under 5% | Smart Turn ends the turn at a 0.6 to 2.0 s pause after "remind me to"; "remind me too" went to the planner |
| 1.4a plain-speech rubric | 7/12 | 71/75 | all | the "what can I say" answer is 53 words, a spoken list of features ("undo, reminders and timers, notes, ..."); the only failing reply |
| 2.1a word error rate | 22.5% | 5.3% | 8% | final pass; wake phrase heard as "Dear Yorks", "Here he was", "The works" |
| 2.1b wake phrase found | 6/9 | 28/28 | 19 in 20 | desk microphone 4 of 4; Bluetooth headset 2 of 5 ("Dear remind me", "Here is your what can I build"); two found ones were doubled ("Hey Heliox, Hey lost") by the Parakeet and Whisper merge |
| 2.1c no wake needed in an open conversation | - | 0/1 | all | follow-up after a local answer was not acted on |
| 2.2 not addressed (62 min of speech) | 0 of 3 real clips | 264 dispatches, 514 spoken lines | 0 | 28 wake-free "command intent" requests (continuous conversation is on), 11 wake-word false hits ("Hello, Alex", "Helix", Heliox named in the third person), then 225 more inside the 30 s follow-up window each reply opens |
| 3.1 spoken slip raised as a card | 1/5 | 1/2 | all | unwoken headset sentences never reach Live; one invoice take was heard as "twenty first"; one correct take raised no card, possibly because Live does not repeat a slip it already raised in the same run (the bench keeps one Live instance) |
| 4.1 simple request under way | never for unwoken sentences; otherwise owed (laptop) | runner: 5.06 s (synthetic voice, GitHub runner (4 cores), CPU only); laptop owed | 3 s | |
| 4.2 control words | 12/12 | 10/11 per model | all | undo after a correction: Parakeet and Whisper merged into "Hey Heliox, Hey Heliocs, undo that", so the undo went to the planner |
| Q fast-layer quality (canned stubs) | - | 9/9 per model | all | |
| Live Bench (text) | precision 1.00, recall 1.00 (personal policy, 27/27) | | keep | |
| Memory Bench | 11/11 | | 10/10 | |

Kinds of model: control words and local answers behaved the same under cloud, local and none. With no model a
plan request was answered with the "configure a provider" message (the acknowledgement plays first; its timing is owed).

Derived conditions on the owner's voice (WER / wake found): Bluetooth call mode 35% / 4 of 9, laptop fan 27% /
6 of 9, another voice 10 dB under 112% / 2 of 9 (the background voice's words were transcribed instead),
12 dB quieter 57% / 3 of 9, 10% faster 33% / 6 of 9.

## Timing on a GitHub runner (first run, 12 November)

The weekly timing workflow (`.github/workflows/conversation-timing.yml`) ran once by hand: run 43197018718 on
0a8edad, ubuntu, 4 cores, CPU only, nothing else running, the load check passed before every scenario. Every
number here is "synthetic voice, GitHub runner (4 cores), CPU only". It tells how long Heliox's own stages take on a small CPU-only machine with a quiet
processor. It does not tell how the owner's laptop behaves (another processor, other load), how a real voice or
microphone behaves, or how long the output device takes to make a sound: the runner has no audio device, so the
listener is fed from files and Heliox's voice is consumed silently, as everywhere in the bench.

| line | runner (cloud stub) | local stub | no model | target | stage that dominates |
|---|---|---|---|---|---|
| 1.1a local answer starts | 5.00 s (n=11) | 5.36 s (n=1) | 5.49 s (n=1) | 0.8 s | synthesis |
| 1.1b model answer starts (median) | 3.22 s (n=13) | 3.20 s (n=1) | 4.84 s (n=1) | 1.5 s | synthesis, then recognition |
| 1.1c model answer starts (worst) | 3.55 s | 3.20 s | 4.84 s | 2.5 s | as 1.1b |
| 1.1d longest silence | 6.24 s (n=24) | 30.20 s (n=2) | 5.49 s (n=2) | 2.5 s | see below |
| 1.2b voice stops after barge-in | 0.34 s (n=6) | 0.32 s | 0.34 s | 0.4 s | the barge-in gate |
| 4.1 simple request under way | 5.06 s (n=24) | 1.97 s (n=2) | 1.68 s (n=2) | 3 s | see below |

Stage medians, wall clock (the audio clock agreed within 0.05 s everywhere, so the listener kept up with the
audio throughout):

| stage | 1.1a, cloud (n=11) | 1.1b, cloud (n=13) | 1.2b, cloud (n=6) |
|---|---|---|---|
| endpoint (last word to end of turn) | 0.69 s | 0.78 s | |
| recognition (Parakeet for the words, the Whisper wake check beside it) | 0.80 s | 1.10 s | |
| decision (words to text to say; for 1.1a the stub fast model's 0.6 s first token is in here) | 0.74 s | 0.0 s | |
| planner call to the acknowledgement | | 0.04 s | |
| synthesis (text to the first sound, Kokoro on the CPU) | 2.46 s | 1.24 s | |
| onset (first word to voice activity) | | | 0.06 s |
| gate (voice activity to barge-in accepted) | | | 0.27 s |
| cut (barge-in to silence) | | | 0.00 s |

What this says:
- 1.1a: half of the five seconds is synthesis. The local answers are whole sentences of 15 to 25 words, and the
  voice synthesises a sentence completely before it plays any of it. Halving synthesis moves 1.1a most (about
  1.2 s); halving recognition would save 0.4 s.
- 1.1b: the first sound is the spoken acknowledgement ("Okay, opening Notepad."), which is short, so synthesis is
  1.24 s; recognition (1.10 s) is next. Halving either moves 1.1b by about 0.6 s; neither alone reaches 1.5 s.
- 1.1d at 30.20 s on the local model is the bench's own assumption, not something measured: the stub planner for
  a local model takes 2.5 s plus 400 tokens at 12 tokens a second (35.8 s), and Heliox said "I'm working out the
  steps." once and nothing more until the result. What it shows is real: during a long plan nothing is said after
  the first progress line.
- 4.1 differs by kind of model only because of which scenarios ran: on the cloud stub it includes the synthetic
  questions answered by the conversation layer ("what did I do this morning?"), where the first sign of progress
  is the spoken answer itself (5.0 s, synthesis again); the local and no-model lines have only the two
  turn-taking scenarios (n=2), whose acknowledgement or local answer comes sooner.
- 1.2b passes on the runner (0.34 s): the gate took 0.27 s, against 0.42 s in the laptop runs before the speech
  agent's capture-side voice activity landed; which of the gate's holds the bench's silent output selects was not
  checked.

The bench's own share: the stub models' delays (cloud first token 0.6 s; the local planner's 35.8 s), the n=1 and
n=2 lines for local and no model, and the first scenario of the run (c-syn-stop, 3.9 s to "Okay, stopped." where
the others took 3.5 s), which carries a cold voice worker. The file-fed microphone kept real-time pace (the two
clocks agree).

## Soak, 20 minutes (12 November)

`soak --minutes 20` ran 24 minutes of wall time (16 activities: requests, barge-ins, control words, two five-minute
not-addressed blocks), sampling every 10 s. Resident memory of the bench process went from 1130 MB after warm-up
to 1019 MB (peak 1188 MB, no upward trend); the voice worker from 316 MB to 130 MB. Handles 701 to 710, threads
106 to 105, asyncio tasks 10 to 7. The voice worker started once and never restarted. The audio feed fell behind
real time at moments, an effect of the load (not quoted). Missed: the two Bluetooth-headset launch-call sentences, which are never woken (see 2.1b).
At the end Heliox still heard and answered aloud. Twenty minutes cannot show a slow leak; the two-hour run is
`python -m benchmarks.conversation_bench soak --minutes 120 --audio <folder>`.

What could not be compressed: speech plays at real time (endpointing, Smart Turn and follow-up windows run on the
audio clock and the recognisers must keep up). Only idle time between activities is compressed, to 2 s.

## Platform-specific code found on the spoken path

From `platform.json` (the run itself was on Windows only, and nothing Heliox said read as "not available"):

- `pilot/system/accessibility.py` `foreground_window`: Windows only, returns None on macOS and Linux.
- `pilot/live/focused_text.py` `window_exists`: only Windows checks; elsewhere every window counts as open.
- `pilot/system/call_detection.py`: Windows, macOS (CoreAudio) and Linux (PulseAudio streams) paths exist in the
  working tree today; only the Windows one has run here.
- `pilot/system/local_tts_worker.py` `_default_spawn`: a Windows-only process flag; behaviour is the same elsewhere.

## Re-baseline, 12 November afternoon (addressing rule, conversation layer up)

Measured on the working tree: base d8fa4ec plus the uncommitted listener and addressing patch that was later
committed as 789b7d6 (the re-baseline started at 11:29). The audio-level unaddressed hour ran from 12:13 on a
tree equal to ff0a8d1. The conversation layer was up with a stub fast model (10 stub calls during the run).
Only lines that do not depend on time are given; every timing line is owed, because the machine was busy. Real
voice is split by microphone: the desk microphone measures recognition, and the Bluetooth headset (narrowband
call mode, which the product now detects and warns about) measures the warning.

| line | real, desk | real, headset | synthetic | changed since the first baseline |
|---|---|---|---|---|
| 2.1a word error rate (as recorded) | 16% (9 edits / 58 words) | 34% | - | desk 12% to 16%, headset 36% to 34%: the same recognisers, run-to-run variation on a loaded CPU |
| 2.1b wake phrase found (as recorded) | 4/4 | 2/5 | 4/4 | none |
| 2.1b in derived conditions (desk) | Bluetooth call mode 2/4, fan 4/4, another voice 1/4 | | | another voice 2/4 to 1/4 |
| 1.3 split at a pause | 0/9 | 0/5 | 4/4 split (0.6 to 2 s pauses) | none |
| 1.2a talks over the person | 0 | 0 | 1 in 13 scenarios | 0 to 1 (synthetic) |
| 1.4a plain-speech rubric | 8/8 | 2/2 | 28/28 | was 7/12 real and 71/75 synthetic: the "what can I say" answer is short now |
| 2.1c no wake needed in an open conversation | - | - | 0/1 | none: the follow-up after a local answer still gets no reply |
| 2.2 not addressed, real fragments | 0 of 3 | | | none |
| 2.2 open-conversation remarks to a colleague | | | 0 of 3 | none |
| 3.1 spoken slip raised as a card | 0/3 | 0/2 | - | was 1/3 desk: remarks in an open conversation are now ignored, so Live never hears the invoice slip |
| 4.2 control words | 12/12 (all three models) | | 10/11 per model | the undo after a correction is now heard as "undo that" (merge fixed), but nothing is said back |
| 2.2 the unaddressed hour through the real listener (13 blocks, 67 min of synthetic speech, 1130 utterances recognised) | | | 0 on a synthetic hour (dispatched, spoken, stored); no real recording of a meeting, a television or a second person exists yet | was 264 dispatched and 514 spoken lines |
| Transcript replay, the hour plus the owner's clips | 0 of 981 synthetic not-addressed lines dispatched; addressed 7/9 (desk 4/4, headset 3/5); control 4/4 | | | first baseline at audio level: 264 dispatches |

Also seen: "Hey Heliox, open Notepad" (synthetic, one voice) was heard as "opened Notepad" and answered by the
conversation layer as a remark, not sent to the planner. With no model, the layer said that talking things
through needs a language model and where to set one up.

## Round 2, 12 November evening (fc60d65, timing-free)

| line | case | result | moved since the afternoon? |
|---|---|---|---|
| 2.1c | follow-up without the wake phrase after a local answer ("And what about reminders?") | taken and answered | yes, 0/1 to 1/1 |
| 4.2 | "Hey Heliox, undo that." after a correction, once the task has ended | recognised, then handed to Live as overheard; nothing dispatched, nothing said | no |
| 1.2a, 1.3 | "Hey Heliox, remind me to ... call the bank tomorrow morning." with a 0.6 to 2.0 s pause | the turn still ends at the pause; at 0.6 and 1.0 s Heliox's acknowledgement starts while the person is still talking (no barge-in); at 1.5 and 2.0 s the second half is taken as "Stop" | no |
| 2.1a | "Hey Heliox, open Notepad." in one synthetic voice | heard as "opened Notepad", answered by the conversation layer as a remark | no |

Line 3.1 now means a slip in something said to Heliox. A slip in a remark to someone else must not raise a card,
because Live never hears speech that was not addressed to it (the privacy line). Scenarios are in
`scenarios/slips.json`; they use synthetic voices, and the owner's real voice for them is still owed.

| case | expected | today |
|---|---|---|
| "Hey Heliox, remind me on Tuesday the nineteenth to send the report." (the 19th is a Thursday) | card | card |
| "Hey Heliox, remind me that the invoice is due on the thirty-first of November." | card | card |
| "Hey Heliox, put the review in for Monday the nineteenth at ten." | card | no card: the conversation layer answers it and Live never sees it |
| "the invoice for Acme is 2,400", then "email the Acme invoice for 4,200" | card | no card; Heliox says "Emailing the acme invoice for 4 200." |
| the invoice slip said to a colleague in an open conversation (synthetic, and the three real desk takes) | no card | no card, 4 of 4 |

The wake phrase with another person talking behind it, on the four real desk sentences (two-voice background mixed
at a speech-to-background level, two background positions each, the listener's own recognisers and addressing
rule): 8 of 8 taken at 30, 20 and 15 dB; 4 of 8 at 10 dB; 2 of 8 at 5 dB; 1 of 8 at 0 dB. It breaks between 15
and 10 dB, mostly because the name is replaced by the background voice's words; twice the name survived but not
at the start, and the rule correctly declined it. Only the Whisper wake check ever writes "Heliox"; Parakeet does
not, even at 30 dB.

The real word error rate rests on 58 words from three distinct sentences on the desk microphone and 44 words on the
headset. A desk number good to about 2.5 points near 8% needs about 40 different sentences (about 500 words) per
microphone, plus about 20 wake sentences per microphone for the 19-in-20 line: what the recording helper below
collects.

## Split sentences and slips on 008f1a0 (synthetic voice, timing-free)

With the listener wired as the server wires it, a sentence split at a pause is withdrawn and joined: on the four
pause scenarios ("Hey Heliox, remind me to | call the bank tomorrow morning", 0.6 to 2.0 s) the first half's
planner call is cancelled and one planner call carries the joined sentence, and Heliox never starts speaking over
the person (0 of 4). The first half's progress line ("I'm working out the steps.") is said again for the joined
sentence at 1.0, 1.5 and 2.0 s. Of the 23 corpus lines the listener cuts at a comma, said to Heliox, 21 end as one
joined turn; the two others are three-part sentences, joined once and then continued again. The five slip
scenarios: 5 of 5 as expected (the four slips said to Heliox held with a question before anything acts, the remark
to a colleague silent).

The bench now logs a withdrawal (with the server's answer and the gap), a cancelled planner call, and a turn the
server starts itself; before, it counted the cancelled first half as an action and missed the joined turn.

## Recording your own voice

The bench needs more of your own voice before two lines mean anything: how many words Heliox gets wrong, and how
often it hears its name. One pass takes about fifteen minutes per microphone. Open a terminal in the `daemon`
folder and run `.venv\Scripts\python.exe -m benchmarks.conversation_bench.record --mic desk` (on macOS or Linux,
`.venv/bin/python` instead). Before it starts it shows which microphone it is about to record from: the one
Heliox is set to use, or the one you name with `--device`. It changes no setting and plays nothing. It then shows
one sentence at a time: press Enter, say the sentence the way you would say it to Heliox, press Enter again, then
keep it, redo it or skip it. There are three sets: 41 everyday sentences, 20 sentences that start with "Hey
Heliox", and five minutes where you let a television or another person talk near the microphone without speaking
to Heliox. You can stop at any point with q; next time it carries on where you left off. If the microphone turns
out to be a Bluetooth headset in call mode while you called it something else, it stops and asks you to run it
again with `--mic headset`.

Two more sets measure conversation itself. `--set followup` (about five minutes, run as part of the default pass)
shows fifteen things Heliox might say, one at a time; read each, press Enter at the moment its voice would end, and
answer as the prompt says: at once, after thinking for a few seconds, or to someone else in the room. Heliox does
not need to be running, and the helper stores how long you took to start answering, so the bench can check
whether replies are taken inside the follow-up window and remarks to other people are left alone. `--set echo`
(about five minutes) is the only set that makes sound, so it runs only when you ask for it by name, warns you
first and asks you to type yes. Use open speakers at your usual volume. For each take it plays a sentence in
Heliox's voice when you press Enter, and you say "Wait, stop.", cough, type loudly or stay silent, as prompted;
the bench then checks that only "Wait, stop." stops Heliox, and how fast.

Then do the same with every other microphone you use, under another name (`--mic headset`, `--mic laptop`), so the
results can be compared. The recordings stay in your private audio folder on this computer, never in the
repository and never uploaded. When you finish, the helper prints what you recorded and the command that runs the
bench on it; the scorecard then shows your word error rate and wake-phrase rate for each microphone separately.
