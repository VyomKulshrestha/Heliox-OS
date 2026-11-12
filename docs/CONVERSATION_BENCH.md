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
ran at 75 to 100 percent CPU for most of the run, which slows recognition and synthesis; the timings below are
what a busy machine gives, not a quiet one. Real-voice understanding was run as recorded and under five derived
conditions (bt-call, fan-10db, voice-10db, quiet-12db, fast-1.1). Full numbers per condition and model are in the
results folder's `scorecard.txt`.

| line | real voice, as recorded | synthetic | target | stage responsible (from the timelines) |
|---|---|---|---|---|
| 1.1a local answer starts | 5.7 s median | 6.9 s | 0.8 s | recognition after the endpoint 2.7 s median (final pass on a busy CPU), the endpoint 0.6 s, synthesis 0.4 to 1.1 s; the answer itself takes 0.03 s |
| 1.1b/c model answer starts | 30.8 s (n=1) | 6.0 s median, 19.7 s worst | 1.5 s / 2.5 s | the spoken acknowledgement is scheduled 2.5 s after dispatch, after recognition of 2.6 s |
| 1.1d longest silence | 30.8 s | 27.6 s | 2.5 s | same as above; nothing is said while the final pass runs |
| 1.2a talks over the person | 0 | 0 | 0 | |
| 1.2b voice stops after barge-in | 0.48 s (0.81 s once, cold) | 0.49 s | 0.4 s | `BargeInGate` waits 0.4 s of speech with open speakers (0.25 s with headphones); voice activity itself is seen in 0.06 s |
| 1.2c interruption heard | 12/12 | 10/10 | all | |
| 1.3 sentence split at a pause | 0/14 | 4/4 | under 5% | Smart Turn ends the turn at a 0.6 to 2.0 s pause after "remind me to"; "remind me too" went to the planner |
| 1.4a plain-speech rubric | 7/12 | 71/75 | all | the "what can I say" answer is 53 words, a spoken list of features ("undo, reminders and timers, notes, ..."); the only failing reply |
| 2.1a word error rate | 22.5% | 5.3% | 8% | final pass; wake phrase heard as "Dear Yorks", "Here he was", "The works" |
| 2.1b wake phrase found | 6/9 | 28/28 | 19 in 20 | desk microphone 4 of 4; Bluetooth headset 2 of 5 ("Dear remind me", "Here is your what can I build"); two found ones were doubled ("Hey Heliox, Hey lost") by the Parakeet and Whisper merge |
| 2.1c no wake needed in an open conversation | - | 0/1 | all | follow-up after a local answer was not acted on |
| 2.2 not addressed (62 min of speech) | 0 of 3 real clips | 264 dispatches, 514 spoken lines | 0 | 28 wake-free "command intent" requests (continuous conversation is on), 11 wake-word false hits ("Hello, Alex", "Helix", Heliox named in the third person), then 225 more inside the 30 s follow-up window each reply opens |
| 3.1 spoken slip raised as a card | 1/5 | 1/2 | all | unwoken headset sentences never reach Live; one invoice take was heard as "twenty first"; one correct take raised no card, possibly because Live does not repeat a slip it already raised in the same run (the bench keeps one Live instance) |
| 4.1 simple request under way | never (not woken) | 18.1 s worst | 3 s | recognition plus a 2.5 s acknowledgement delay; with a 10 s plan the result comes later still |
| 4.2 control words | 12/12 | 10/11 per model | all | undo after a correction: Parakeet and Whisper merged into "Hey Heliox, Hey Heliocs, undo that", so the undo went to the planner |
| Q fast-layer quality (canned stubs) | - | 9/9 per model | all | |
| Live Bench (text) | precision 1.00, recall 1.00 (personal policy, 27/27) | | keep | |
| Memory Bench | 11/11 | | 10/10 | |

Kinds of model: control words and local answers behaved the same under cloud, local and none. With no model a
plan request was answered with the "configure a provider" message in 9.5 s (the acknowledgement plays first).

Derived conditions on the owner's voice (WER / wake found): Bluetooth call mode 35% / 4 of 9, laptop fan 27% /
6 of 9, another voice 10 dB under 112% / 2 of 9 (the background voice's words were transcribed instead),
12 dB quieter 57% / 3 of 9, 10% faster 33% / 6 of 9.

## Soak, 20 minutes (12 November)

`soak --minutes 20` ran 24 minutes of wall time (16 activities: requests, barge-ins, control words, two five-minute
not-addressed blocks), sampling every 10 s. Resident memory of the bench process went from 1130 MB after warm-up
to 1019 MB (peak 1188 MB, no upward trend); the voice worker from 316 MB to 130 MB. Handles 701 to 710, threads
106 to 105, asyncio tasks 10 to 7. The voice worker started once and never restarted. The audio feed fell at most
2.8 s behind real time. Missed: the two Bluetooth-headset launch-call sentences, which are never woken (see 2.1b).
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
