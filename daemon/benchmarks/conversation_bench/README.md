# Conversation Bench (code)

The full description, how to run it on Windows, macOS and Linux, and the latest baseline are in
`docs/CONVERSATION_BENCH.md`. This folder:

| file | what it is | imports |
|---|---|---|
| `scenarios/*.json` | the scenarios: text, state, expectations, per-model overrides; never audio | - |
| `scenarios.py` | loads and checks scenario files | stdlib |
| `audio.py` | audio folder and `path#start-end` references (pure path logic, both rule sets) | stdlib |
| `wavio.py` | WAV read and write without OS audio libraries | numpy, scipy (core deps) |
| `conditions.py` | derived listening conditions (headset call mode, noise, other voices, level, speed) | numpy, scipy |
| `models.py` | kinds of model (cloud, local, none) as stubbed delays, and their pass conditions | stdlib |
| `timeline.py` | the event timeline and what is measured from it | stdlib |
| `scoring.py` | WER, tolerance matching, scenario scoring, the scorecard | stdlib |
| `rubric.py` | the plain-speech rubric and the opt-in model-judge prompt | stdlib |
| `targets.py` | scorecard lines and targets from the program brief | stdlib |
| `corpus.py` | the hour of not-addressed speech, as text | stdlib |
| `quality.py` | fast-layer quality cases (pending until `pilot.conversation.bench_reply` exists) | stdlib |
| `platform_scan.py` | which spoken-path code branches by operating system | stdlib |
| `driver.py` | drives the real listener and dispatch with recorded audio | the daemon, speech models |
| `synth.py` | renders synthetic speech with local Kokoro on the CPU | kokoro |
| `soak.py` | the long mixed run and its resource sampling | psutil (optional) |
| `report.py`, `__main__.py` | printing and the command line | stdlib |

Adding a scenario: put the audio in the private audio folder, add an entry to the right JSON file with the
sentence actually said as `text`, and run `python -m benchmarks.conversation_bench list` to check its audio is
found. Mark a recording whose words are not known for sure with `"reference_confirmed": false`; it is run but
kept out of the word error rate.
