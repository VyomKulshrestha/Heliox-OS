# Reliability: what happens when something fails

This page lists the faults Heliox handles, what it does about each, what you see or hear, how it recovers, and how
that is tested. Every row names its test; a behaviour without a test is listed under [Not covered](#not-covered),
not in the table. Tests run on CI on Windows, macOS and Linux unless the row says otherwise.

How each fault is tested:

- **Real:** done for real on a machine, as described.
- **Drill:** the real daemon objects with the failure forced from outside, such as a killed process, a database
  that refuses writes, or a provider stub that fails.
- **Fakes:** unit tests with stand-ins for the GPU, the provider or the clock.

The test files are in `daemon/tests/` unless they end in `.test.ts`, which are in `tauri-app/ui/src/`.

## The daemon itself

| Fault | What Heliox does | What you see or hear | How it recovers | Tested |
|---|---|---|---|---|
| The daemon process exits or is killed while the app is open | The desktop app, which started it, starts it again after 2 s, doubling to 60 s between attempts. A daemon you started yourself from a terminal is never touched. | One line in the conversation: "Heliox's background service stopped. Starting it again in 2 s." The window reconnects by itself. | The back-off resets after five quiet minutes. | Fakes: `supervisor.rs` unit tests (back-off, notice wording); `session.supervisor.test.ts`. The restart itself has not been run on a real crash. |
| The daemon stops answering (its event loop is stuck) | After a 300 s start-up grace, a minute without an answer to its ping counts as a hang: the app kills it and its direct children, then starts it again. | "Heliox's background service stopped answering, so it is being restarted." | As above. | Fakes: `supervisor.rs` (`a_minute_without_an_answer_after_start_up_is_a_hang`, `silence_during_start_up_is_not_a_hang`). |
| The computer sleeps and the clock jumps | The hang rule counts its own 5 s checks, not clock time, so a jump never looks like a minute of silence. | Nothing. | Nothing to recover. | Fakes: `supervisor.rs` (the rule only ever sees fixed check intervals). Not tested on a real sleep. |
| Start-up is slow (a loaded machine, a first run) | The daemon listens first. `ping`, `health`, `ready` and `startup_status` answer at once; anything else waits until start-up has finished and is then answered as usual. | "Starting: opening Heliox's records" (then the next stage) instead of "Reconnecting...". | Becomes "Online" when ready. If start-up fails, waiting requests get the reason in plain words. | Drill: `test_startup_readiness.py`; `ConnectionBadge.test.ts`. |
| A task was running when the daemon died | Nothing starts again by itself. Steps that may or may not have happened are marked uncertain and never repeated. | A card: "Heliox stopped while it was doing X. 1 step was done, 2 were left." with Continue (or "Plan it again") and Leave it. Voice can say "leave it" but never "continue". | Only your Continue button restarts it. Leave it writes a line to the activity log. A reconnect to the same, still-running daemon carries on as before. | Drill: `test_interrupted_task_offer.py`, `test_resume_partial_approval.py`; `session.interrupted.test.ts`. |
| An approval was open when the connection dropped or the daemon stopped | Approvals fail closed: the dialog closes and nothing in it is approved. | "Lost the connection to Heliox, so the approval dialog was closed. Nothing in it was approved." | A task still waiting after the reconnect shows its approval again; it must be approved again. | Drill: `test_pending_resolution_lock.py`, `test_confirmation_unanswered.py`; `session.confirm.test.ts`. |
| The daemon is stopped while things are in flight | Speech stops first, then the conversation turn, then voice turns, then the running task, then the subsystems; requests still open are cancelled before any store closes. Each step waits at most 10 s and the whole stop at most 30 s; anything slower is logged and left behind. | Speech stops at once. | Not needed: stop() always completes. | Fakes: `test_daemon_stop_releases_workers.py`, `test_server_background_lifecycle.py`. |
| Two hours of use | No list or dictionary reachable from the server grew without bound. A thread blocked on the voice worker, which kept the process alive 300 s after stop, was fixed. | Nothing. | Not needed. | Drill: a soak script (two simulated hours at ten times the density of real use), run by hand. It is not part of CI. |

## Local models and the graphics card

| Fault | What Heliox does | What you see or hear | How it recovers | Tested |
|---|---|---|---|---|
| A CUDA fault, out-of-memory error or kill in Gnomon's process | Gnomon runs in its own worker process, so the daemon stays up. The review in progress counts as not done, but an approval it had already asked for still stands. | The approval dialog says "Not loaded, did not review this plan. Gnomon's process stopped. Restarting it in 5 s." | The worker restarts after 5 s, doubling to 5 minutes. | Drill (Windows): the worker's interpreter was killed in the middle of a review. Fakes: `test_gnomon_worker.py` (crash, hang, garbage, driver error). |
| Repeated GPU faults (3 within 10 minutes) | The graphics card is not used again until Heliox restarts. | Settings, Models on this computer: "Stopped using the graphics card for this session after 3 faults in 10 minutes." | Restart Heliox. | Fakes: `test_gpu_arbiter.py`, `test_gnomon_worker.py`. |
| The card is hot (83 °C or more by default), short of memory, or used by another Heliox process | Nothing new starts on the card. The voice uses the processor instead; Gnomon waits and checks again every minute. | Settings shows the card's temperature and why a model is waiting. | By itself, once the card is free. | Fakes: `test_gpu_arbiter.py`. Real (Windows): the card at 85 °C was refused. |
| No NVIDIA card (CPU only, an integrated GPU, or a Mac) | Gnomon is reported as not supported, without starting a worker to find out. The voice, recognisers and screen reading run on the processor or use the system's own engines. | "Not on this computer" in grey, with the reason. It is not shown as an error. | Not applicable. | Fakes: `test_gpu_arbiter.py`, `test_gnomon_worker.py`, `test_model_health.py`; `LocalModelsPanel.test.ts`. |
| Less than 2 GB of memory free for the local voice | The local voice is not started. | Heliox speaks with the system voice. | The local voice is tried again after 120 s. | Fakes: `test_local_voice_lifecycle.py`. |
| The local voice worker crashes or hangs | It is killed. If it was on the graphics card, the crash counts as a GPU fault. | The system voice speaks the sentence. | Tried again after 120 s. | Fakes: `test_local_voice_lifecycle.py`, `test_model_health.py`. |
| The daemon itself opening CUDA (OCR, embeddings, the hardware check) | Embeddings and Chroma always run on the processor. Screen reading uses the card only when the GPU arbiter allows it, and repeats the read on the processor after a CUDA error. The hardware check reads `nvidia-smi` and never opens CUDA. | Nothing. | The read is repeated on the processor. | Fakes: `test_no_cuda_in_daemon.py`, `test_vision_ocr_resources.py`. |

## Network and model providers

| Fault | What Heliox does | What you see or hear | How it recovers | Tested |
|---|---|---|---|---|
| No connection, or a DNS failure | The provider is not asked again for 30 s; things Heliox handles itself (reminders, timers, to-dos, undo) keep working. | "I can't reach the language model right now, so I can only do what I handle myself, like reminders, to-dos, timers and undo." Then a shorter line on later turns. | By itself, after the hold. | Drill: provider stub, run by hand. Fakes: `test_conversation_models.py` (`test_after_a_connection_failure_the_next_turn_falls_back_at_once`). |
| A cloud provider that accepts the request but never answers | Given up on after 4.5 s and held for 30 s. A local model gets 90 s and says once that it is loading. | "One moment." at 2 s, then a fallback sentence. | By itself. | Fakes: `test_conversation_layer.py` (`test_a_provider_that_never_answers_is_given_up_on_and_held`). |
| The connection drops in the middle of a reply | The reply stops where the connection broke. | "I lost the connection there." | The next turn tries again. | Fakes: `test_conversation_layer.py` (`test_a_reply_cut_by_the_network_says_so`). |
| Rate limit (429) | Held for the time the provider's Retry-After header gives, or 20 s without one. | "The language model is busy right now. Try me again in a moment." | By itself. | Fakes: `test_conversation_models.py`. |
| The provider account is out of credit (Anthropic 400, OpenAI 429 `insufficient_quota`, OpenRouter 402) | Held for 10 minutes. This is kept separate from Heliox's own monthly budget. | "Your <provider> account is out of credit. Add credit there, or choose another model in Settings, under Model." | By itself once credit is added. | Fakes: `test_conversation_models.py`, `test_conversation_layer.py`, `test_provider_error_safety.py`. |
| The neural sidecar loses the daemon (a slow call, a restart) | It reconnects after 1 s, doubling to 60 s. The daemon disarmed when the sidecar went away, so nothing is armed again by itself. | The neural status says "The neural sidecar lost its link at HH:MM:SS and reconnects by itself. Arm it again once it is back." | Reconnects by itself; you arm it again. | Fakes: `test_neurod_bridge.py`, `test_neural_sidecar_status.py`. Drill: `test_ipc_integration.py`, run 30 times next to CPU load. |
| A neural preview lapses before its commit | The preview is dropped and the session is armed again; nothing is committed. | The next preview appears. | By itself. | Fakes: `test_neural_gate.py`. |

## Disk and storage

| Fault | What Heliox does | What you see or hear | How it recovers | Tested |
|---|---|---|---|---|
| The disk is full while Heliox runs (a reminder, to-do, note, profile or settings cannot be saved) | Says so once, keeps answering, and does not pass the request on as if it had not been heard. | "The disk is full, so Heliox could not save that. Free some space; Heliox keeps working meanwhile." | The first write that works says "Heliox can save again." | Drill: every write failing, run by hand. Fakes: `test_storage_health.py`. |
| SQLite cannot grow (a full disk under its write-ahead log) | As above. Any transaction left open by the failed write is rolled back, so later writes are not blocked. | As above. | As above. | Real SQLite error (its page limit): `test_storage_health.py`. |
| The data folder or database turns read-only while running | As above. | "Heliox's data folder cannot be written to, so it could not save that. Check the folder's permissions or that the drive is not read-only; Heliox keeps working meanwhile." | As above. | Real SQLite read-only error: `test_storage_health.py`. |
| The data folder cannot be written at start-up | The daemon still listens and answers, says why, and checks again every 30 s. | The sentence above, then "Heliox can save again and is fully started." | Starts fully by itself once the folder can be written. | Fakes: `test_storage_health.py`. |
| A model download cannot be written | The reason is put in plain words. | Settings, Models on this computer: "The disk is full, so Heliox could not save the model download..." | When you try again after freeing space. | Fakes: `test_storage_health.py` (`plain_reason`). |

## Time and reminders

| Fault | What Heliox does | What you see or hear | How it recovers | Tested |
|---|---|---|---|---|
| Reminders came due while the computer slept or Heliox was off | None is ever dropped. They go out once, together; one more than a week old is also written to the activity log. | One card listing each reminder with when it was due, with Done and Remind me again. At most one spoken sentence, e.g. "While this computer was asleep, two reminders came due: call mum and stretch." | Not applicable; no reminder fires twice. | Fakes (injected clock): `test_reminders.py`, `test_repeating_and_app_reminders.py`, `test_missed_reminders.py`; `session.reminders.test.ts`. |

## Microphone

| Fault | What Heliox does | What you see or hear | How it recovers | Tested |
|---|---|---|---|---|
| The capture stream fails or stops delivering (a call app takes the microphone, a device is unplugged) | The stream is reopened. | The problem is reported once, and the recovery once. | Reopens by itself, backing off while it brings no sound. | Fakes: `test_audio_health.py`. |

## By system

- **Windows:** a virtual environment's `python.exe` is a launcher that starts the real interpreter as its child. The
  app and the Gnomon supervisor therefore kill the process together with its children, so no daemon or worker is
  left holding the port or the graphics card.
- **macOS and Linux:** the virtual environment's `python3` is the interpreter itself. Worker processes end when the
  daemon dies (their pipe closes, and a parent watch notices). A folder made read-only keeps working for files that
  are already open, so the first failure shows when SQLite creates a new file.
- **Apple Silicon:** memory is shared with the graphics processor. The question is free system memory, and Gnomon is
  not supported (no CUDA).

## Not covered

- A real GPU reset under load is not tested: the card that would show it is failing, and stress runs are not
  allowed on it. Gnomon's real model was never loaded through the worker during this work.
- A truly full disk, or a drive remounted read-only, is not tested. The tests use SQLite's own errors and fakes.
- The app restarting a daemon that really crashed has not been run on any system. The Rust is type-checked and linted
  on Windows only; CI compiles it on its runners.
- A real sleep and wake has not been tested.
