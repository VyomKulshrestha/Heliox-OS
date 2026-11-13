# First run

What happens the first time someone opens Heliox on Windows, macOS or Linux, and why.

Each testable sentence says how it is known, in the words of the legend at the top of [PLATFORM_SUPPORT.md](PLATFORM_SUPPORT.md): Run for real (with the CI job and check, or the owner's Windows machine and the date), Fakes only, or Not tested. "CI on 3793a0d" is the CI run on that commit (12 November); "CI on 0a8edad" is the run on that commit (run 43196949417).

## What was wrong

The desktop app used to set itself up in the background as soon as it opened. It ran `python3 -m venv` (`python` on Windows) and then `pip install pilot-daemon[all]`, which is several gigabytes: PyTorch, speech models' libraries, OCR, Chromium.

- **Nothing was asked.** The download happened without the person choosing it, which breaks the rule that every optional download is offered with its size and source.
- **A missing Python was not handled:**
  - On a stock Mac, `python3` is Apple's stub: it opens the "install command line developer tools" dialog, and the Python it installs is 3.9, too old.
  - On Debian and Ubuntu without `python3-venv`, the environment could not be created.
  - On Windows without Python, the `python` command is the Microsoft Store stub.
- **Failures were invisible.** Every failure went only to the app's stderr while the window sat on "Connecting".

## What happens now

1. **Nothing is installed when the app opens.** If Heliox's environment is missing and no daemon answers, the app shows one screen: "Set up Heliox" (`FirstRunInstall.svelte`). Fakes only: the screen in vitest; the built app has not been run through it.
2. **The screen says what setting up Python will take on this machine.** The app learns this from the setup script's `probe` mode, which looks and installs nothing. The probe: Run for real (CI `first-run` on all three systems; the owner's Windows machine, 12 November).
   - **A suitable Python is already there.** Suitable means a version inside the tested range in `tauri-app/src-tauri/setup/python-versions.json` (3.11 to 3.13 on 12 November 2026), with `venv` and `ensurepip`. Heliox uses it to create its own environment and changes nothing about it.
     - The upper bound matters as much as the lower one. A newer Python can lack prebuilt wheels for the core's dependencies: numpy 2.2.6, scipy 1.15 and river 0.23 stop at 3.13. pip then tries to compile them, which failed on CI's Mac with Homebrew's Python 3.15 ("Unknown compiler(s): gfortran").
     - The same file feeds `setup.rs`, which passes the range to the scripts, and the first-run CI job. `daemon/tests/test_python_versions.py` checks that the CI test matrix stays inside the range and that `requires-python` agrees with it.
     - On macOS, `/usr/bin/python3` is never run. Run for real (CI on e51aead: `probe-macos/never-runs-usr-bin-python3`, the probe traced with PATH limited to /usr/bin and /bin; it chose Homebrew's Python 3.13). Every hosted Mac has Homebrew's Python, so "download" itself cannot be shown there.
     - On Windows, the Store stub is skipped. Run for real on the owner's Windows machine (12 November): with only a file in a `Microsoft\WindowsApps` folder named `python.exe` on PATH, the probe chose "download" (`probe-windows/only-the-store-stub-means-download`); and in CI on e51aead on windows-latest.
   - **If the core still cannot be installed on that Python, Heliox falls back once.** It says so ("trying again with Heliox's own Python"), removes the environment, downloads its private Python and tries again. Only then does it report a failure. Run for real on the owner's Windows machine (12 November); on macOS and Linux: Not tested.
   - **No suitable Python.** Heliox downloads a private one. Before that, the screen shows the size and the source:
     - uv 0.13.5 from `github.com/astral-sh/uv`. The screen shows its exact size from the probe. Its SHA-256 for each platform is fixed in the script and checked before uv is unpacked. Run for real (CI `first-run`, private Python, on all three systems).
     - Python 3.12 from python-build-standalone, downloaded by `uv python install`, which checks its own checksums.
     - Both stay in Heliox's folder: `~/.config/heliox-os/tools`, `python` and `env`. They take about 110 to 160 MB on disk. Run for real: 112 MB on Windows, 119 MB on macOS and 157 MB on Linux (`firstRunSizes.json`, measured by run 43162118953).
     - The system Python and the PATH are not touched. Not tested: no check compares them before and after.
3. **The person chooses the optional groups.** The core is always installed. The screen shows its download and, where CI has measured it, its size on disk once installed: about 450 MB on Windows, 430 MB on macOS and 530 MB on Linux. Run for real: 450, 426 and 529 MB in `firstRunSizes.json`, measured by run 43162118953. The optional groups are:

   | Group | What it gives | Preselected |
   |---|---|---|
   | macOS readers / Linux readers (`macos` / `linux` extra) | Live reads the field you type in, the app in front, the screen text, window actions | yes, on macOS and Linux |
   | Voice (`voice`) | speech in and out, locally, with Whisper base; the speech models below are offered with it | yes, when the computer has a microphone |
   | Voice on the graphics card (`voice-gpu` extra; Windows and Linux) | Whisper's CUDA 12 libraries (NVIDIA cuBLAS and cuDNN from PyPI) and, on Windows, the same PyTorch version's CUDA 12.6 build from download.pytorch.org: speech recognition and the voice on the NVIDIA card. Offered only when the probe finds one (`@@gpu nvidia <name>` from `nvidia-smi`; never on macOS). About 3.7 GB on Windows (cuBLAS 553 MB, cuDNN 696 MB, PyTorch 2.5 GB, from the package indexes' own sizes); on Linux about nothing, since PyPI's PyTorch already brings both libraries. Fakes only: unit tests; CI shows the runners, which have no card, choose the processor and say why (`first-run/gpu-probe`, `first-run/gpu-status`) | no |
   | Speech models (inside Voice: Parakeet, live captions, Smart Turn) | from the daemon's `speech_models.json`, each with its size, licence, source and language; the daemon downloads them, checks their checksums and can stop and resume | none, not even Parakeet |
   | Screen reading extras (`vision`) | EasyOCR, OpenCV, MediaPipe, where the system's own text recognition is not enough | no |
   | Browser automation (`browser` and Chromium) | Heliox works in web pages in its own Chromium | no |
   | Built-in local model runner (`llamacpp`) | GGUF models inside Heliox; builds llama.cpp, so it needs a C and C++ compiler; not needed with Ollama or a cloud provider | no |
   | Everything else | documents, network discovery, remote machines, plugin sandbox, typing supervision, memory search | no |

   Installed for real in CI: the core, and the macOS and Linux readers (CI `first-run`). The voice, screen, browser, local and everything-else groups: Not tested by a real install, and their sizes are estimates. The speech models' downloads are the speech workstream's; this page cites no run of them.

   Nothing is downloaded until the person presses "Set up Heliox". The total is shown next to the button. Fakes only (vitest).

   **Where the sizes come from.** No number is written into the screen's text:
   - The install groups' sizes are in `tauri-app/ui/src/lib/firstRunSizes.json`, per system. Each entry has a download size and, when known, a size on disk, and an entry measured by CI names the run.
   - The first-run job measures them. It points pip's cache at an empty folder, and after each step the scripts print the cache's size (the download) and Heliox's folder's size (the disk).
   - `daemon/tests/platform_real/first_run_sizes.py update` writes the job's log into the file.
   - The speech models' sizes come from the daemon's manifest.
   - Every number on screen says "about".
   - The measuring itself: Run for real (CI `first-run`, `FIRST_RUN_SIZES` lines).
4. **Each step shows its progress, and a failure shows its reason, what to do, and Retry.** Progress is the step and the last line of pip's output. "Show details" gives the last 40 lines. The screen: Fakes only (vitest); the lines it reads: Run for real (the scripts' `@@` lines in CI `first-run`).
   - **The core decides whether setup succeeded.** When it is in, the app starts the daemon itself and its supervisor watches it from then on; no restart is needed. Not tested in the app; the start through `launch_daemon` after a repair: Run for real on all three systems (CI on e51aead, `real-removal`: `real_repair_through_the_apps_code` passed).
   - **A failed optional group never blocks the app.** The person can retry it on the spot or later, from Settings > Optional parts. The script carrying on past a failed group: Run for real on the owner's Windows machine (12 November); the retry: Fakes only (vitest).
5. **Once the daemon answers,** the screen asks it for each speech model the person ticked (`speech_model_download`). It keeps the ticks until each has been accepted, across a restart, because the daemon allows one download at a time. Progress then shows in Settings > Voice. The screen also starts the first conversation if the profile is empty (`getting_to_know`: status, then start). Otherwise the daemon offers that conversation later as a card. Fakes only (vitest).
6. **Settings > Optional parts** offers every group again, each with its size and source. A group added there is used after Heliox restarts. Fakes only (vitest).

The app can only ask for groups from a fixed list: `group_extras` in `src/setup.rs`. Each version is pinned to the app's own release (`pilot-daemon==<app version>`). `HELIOX_DAEMON_SOURCE` points the install at a local checkout, for development and CI.

## Updating an older install

A person who installed an earlier version has an environment with an older daemon. The old desktop app installed `pilot-daemon[all]==<its version>`. When the app's own version is newer than the daemon's, the app offers an update.

1. **Detection costs nothing.** At start, the app reads the daemon's version from the `pilot_daemon-<version>.dist-info` folder in the environment. If that version is older than the app's, the "Update Heliox" screen says which version is installed and which this app is. A newer daemon is never downgraded. "Not now" keeps the installed version for this run, and the offer comes back at the next start. Fakes only (Rust unit tests and vitest).
2. **The plan changes nothing.** "See what changes" runs `update.py plan`, which uses `pip install --dry-run --report` with `--upgrade-strategy only-if-needed`. The screen lists every package that will change (from → to, or new) and the download size the package index reports. Run for real (the owner's Windows machine, 12 November; CI `update-path` on ff0a8d1, all three systems: "update installed 0.15.0 target 0.16.0 groups platform", then "Updated to 0.16.0, keeping platform").
3. **The person's parts are kept.** The groups come from `~/.config/heliox-os/setup.json`, which first-run setup and every update now write. For an install made before that file existed, they are inferred from what is installed: `faster-whisper` means voice, `easyocr` means screen reading, and so on. So an old `[all]` install keeps every group. Fakes only (`test_update_path.py`).
4. **Unchanged packages are not touched.** The update runs in place with `only-if-needed`, so only the packages in the plan are replaced.
5. **The update is all or nothing.** Before pip runs, `update.py apply` keeps a copy of the environment made of hard links, so no data is copied. pip replaces a file by removing it and writing a new one, so the copy keeps the old files intact. After installing, it checks that `pilot.server` imports and reports the new version.
   - On success it deletes the copy and records the version and groups.
   - On any failure it puts the copy back, so the environment is exactly as it was, and says so with a Retry. Retry plans again, and "Keep the current version" closes the screen. The rollback: Run for real (the owner's Windows machine, 12 November, and CI `update-path` on ff0a8d1, all three systems: "The update did not install cleanly, so Heliox put the previous version back."); the screen: Fakes only (vitest).
6. **The daemon is stopped for the update and started again after it.** On Windows its files are locked while it runs. The supervisor is paused for the duration (`DAEMON_PAUSED` in `setup.rs`) so it does not restart the daemon into a half-changed environment. If a daemon the app did not start is listening, the update refuses and asks the person to quit it. Not tested: the app has not driven an update.

`update.py` runs with the environment's base Python, named in `env/pyvenv.cfg`, never the environment's own interpreter. An environment whose Python no longer starts, for example after Homebrew upgraded the Python it was made from, is sent to Repair, which rebuilds it.

**How it is tested:**
- **Real run on Windows (12 November).** A venv with a small fake `pilot-daemon` 0.1.0 was updated to a fake 99.0.0 that installs but has no `pilot.server`. The update was rolled back: the environment reported 0.1.0 afterwards and no copy was left behind. A clean update to 0.2.0 then succeeded and recorded the groups file.
- **Unit tests.** `daemon/tests/test_update_path.py` covers group inference, the requirement, which packages count as changed, version order, and the hard-link copy surviving a pip-style file replacement and restoring exactly. The Rust tests cover version detection, `pyvenv.cfg` and the plan lines; vitest covers the update screen's states.
- **CI, job `update-path`** (non-blocking; ubuntu, macOS and Windows). It installs the released `pilot-daemon` 0.15.0 as an older app did, then does three things:
  - It applies a fake 99.0.0 that breaks the daemon, and checks the rollback left 0.15.0 with an identical package list.
  - It plans and applies the update to this checkout.
  - It checks that the updated daemon answers an authenticated ping. Run for real only from b10b882 on: before it, the check passed on the daemon's first message, a start-up notification. In CI on 0a8edad the macOS daemon did not answer within 240 s (the robustness workstream's).
- **Not tested:** the desktop app driving the update (Rust compiled and unit-tested here, not run as an app).

## Repair

At start, the app checks that the environment's own Python starts and sees the daemon package. This is cheap: it does not import the daemon. Run for real in the scripts (CI `first-run`, the repair step); through the app's `startup_check`: Run for real on all three systems (CI on e51aead, `real-removal`).

If the environment is broken, the first-run screen shows "Repair Heliox" and says which parts it will build again. Typical causes: Homebrew upgraded the Python it was made from, the private Python was deleted, or part of the environment was removed. The screen: Fakes only (vitest).

Repair runs the first-run script with the recorded groups. The script then:
- sets the broken environment aside as `env.broken`;
- builds a new one, with the same Python rules as first run;
- installs the core and the groups;
- deletes `env.broken` only once the core is in.

Settings and data are not touched. Run for real on all three systems around the app-code repair (CI on e51aead, `real-removal`: `repair/settings-kept`, `repair/data-kept`, `repair/every-settings-and-data-file-unchanged`, 21 files on macOS and ubuntu, 29 on Windows).

**How it is tested:**
- **Real run on Windows (12 November).** A venv whose `python.exe` was deleted was detected ("no longer starts; building it again"), rebuilt from the system Python and given its core.
- **CI, job `first-run`.** On all three systems, with the runner's Python and with a private one, the job deletes the environment's interpreter after setup. It then reruns the script and checks three things: that the repair was detected, that `env.broken` is gone, and that the rebuilt daemon answers. Run for real: in CI on 3793a0d `first-run/daemon-answers` and `first-run/daemon-ready` passed after the repair on all six combinations; in CI on 0a8edad on ubuntu and Windows (the macOS daemon did not answer in that run, the robustness workstream's).
- **CI, job `real-removal`, through the app's own Rust code.** In CI on 0a8edad the first run and the repair passed on ubuntu and Windows; the job then failed in its credential part (below). On all three systems, a headless run of the same functions the Repair button uses (`startup_check`, `run_command` with the recorded groups, `read_lines` into the screen's state, `write_groups`, then `launch_daemon`), after the job deleted the environment's interpreter. It checks that the start-up check finds `needs_repair`, that the repair finishes with the core in, that `env.broken` is gone, that the groups are kept, and that the daemon starts from the result the way the app starts it.
- **What no CI run proves:** the window itself (the "Repair Heliox" screen, its button, the `setup-progress` events reaching it) and `setup_repair`'s own two lines, which pause the supervisor and stop the daemon the app holds. A headless runner has no window to press; vitest covers the screen, and the Rust above covers everything the button starts.
- **Unit tests.** The start-up decision (repair before update, nothing to do, update) has Rust unit tests; the screen has vitest.

## Removing Heliox's files

Settings > "Remove Heliox's files" lists everything Heliox put on this computer, with sizes, before anything is deleted. The list is computed by `src/uninstall.rs` from the home folder and the system's app-data folders. The UI cannot add a path to it. Heliox's settings, data and logs folders follow `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `XDG_STATE_HOME` as the daemon does (`pilot.config`), on every system: an empty or relative value is ignored, as the XDG specification says, and outside the home folder removal acts only inside the variable's own `heliox-os` folder, never on the folder the variable names or anything beside it (Rust tests, on a real file system with the folders outside the home). Run for real on a real file system in the Rust tests (below).

Every folder in the list carries a name that is Heliox's alone, and the code says why beside each one: `heliox-os`, `.heliox`, `heliox` under `.cache`, `com.helioxos.app`. When the home folder is not known, nothing is listed and nothing is removed. As a second lock, every path must be absolute and at least two folders below the root. Fakes only (Rust unit tests).

| Where | What | Kind |
|---|---|---|
| `~/.config/heliox-os/env`, `python`, `tools`, `uv-cache`, `setup` | the environment, private Python, uv and setup scripts; always under the home folder, whatever `XDG_CONFIG_HOME` says | program files |
| `$XDG_CONFIG_HOME/heliox-os`, by default `~/.config/heliox-os` (the rest) | settings, keys, restrictions, shortcut | the person's data |
| `~/.config/heliox-os/setup.json`, when `XDG_CONFIG_HOME` points elsewhere | the parts setup installed | program files |
| `~/.config/heliox-os` (the rest), when `XDG_CONFIG_HOME` points elsewhere | settings from before the variable was set, which the daemon no longer reads; listed with their size, kept with "keep my data"; the folder goes once it is empty | the person's data |
| `$XDG_DATA_HOME/heliox-os/models`, by default under `~/.local/share` | downloaded speech models | program files |
| `$XDG_DATA_HOME/heliox-os`, by default `~/.local/share/heliox-os` (the rest) | memory, timeline, receipts, undo copies | the person's data |
| `$XDG_STATE_HOME/heliox-os`, by default `~/.local/state/heliox-os` | logs | the app's working files |
| `~/.cache/heliox`, `~/.heliox` | persona, voice calibration, plugins, screenshots, quarantined files | the person's data |
| `~/.pilot/usage_patterns.json`, `~/.pilot/failure_memory.json` | usage patterns and failure memory: only these two files, because another program could use a folder called `.pilot`; the folder goes only if that leaves it empty | the person's data |
| The system credential store, under four services named in `pilot.system.credential_cleanup` | API keys, SSH keys and passphrases, paired devices, the receipt signing key, study and recording keys | the person's data |
| Runtime folder: `%LOCALAPPDATA%\heliox-os`, `~/Library/Caches/heliox-os` | runtime files | the app's working files |
| The app's web view data: `%LOCALAPPDATA%\com.helioxos.app`; `~/Library/Application Support`, `Caches` and `WebKit` under `com.helioxos.app`; `~/.local/share` and `~/.cache` under `com.helioxos.app` | interface settings stored in the browser view | the app's working files |

The kinds: **program files** are what setup installs or downloads and Heliox runs; **the app's working files** are what the app writes while it runs (logs, runtime files, the web view's storage and cache), safe to delete; **the person's data** is theirs. "Remove the app's files" removes the first two kinds. Until 13 November 2026 this table called the first two kinds together "app", and the linux-package check and the package's post-remove message needed to tell them apart.

**The person chooses** between "remove the app's files, keep my data" (the default) and "remove everything". They see the total size, and they tick that they understand the files will be deleted. Only then does the button work. Fakes only (vitest).

**Before deleting**, the daemon is stopped and the supervisor paused. Not tested: the app has not driven a removal.

**Saved keys.** With "remove everything", the saved keys go before any file is deleted:
- They are removed from the credential store by the daemon's own code, `python -m pilot.system.credential_cleanup remove`, using the environment that still exists, with a 30 s limit. Fakes only (a fake keyring); against the real stores: Run for real on all three systems (CI on e51aead; it had failed in CI on 0a8edad for three reasons, see How it is tested).
- The daemon finds the entries by service where the store allows it: Windows' credential list, macOS's `security` tool, the Linux Secret Service. It also tries the accounts it knows by name. Windows' search: Run for real on the owner's machine (12 November, names listed, nothing removed); macOS and Linux: Run for real (CI on e51aead).
- Anything it cannot remove is listed by service and account, with how to remove it on that system.
- If the environment cannot run, or the step fails or runs out of time, the screen lists the four services by name instead. That list is `setup/credential_services.json`, which a test keeps equal to the daemon's own list. Fakes only (Rust unit tests); the case of an environment already gone: Run for real (CI on e51aead).

With "keep my data", the keys stay, and the screen says which services hold them. Run for real against the real stores (CI on e51aead: `credentials/kept-with-the-data` on all three systems).

**After the removal**, the window shows what was removed and what failed. On Windows, the app's own web view folder is expected to fail while the app runs (Not tested: no run has seen it), and the screen says it goes when the app is uninstalled. It also shows which saved keys remain and the manual steps, with a Quit button instead of a reconnecting screen. Fakes only (vitest).

**Deletion never follows a link out of Heliox's folders.** A symbolic link or junction inside them is removed as a link, and what it points to stays. Rust's `remove_dir_all` does not follow links (Rust 1.58 and later). Run for real on all three file systems (the Rust tests below).

**Heliox leaves these to the person, and the screen says so:**
- the app itself (the system's own uninstall);
- shared caches: pip's cache, and the Hugging Face cache that holds the local voice and Whisper models;
- on macOS, the privacy grants: System Settings, or `tccutil reset All com.helioxos.app`;
- on Wayland, the global shortcut the desktop stores.

**Uninstalling the app itself** (the system's own uninstall, not Settings > "Remove Heliox's files"). No uninstaller deletes anything inside a home folder. Everything in the table above stays. Its program files are the environment folder `~/.config/heliox-os` (`env`, `python`, `tools`, `uv-cache` and `setup`, several hundred MB after a first run) and any downloaded speech models; its working files are the logs and the web view's storage; the rest is the person's data.
- **Linux, .deb and .rpm.** Before removal, the package ends the app and the daemon it started, for everyone who has Heliox open (`tauri-app/src-tauri/linux/pre-remove.sh`). On an upgrade both keep running. After removal it prints, on the terminal the person ran apt or dnf in, once and never on an upgrade, each home folder's environment folder, its rough size and the one `rm -rf` command that deletes those program files; that the settings and data beside them are kept; and, in one more sentence, the folders of the app's working files at their default places (`~/.local/state/heliox-os`, `~/.local/share/com.helioxos.app`, `~/.cache/com.helioxos.app`) as safe to delete too (`linux/post-remove.sh`). It deletes nothing. Not tested yet: the linux-package workflow checks both on Ubuntu 24.04 (the `deb/remove-*` lines) and has not run. On Fedora: Not tested.
- **Linux, AppImage.** Deleting the file removes the app. Nothing ends a running daemon, and nothing says what is left. Not tested.
- **Windows.** The installer's uninstaller removes the app. Tauri's uninstaller can delete the web view's data (`com.helioxos.app`), but not the Heliox folders in the table, and it does not name them. Not tested.
- **macOS.** Moving the app to the Bin removes only the app. The folders stay, and nothing says so. Not tested.

Heliox creates no login item, service or scheduled task. Not tested yet: the real-removal job's home listing would show a launch agent or a systemd user unit, and on Windows `removal/no-autostart-entry-of-heliox` compares scheduled tasks, the Run and RunOnce keys and the Startup folders before setup, after it and after removal: Run for real (CI on e51aead, nothing new after setup or after removal). macOS and Linux: the home listing passed, nothing of Heliox left.

**How it is tested:**
- **Rust tests on a real file system.** On Windows here, on Linux in the Rust job and on macOS in the `rust-macos` job, the tests build a home folder with an environment, settings and a quarantined file, plus a link pointing outside. Removing with "keep my data" deletes the environment and keeps the settings and the quarantined file. Removing everything deletes Heliox's folders. In both cases the file outside is untouched. Run for real: the Rust suite on Windows here, in the `rust` job on Linux, and in `rust-macos` (8 removal tests passed in CI on 3793a0d).
- **Credential cleanup with a fake keyring** (`daemon/tests/test_credential_cleanup.py`): entries are removed by service search and by known name, an entry the store refuses is listed with how to remove it, and a store that cannot be searched still has its known names removed. The search by service also ran for real on Windows here, listing names only.
- **Credential cleanup against real stores, CI job `real-removal`** (`daemon/tests/platform_real/credential_store_check.py`). In CI on 0a8edad it failed on all three systems: on macOS and Windows the store already held the daemon's own learner key (`heliox_learner_snapshot`, written at first run), which the check took for an entry it must not touch; on Windows an empty store's search raised error 1168. The check now records the daemon's entries on a runner and requires them gone too, the cleanup knows the learner key's old entry, and 1168 is an empty list: Run for real on all three systems in CI on e51aead (`credentials/cleanup-reports-each-entry-removed`, `credentials/system-tool-finds-none-left`). On each runner the check writes one test entry under each of the four services with `keyring`, then sees them with the system's own tool: `cmdkey /list` on Windows, `security find-generic-password` in the macOS login keychain, `secret-tool lookup` on Linux with a GNOME keyring unlocked for a `dbus-run-session` (`with_keyring.sh`). It runs the daemon's cleanup and checks with the same tool that none is left. Before writing, it searches the store for any Heliox entry and refuses to go on if one exists, so only entries the check created are ever touched.
- **A real removal, CI job `real-removal`** (`daemon/tests/platform_real/real_removal.sh`). In CI on 0a8edad the Linux home listing reported the checks' own state files, kept in the job's temporary folder inside the runner's home; they are now left out by their exact paths: Run for real on macOS and ubuntu (CI on e51aead, `removal/nothing-of-heliox-left-in-home` passed twice on each). On Windows the listing crashed in CI on e51aead, as the checkout is on D: and the home on C:; a path on another drive is now left out, and in CI on 105b573 `removal/nothing-of-heliox-left-in-home` passed on Windows too, both times: Run for real on all three systems. After a first run and a repair through the app's Rust code in the runner's real home folder, `uninstall::remove_heliox` runs through an ignored Rust test that runs only under GitHub Actions:
  - "everything" while the environment exists: the test keys go through the daemon's code, then the files;
  - setup again, then "keep my data": the environment goes, the data folders and the keys stay;
  - then "everything" with the environment already gone: the files go and the four services are listed for the person, as the screen would show;
  - each time, `home_snapshot.py` lists what the run left in the home folder (four levels down) against a listing taken before setup. Anything left with Heliox's names fails the job; anything else is listed as another program's (pip's cache, for example).
- **Rust tests** also cover no home folder, the safe-location lock, `.pilot` keeping another program's file, the credential report and the Windows web view note.
- **Vitest:** the screen's choice, confirmation, the after-removal screen, and the three replies: a bare ok, a refusal, no reply.
- **Not tested:** the window driving a removal (the confirmation, the button, the after-removal screen over a real removal), and the step in `heliox_files_remove` that pauses the supervisor and stops the daemon the app holds; in the CI job the daemon is stopped by the job first.

## Where the code is

- **Steps:** `tauri-app/src-tauri/setup/first-run.sh` (macOS, Linux; POSIX sh) and `first-run.ps1` (Windows PowerShell 5.1). They take the same inputs and print the same `@@` lines. The app embeds them, writes them to `~/.config/heliox-os/setup/` and runs them; CI runs the same files.
- **Runner and state:** `tauri-app/src-tauri/src/setup.rs`.
  - Commands: `setup_status` and `setup_run`.
  - Event: `setup-progress`.
  - Its line parser and phase logic have unit tests.
- **Screen:** `tauri-app/ui/src/lib/components/FirstRunInstall.svelte`.
  - The choices and every state live in the pure `tauri-app/ui/src/lib/firstRun.ts`.
  - Vitest: `FirstRunInstall.test.ts`.
  - The screen renders only inside the desktop app, so no visual baseline changes.
- **Settings:** `OptionalParts.svelte`.

## How it is tested

- **Real run on Windows (this machine, 12 November):**
  - `first-run.ps1 probe` found the installed Python 3.12.6.
  - `HELIOX_PYTHON=private` downloaded uv, checked its checksum, installed a private Python 3.12.15 and created the environment.
  - It installed a stand-in core package and one optional group, and reported a group that does not exist as failed (exit code 2) while the others carried on.
  - A second run skipped the existing environment.
- **CI, job `first-run`** (non-blocking; ubuntu-24.04, macos-latest and windows-latest, each with the runner's Python and with a private Python):
  - It runs the script in an empty home folder with the core from the checkout, plus the platform readers on macOS and Linux.
  - It starts the daemon it installed and checks that the daemon answers an authenticated `ping` (`daemon/tests/platform_real/first_run_rpc_check.py`). Run for real: `first-run/daemon-answers` and `first-run/daemon-ready` passed on all six combinations in CI on 3793a0d.
  - It prints the time taken and the size of each folder, which replaces the estimates above once measured.
- **Real run on Windows (12 November, second round):** a Python range that excludes every Python on the machine leads to the private download, and a core that cannot be installed on the system Python falls back to a private Python once before failing.
- **Not tested:**
  - The desktop app driving the scripts (Rust compiled only by CI's Rust job).
  - The real voice, screen, browser, local and "everything else" groups.
  - A Mac or Linux desktop with nothing installed: hosted runners always have some Python, so the private path is forced with `HELIOX_PYTHON=private`.

## What an unsigned build asks of the person

Signing costs money and accounts, so it is the owner's decision. Until then, this is what someone is expected to see. Not tested with Heliox's own build: the steps below come from Apple's and Microsoft's documented behaviour.

**macOS (15 Sequoia and later), with an app that is ad-hoc signed and not notarised:**

1. Opening it shows "“Heliox-OS” Not Opened. Apple could not verify “Heliox-OS” is free of malware…", with Done and Move to Trash. Since Sequoia, Control-click > Open no longer offers a way past it.
2. The person opens System Settings > Privacy & Security, scrolls to "“Heliox-OS” was blocked to protect your Mac", presses **Open Anyway**, and enters their password.
3. On the next launch they press **Open Anyway** once more.
4. If the download was modified or its signature broken, macOS says the app "is damaged and can't be opened". The way out is `xattr -dr com.apple.quarantine /Applications/Heliox-OS.app` in Terminal, which is not something to ask of most people.
5. An ad-hoc signature changes with every build. Microphone, Accessibility and Screen Recording grants can stop applying after an update, and the person turns them off and on again in System Settings.

**Windows 10 and 11, with an unsigned installer and app:**

1. Microsoft Defender SmartScreen shows "Windows protected your PC". The person presses **More info**, then **Run anyway**.
2. The installer's UAC prompt says "Publisher: Unknown" (an install for all users).
3. On Windows 11 with Smart App Control on, which is the default on some fresh installs, an unsigned app is blocked with no "run anyway". The only way through is turning Smart App Control off, which cannot be turned back on without resetting Windows.

**Linux:** no signing prompt. The .deb and .rpm declare their dependencies (Not tested against a built package). The AppImage needs `chmod +x` or the file manager's "Allow executing as program".

## What only a person at a real machine can show

No hosted runner can show these; each needs someone at a real computer:

1. The built app driving setup, an update, a repair and a removal in its own window: the screens, their buttons, the progress reaching them, and the daemon the app holds being stopped and the supervisor paused.
2. A Mac and a Windows PC with no Python at all: the probe choosing Heliox's own Python, and no developer-tools dialog on the Mac.
3. The unsigned-build dialogs with Heliox's own downloads: Gatekeeper on macOS 15 (and "Open Anyway" twice), SmartScreen and the "Publisher: Unknown" UAC prompt on Windows, and Smart App Control on a fresh Windows 11.
4. On Windows, removing Heliox's files while the app runs: whether its web view folder is held open.
5. The permission prompts, a real GNOME or KDE desktop and two audio devices: see the same list in [PLATFORM_SUPPORT.md](PLATFORM_SUPPORT.md).
