# First run

What happens the first time someone opens Heliox on Windows, macOS or Linux, and why.

## What was wrong

The desktop app used to set itself up in the background as soon as it opened. It ran `python3 -m venv` (`python` on Windows) and then `pip install pilot-daemon[all]`, which is several gigabytes: PyTorch, speech models' libraries, OCR, Chromium.

- **Nothing was asked.** The download happened without the person choosing it, which breaks the rule that every optional download is offered with its size and source.
- **A missing Python was not handled:**
  - On a stock Mac, `python3` is Apple's stub: it opens the "install command line developer tools" dialog, and the Python it installs is 3.9, too old.
  - On Debian and Ubuntu without `python3-venv`, the environment could not be created.
  - On Windows without Python, the `python` command is the Microsoft Store stub.
- **Failures were invisible.** Every failure went only to the app's stderr while the window sat on "Connecting".

## What happens now

1. **Nothing is installed when the app opens.** If Heliox's environment is missing and no daemon answers, the app shows one screen: "Set up Heliox" (`FirstRunInstall.svelte`).
2. **The screen says what setting up Python will take on this machine.** The app learns this from the setup script's `probe` mode, which looks and installs nothing.
   - **A suitable Python is already there.** Suitable means a version inside the tested range in `tauri-app/src-tauri/setup/python-versions.json` (3.11 to 3.13 on 12 November 2026), with `venv` and `ensurepip`. Heliox uses it to create its own environment and changes nothing about it.
     - The upper bound matters as much as the lower one. A newer Python can lack prebuilt wheels for the core's dependencies: numpy 2.2.6, scipy 1.15 and river 0.23 stop at 3.13. pip then tries to compile them, which failed on CI's Mac with Homebrew's Python 3.15 ("Unknown compiler(s): gfortran").
     - The same file feeds `setup.rs`, which passes the range to the scripts, and the first-run CI job. `daemon/tests/test_python_versions.py` checks that the CI test matrix stays inside the range and that `requires-python` agrees with it.
     - On macOS, `/usr/bin/python3` is never run.
     - On Windows, the Store stub is skipped.
   - **If the core still cannot be installed on that Python, Heliox falls back once.** It says so ("trying again with Heliox's own Python"), removes the environment, downloads its private Python and tries again. Only then does it report a failure.
   - **No suitable Python.** Heliox downloads a private one. Before that, the screen shows the size and the source:
     - uv 0.13.5 from `github.com/astral-sh/uv`. The screen shows its exact size from the probe. Its SHA-256 for each platform is fixed in the script and checked before uv is unpacked.
     - Python 3.12 from python-build-standalone, downloaded by `uv python install`, which checks its own checksums.
     - Both stay in Heliox's folder: `~/.config/heliox-os/tools`, `python` and `env`. They take about 110 MB on disk, measured on CI.
     - The system Python and the PATH are not touched.
3. **The person chooses the optional groups.** The core is always installed. The screen shows its download and, where CI has measured it, its size on disk: about 460 to 530 MB once installed. The optional groups are:

   | Group | What it gives | Preselected |
   |---|---|---|
   | macOS readers / Linux readers (`macos` / `linux` extra) | Live reads the field you type in, the app in front, the screen text, window actions | yes, on macOS and Linux |
   | Voice (`voice`) | speech in and out, locally, with Whisper base; the speech models below are offered with it | yes, when the computer has a microphone |
   | Speech models (inside Voice: Parakeet, live captions, Smart Turn) | from the daemon's `speech_models.json`, each with its size, licence, source and language; the daemon downloads them, checks their checksums and can stop and resume | none, not even Parakeet |
   | Screen reading extras (`vision`) | EasyOCR, OpenCV, MediaPipe, where the system's own text recognition is not enough | no |
   | Browser automation (`browser` and Chromium) | Heliox works in web pages in its own Chromium | no |
   | Built-in local model runner (`llamacpp`) | GGUF models inside Heliox; builds llama.cpp, so it needs a C and C++ compiler; not needed with Ollama or a cloud provider | no |
   | Everything else | documents, network discovery, remote machines, plugin sandbox, typing supervision, memory search | no |

   Nothing is downloaded until the person presses "Set up Heliox". The total is shown next to the button.

   **Where the sizes come from.** No number is written into the screen's text:
   - The install groups' sizes are in `tauri-app/ui/src/lib/firstRunSizes.json`, per system. Each entry has a download size and, when known, a size on disk, and an entry measured by CI names the run.
   - The first-run job measures them. It points pip's cache at an empty folder, and after each step the scripts print the cache's size (the download) and Heliox's folder's size (the disk).
   - `daemon/tests/platform_real/first_run_sizes.py update` writes the job's log into the file.
   - The speech models' sizes come from the daemon's manifest.
   - Every number on screen says "about".
4. **Each step shows its progress, and a failure shows its reason, what to do, and Retry.** Progress is the step and the last line of pip's output. "Show details" gives the last 40 lines.
   - **The core decides whether setup succeeded.** When it is in, the app starts the daemon itself and its supervisor watches it from then on; no restart is needed.
   - **A failed optional group never blocks the app.** The person can retry it on the spot or later, from Settings > Optional parts.
5. **Once the daemon answers,** the screen asks it for each speech model the person ticked (`speech_model_download`). It keeps the ticks until each has been accepted, across a restart, because the daemon allows one download at a time. Progress then shows in Settings > Voice. The screen also starts the first conversation if the profile is empty (`getting_to_know`: status, then start). Otherwise the daemon offers that conversation later as a card.
6. **Settings > Optional parts** offers every group again, each with its size and source. A group added there is used after Heliox restarts.

The app can only ask for groups from a fixed list: `group_extras` in `src/setup.rs`. Each version is pinned to the app's own release (`pilot-daemon==<app version>`). `HELIOX_DAEMON_SOURCE` points the install at a local checkout, for development and CI.

## Updating an older install

A person who installed an earlier version has an environment with an older daemon. The old desktop app installed `pilot-daemon[all]==<its version>`. When the app's own version is newer than the daemon's, the app offers an update.

1. **Detection costs nothing.** At start, the app reads the daemon's version from the `pilot_daemon-<version>.dist-info` folder in the environment. If that version is older than the app's, the "Update Heliox" screen says which version is installed and which this app is. A newer daemon is never downgraded. "Not now" keeps the installed version for this run, and the offer comes back at the next start.
2. **The plan changes nothing.** "See what changes" runs `update.py plan`, which uses `pip install --dry-run --report` with `--upgrade-strategy only-if-needed`. The screen lists every package that will change (from → to, or new) and the download size the package index reports.
3. **The person's parts are kept.** The groups come from `~/.config/heliox-os/setup.json`, which first-run setup and every update now write. For an install made before that file existed, they are inferred from what is installed: `faster-whisper` means voice, `easyocr` means screen reading, and so on. So an old `[all]` install keeps every group.
4. **Unchanged packages are not touched.** The update runs in place with `only-if-needed`, so only the packages in the plan are replaced.
5. **The update is all or nothing.** Before pip runs, `update.py apply` keeps a copy of the environment made of hard links, so no data is copied. pip replaces a file by removing it and writing a new one, so the copy keeps the old files intact. After installing, it checks that `pilot.server` imports and reports the new version.
   - On success it deletes the copy and records the version and groups.
   - On any failure it puts the copy back, so the environment is exactly as it was, and says so with a Retry. Retry plans again, and "Keep the current version" closes the screen.
6. **The daemon is stopped for the update and started again after it.** On Windows its files are locked while it runs. The supervisor is paused for the duration (`DAEMON_PAUSED` in `setup.rs`) so it does not restart the daemon into a half-changed environment. If a daemon the app did not start is listening, the update refuses and asks the person to quit it.

`update.py` runs with the environment's base Python, named in `env/pyvenv.cfg`, never the environment's own interpreter. An environment whose Python no longer starts, for example after Homebrew upgraded the Python it was made from, is sent to Repair, which rebuilds it.

**How it is tested:**
- **Real run on Windows (12 November).** A venv with a small fake `pilot-daemon` 0.1.0 was updated to a fake 99.0.0 that installs but has no `pilot.server`. The update was rolled back: the environment reported 0.1.0 afterwards and no copy was left behind. A clean update to 0.2.0 then succeeded and recorded the groups file.
- **Unit tests.** `daemon/tests/test_update_path.py` covers group inference, the requirement, which packages count as changed, version order, and the hard-link copy surviving a pip-style file replacement and restoring exactly. The Rust tests cover version detection, `pyvenv.cfg` and the plan lines; vitest covers the update screen's states.
- **CI, job `update-path`** (non-blocking; ubuntu, macOS and Windows). It installs the released `pilot-daemon` 0.15.0 as an older app did, then does three things:
  - It applies a fake 99.0.0 that breaks the daemon, and checks the rollback left 0.15.0 with an identical package list.
  - It plans and applies the update to this checkout.
  - It checks that the updated daemon answers an authenticated ping.
- **Not tested:** the desktop app driving the update (Rust compiled and unit-tested here, not run as an app).

## Repair

At start, the app checks that the environment's own Python starts and sees the daemon package. This is cheap: it does not import the daemon.

If the environment is broken, the first-run screen shows "Repair Heliox" and says which parts it will build again. Typical causes: Homebrew upgraded the Python it was made from, the private Python was deleted, or part of the environment was removed.

Repair runs the first-run script with the recorded groups. The script then:
- sets the broken environment aside as `env.broken`;
- builds a new one, with the same Python rules as first run;
- installs the core and the groups;
- deletes `env.broken` only once the core is in.

Settings and data are not touched.

**How it is tested:**
- **Real run on Windows (12 November).** A venv whose `python.exe` was deleted was detected ("no longer starts; building it again"), rebuilt from the system Python and given its core.
- **CI, job `first-run`.** On all three systems, with the runner's Python and with a private one, the job deletes the environment's interpreter after setup. It then reruns the script and checks three things: that the repair was detected, that `env.broken` is gone, and that the rebuilt daemon answers.
- **Unit tests.** The start-up decision (repair before update, nothing to do, update) has Rust unit tests; the screen has vitest.

## Removing Heliox's files

Settings > "Remove Heliox's files" lists everything Heliox put on this computer, with sizes, before anything is deleted. The list is computed by `src/uninstall.rs` from the home folder and the system's app-data folders. The UI cannot add a path to it.

Every folder in the list carries a name that is Heliox's alone, and the code says why beside each one: `heliox-os`, `.heliox`, `heliox` under `.cache`, `com.helioxos.app`. When the home folder is not known, nothing is listed and nothing is removed. As a second lock, every path must be absolute and at least two folders below the root.

| Where | What | Kind |
|---|---|---|
| `~/.config/heliox-os/env`, `python`, `tools`, `uv-cache`, `setup` | the environment, private Python, uv and setup scripts | app |
| `~/.config/heliox-os` (the rest) | settings, keys, restrictions, shortcut | data |
| `~/.local/share/heliox-os/models` | downloaded speech models | app |
| `~/.local/share/heliox-os` (the rest) | memory, timeline, receipts, undo copies | data |
| `~/.local/state/heliox-os` | logs | app |
| `~/.cache/heliox`, `~/.heliox` | persona, voice calibration, plugins, screenshots, quarantined files | data |
| `~/.pilot/usage_patterns.json`, `~/.pilot/failure_memory.json` | usage patterns and failure memory: only these two files, because another program could use a folder called `.pilot`; the folder goes only if that leaves it empty | data |
| The system credential store, under four services named in `pilot.system.credential_cleanup` | API keys, SSH keys and passphrases, paired devices, the receipt signing key, study and recording keys | data |
| Runtime folder: `%LOCALAPPDATA%\heliox-os`, `~/Library/Caches/heliox-os` | runtime files | app |
| The app's web view data: `%LOCALAPPDATA%\com.helioxos.app`; `~/Library/Application Support`, `Caches` and `WebKit` under `com.helioxos.app`; `~/.local/share` and `~/.cache` under `com.helioxos.app` | interface settings stored in the browser view | app |

**The person chooses** between "remove the app's files, keep my data" (the default) and "remove everything". They see the total size, and they tick that they understand the files will be deleted. Only then does the button work.

**Before deleting**, the daemon is stopped and the supervisor paused.

**Saved keys.** With "remove everything", the saved keys go before any file is deleted:
- They are removed from the credential store by the daemon's own code, `python -m pilot.system.credential_cleanup remove`, using the environment that still exists, with a 30 s limit.
- The daemon finds the entries by service where the store allows it: Windows' credential list, macOS's `security` tool, the Linux Secret Service. It also tries the accounts it knows by name.
- Anything it cannot remove is listed by service and account, with how to remove it on that system.
- If the environment cannot run, or the step fails or runs out of time, the screen lists the four services by name instead. That list is `setup/credential_services.json`, which a test keeps equal to the daemon's own list.

With "keep my data", the keys stay, and the screen says which services hold them.

**After the removal**, the window shows what was removed and what failed. On Windows, the app's own web view folder fails while the app runs; the screen says it goes when the app is uninstalled. It also shows which saved keys remain and the manual steps, with a Quit button instead of a reconnecting screen.

**Deletion never follows a link out of Heliox's folders.** A symbolic link or junction inside them is removed as a link, and what it points to stays. Rust's `remove_dir_all` does not follow links (Rust 1.58 and later).

**Heliox leaves these to the person, and the screen says so:**
- the app itself (the system's own uninstall);
- shared caches: pip's cache, and the Hugging Face cache that holds the local voice and Whisper models;
- on macOS, the privacy grants: System Settings, or `tccutil reset All com.helioxos.app`;
- on Wayland, the global shortcut the desktop stores.

Heliox creates no login item, service or scheduled task.

**How it is tested:**
- **Rust tests on a real file system.** On Windows here, on Linux in the Rust job and on macOS in the `rust-macos` job, the tests build a home folder with an environment, settings and a quarantined file, plus a link pointing outside. Removing with "keep my data" deletes the environment and keeps the settings and the quarantined file. Removing everything deletes Heliox's folders. In both cases the file outside is untouched.
- **Credential cleanup.** Tested with a fake keyring backend (`daemon/tests/test_credential_cleanup.py`): entries are removed by service search and by known name, an entry the store refuses is listed with how to remove it, and a store that cannot be searched still has its known names removed. The search by service ran for real on Windows here, listing names only. Removal has not run against a real credential store on any system.
- **Rust tests** also cover no home folder, the safe-location lock, `.pilot` keeping another program's file, the credential report and the Windows web view note.
- **Vitest:** the screen's choice, confirmation, the after-removal screen, and the three replies: a bare ok, a refusal, no reply.
- **Not tested:** the desktop app driving a real removal (it would remove the runner's own install).

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
  - It starts the daemon it installed and checks that the daemon answers an authenticated `ping` (`daemon/tests/platform_real/first_run_rpc_check.py`).
  - It prints the time taken and the size of each folder, which replaces the estimates above once measured.
- **Real run on Windows (12 November, second round):** a Python range that excludes every Python on the machine leads to the private download, and a core that cannot be installed on the system Python falls back to a private Python once before failing.
- **Not tested:**
  - The desktop app driving the scripts (Rust compiled only by CI's Rust job).
  - The real voice, screen, browser, local and "everything else" groups.
  - A Mac or Linux desktop with nothing installed: hosted runners always have some Python, so the private path is forced with `HELIOX_PYTHON=private`.

## What an unsigned build asks of the person

Signing costs money and accounts, so it is the owner's decision. Until then, this is what someone sees.

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

**Linux:** no signing prompt. The .deb and .rpm declare their dependencies. The AppImage needs `chmod +x` or the file manager's "Allow executing as program".
