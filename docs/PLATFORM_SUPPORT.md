# Platform support

Heliox runs on Windows, macOS and Linux. This page says what each part of the companion does on each system, where the code is, what the person has to grant or install, and how each claim is known. Nobody on the project develops on a Mac or a Linux desktop, so how a claim is known matters as much as the claim.

How each claim is known, everywhere on this page (one legend for every section and table):

- Run for real: it ran against the real system interface, and the page names what ran it. That is a GitHub-hosted runner (the CI job and the check's name, with the commit whose CI run it was) or the owner's Windows machine (with the date where it was recorded).
- Fakes only: only the decision logic ran, with the platform layer replaced by a fake.
- Not tested: nothing has run it.

A claim marked Fakes only or Not tested describes what the code is built to do, not what anyone has seen it do. Where one cell needs two markers, each part says which. "CI on 3793a0d" is the CI run on commit 3793a0d (12 November), and "CI on 0a8edad" the run on commit 0a8edad (run 43196949417), and "CI on e51aead" the run on commit e51aead (run 43209796125); their CHECK lines are quoted below.

| Feature | Windows | macOS | Linux | Code | Grant or install | How it is known |
|---|---|---|---|---|---|---|
| Reading the focused text field (Live) | Run for real: UI Automation through a long-lived PowerShell helper. Its read cost has not been measured on an idle machine | Run for real: Accessibility API in process (pyobjc), with Accessibility granted. Without the grant: Fakes only | Run for real with a GTK 3 window, under X11 and under sway (Wayland): AT-SPI over D-Bus (jeepney). GTK 4: typed text Run for real under X11 (CI on 008f1a0, `gtk4/reads-typed-text`); in that run its password entry was read as an ordinary field, now refused by its numeric role (Fakes only until the next run, see Password fields below). Firefox: the check never saw its window in that run (Not tested). Qt, Chromium and Electron apps, and GNOME's and KDE's compositors: Not tested | `daemon/pilot/live/focused_text*.py` | macOS: Accessibility for Heliox-OS; the app explains it and offers the system prompt once (Fakes only: the runner has the grant). Linux: `at-spi2-core`; Qt, Chromium, Electron and Firefox apps need toolkit accessibility on (Not tested) | Windows: the owner's machine. macOS, CI on 3793a0d, `platform-readers`: `macos/reads-typed-text`, `macos/secure-field-refused`, `macos/identity-stable-across-title-change` passed; `macos/read-cost` median 0.1 ms. Linux, same run: the `x11/` and `wayland/` checks of `linux_atspi_check.py` passed; `x11/read-cost` median 1.6 ms, `wayland/read-cost` median 1.4 ms. Read costs vary between runs; these are one run's |
| Foreground app and window (timeline, Live) | Run for real: `win32gui` | Run for real with Accessibility: the app and the title. Without the grant, the app from LaunchServices (`lsappinfo`, the app owning the menu bar), then the Quartz window list, and no title: each source Run for real on its own, the order without the grant Fakes only. AppleScript when pyobjc is missing: Not tested | Run for real: AT-SPI focus under X11 and sway. The `xdotool` fallback (X11): Fakes only until `windows-x11/foreground-falls-back-to-xdotool` has run | `daemon/pilot/agents/screen_vision.py`, `focused_text_macos.py`, `focused_text_linux.py` | As above | Windows: the owner's machine. macOS, CI on 3793a0d: `macos/frontmost-app-from-launch-services` and `macos/frontmost-app-from-window-list` passed with a test window in front. Linux, same run: `x11/foreground-title-follows`, `wayland/foreground-title-follows`. The xdotool check (`linux_windows_check.py`, with the reader's bus pointed nowhere) is new, result to come |
| Application display names | Run for real: the file description | Run for real: the bundle display name (CI on e51aead: `apps/macos-bundle-name-textedit`, `apps/macos-bundle-name-safari`, and `apps/macos-names-the-running-process` named the running TextEdit) | Fakes only: the desktop entry's `Name`. A real check (`apps/linux-names-a-desktop-entry`, and the runner's own entries) is new, result to come | `daemon/pilot/system/app_names.py` | Nothing | Windows: the owner's machine. macOS and Linux: pure parsers over a path or file text, plus `tests/platform_real/apps_check.py` |
| Activity timeline | Fakes only, apart from the foreground row above | Fakes only, apart from the foreground row | Fakes only, apart from the foreground row | `daemon/pilot/workflows/activity_timeline.py` | Nothing more | Unit tests on all three |
| macOS permissions status | Not applicable | Run for real for reading without a prompt: on the runner Microphone, Accessibility and Screen Recording read "granted" and Automation "cannot_tell". The messages for "not granted": Fakes only | Not applicable | `daemon/pilot/system/macos_permissions.py` | Nothing | CI on 3793a0d: `macos/permissions-read-without-prompt` (7 ms) |
| Reading the screen (OCR) | Run for real: Windows OCR | Run for real with the `macos` extra: the Vision framework in process (pyobjc), no developer tools needed. Without pyobjc: tesseract Run for real (a second venv without pyobjc, CI on e51aead: `ocr-macos/tesseract-reads-the-line`). Vision through a Swift helper: in CI on e51aead the script, interpreted with `swift` on every read, timed out at 20 s (the in-process route took 3.3 s); it is now compiled once with `swiftc`, in the background (up to 180 s): the read that starts it goes on to tesseract and says a one-time preparation is under way, a failed compile is not tried again until Heliox restarts, and each later read runs the helper within 20 s (Fakes only; the real check is fixed after that run, result to come). EasyOCR: Not tested. Without Screen Recording the capture shows only the wallpaper, reported by the permission status: Fakes only | tesseract, EasyOCR, pytesseract, with capture by `pyautogui`, `gnome-screenshot` or `scrot` on X11: Fakes only, real check new in this round, result to come (`ocr-linux/` checks: scrot and tesseract read a rendered line). On Wayland, through the desktop's Screenshot portal: Run for real under sway. On sway, Hyprland and other wlroots desktops the portal also needs xdg-desktop-portal-gtk (Run for real: without an Access backend the front end offered no Screenshot) and grim (the wlr backend runs it) | `daemon/pilot/system/vision.py`, `daemon/pilot/system/macos_vision_ocr.py`, `daemon/pilot/system/screen.py` | macOS Screen Recording. Linux: tesseract; on Wayland, xdg-desktop-portal with the desktop's backend (and xdg-desktop-portal-gtk and grim on wlroots desktops) | Windows: the owner's machine. macOS, CI on 3793a0d: `macos/ocr-vision-in-process`, `macos/ocr-through-vision-module` (a rendered line read back). Linux, same run: `portal/screenshot` failed with response 2: the request reached xdg-desktop-portal-wlr, whose grim was missing ("Operation not permitted"); with grim installed, CI on 0a8edad: `portal/screenshot` passed (4789 bytes in 68 ms). GNOME's and KDE's consent dialogs: Not tested |
| Call detection (quiet in calls; `current_call()` for the listener and Live) | Run for real: the microphone-users store, meeting processes (Zoom's `CptHost.exe`) and call-window titles (Meet, Zoom, Teams, WhatsApp, Slack, Webex, Discord) | macOS 14 and later, CoreAudio's list of processes recording input (through ctypes), Heliox's own processes excluded: the reader Run for real only as far as answering on the runner without error; a call being seen: Not tested. Older macOS (a busy default input counts only when Heliox is not listening and a call app is open): Not tested | Run for real with PulseAudio: the recording streams from `pactl list source-outputs`, recordings of the output and Heliox's own PIDs excluded. PipeWire's pipewire-pulse: Not tested. Without `pactl` it cannot tell and stays "not in a call": Fakes only | `daemon/pilot/system/call_detection.py` | Nothing on any system (no permission, no extra package); Linux: `pulseaudio-utils` or `pipewire-pulse` for `pactl` | Windows: the owner's machine. Linux, CI on 3793a0d, `platform-readers` (quiet in calls step): a null source with `parecord` recording from it is seen as a call (3 real tests passed). macOS: the real reader on the runner; the rest with fakes |
| Speech capture and playback | Run for real: capture through WASAPI (`soundcard`), then PortAudio; playback through one PortAudio output line kept open between sentences | Fakes only: PortAudio (Core Audio) both ways. A refused or not-yet-asked microphone permission is said at once (`microphone_state()`); the bundle declares the microphone use | Capture Run for real: PortAudio over PulseAudio; a chosen source is opened through the ALSA `pulse` device with `PULSE_SOURCE`. Playback Run for real (CI on e51aead: `audio-linux/playback-reaches-the-default-sink`, a tone through Heliox's output line recorded from a null sink's monitor: peak 13978, rms 6897). Plain ALSA and PipeWire: Not tested. Needs `libportaudio2`, and `libasound2-plugins` for the `pulse` device | `daemon/pilot/system/voice.py`, `daemon/pilot/system/audio_health.py` | macOS: Microphone. Linux: `libportaudio2` | Windows: the owner's machine. Linux, CI on 3793a0d, `platform-readers`: soundcard names the default source by its source name; PortAudio's `pulse` device opens the source `PULSE_SOURCE` names |
| Microphone and speaker recovery | Fakes only: a failed, stalled or silent stream is reopened (backing off to once every 30 s); the default microphone and speakers are followed live. The reopen against real hardware: Not tested (steps in device_check_steps) | Fakes only, the same way; default changes read through CoreAudio (`soundcard`); PortAudio's device list is re-read on reopen | Fakes only, the same way; default changes read through PulseAudio or PipeWire (`soundcard`); PortAudio's device list is re-read on reopen. A new default microphone followed during capture: Fakes only. The real check failed in CI on e51aead for a fault of the check (its two test microphones never loaded), fixed after it, result to come. Two microphones with the same name: see Gaps that remain | `daemon/pilot/system/audio_health.py`, `daemon/pilot/system/voice.py` | Nothing | Unit tests on all three, including the 0x88890004 "device invalidated" case |
| Microphone kind (Bluetooth call mode, very quiet, clipping) | The judgement Run for real on the owner's recordings: nothing above 4 kHz is call mode; a headset in call mode was flagged after two utterances, the desk microphone never. Said once in the voice bar with a one-button switch to another microphone, remembered per headset by its WASAPI name: Fakes only | The same judgement from the sound; the switch remembered by the Core Audio device name (the name PortAudio also uses; CoreAudio's numeric ids change on replug): Fakes only | The same judgement; the switch remembered by the PulseAudio source name, and the chosen source opened through `PULSE_SOURCE`: the names and the opening Run for real, the offer Fakes only | `daemon/pilot/system/mic_quality.py`, `daemon/pilot/system/voice.py` | Nothing | Windows: the owner's recordings. Linux names and opening through `PULSE_SOURCE`: CI on 3793a0d, `platform-readers` |
| Wake phrase | Run for real | Fakes only, and only as far as capture works | Fakes only, and only as far as capture works | `daemon/pilot/system/voice.py` | As for capture | Windows: the owner's machine |
| Speaking | Run for real: Kokoro or Pocket TTS, then System.Speech | The same local engines: Fakes only. Then `say`: Run for real to a file (CI on e51aead: `speech-macos/say-writes-audio`, 2.14 s of audio) | Fakes only: the same local engines, then `espeak-ng` or `espeak`; with neither, Heliox says once in the log that no voice is available and carries on in text. Both: real check new in this round, result to come (`speech-linux/espeak-writes-audio`, `speech-linux/no-voice-said-once-text-carries-on`) | `daemon/pilot/system/voice.py`, `daemon/pilot/system/linux_voice.py` | Linux: `espeak-ng` for the fallback voice | Windows: the owner's machine |
| Notifications and cards | In-app cards, OS toasts through the Tauri plugin, and the `notify` balloon: Run for real before the `notify` result change; the change itself Fakes only | Fakes only: cards; toasts may not appear for an unsigned app (Not tested); the `notify` action reports failure instead of "sent" (Fakes only); osascript's result Run for real (CI on e51aead: `notify-macos/osascript-result`; whether the banner showed cannot be seen on a runner) | Fakes only: cards; toasts need a notification daemon; `notify` reports a missing `notify-send`, a missing notification daemon, and "sent" only when a daemon took it: real check new in this round, result to come (`notify-linux/` checks, with dunst) | `daemon/pilot/system/notify.py`, `daemon/pilot/agents/executor.py` | Notification permission (asked by the app) | Unit tests for the command and its result |
| Global hotkey | Run for real: Ctrl+Space | Default Option+Space (Cmd+Space is Spotlight's); Settings shows shortcuts in the Mac's key names (Option, Command, Control) and accepts either spelling (Fakes only, vitest); registering it, and a failed registration shown in Settings: Not tested | Ctrl+Space on X11: Not tested. On Wayland the daemon binds it through the GlobalShortcuts portal (GNOME 48+, KDE Plasma 5.27+, Hyprland), where the desktop confirms the keys: the client Run for real against a fake portal on a real session bus, real desktops Not tested. Elsewhere on Wayland (sway, older GNOME, Xfce) Settings says the desktop has no such portal: the message Run for real under sway | `tauri-app/src-tauri/src/hotkey.rs` | Nothing | Rust compiled and linted by the CI `rust` job. CI on 3793a0d: `portal-fake/bind`, `portal-fake/activated`, `portal/global-shortcuts-missing-is-explained` |
| Window actions (list, focus, minimise, maximise, close) | Run for real: user32 and UI Automation | Run for real with the `macos` extra and Accessibility: list, maximise, minimise, close (the window only), and focus, which tries activation, then AXFrontmost and AXRaise, then LaunchServices, checking the front app by pid after each. Without pyobjc, AppleScript through System Events: Not tested | Fakes only: `wmctrl` and `xdotool`, X11 only. A real check under Xvfb with openbox and two GTK windows (`windows-x11/` checks) is new, result to come | `daemon/pilot/system/window_mgr.py`, `daemon/pilot/system/macos_windows.py` | macOS: Accessibility | Windows: the owner's machine. macOS, CI on 3793a0d: the `macos/windows-` checks passed. On the hosted Mac the activation request was accepted and ignored, AXFrontmost did not take either, and LaunchServices (`open -b`) brought Finder forward (`last_route` in the log). Linux: `tests/platform_real/linux_windows_check.py` |
| Naming a control to click (accessibility snapshot) | Run for real: UI Automation; a click by element ref, or a press in the background | The snapshot of the front window's controls through the Accessibility API, never a field's value: Run for real. Labels from placeholder or help text, and the press with AXPress after checking the window again (a window Heliox never reads gets no press and no click): Run for real. The window's own buttons named by their subrole: Run for real (CI on e51aead: `Button "close window"` in the snapshot). Without a press action the mouse clicks the centre: Not tested; in CI on e51aead the check found another app in front and did not reach its label, fixed after it, result to come | The snapshot through AT-SPI under X11 and sway: Run for real. The control's own AT-SPI action ("click", "press", "activate") after the same window check: Run for real under X11 and sway. Without an action the mouse clicks the centre on X11: Not tested, real check new in this round, result to come (`x11/centre-click-reaches-a-control-without-a-press-action`). On Wayland a control with no action is refused with a message, as the pointer belongs to the compositor: Fakes only | `daemon/pilot/system/accessibility.py`, `daemon/pilot/system/ax_snapshot_platforms.py` | macOS: Accessibility. Linux: AT-SPI | Windows: the owner's machine. CI on 3793a0d: `macos/snapshot-finds-the-button`, `-never-has-values`, `-ref-resolves-in-the-same-window` and the same three under `x11/` and `wayland/` passed. CI on 0a8edad: `macos/snapshot-names-fields-by-placeholder` passed (`Edit "message"`, `Edit "password"`); `press-ran-the-buttons-handler` (presses 0 -> 1, "through the Accessibility API (AXPress)" and "through AT-SPI (click)", the cursor did not move) and `press-refused-in-a-password-managers-window` passed on the Mac, X11 and sway |
| Undo and protected folders | Fakes only | Run for real on APFS: folder checks ignore case, as APFS does (CI on e51aead: `protected-macos/apfs-treats-both-spellings-as-one-file`, `protected-macos/other-case-is-refused`) | Fakes only: case-sensitive, as Linux file systems are | `daemon/pilot/security/path_rules.py`, `daemon/pilot/security/validator.py`, `daemon/pilot/workflows/undo_journal.py` | Nothing | Both rule sets with fakes on every system |
| Launching apps (simple actions) | Run for real: Start menu shortcuts, PATH, Store apps | Run for real: installed bundles matched by file name and the bundle's own name, then `open -a` (CI on e51aead: `apps/macos-finds-textedit-among-bundles`, `apps/macos-launch-textedit`); arguments after `--args`: Fakes only | Fakes only: a path, PATH, then desktop entries by shown name or id, started from their `Exec` line with arguments in place of `%f %F %u %U`; terminal apps open in the system terminal; ambiguous names fail closed. A real launch by a desktop entry, with its arguments checked (`apps/linux-launch-by-desktop-entry-with-arguments`), is new, result to come | `daemon/pilot/system/applications.py`, `daemon/pilot/system/desktop_entry.py` | Nothing | Windows: the owner's machine. macOS and Linux: Exec parsing, matching and the command started with fakes, plus `tests/platform_real/apps_check.py` |

## What Heliox Live can notice on each system

Live checks four things, each from a different source. What it can see depends on that source being available, so
the same slip can be caught on one system and missed on another. From the code (`daemon/pilot/live/`) as of this
page, with what has been shown for each claim:

- **Run for real**: seen on that system in a real run, named with its job and check;
- **Fakes only**: the decision is tested with stand-ins (recorded or typed text, fake readers), never seen there;
- **Not tested**.

The end-to-end check (`tests/platform_real/live_end_to_end_check.py`, in the `platform-readers` job: an excluded
app, a slip that must become a card with its evidence, the password field, the window closed) passed all four on
Linux under X11 and sway on 12 November. On macOS the slip did not become a card (the reader reads the field in its
own check, so the gap is between the reader and Live); that is being found out.

| Slip | Comes from | Windows | macOS | Linux |
|---|---|---|---|---|
| In what you type: a weekday that doesn't match its date, a date that doesn't exist, a wrong sum, a wrong name for someone in your profile, a doubled word, "attached" with nothing attached, a key pasted into a message, catch-pack phrases, a promise with a deadline, rewriting the same sentence | The focused text field (the reader above) | Meant for any app whose fields UI Automation exposes (the reader was built against Notepad on the development machine). Notepad, Office, browsers, chat apps: **Not tested** by a recorded check. The detectors: **Fakes only** (typed text, the Live Bench) | Meant for apps that expose their fields to Accessibility, once it is granted to Heliox-OS. Reading a field: **Run for real** on an AppKit text field (`platform-readers`, `macos_ax_check.py`, "reads-typed-text"). The slip becoming a card: **Run for real: fails** (end-to-end check, "live-typed-slip-becomes-a-card": no card). Safari, Chrome, Electron apps: **Not tested** | Meant for apps on the accessibility bus. Reading a field: **Run for real** on a GTK 3 field under X11 and sway (`platform-readers`, `linux_atspi_check.py`, "reads-typed-text"). The slip becoming a card, an excluded app and the password field giving none, the card withdrawn when the window closes: **Run for real** under X11 and sway (`live_end_to_end_check.py`). GTK 4, Qt, Chromium, Electron, Firefox: **Not tested** |
| A risky command before you press Enter | The focused field of a terminal | Meant for Windows Terminal, the console and PowerShell windows. The detector: **Fakes only**; reading those windows: **Not tested** by a recorded check | Meant for Terminal.app and iTerm2 (other terminals expose nothing): **Not tested** | Meant for VTE terminals (GNOME Terminal, Tilix, Terminator, Xfce Terminal) and partly Konsole; not Alacritty, kitty, WezTerm, foot or xterm: **Not tested** |
| In what you say to Heliox: a wrong weekday or impossible date, a wrong sum, a wrong name, a promise without a reminder, a date that differs from the one you just typed | Speech, after the wake phrase, the microphone button, or in the 8 s follow-up | **Fakes only**: the catches run on recognised text, including real recogniser output replayed from recordings (Live Bench); no recorded voice session has shown a spoken catch | **Fakes only**, as on Windows; it needs capture to work (PortAudio, microphone permission) | **Fakes only**, as on Windows; it needs capture to work (PortAudio with ALSA, PulseAudio or PipeWire) |
| An error on screen, with the fix | Reading the screen (OCR); off by default | Windows OCR. The card: **Not tested** | The Vision framework, once Screen Recording is granted. Vision reading a rendered line: **Run for real** (`macos_ax_check.py`, "ocr-vision-in-process"). The card: **Not tested** | tesseract on X11; on Wayland through the desktop's Screenshot portal, not yet proven (see above). The card: **Not tested** |
| A risky command or a key typed anywhere (supervision's keyboard hook) | A global keyboard hook (`pynput`); off by default, its own opt-in | **Fakes only** in Live's tests | Needs Input Monitoring for Heliox-OS: **Not tested** | X11 only; Wayland gives apps no global keyboard: **Not tested** |
| Switching back and forth between two windows | The foreground window | The offer: **Fakes only** | Which app is in front: **Run for real** (`macos_ax_check.py`, "frontmost-app-from-launch-services"). The offer: **Fakes only** | The window title following focus: **Run for real** under X11 and sway (`linux_atspi_check.py`, "foreground-title-follows"). The offer: **Fakes only** |
| Staying quiet in calls | Call detection (above) | Live holding back in a call: **Fakes only** | macOS 14 and later, partly before: **Not tested** | Another app recording seen through `pactl`: **Run for real** (`platform-readers`, "Linux, quiet in calls"). Live holding back: **Fakes only** |

What Live never sees, on any system: password fields, Heliox's own window, password managers and apps you excluded
from the timeline, text inside images or video without screen reading on, the body of a web page you are reading
(only fields you type in), other fields of a form than the one you are in (so it cannot check a mail's recipients),
and speech said near the microphone that was not for Heliox (see below). Password fields: **Run for real** on GTK
under X11 and sway ("password-refused") and on an AppKit secure field ("secure-field-refused"). Heliox's own
window, password managers and excluded apps: **Fakes only** (matched by name; the end-to-end check's excluded app
is pending). The body of a web page: **Not tested**.

**Terminals that expose nothing (the Live terminal check).** On Windows the console and Windows Terminal are meant
to be read through UI Automation (**Not tested** by a recorded check; mintty, used by Git Bash, and ConEmu: **Not
tested**). On Linux kitty, Alacritty, WezTerm, foot and xterm expose no text, and on macOS only Terminal.app and
iTerm2 are meant to be read, so a risky command typed elsewhere was noticed only after it ran. Settings, Live, Terminal check adds an
opt-in hook to the shell instead (`daemon/pilot/live/shell_line.py`), after showing exactly what it adds and how to
take it out:

| Shell | What it does | What is added |
|---|---|---|
| zsh (5.3 or later; only 5.9 has been run) | A ZLE `line-pre-redraw` hook sends the line being edited each time it changes; Live checks it like a terminal field and the card comes at the first pause, before Enter. In a real zsh: **Run for real** on ubuntu and macOS, zsh 5.9 (see below) | One line in `~/.zshrc` (or `$ZDOTDIR/.zshrc`) that sources `~/.local/share/heliox/shell/heliox-live.zsh`, marked "# Heliox Live terminal check" |
| fish | fish has no hook on every change, so Enter asks Heliox first (through `nc -U -w 1`: at most one second for Heliox plus the time to start `nc`, measured at 1.01 s on ubuntu and 1.13 s on macOS against a Heliox that never answers, and no wait at all when Heliox is not running); a risky command is not run, its warning is printed, and Enter again runs it. In a real fish: **Run for real** on ubuntu (fish 3.7, netcat-openbsd) and macOS (fish 4.9, the system's own nc) (see below) | `~/.config/fish/conf.d/heliox-live.fish`; no start-up file is changed |
| bash | Nothing yet. Plain bash only offers hooks after Enter, too late; with ble.sh installed it could work like zsh, which is not built | Nothing |

The line travels over a Unix socket in a folder only the person can open (mode 0700, owned by them; Heliox refuses a
folder anyone else owns or can open; the socket is 0600), never over the network, and only while Live is on: with
Live off the socket does not exist and the hook stays silent. Never sent: a line that starts with a space, a line in
zsh's `vared` or a program's own prompt (sudo and password prompts never use the line editor), and anything from the
first token-looking string onwards. It is kept in memory for 3 seconds and never sent to a model.

What has been shown: **Run for real**, the socket folder 0700 and the socket 0600, and the socket gone once Heliox
closes it (`platform-readers` on ubuntu, `tests/platform_real/shell_integration_check.py`). **Fakes only**, the
decisions above (what is never sent, the 3 seconds) on all three systems, and the socket opening and closing with
Live in unit tests on Linux and macOS. **Run for real** on ubuntu and macOS, all 26 checks passing on both
(the `shell-check` workflow, run 43209835182 on e51aead, 12 November; `tests/platform_real/shell_integration_check.py` and `shell_prompts_check.py`):

- zsh 5.9: the line being typed reaches Heliox before Enter; a line started with a space is never sent and clears
  the last one; what is typed inside `vared`, at `read -s` and at a program's own password prompt is never sent
  (Python's `getpass` stands in for sudo, which asks the runner for no password); with Live off nothing is taken;
  `zsh/net/socket` and `zsh/zselect` load in `/bin/zsh` on macOS and `/usr/bin/zsh` on ubuntu.
- fish 3.7 (ubuntu) and 4.9 (macOS): Settings' test offers Add; Enter on a risky command prints Heliox's warning and
  does not run it; Enter again runs it; an ordinary command runs at once.
- Both: the dry run reaches Heliox (zsh, and fish with netcat-openbsd and with Apple's own nc); fish with no nc is
  refused and says what to do; with Heliox hung Enter runs the command after 1.01 s (ubuntu) and 1.13 s (macOS),
  and with Heliox not running after 0.05 s and 0.15 s; zsh keeps taking keys with 2 MB of lines nobody reads; the
  socket is gone once Heliox closes it.

Earlier runs found and fixed: the zsh hook stayed silent where no start-up file had loaded `zsh/zle`, and fish on
macOS called Apple's nc with a flag it lacks (the dry run now finds the flags by trying).

**kitty and WezTerm, read directly.** For people who would rather not change their shell, kitty and WezTerm are
read directly (`daemon/pilot/live/terminal_text.py`), while one of them is the window in front, Live is on and
reading fields is on: WezTerm through `wezterm cli` (the focused pane, its cursor row, its text; nothing to set up,
on every system), kitty through `kitty @ ls` and `kitty @ get-text --add-cursor` over its Unix socket, which kitty
opens only with `allow_remote_control socket-only` and `listen_on unix:/tmp/kitty` in kitty.conf (Settings says so;
Heliox never edits kitty.conf, and never uses a TCP `listen_on`). Only the cursor's line is kept, from after the
prompt sign, with the shell hook's rules (no line starting with a space, nothing from a key-looking string on). It
is asked in the background at most every 0.6 s, each call is killed after 1 s, and the line is kept in memory for
2 s. What has been shown: **Fakes only**, the answers of both terminals as recorded output, on all three systems.
**Not tested** yet, real kitty under Xvfb in the `platform-readers` job (`tests/platform_real/terminal_text_check.py`,
first run pending), and real WezTerm anywhere (it is not in Ubuntu's archive, so that step skips it).

Add is offered only after a dry run on the person's own computer (`shell_line.dry_run`, the `live_shell` "test"
action) is meant to prove the hook would reach Heliox: zsh must load `zsh/net/socket` and `zsh/zselect` (a zsh built
without them would leave the hook silent) and get an answer through the socket; fish must find an `nc` that talks
to a Unix socket and waits for the answer, and that same `nc` must give up within the bound on a socket that
accepts and never answers. When it cannot, Settings says why and what to do, and install refuses as well: a hook
that would stay silent, or an Enter that could hang, is never added. What to do: on macOS nothing needs installing
for `nc`, as the system's own `/usr/bin/nc` talks to a Unix socket (Homebrew's `netcat` is GNU netcat, which cannot,
so if it comes first, `brew uninstall netcat`); for zsh's modules, `brew install zsh`. On Linux, the distribution's
OpenBSD netcat: `netcat-openbsd` on Debian, Ubuntu and Alpine, `netcat` on Fedora, `openbsd-netcat` on Arch. A hung
Heliox is meant never to hold zsh either: the hook writes only when the socket is writable this instant (`zselect -t
0`), and otherwise drops the line and leaves Heliox alone for 30 seconds. Settings states fish's bound, and what the
test measured.

What has been shown: **Run for real** on ubuntu and macOS (the `shell-check` workflow, run 43209835182 on e51aead, 12 November): the dry run, the hung and dead Heliox through
a real fish, and zsh against a Heliox that stopped reading, as listed above. The decisions: **Fakes only** (unit
tests, including an `nc` that never gives up).

Speech heard near the microphone but not said to Heliox (in an open conversation, a sentence for someone else)
reaches Live only as text and is not checked: no card, nothing kept, never stored as conversation or sent to any
model. Outside an open conversation Live does not hear it at all. **Fakes only**: unit tests with recognised text;
no recorded conversation has shown it.

## Reading the focused field

Every platform reader satisfies `FocusedTextSource` (`daemon/pilot/live/focused_text.py`): `read()`, `status()`, `close()` and `available`. `create_reader()` picks the one for the system. Each reader turns what its accessibility interface says about the focused element into `ElementFacts`, and `focused_text_policy.admit` makes the one decision for all three: text passes only from a writable field or a terminal, never from a secure field, and only the last 6,000 characters. The window-level rule, `never_read`, refuses Heliox's own window, password managers, the system credential stores (Keychain Access, Passwords, Seahorse, KDE Wallet) and apps the person excluded from the timeline before any reader runs. The decision itself: Fakes only, on all three systems; the secure field refused: Run for real (CI on 3793a0d, `macos/secure-field-refused`, `x11/password-refused`, `wayland/password-refused`).

Field identity has to stay the same while the person types and the window title changes:

- Windows: the UI Automation runtime id (Run for real on the owner's machine).
- macOS: the process id plus `CFHash` of the accessibility element, which hashes what the element refers to (Run for real: `macos/identity-stable-across-title-change`).
- Linux: the application's bus name plus the element's object path (Run for real: `x11/identity-stable-across-title-change`, `wayland/identity-stable-across-title-change`).

The unit tests check the property with fakes: the same field keeps its identity across typing and a title change, and a different field gets a different one.

Live drops a card early when the window its field lived in has closed. Windows asks whether the window handle in the UI Automation id still exists (Not tested on its own). macOS asks the Accessibility API about the field's window: its element must still answer, and its window (`kAXWindowAttribute`) must still be one of its app's windows (`kAXWindowsAttribute`, which counts a minimised window). Asking the element alone was not enough: in CI on 3793a0d the Mac's text field kept answering after its window closed (`macos/window-gone-after-close` failed), because the app was still running; with the fix, CI on 0a8edad: `macos/window-gone-after-close` passed (the element still answers, its window is no longer one of the app's): Run for real. Linux asks AT-SPI whether the field's object still exists (its application left the bus, or the object answers unknown or defunct): Run for real, `x11/window-gone-after-close` and `wayland/window-gone-after-close` passed in CI on 3793a0d. When a system cannot tell, the card stays (Fakes only). A window on another Space: Not tested on macOS.

### macOS

The reader uses pyobjc (`pip install 'pilot-daemon[macos]'`, included in `all`). pyobjc is Apple-platform-only, maintained, and ships prebuilt wheels, so nothing is compiled on the person's machine. It reads `AXFocusedUIElement` from the system-wide element with a 0.25 s messaging timeout, meant to keep a hung app from stalling Live (Not tested with a hung app: in CI on e51aead the check's read saw no field even before the hang, so its pass proved nothing; the check now requires that read first, fixed after it, result to come). It refuses a field before reading its value when the subrole is `AXSecureTextField` (Run for real) or when secure keyboard entry is on (Fakes only). Terminal.app and iTerm2 are read as terminals (Fakes only). Other terminals expose nothing.

The Accessibility permission belongs to the app "Heliox-OS". The daemon is the app's child process, and macOS attributes a child's requests to the responsible app (Not tested: no run of the built app on a Mac). A daemon started from a terminal during development is attributed to that terminal.

Heliox checks the permission without a prompt (Run for real: `macos/permissions-read-without-prompt`). When it is missing, Live's settings say what to turn on and where, and offer one button that shows the system prompt; that prompt appears at most once per run; Heliox re-checks every 10 s, so turning the permission on takes effect without a restart (all Fakes only: the runner has the grant). Rebuilding the app with an ad-hoc signature can drop the grant.

`macos_permissions.snapshot()` reads microphone, Accessibility, Screen Recording and Automation without prompting. Accessibility and Screen Recording can only say "granted" or "not granted". `microphone_state()` is the interface the audio side uses.

### Linux

The reader speaks AT-SPI directly over D-Bus with jeepney (`pip install 'pilot-daemon[linux]'`, included in `all`). jeepney is pure Python and is already pulled in by SecretStorage. PyGObject and pyatspi were not used because a pip install of them needs a compiler and the cairo and GObject headers.

The reader subscribes to `object:state-changed:focused` events, so each read is four or five calls to the focused application. When no event has arrived, it searches the active window for the focused element, visiting at most 250 objects and searching at most every 3 s.

What the person needs:

- The AT-SPI bus (`at-spi2-core`). Without it, Live says the bus is not running and what to install (Fakes only).
- For Qt, Chromium, Electron and Firefox apps, toolkit accessibility turned on: `gsettings set org.gnome.desktop.interface toolkit-accessibility true`, then restart those apps. Live says this when the session reports accessibility off (Fakes only).

GTK 3 apps: Run for real (the fixture under X11 and sway). GTK 4 apps: typed text Run for real under X11 (CI on 008f1a0, `gtk4/reads-typed-text`). A hung GTK app: `x11/read-of-a-hung-app-returns-within-3s`, real check new in this round, result to come.

Password fields. GTK 3, Qt, Chromium, Electron and Firefox give a password field the role "password text"; GTK 4 does not. It names its roles after ARIA in GetRoleName, so a password entry is a "text box" like any entry. GTK 4 marks the password only in the numeric role (GetRole), and on the GtkPasswordEntry, while focus sits on the text widget inside it. In CI on 008f1a0 the reader read a GTK 4 password entry's text as an ordinary field (`gtk4/password-refused` failed). The reader now asks the numeric role of the focused field and of its two parents, and refuses the field when any of them is a password, or when its own numeric role cannot be read. The press path refuses a password field the same way. The snapshot never reads any field's value, only its role and label. The fix is Fakes only until `gtk4/password-refused` passes under X11 and sway; that check prints DIAG lines with the real role names, numeric roles, states and attributes of the field and its parents. Qt, Chromium, Electron and Firefox: checked by reasoning only, from their documented mapping of password inputs to the "password text" role, which the reader has refused since it was written.

Terminals (all Not tested by a run of the terminal itself through AT-SPI):

- Readable: VTE terminals (GNOME Terminal, Tilix, Terminator, Xfce Terminal).
- Partly readable: Konsole.
- Not readable: Alacritty, kitty, WezTerm, foot and xterm, which expose nothing.

AT-SPI does not depend on the display server: under sway it read the same as under X11 (Run for real); GNOME's and KDE's compositors: Not tested. `xdotool` remains the fallback for the window title on X11 when an app is not on the accessibility bus (Fakes only until `windows-x11/foreground-falls-back-to-xdotool` has run).

## How this is tested

The pure decisions run in the existing `python` job on ubuntu, macOS and Windows: `tests/test_platform_readers.py` and `tests/test_platform_quiet_failures.py`. They cover the privacy decision, identity, change detection, the permission state machine, AT-SPI and Accessibility layers with fakes, the name parsers, notifications, the Linux voice fallback and path rules.

The `platform-readers` job in `.github/workflows/ci.yml` runs the real readers and is non-blocking (`continue-on-error`):

- **ubuntu-24.04:** a session bus, Xvfb, AT-SPI and a GTK 3 window (`tests/platform_real/gtk_fixture.py`). `tests/platform_real/linux_atspi_check.py` types with xdotool and checks five things, all passed in CI on 3793a0d:
  - the typed text is read;
  - the field keeps its identity while the window title changes;
  - the password field is refused;
  - the foreground title follows;
  - the median read cost.

  A second step repeats this in a Wayland session under sway's headless backend. A headless compositor has no keyboard, so no window gets keyboard focus and GTK marks nothing focused (CI showed seat0 with no capabilities and no focus events); the step holds a virtual keyboard open with wtype for the whole run, as every real machine has one, and then every check passes (median read 1.4 ms in CI on 3793a0d). This proves focus under wlroots; GNOME's and KDE's compositors: Not tested.
- **Launching, names and window actions** (ubuntu-24.04 under Xvfb with openbox; macos-latest): `tests/platform_real/apps_check.py` launches TextEdit by name and a test desktop entry with arguments, and reads app names from real bundles and entries; `tests/platform_real/linux_windows_check.py` lists, focuses, maximises, minimises and closes two GTK windows and checks the xdotool foreground fallback. New, result to come.
- **Wayland portals** (ubuntu-24.04, same job): a screenshot through xdg-desktop-portal under sway with xdg-desktop-portal-wlr (Run for real: it failed in CI on 3793a0d for want of grim, and passed in CI on 0a8edad with grim installed), the plain message when a desktop has no GlobalShortcuts portal (Run for real), and the GlobalShortcuts client against a fake portal served on a real session bus (Run for real).
- **macos-latest:** `tests/platform_real/macos_ax_check.py` checks that pyobjc imports and that every permission is read without a prompt. On the hosted runner Microphone, Accessibility and Screen Recording read "granted" and Automation reads "cannot_tell" (CI on 3793a0d), so it also reads a real AppKit window with a text field and a secure field (`macos_fixture.py`): typed text, the secure field refused, identity across a title change, and, with that window in front, the frontmost app named by LaunchServices and by the window list. Because the runner has the grants, the "not granted" path (no reads, the status names the setting, no prompt) is Fakes only.

Make the job required once it has passed on three consecutive pushes to main. Before then, a broken runner image cannot turn CI red.

## First run on macOS and Linux

What a brand-new install meets, in the order the person meets it, with no speech models, no permissions granted, no espeak and no AT-SPI. "Fixed" means the code now explains itself; the marker after each says how that is known.

macOS:

1. Fixed (see [FIRST_RUN.md](FIRST_RUN.md)): without Python 3.11 or later the app no longer runs Apple's `python3` stub; it offers a private Python (uv and python-build-standalone, size and source shown, checksum-checked) and shows any setup failure on screen with what to do. The private Python: Run for real (CI `first-run`, macOS, private). Never running the stub: Not tested (the runner always has a real Python).
2. Fixed (see [FIRST_RUN.md](FIRST_RUN.md)): nothing is installed until the person chooses; the core, then the groups they pick, each with size and source. The screen: Fakes only (vitest); the scripts: Run for real (CI `first-run`).
3. Microphone: the bundle declares why Heliox uses it, so macOS asks on first capture; `microphone_state()` lets the audio side say what to turn on (audio workstream). Not tested.
4. Fixed: Live's field reading names the Accessibility setting and offers the system prompt once. Fakes only.
5. Fixed: reading the screen without Screen Recording used to read the wallpaper and find nothing; it now says what to turn on. The first capture is what makes macOS show its prompt. Fakes only.
6. Fixed: window actions without Accessibility say what to turn on; focus works without it. Fakes only (the runner has the grant).
7. Fixed: the default shortcut no longer collides with Spotlight, and a failed registration is shown in Settings. The key names: Fakes only (vitest); the registration: Not tested.
8. Open, for the owner (it costs money and an Apple account): an unsigned app meets Gatekeeper; [FIRST_RUN.md](FIRST_RUN.md) says what the person has to do.

Linux:

1. Fixed for the .deb: `python3 -m venv` fails on Debian and Ubuntu without `python3-venv`, and microphone capture fails without `libportaudio2`; both are now package dependencies, with `at-spi2-core` (the .rpm gets `portaudio` and `at-spi2-core`). Not tested against a built package. The AppImage cannot declare dependencies, but setup now shows its failures in the app (Fakes only: vitest).
2. Fixed: the same chosen, visible setup as on macOS ([FIRST_RUN.md](FIRST_RUN.md)); a Python without `venv` is skipped in favour of a private one (Not tested: the runner's Python has `venv`).
3. Fixed: Live says when the accessibility bus is missing or toolkit accessibility is off. Fakes only.
4. Fixed: without espeak, speech logs one "no voice available" message and answers in text; whether the app shows that sentence is up to the audio workstream. Fakes only.
5. Fixed: the `notify` action says when `notify-send` is missing instead of claiming success. Fakes only.
6. Partly fixed: on Wayland, screenshots and the global shortcut go through the desktop's portals (where the desktop has them, and the desktop asks the person): the screenshot Run for real under sway (CI on 0a8edad), the shortcut Run for real only against a fake portal. The window tools (`wmctrl`, `xdotool`) are still X11-only and fail with an X11 hint.

## Gaps that remain

| Gap | Documented route | What the person grants | Estimate |
|---|---|---|---|
| Microphone permission on macOS before first capture | Owned by the audio workstream; it uses `microphone_state()` | Microphone | |
| A new default microphone with the same name as the old one is not followed | The capture watch compares `default_microphone_name()`, which on Linux is PulseAudio's description (two identical USB microphones share it); the source name, `default_microphone_key()`, would tell them apart. The speech workstream's | Nothing | |

## What only a person at a real machine can show

No hosted runner can show these; each needs someone at a real computer:

1. On a Mac without the grants, running the built Heliox-OS app: the microphone prompt names Heliox-OS on first use; Live's note names Accessibility, its button shows the system prompt once, and after the person turns Accessibility on the note goes within 10 s and typing in TextEdit raises cards; the Screen Recording prompt at the first screen capture, and Heliox's message before it; window focus without Accessibility.
2. On that Mac: a password field in Safari is never read, and fields in Safari, Chrome and an Electron app are.
3. On a Mac with a microphone: a real call (FaceTime or Zoom) makes Heliox quiet, and it speaks again after.
4. The global shortcut registered by the built app: Option+Space on a Mac, Ctrl+Space on an X11 desktop, and a taken shortcut shown as failed in Settings.
5. A real GNOME 48 or later, or KDE Plasma 5.27 or later, desktop: the shortcut portal's confirmation dialog, and the screenshot portal's consent dialog with Heliox's message when the person says no.
6. Two audio devices: the default microphone or speakers changed while Heliox listens or speaks, a USB microphone unplugged and plugged back, and a Bluetooth headset in call mode flagged with the switch offered.
7. The unsigned-build dialogs and the app driving setup, repair and removal in its own window: see the same list in [FIRST_RUN.md](FIRST_RUN.md).
