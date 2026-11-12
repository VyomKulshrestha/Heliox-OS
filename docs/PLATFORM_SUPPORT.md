# Platform support

Heliox runs on Windows, macOS and Linux. This page says what each part of the companion does on each system, where the code is, what the person has to grant or install, and how that claim was tested. Nobody on the project develops on a Mac or a Linux desktop, so the last column matters: "real run" means the feature ran against the real system interface, "CI run" means it ran on a GitHub-hosted runner in the `platform-readers` job, and "fakes" means only the decision logic was tested, with the platform layer replaced.

"Works" means the feature behaves as designed. "Partly" means it works in some setups, with the limits listed. "No-op" means it silently does nothing. "Fails" means it raises or shows an error.

| Feature | Windows | macOS | Linux | Code | Grant or install | Tested how |
|---|---|---|---|---|---|---|
| Reading the focused text field (Live) | Works: UI Automation through a long-lived PowerShell helper, 5 to 10 ms a read | Works: Accessibility API in process (pyobjc), 2 to 10 ms a read | Works for apps on the accessibility bus, under X11 and Wayland: AT-SPI over D-Bus (jeepney), 1 to 5 ms a read | `daemon/pilot/live/focused_text*.py` | macOS: Accessibility for Heliox-OS; the app explains it and offers the system prompt once. Linux: `at-spi2-core` (shipped by GNOME, KDE and most desktops); Qt, Chromium, Electron and Firefox apps need toolkit accessibility on | Windows: real run. macOS: CI run against the real Accessibility API (the hosted runner grants Accessibility): typed text read, secure field refused, identity kept across a title change, median 0.6 ms; the "permission missing" path with fakes only, since the runner has it granted. Linux X11: CI run (Xvfb, GTK 3, real keystrokes): all checks pass, median 0.7 ms. Linux Wayland: CI run under sway (wlroots) with a virtual keyboard held open: typed text, password refused, identity across a title change, median 1.0 ms. GNOME's and KDE's compositors are not proven |
| Foreground app and window (timeline, Live) | Works: `win32gui` | Works: Accessibility API for app and title; without the permission, the app from LaunchServices (`lsappinfo`, the app owning the menu bar), then the Quartz window list, and no title. AppleScript only when pyobjc is missing | Works: AT-SPI focus (X11 and Wayland), then `xdotool` (X11) | `daemon/pilot/agents/screen_vision.py`, `focused_text_macos.py`, `focused_text_linux.py` | As above | Windows: real run. macOS: CI run for the Accessibility route; the no-permission route (LaunchServices, then the window list) is checked with a test window in front from the next run on. Linux: CI run on X11 (title change followed) |
| Application display names | Works: file description | Works: bundle display name | Works: desktop entry `Name` | `daemon/pilot/system/app_names.py` | Nothing | Windows: real run. macOS and Linux: fakes (pure parsers over a path or file text) |
| Activity timeline | Works | Works as far as the foreground row | Works as far as the foreground row | `daemon/pilot/workflows/activity_timeline.py` | Nothing more | Fakes on all three, plus the foreground row |
| macOS permissions status | Not applicable | Works: microphone, Accessibility, Screen Recording and Automation read without a prompt | Not applicable | `daemon/pilot/system/macos_permissions.py` | Nothing | macOS: CI run (states read, no prompt); messages with fakes |
| Reading the screen (OCR) | Works: Windows OCR | Works with the `macos` extra: the Vision framework in process (pyobjc), no developer tools needed; without pyobjc, Vision through `swift` (needs the Xcode command-line tools), else tesseract or EasyOCR. Without Screen Recording the capture shows only the wallpaper, reported by the permission status | Partly: tesseract, EasyOCR, pytesseract; capture by `pyautogui`, `gnome-screenshot` or `scrot` on X11, and through the desktop's Screenshot portal on Wayland (after a "no", not asked again that run) | `daemon/pilot/system/vision.py`, `daemon/pilot/system/macos_vision_ocr.py`, `daemon/pilot/system/screen.py` | macOS Screen Recording. Linux: tesseract | Windows: real run. macOS: CI run on a real Mac: Vision read a rendered line of text in process and through vision.py; reading order with fakes. Linux: fakes for OCR; the Wayland portal screenshot is a CI run under sway with xdg-desktop-portal-wlr (GNOME's and KDE's consent dialogs not proven) |
| Call detection (quiet in calls; `current_call()` for the listener and Live) | Works: the microphone-users store, meeting processes (Zoom's `CptHost.exe`) and call-window titles (Meet, Zoom, Teams, WhatsApp, Slack, Webex, Discord) | Works on macOS 14 and later: CoreAudio's list of processes recording input (through ctypes), Heliox's own processes excluded. Partly on older macOS: a busy default input counts only when Heliox is not listening and a call app is open | Works with PulseAudio or PipeWire (pipewire-pulse): the recording streams from `pactl list source-outputs`, recordings of the output and Heliox's own PIDs excluded. Without `pactl` it cannot tell and stays "not in a call" | `daemon/pilot/system/call_detection.py` | Nothing on any system (no permission, no extra package); Linux: `pulseaudio-utils` or `pipewire-pulse` for `pactl` | Windows: real run. macOS: fakes, plus the real reader on the CI runner (must answer without error). Linux: fakes, plus a CI run in `platform-readers`: a null source with `parecord` recording from it is seen as a call |
| Speech capture and playback | Works: capture through WASAPI (`soundcard`), then PortAudio; playback through one PortAudio output line kept open between sentences | Partly: PortAudio (Core Audio) both ways. A refused or not-yet-asked microphone permission is said at once (`microphone_state()`); the bundle declares the microphone use | Partly: PortAudio over ALSA, PulseAudio or PipeWire; a chosen PulseAudio source is opened through the ALSA `pulse` device with `PULSE_SOURCE`; needs `libportaudio2`, and `libasound2-plugins` for the `pulse` device | `daemon/pilot/system/voice.py`, `daemon/pilot/system/audio_health.py` | macOS: Microphone. Linux: `libportaudio2` | Windows: real run. Linux: CI run in `platform-readers` (soundcard names the default source by its source name; PortAudio's `pulse` device opens the source `PULSE_SOURCE` names). macOS: fakes |
| Microphone and speaker recovery | Works: a failed, stalled or silent stream is reopened (backing off to once every 30 s); the default microphone and speakers are followed live | Works the same way; default changes read through CoreAudio (`soundcard`); PortAudio's device list is re-read on reopen | Works the same way; default changes read through PulseAudio or PipeWire (`soundcard`); PortAudio's device list is re-read on reopen | `daemon/pilot/system/audio_health.py`, `daemon/pilot/system/voice.py` | Nothing | Fakes on all three, including the 0x88890004 "device invalidated" case; the Windows reopen path against real hardware is still owed (steps in device_check_steps) |
| Microphone kind (Bluetooth call mode, very quiet, clipping) | Works: judged from the sound (nothing above 4 kHz is call mode), said once in the voice bar with a one-button switch to another microphone, remembered per headset by its WASAPI name | Works: same judgement from the sound; the switch is remembered by the Core Audio device name (the name PortAudio also uses; CoreAudio's numeric ids change on replug) | Works: same judgement; the switch is remembered by the PulseAudio/PipeWire source name (stable), and the chosen source is opened through `PULSE_SOURCE` | `daemon/pilot/system/mic_quality.py`, `daemon/pilot/system/voice.py` | Nothing | The judgement on the owner's real recordings (headset in call mode flagged after two utterances; desk microphone never). The offer and the per-headset choice with fakes on all three; Linux names and opening through `PULSE_SOURCE`: CI run in `platform-readers` |
| Wake phrase | Works | As far as capture works | As far as capture works | `daemon/pilot/system/voice.py` | As for capture | Fakes off Windows |
| Speaking | Works: Kokoro or Pocket TTS, then System.Speech | Works: same local engines, then `say` | Works: same local engines, then `espeak-ng` or `espeak`; with neither, Heliox says once in the log that no voice is available and carries on in text | `daemon/pilot/system/voice.py`, `daemon/pilot/system/linux_voice.py` | Linux: `espeak-ng` for the fallback voice | Fakes off Windows |
| Notifications and cards | Works: in-app cards; OS toasts through the Tauri plugin; the `notify` action shows a balloon | Partly: cards work; toasts may not appear for an unsigned app; the `notify` action now reports failure instead of "sent" | Partly: cards work; toasts need a notification daemon; `notify` reports a missing `notify-send` | `daemon/pilot/system/notify.py`, `daemon/pilot/agents/executor.py` | Notification permission (asked by the app) | Fakes for command and result; Windows: real run before this change |
| Global hotkey | Works: Ctrl+Space | Works: new default Option+Space (Cmd+Space is Spotlight's); Settings shows shortcuts in the Mac's key names (Option, Command, Control) and accepts either spelling; a failed registration is shown in Settings | Works on X11 (Ctrl+Space). On Wayland the daemon binds it through the GlobalShortcuts portal (GNOME 48+, KDE Plasma 5.27+, Hyprland), where the desktop confirms the keys; elsewhere on Wayland (sway, older GNOME, Xfce) Settings says the desktop has no such portal | `tauri-app/src-tauri/src/hotkey.rs` | Nothing | Rust compiled and linted by the CI `rust` job; key names and both spellings with vitest; registration not exercised on macOS or Linux; the Wayland portal client against a fake portal on a real session bus in CI, real desktops unproven |
| Window actions (list, focus, minimise, maximise, close) | Works: user32 and UI Automation | Works with the `macos` extra: Accessibility API (everything but focus needs Accessibility); focus tries activation, then AXFrontmost and AXRaise, then LaunchServices, checking the front app by pid after each; close closes that window, not the app. Without pyobjc, AppleScript through System Events | Partly: `wmctrl` and `xdotool`, X11 only | `daemon/pilot/system/window_mgr.py`, `daemon/pilot/system/macos_windows.py` | macOS: Accessibility | Windows: real run. macOS: CI run on a real AppKit window: list, maximise, minimise and close (which closes the window only) pass; focus passes for an app already in front and for another app (Finder). On the hosted Mac the activation request was accepted and ignored, AXFrontmost did not take either, and LaunchServices (`open -b`) was the route that brought Finder forward (`last_route` in the log). Linux: fakes |
| Undo and protected folders | Works | Works: folder checks now ignore case, as APFS does | Works: case-sensitive, as Linux file systems are | `daemon/pilot/security/path_rules.py`, `daemon/pilot/security/validator.py`, `daemon/pilot/workflows/undo_journal.py` | Nothing | Both rule sets with fakes on every system |
| Launching apps (simple actions) | Works: Start menu shortcuts, PATH, Store apps | Works: installed bundles matched by file name and the bundle's own name, then `open -a`; arguments after `--args` | Works: a path, PATH, then desktop entries by shown name or id, started from their `Exec` line with arguments in place of `%f %F %u %U`; terminal apps open in the system terminal; ambiguous names fail closed | `daemon/pilot/system/applications.py`, `daemon/pilot/system/desktop_entry.py` | Nothing | Windows: real run. macOS and Linux: fakes (Exec parsing, matching, the command started) |

## Reading the focused field

Every platform reader satisfies `FocusedTextSource` (`daemon/pilot/live/focused_text.py`): `read()`, `status()`, `close()` and `available`. `create_reader()` picks the one for the system. Each reader turns what its accessibility interface says about the focused element into `ElementFacts`, and `focused_text_policy.admit` makes the one decision for all three: text passes only from a writable field or a terminal, never from a secure field, and only the last 6,000 characters. The window-level rule, `never_read`, refuses Heliox's own window, password managers, the system credential stores (Keychain Access, Passwords, Seahorse, KDE Wallet) and apps the person excluded from the timeline before any reader runs.

Field identity has to stay the same while the person types and the window title changes:

- Windows: the UI Automation runtime id.
- macOS: the process id plus `CFHash` of the accessibility element, which hashes what the element refers to.
- Linux: the application's bus name plus the element's object path.

The tests check the property: the same field keeps its identity across typing and a title change, and a different field gets a different one.

### macOS

The reader uses pyobjc (`pip install 'pilot-daemon[macos]'`, included in `all`). pyobjc is Apple-platform-only, maintained, and ships prebuilt wheels, so nothing is compiled on the person's machine. It reads `AXFocusedUIElement` from the system-wide element with a 0.25 s messaging timeout, so a hung app cannot stall Live. It refuses a field before reading its value when the subrole is `AXSecureTextField` or when secure keyboard entry is on. Terminal.app and iTerm2 are read as terminals. Other terminals expose nothing.

The Accessibility permission belongs to the app "Heliox-OS". The daemon is the app's child process, and macOS attributes a child's requests to the responsible app. A daemon started from a terminal during development is attributed to that terminal.

Heliox checks the permission without a prompt. When it is missing, Live's settings say what to turn on and where, and offer one button that shows the system prompt. That prompt appears at most once per run. Heliox re-checks every 10 s, so turning the permission on takes effect without a restart. Rebuilding the app with an ad-hoc signature can drop the grant.

`macos_permissions.snapshot()` reads microphone, Accessibility, Screen Recording and Automation without prompting. Accessibility and Screen Recording can only say "granted" or "not granted". `microphone_state()` is the interface the audio side uses.

What a Mac user should check first, because CI cannot:

1. The microphone prompt names Heliox-OS on first use.
2. After turning on Accessibility for Heliox-OS, Live's note disappears within 10 s and typing in TextEdit raises cards.
3. A password field in Safari is never read.

### Linux

The reader speaks AT-SPI directly over D-Bus with jeepney (`pip install 'pilot-daemon[linux]'`, included in `all`). jeepney is pure Python and is already pulled in by SecretStorage. PyGObject and pyatspi were not used because a pip install of them needs a compiler and the cairo and GObject headers.

The reader subscribes to `object:state-changed:focused` events, so each read is four or five calls to the focused application. When no event has arrived, it searches the active window for the focused element, visiting at most 250 objects and searching at most every 3 s.

What the person needs:

- The AT-SPI bus (`at-spi2-core`). Without it, Live says the bus is not running and what to install.
- For Qt, Chromium, Electron and Firefox apps, toolkit accessibility turned on: `gsettings set org.gnome.desktop.interface toolkit-accessibility true`, then restart those apps. Live says this when the session reports accessibility off.

GTK 3 and 4 apps always work.

Terminals:

- Readable: VTE terminals (GNOME Terminal, Tilix, Terminator, Xfce Terminal).
- Partly readable: Konsole.
- Not readable: Alacritty, kitty, WezTerm, foot and xterm, which expose nothing.

AT-SPI does not depend on the display server, so Wayland sessions read the same as X11. `xdotool` remains the fallback for the window title on X11 when an app is not on the accessibility bus.

## How this is tested

The pure decisions run in the existing `python` job on ubuntu, macOS and Windows: `tests/test_platform_readers.py` and `tests/test_platform_quiet_failures.py`. They cover the privacy decision, identity, change detection, the permission state machine, AT-SPI and Accessibility layers with fakes, the name parsers, notifications, the Linux voice fallback and path rules.

The `platform-readers` job in `.github/workflows/ci.yml` runs the real readers and is non-blocking (`continue-on-error`):

- **ubuntu-24.04:** a session bus, Xvfb, AT-SPI and a GTK 3 window (`tests/platform_real/gtk_fixture.py`). `tests/platform_real/linux_atspi_check.py` types with xdotool and checks five things:
  - the typed text is read;
  - the field keeps its identity while the window title changes;
  - the password field is refused;
  - the foreground title follows;
  - the median read cost.

  A second step repeats this in a Wayland session under sway's headless backend. A headless compositor has no keyboard, so no window gets keyboard focus and GTK marks nothing focused (CI showed seat0 with no capabilities and no focus events); the step holds a virtual keyboard open with wtype for the whole run, as every real machine has one, and then every check passes (median read 1.0 ms). This proves focus under wlroots; GNOME's and KDE's compositors are not proven.
- **Wayland portals** (ubuntu-24.04, same job): a real screenshot through xdg-desktop-portal under sway with xdg-desktop-portal-wlr, the plain message when a desktop has no GlobalShortcuts portal, and the GlobalShortcuts client against a fake portal served on a real session bus.
- **macos-latest:** `tests/platform_real/macos_ax_check.py` checks that pyobjc imports and that every permission is read without a prompt. The hosted runner grants Accessibility, Microphone and Screen Recording to the job, so it also reads a real AppKit window with a text field and a secure field (`macos_fixture.py`): typed text, the secure field refused, identity across a title change, and, with that window in front, the frontmost app named by LaunchServices and by the window list without any permission. Because the runner has the permission, the "not granted" path (no reads, the status names the setting, no prompt) is covered only by fakes.

Make the job required once it has passed on three consecutive pushes to main. Before then, a broken runner image cannot turn CI red.

## First run on macOS and Linux

What a brand-new install meets, in the order the person meets it, with no speech models, no permissions granted, no espeak and no AT-SPI. "Fixed" items now explain themselves; the rest are open.

macOS:

1. Fixed (see [FIRST_RUN.md](FIRST_RUN.md)): without Python 3.11 or later the app no longer runs Apple's `python3` stub; it offers a private Python (uv and python-build-standalone, size and source shown, checksum-checked) and shows any setup failure on screen with what to do.
2. Fixed (see [FIRST_RUN.md](FIRST_RUN.md)): nothing is installed until the person chooses; the core, then the groups they pick, each with size and source.
3. Microphone: the bundle declares why Heliox uses it, so macOS asks on first capture; `microphone_state()` lets the audio side say what to turn on (audio workstream).
4. Fixed: Live's field reading names the Accessibility setting and offers the system prompt once.
5. Fixed: reading the screen without Screen Recording used to read the wallpaper and find nothing; it now says what to turn on. The first capture is what makes macOS show its prompt.
6. Fixed: window actions without Accessibility say what to turn on; focus works without it.
7. Fixed: the default shortcut no longer collides with Spotlight, and a failed registration is shown in Settings.
8. Open, for the owner (it costs money and an Apple account): an unsigned app meets Gatekeeper; [FIRST_RUN.md](FIRST_RUN.md) says exactly what the person has to do.

Linux:

1. Fixed for the .deb: `python3 -m venv` fails on Debian and Ubuntu without `python3-venv`, and microphone capture fails without `libportaudio2`; both are now package dependencies, with `at-spi2-core` (the .rpm gets `portaudio` and `at-spi2-core`). The AppImage cannot declare dependencies, but setup now shows its failures in the app.
2. Fixed: the same chosen, visible setup as on macOS ([FIRST_RUN.md](FIRST_RUN.md)); a Python without `venv` is skipped in favour of a private one.
3. Fixed: Live says when the accessibility bus is missing or toolkit accessibility is off.
4. Fixed: without espeak, speech logs one "no voice available" message and answers in text; whether the app shows that sentence is up to the audio workstream.
5. Fixed: the `notify` action says when `notify-send` is missing instead of claiming success.
6. Partly fixed: on Wayland, screenshots and the global shortcut now go through the desktop's portals (where the desktop has them, and the desktop asks the person). The window tools (`wmctrl`, `xdotool`) are still X11-only and fail with an X11 hint.

## Gaps that remain

| Gap | Documented route | What the person grants | Estimate |
|---|---|---|---|
| Microphone permission on macOS before first capture | Owned by the audio workstream; it uses `microphone_state()` | Microphone | |
