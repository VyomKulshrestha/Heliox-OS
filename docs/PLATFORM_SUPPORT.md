# Platform support

Heliox runs on Windows, macOS and Linux. This page says what each part of the companion does on each system, where the code is, what the person has to grant or install, and how that claim was tested. Nobody on the project develops on a Mac or a Linux desktop, so the last column matters: "real run" means the feature ran against the real system interface, "CI run" means it ran on a GitHub-hosted runner in the `platform-readers` job, and "fakes" means only the decision logic was tested, with the platform layer replaced.

"Works" means the feature behaves as designed. "Partly" means it works in some setups, with the limits listed. "No-op" means it silently does nothing. "Fails" means it raises or shows an error.

| Feature | Windows | macOS | Linux | Code | Grant or install | Tested how |
|---|---|---|---|---|---|---|
| Reading the focused text field (Live) | Works: UI Automation through a long-lived PowerShell helper, 5 to 10 ms a read | Works: Accessibility API in process (pyobjc), 2 to 10 ms a read | Works for apps on the accessibility bus, under X11 and Wayland: AT-SPI over D-Bus (jeepney), 1 to 5 ms a read | `daemon/pilot/live/focused_text*.py` | macOS: Accessibility for Heliox-OS; the app explains it and offers the system prompt once. Linux: `at-spi2-core` (shipped by GNOME, KDE and most desktops); Qt, Chromium, Electron and Firefox apps need toolkit accessibility on | Windows: real run. macOS: CI run against the real Accessibility API (the hosted runner grants Accessibility): typed text read, secure field refused, identity kept across a title change, median 0.6 ms; the "permission missing" path with fakes only, since the runner has it granted. Linux X11: CI run (Xvfb, GTK 3, real keystrokes): all checks pass, median 0.7 ms. Linux Wayland: not proven yet; Weston headless has no seat, so the step now uses sway headless and wtype |
| Foreground app and window (timeline, Live) | Works: `win32gui` | Works: Accessibility API for app and title; without the permission, the app from LaunchServices (`lsappinfo`, the app owning the menu bar), then the Quartz window list, and no title. AppleScript only when pyobjc is missing | Works: AT-SPI focus (X11 and Wayland), then `xdotool` (X11) | `daemon/pilot/agents/screen_vision.py`, `focused_text_macos.py`, `focused_text_linux.py` | As above | Windows: real run. macOS: CI run for the Accessibility route; the no-permission route (LaunchServices, then the window list) is checked with a test window in front from the next run on. Linux: CI run on X11 (title change followed) |
| Application display names | Works: file description | Works: bundle display name | Works: desktop entry `Name` | `daemon/pilot/system/app_names.py` | Nothing | Windows: real run. macOS and Linux: fakes (pure parsers over a path or file text) |
| Activity timeline | Works | Works as far as the foreground row | Works as far as the foreground row | `daemon/pilot/workflows/activity_timeline.py` | Nothing more | Fakes on all three, plus the foreground row |
| macOS permissions status | Not applicable | Works: microphone, Accessibility, Screen Recording and Automation read without a prompt | Not applicable | `daemon/pilot/system/macos_permissions.py` | Nothing | macOS: CI run (states read, no prompt); messages with fakes |
| Reading the screen (OCR) | Works: Windows OCR | Works with the `macos` extra: the Vision framework in process (pyobjc), no developer tools needed; without pyobjc, Vision through `swift` (needs the Xcode command-line tools), else tesseract or EasyOCR. Without Screen Recording the capture shows only the wallpaper, reported by the permission status | Partly: tesseract, EasyOCR, pytesseract; capture by `pyautogui`, `gnome-screenshot` or `scrot`; Wayland not handled | `daemon/pilot/system/vision.py`, `daemon/pilot/system/macos_vision_ocr.py`, `daemon/pilot/system/screen.py` | macOS Screen Recording. Linux: tesseract | Windows: real run. macOS: CI run on a real Mac: Vision read a rendered line of text in process and through vision.py; reading order with fakes. Linux: fakes |
| Call detection | Works | No-op (being built in another workstream) | No-op (same) | `daemon/pilot/system/call_detection.py` | None on Windows | Fakes |
| Speech capture and playback | Works: WASAPI, then PortAudio | Partly: PortAudio; the bundle now declares the microphone use, and `microphone_state()` tells the audio side whether it was granted | Partly: PortAudio over ALSA, PulseAudio or PipeWire; needs `libportaudio2` | `daemon/pilot/system/voice.py`, `daemon/pilot/system/audio_health.py` | macOS Microphone (asked by the system on first use). Linux: PortAudio | Fakes off Windows |
| Wake phrase | Works | As far as capture works | As far as capture works | `daemon/pilot/system/voice.py` | As for capture | Fakes off Windows |
| Speaking | Works: Kokoro or Pocket TTS, then System.Speech | Works: same local engines, then `say` | Works: same local engines, then `espeak-ng` or `espeak`; with neither, Heliox says once in the log that no voice is available and carries on in text | `daemon/pilot/system/voice.py`, `daemon/pilot/system/linux_voice.py` | Linux: `espeak-ng` for the fallback voice | Fakes off Windows |
| Notifications and cards | Works: in-app cards; OS toasts through the Tauri plugin; the `notify` action shows a balloon | Partly: cards work; toasts may not appear for an unsigned app; the `notify` action now reports failure instead of "sent" | Partly: cards work; toasts need a notification daemon; `notify` reports a missing `notify-send` | `daemon/pilot/system/notify.py`, `daemon/pilot/agents/executor.py` | Notification permission (asked by the app) | Fakes for command and result; Windows: real run before this change |
| Global hotkey | Works: Ctrl+Space | Works: new default Option+Space (Cmd+Space is Spotlight's); Settings shows shortcuts in the Mac's key names (Option, Command, Control) and accepts either spelling; a failed registration is shown in Settings | Partly: Ctrl+Space works on X11; Wayland has no global shortcuts for apps | `tauri-app/src-tauri/src/hotkey.rs` | Nothing | Rust compiled and linted by the CI `rust` job; key names and both spellings with vitest; registration not exercised on macOS or Linux |
| Window actions (list, focus, minimise, maximise, close) | Works: user32 and UI Automation | Works with the `macos` extra: Accessibility API (focus needs no permission; the rest need Accessibility); close closes that window, not the app. Without pyobjc, AppleScript through System Events | Partly: `wmctrl` and `xdotool`, X11 only | `daemon/pilot/system/window_mgr.py`, `daemon/pilot/system/macos_windows.py` | macOS: Accessibility | Windows: real run. macOS: CI run on a real AppKit window; fakes for the rest. Linux: fakes |
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

  A second step repeats this in a Wayland session under sway's headless backend, typing with wtype. The first attempt used Weston's headless backend, which has no seat: GTK logged `gdk_seat_get_keyboard` assertions and no element ever took focus, so focus under Wayland is not yet proven by CI. AT-SPI itself is the same D-Bus protocol under both, and the reader has no display-server code.
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
6. Open: on Wayland the global shortcut and the X11 window tools (`wmctrl`, `xdotool`) do nothing or fail with an X11-only hint, and screen capture falls back to `gnome-screenshot`; see the gaps below.

## Gaps that remain

| Gap | Documented route | What the person grants | Estimate |
|---|---|---|---|
| Screen capture and OCR on Wayland | `org.freedesktop.portal.Screenshot` (one image, consent per call or a remembered grant on GNOME 46 and later) or `org.freedesktop.portal.ScreenCast` with PipeWire for repeated capture, over D-Bus with jeepney | A portal consent dialog, once per session for ScreenCast with `persist_mode` | 3 to 4 days, with a CI run under Weston plus `xdg-desktop-portal` with a stub backend |
| Global hotkey on Wayland | `org.freedesktop.portal.GlobalShortcuts` (GNOME 48 and later, KDE Plasma 5.27 and later); elsewhere ask the person to bind a desktop shortcut to `heliox --toggle` | The portal's confirmation dialog | 2 to 3 days, Rust side |
| Call detection on macOS and Linux | Owned by another workstream | | |
| Microphone permission on macOS before first capture | Owned by the audio workstream; it uses `microphone_state()` | Microphone | |
