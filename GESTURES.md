# 🤚 Heliox OS — Gesture System v3 (30+ Gestures)

Heliox OS includes a state-of-the-art webcam-based hand gesture recognition engine powered by [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html). It supports **30+ gestures** including both static poses and real-time motion tracking.

## Architecture

```
Webcam Feed
    │
    ▼
MediaPipe Hands (21 landmarks per hand)
    │
    ▼
Spatial/World-Model Layer (spatialModel.ts)
    │   ├─ One Euro filter — temporal smoothing of landmark positions
    │   │  (feeds static-pose classification; motion buffers below stay raw)
    │   └─ Hand quality score — MediaPipe's own detection confidence ×
    │      geometric self-consistency check (catches occluded/edge-on poses)
    │
    ├──► classifyGesture() ──► Static Pose Detection
    │                              (finger extension patterns,
    │                               tip distances, orientation-invariant
    │                               thumb-extension check)
    │
    ├──► detectCircularMotion() ──► Circular Gesture Detection
    │                                  (cross product analysis on
    │                                   12-point index finger buffer, raw landmarks)
    │
    └──► detectPushPull() ──► Z-Axis Depth Detection
                                (8-point wrist Z-history, raw landmarks)
    │
    ▼
Quality Gate (confidence × hand-quality score; low-quality frames don't
              advance the frame stabilizer below)
    │
    ▼
Frame Stabilizer (5 consecutive identical frames required)
    │
    ▼
Cooldown Gate (1200ms between triggers)
    │
    ▼
executeGestureAction() ──► session.sendCommand / session.confirm
    │
    ▼
UI Feedback (emoji badge, particle burst, gesture history)
```

## Debouncing & Stability

The gesture engine uses a **4-layer anti-jitter system** to prevent false triggers:

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Temporal Filtering** | One Euro filter smooths landmark positions before classification | Reduces single-frame jitter without adding perceptible lag |
| **Quality Gate** | Confidence scaled by detection + geometric quality; sub-threshold frames don't advance the stabilizer | Prevents a degenerate/occluded pose from misfiring |
| **Frame Stabilizer** | Gesture must be detected for 5 consecutive frames | Eliminates transition noise |
| **Cooldown Gate** | 1200ms lockout after each trigger | Prevents double-fires |
| **Buffer Clearing** | Motion buffers reset after circular/push gestures | Prevents re-triggering |

## Spatial/World-Model Layer

`lib/gesture/spatialModel.ts` sits between raw MediaPipe landmark output and
the classifiers above. It's pure TypeScript with no MediaPipe/DOM dependency
(and has its own unit tests — see `spatialModel.test.ts`), covering three
things:

1. **Temporal filtering** — a One Euro filter (adaptive: heavy smoothing at
   rest, low lag during fast motion) smooths landmark positions before static-
   pose classification. Swipe/circular/push-pull motion buffers deliberately
   stay on **raw** (unfiltered) landmarks, since their thresholds are already
   tuned against raw jitter and filtering would risk damping the fast motion
   they're built to detect.
2. **Orientation/handedness-invariant thumb detection** — replaces the old
   `thumb_tip.x < thumb_ip.x` check (which silently assumed a right hand
   facing the camera and misclassified left hands or a rotated wrist) with a
   hand-size-normalized distance ratio from the thumb tip to the index MCP.
   Tucked-against-the-palm vs. extended-out reads the same regardless of
   which hand or which way it's rotated.
3. **Hand quality scoring** — combines MediaPipe's own per-hand detection
   confidence with a geometric self-consistency check (do finger tip-to-MCP
   distances fall in a plausible range) into a single score that scales down
   reported gesture confidence, so a degenerate/occluded/edge-on hand pose
   gets suppressed rather than misfiring at full confidence.

This intentionally does **not** re-express every `classifyGesture()`
threshold in a fully hand-local, scale-normalized coordinate frame — that
would touch ~20 empirically-tuned distance constants with no way to validate
the recalibration against real camera input outside a live testing session,
and risks silently breaking gestures that work today. If you want to take
that further, add a recorded-landmark-sequence regression fixture per gesture
first so the recalibration can actually be verified.

---

## Gesture Cursor Control (continuous cursor bridge)

A separate, **off-by-default** mode that continuously drives the real OS
mouse cursor from hand position, rather than firing discrete named gestures.
Enable it in Settings → Gesture Cursor Control (`gesture_cursor.enabled`),
then toggle "Cursor Mode" in the gesture panel while the engine is running —
it is never enabled by a gesture itself.

**While active:**
- The index fingertip's filtered position, blended with its predicted
  near-future position (`prediction_ms`/`blend` settings — see "Prediction"
  below), drives the OS cursor.
- **Pinch fires a click** — evaluated against the *predicted* pinch distance,
  so the click registers a little before the pinch pose has fully, stably
  closed. This is the concrete case of "fire before a gesture completes."
- Every other discrete gesture (all 31 above) is **suppressed** — reaching
  for a swipe or a thumbs-up while pointing would otherwise misfire
  constantly against the cursor-tracking logic.
- **Open palm exits cursor mode immediately** (its existing "Cancel/Stop"
  meaning), checked first every frame, before anything else. The gesture
  panel's stop button and the "Cursor Mode" toggle button both work too.

**Coordinate mapping:** the camera video is mirrored (`scaleX(-1)`) for
natural selfie-view display, but MediaPipe processes the raw, unmirrored
frame — the x coordinate is flipped when mapping to screen pixels so moving
your hand right visually moves the cursor right, matching what you see on
screen rather than the raw camera data.

**Where cursor movement actually comes from:** a native Rust Tauri command
(`move_gesture_cursor`/`click_gesture_cursor` in
`tauri-app/src-tauri/src/commands.rs`, using the `enigo` crate) — not the
Python daemon. The daemon's existing `mouse_move` defaults to a 300ms
pyautogui tween plus a 50ms pause after every call, and the Tauri↔daemon
bridge opens a fresh WebSocket connection per invocation — both make a
sustained ~30fps cursor stream impractical. `enigo` runs in-process instead.
A `cursor_move`/`cursor_click` daemon RPC pair still exists as a degraded
fallback for testing the wiring in a plain browser dev session without a
compiled Tauri binary — see IPC_MESSAGE_FORMATS.md.

**Prediction**: reuses the same `predictAhead()` kinematic extrapolation
described above — a constant-velocity estimate, not a generative model.
`gesture_cursor.blend` (default `0.3`) controls how much of the predicted
vs. current position feeds the cursor; kept modest by default because the
predictor's velocity estimate is empirically amplified for a sustained
motion (see `spatialModel.test.ts`), so a larger blend risks visible
overshoot until this is tuned against real camera data.

**Safety notes:** this is the first gesture feature that continuously drives
real OS input without a per-action confirmation gate, so treat the escape
hatches above as load-bearing, not optional — cursor coordinates are also
clamped to screen bounds before every move, and `Enigo::new()` failing (e.g.
no display session) degrades to a clear error rather than a crash.

---

## Gesture Calibration (on-device continual learning)

A **lightweight, on-by-default** personalization loop that nudges exactly
two of the ~20 hardcoded static-pose constants —
`PINCH_DISTANCE_THRESHOLD` (Pinch/OK) and `THUMB_EXTENDED_RATIO`
(Thumbs Up/Down) — toward an individual user's hand anatomy and pinch
style. These two were chosen because they depend on the hand itself, unlike
e.g. swipe velocity or cooldown timing, which are motion-preference/safety
knobs left untouched.

This is **not** model retraining — MediaPipe stays frozen — and there is no
new "was this right? 👍/👎" prompt. Instead it reads an implicit signal
already latent in normal usage: **gesture reversal detection**. If a
calibrated gesture (pinch/OK/thumbs-up/thumbs-down) fires and is
immediately followed by a semantically contradictory gesture (e.g. an open
palm right after a pinch) within `REVERSAL_WINDOW_MS` (2500ms), that's read
as an implicit misfire and simply isn't reinforced. If nothing contradicts
it in time, the gesture's measured metric (pinch distance / thumb ratio at
the moment it fired) feeds an exponential moving average (`EMA_ALPHA =
0.08`).

The learned EMA only takes effect once at least `MIN_SAMPLES_TO_APPLY`
(8) confirmed observations exist, and is always clamped to
`[0.6x, 1.4x]` of the shipped default — so a single unusual session can't
swing recognition, and drift stays bounded and reversible. Below the sample
floor, the shipped constant is used unchanged.

**Storage:** exclusively browser `localStorage`
(`heliox_gesture_calibration`) — gesture recognition never leaves the
frontend today, so this personalization never needs to either. Nothing is
transmitted anywhere. Visible and resettable from Settings → Gesture
Calibration (`adaptive_calibration.gesture_enabled`), which also shows the
current pinch/thumb sample counts.

See `tauri-app/ui/src/lib/gesture/calibration.ts` for the implementation
and `calibration.test.ts` for coverage of the EMA math, clamping, and
reversal-pairing logic.

---

## Static Pose Gestures (21)

These are recognized by analyzing which fingers are extended, curled, or touching.

| # | Gesture | Emoji | How To Do It | System Action |
|---|---------|-------|--------------|---------------|
| 1 | **Open Palm** | ✋ | All fingers and thumb extended | Cancel / Stop current task |
| 2 | **Thumbs Up** | 👍 | Only thumb up, fist closed | Confirm AI plan |
| 3 | **Thumbs Down** | 👎 | Only thumb down, fist closed | Deny / Reject AI plan |
| 4 | **Peace Sign** | ✌️ | Index + middle finger up | Toggle voice mode |
| 5 | **Fist** | 👊 | All fingers curled tight | Execute last command |
| 6 | **Point Up** | 👆 | Only index finger raised | Scroll up |
| 7 | **Rock Sign** | 🤟 | Index + pinky extended | Show system info |
| 8 | **OK Sign** | 👌 | Thumb tip touches index tip, others up | Acknowledge / Accept |
| 9 | **Call Me** | 🤙 | Thumb + pinky extended, others curled | Open settings panel |
| 10 | **Finger Gun** | 🔫 | Thumb + index extended horizontally | Take a screenshot |
| 11 | **Pinch** | 🤏 | Thumb + index tips touching, others curled | Grab / Select element |
| 12 | **Middle Finger** | 🖕 | Only middle finger extended | **Emergency Stop** all tasks |
| 13 | **Pinky Up** | 🌸 | Only pinky finger extended | Fancy mode trigger |
| 14 | **Vulcan Salute** | 🖖 | All 4 fingers up with gap between middle and ring | Run full system diagnostics |
| 15 | **Crossed Fingers** | 🤞 | Index + middle up, tips touching | Surprise me random action |
| 16 | **Snap Ready** | 🫰 | Thumb touching middle finger, others curled | Quick Launch most-used app |
| 17 | **Devil Horns** | 🤘 | Index + pinky spread wide, no thumb | Open music player / Play music |
| 18 | **Palm Down** | 🫳 | All fingers extended, tips pointing down | **Mute** volume to 0 |
| 19 | **Palm Up** | 🫴 | All fingers extended, tips pointing up high | **Unmute** volume to 50% |
| 20 | **Three Up** | 🔆 | Index + middle + ring up (no pinky, no thumb) | Brightness up 20% |
| 21 | **Four Up** | 🔅 | All 4 fingers up (no thumb) | Brightness down 20% |

---

## Motion-Based Gestures (10)

These detect **hand movement over time** using position history buffers.

| # | Gesture | Emoji | How To Do It | System Action |
|---|---------|-------|--------------|---------------|
| 22 | **Swipe Left** | 👈 | Open palm, move hand left quickly | Previous tab |
| 23 | **Swipe Right** | 👉 | Open palm, move hand right quickly | Next tab |
| 24 | **Swipe Up** | ⬆️ | Open palm, move hand up quickly | Scroll up fast |
| 25 | **Swipe Down** | ⬇️ | Open palm, move hand down quickly | Scroll down fast |
| 26 | **Circular Clockwise** | 🔄 | Draw a circle with index finger (CW) | **Volume Up** (+15%) |
| 27 | **Circular Counter-CW** | 🔃 | Draw a circle with index finger (CCW) | **Volume Down** (-15%) |
| 28 | **Palm Push** | 🫸 | Open palm, push hand toward screen | **Confirm** AI action |
| 29 | **Palm Pull** | 🫷 | Open palm, pull hand away from screen | **Cancel** AI action |
| 30 | **Two-Finger Swipe Left** | ⏪ | Peace sign, swipe left | Switch workspace left |
| 31 | **Two-Finger Swipe Right** | ⏩ | Peace sign, swipe right | Switch workspace right |

---

## Detection Algorithms

### Static Pose Detection

Each of the 21 hand landmarks from MediaPipe is analyzed (landmarks are
temporally filtered by the spatial/world-model layer before this step — see
above):
- **Finger extension**: `tip.y < pip.y` means the finger is extended (up)
- **Thumb extension**: hand-size-normalized distance from thumb tip to index
  MCP (`spatialModel.ts`'s `isThumbExtended()`) — orientation and handedness
  invariant, unlike a raw `x` coordinate comparison
- **Tip distance**: `dist(thumb_tip, index_tip)` for OK/Pinch gestures
- **Orientation**: Comparing average fingertip Y vs wrist Y for palm up/down

Gestures are checked **most-specific-first** to prevent ambiguity:
1. Orientation-dependent (Palm Down/Up)
2. Multi-touch (OK, Pinch, Snap Ready, Crossed)
3. Single-finger (Middle, Pinky)
4. Compound (Devil Horns, Finger Gun, Call Me)
5. Simple (Fist, Palm, Peace, Point)

### Circular Motion Detection

Uses a **12-point index finger position buffer**:

```
1. Compute centroid: cx = mean(x), cy = mean(y)
2. Compute radius of each point from centroid
3. Check circularity: stddev(radii) < 50% of avgRadius
4. Determine direction via cross product sum:
   crossSum > 0 → Clockwise  → Volume Up
   crossSum < 0 → Counter-CW → Volume Down
```

### Palm Push/Pull (Z-Axis Depth)

Uses an **8-point wrist history with Z-coordinate** from MediaPipe:

```
1. Compare Z-depth: newest.z - oldest.z
2. Requires elapsed time: 100ms < elapsed < 600ms
3. All 4 fingers must be extended (open palm pose)
4. dz < -0.06 → Push forward (confirm)
5. dz > +0.06 → Pull back (cancel)
```

---

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `REQUIRED_FRAMES` | 5 | Consecutive frames needed to confirm gesture |
| `GESTURE_COOLDOWN_MS` | 1200 | Milliseconds between gesture triggers |
| `MOTION_BUFFER_SIZE` | 20 | Max frames in wrist/index history buffers |
| `MAX_TRAIL_LENGTH` | 60 | Max points in air-drawing trail |
| `modelComplexity` | 0 | MediaPipe model (0=lite, 1=full) |
| `minDetectionConfidence` | 0.6 | Minimum hand detection confidence |
| `minTrackingConfidence` | 0.5 | Minimum hand tracking confidence |

---

## Adding New Gestures

To add a new gesture:

1. **Define the finger pattern** in `classifyGesture()` inside `GestureControl.svelte` — place it in priority order
2. **Add an emoji** to the `GESTURE_EMOJIS` map
3. **Add an action** in `executeGestureAction()` — call `session.sendCommand()` or `session.addSystemMessage()`
4. **Handle in `App.svelte`** if it needs to trigger UI navigation (tab switching, etc.)

---

## Source Files

| File | Role |
|------|------|
| `tauri-app/ui/src/lib/components/GestureControl.svelte` | Core gesture engine + gesture-cursor bridge |
| `tauri-app/ui/src/lib/gesture/spatialModel.ts` | Spatial/world-model layer — temporal filtering, thumb-extension check, hand quality scoring, kinematic prediction |
| `tauri-app/ui/src/lib/gesture/spatialModel.test.ts` | Unit tests for the spatial model's pure functions |
| `tauri-app/ui/src/lib/gesture/calibration.ts` | On-device gesture calibration — EMA, reversal detection, localStorage store |
| `tauri-app/ui/src/lib/gesture/calibration.test.ts` | Unit tests for calibration EMA/clamping/reversal-pairing logic |
| `tauri-app/ui/src/lib/utils/runtime.ts` | `isTauriRuntime()` — used to pick the native vs. daemon-RPC cursor path |
| `tauri-app/ui/src/lib/stores/settings.ts` | `gesture_cursor`, `adaptive_calibration` settings sections |
| `tauri-app/ui/src/lib/components/SettingsPanel.svelte` | Gesture Cursor Control + Gesture/Voice Calibration settings UI |
| `tauri-app/src-tauri/src/commands.rs` | `move_gesture_cursor`/`click_gesture_cursor` Tauri commands (enigo) |
| `daemon/pilot/server.py` | `cursor_move`/`cursor_click`, `reset_wake_calibration`/`list_wake_variants` RPC handlers |
| `daemon/pilot/config.py` | `GestureCursorConfig`, `AdaptiveCalibrationConfig` |
| `daemon/pilot/system/voice_calibration.py` | On-device wake-word calibration — Levenshtein near-miss detection, promotion, JSON store |
| `tauri-app/ui/src/App.svelte` | Gesture to UI navigation handler |
| `tauri-app/src-tauri/tauri.conf.json` | CSP allowing MediaPipe CDN |
