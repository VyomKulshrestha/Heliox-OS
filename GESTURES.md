# 🤚 Heliox OS — Gesture System v3 (30+ Gestures)

Heliox OS includes an experimental webcam-based hand gesture recognition engine powered by [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js), with a legacy MediaPipe Hands compatibility backend. It supports **30+ gesture mappings** including static poses and motion rules; physical accuracy varies by camera, user, lighting, pose, and background.

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
Temporal Hand Verifier (MediaPipe proves hand presence; continuity evidence
                        can only reduce or reject a candidate)
    │
    ▼
Frame Stabilizer (5 consecutive identical frames required)
    │
    ▼
Cooldown Gate (1200ms between triggers)
    │
    ▼
executeGestureAction() ──► built-in control action or reviewed workflow binding
    │
    ▼
UI Feedback (emoji badge, particle burst, gesture history)
```

## Debouncing & Stability

The gesture engine uses a layered anti-jitter system to reduce false triggers:

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Temporal Filtering** | One Euro filter smooths landmark positions before classification | Reduces single-frame jitter without adding perceptible lag |
| **Quality Gate** | Confidence scaled by detection + geometric quality; sub-threshold frames don't advance the stabilizer | Prevents a degenerate/occluded pose from misfiring |
| **Temporal Hand Verification** | Three-to-eight privacy-preserving hand signatures bound to one gesture candidate; resets on no-hand, neutral, or candidate change | Rejects face/background false positives, pose transitions, and discontinuous motion; never creates a gesture |
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
   gets suppressed rather than misfiring at full confidence. Rejection is
   user-visible as missing confidence, low confidence, incomplete geometry,
   too-small framing, or implausible/occluded geometry. Losing the hand also
   clears every raw motion buffer so a newly acquired hand cannot inherit an
   old swipe, circle, push, or cursor trail.

This intentionally does **not** re-express every `classifyGesture()`
threshold in a fully hand-local, scale-normalized coordinate frame — that
would touch ~20 empirically-tuned distance constants with no way to validate
the recalibration against real camera input outside a live testing session,
and risks silently breaking gestures that work today. If you want to take
that further, add a recorded-landmark-sequence regression fixture per gesture
first so the recalibration can actually be verified.

---

## 3D World-Model Layer (MediaPipe Tasks)

An **explicit backend switch**, `vision.mediapipe_backend` (`"tasks"` default
/ `"legacy"` compatibility mode), between the two coordinate systems the gesture engine can run
on:

- **`"legacy"`** (compatibility mode) — the `@mediapipe/hands` callback API. Only
  ever exposes normalized image-space landmarks (`x`/`y` in `[0,1]`, `z`
  relative and unitless). This is the "Spatial/World-Model Layer" described
  above — a 2D-normalized-space temporal-filtering + kinematic-extrapolation
  layer, not real 3D.
- **`"tasks"`** — `@mediapipe/tasks-vision`'s `HandLandmarker`
  (`runningMode: "VIDEO"`, `detectForVideo()`), which additionally exposes
  `worldLandmarks`: real-metric-scale 3D coordinates in **meters**,
  hand-center-relative. This is what makes a genuine 3D world-model layer
  possible — the legacy API never gives you metric scale at all.

Flipping the setting requires stopping/restarting the gesture engine (not a
hot-swap mid-session) — see `VisionConfig.mediapipe_backend` in
`daemon/pilot/config.py`.

**What the `"tasks"` backend adds** (`lib/gesture/worldModel.ts`, additive
and separate from `spatialModel.ts`) — and how `GestureControl.svelte`'s
`handleFrameResult()` actually calls each one, live, per frame:

- `toWristRelative3D()` / `handSize3D()` / `pinchDistance3D()` — re-anchors
  `worldLandmarks` (hand-center-relative by default) to the wrist, then
  measures a metric (meters), camera-distance-invariant hand size and
  thumb-to-index-tip distance. Used as a **confirmation signal**: when the
  2D classifier reports `"ok"` or `"pinch"`, the metric pinch-to-hand-size
  ratio is checked against a wide tolerance band, and confidence is only
  ever *reduced* (never raised) if the metric reading strongly disagrees —
  catching 2D-projection false positives where the thumb and index tip
  merely overlap in the camera's view without truly being close in real
  depth. The tuned 2D `PINCH_DISTANCE_THRESHOLD` check itself is untouched.
- `detectPushPull3D()` — a metric-threshold push/pull detector over a raw
  (unfiltered) world-space wrist-position buffer, gated on the same
  all-fingers-extended pose check the 2D version uses. Under the `"tasks"`
  backend this **replaces** the ad hoc `±0.06` normalized-z check entirely
  (the 2D `detectPushPull()` is not called at all in that case) — under
  `"legacy"` the 2D check runs exactly as before, since `worldLandmarks` is
  always `null` there.
- `WorldModelFilterBank` — like `LandmarkFilterBank`, but **coupled** across
  x/y/z: one shared 3D velocity vector per landmark (a single adaptive
  cutoff derived from combined 3D speed), instead of three independently
  extrapolated axes. Currently used to temporally smooth `worldLandmarks`
  before the metric pinch check above (paralleling `landmarkFilter`'s role
  in 2D classification); `predictAhead()` is unit-tested
  (`worldModel.test.ts`) but not yet consumed by a live predictive feature
  the way `LandmarkFilterBank.predictAhead()` feeds the gesture-cursor
  bridge — a natural next step, not yet wired.

**What stays untouched, on the existing 2D path, regardless of which
backend is active**: every `classifyGesture()` static-pose threshold
(finger-extension patterns, orientation checks, the tuned pinch/thumb
distance constants themselves), the gesture-cursor bridge, and gesture
calibration. `GestureControl.svelte` funnels both backends through one
backend-agnostic `handleFrameResult(landmarks, worldLandmarks,
handednessScore)` — the `"legacy"` path always passes `worldLandmarks:
null`, so none of the above 3D logic ever executes for it, and its behavior
is bit-for-bit unchanged. None of the ~20 empirically-tuned 2D distance
constants were re-expressed in 3D (see the caution in the "Spatial/World-
Model Layer" section above — that still applies here unchanged); the 3D
layer only ever adds a confirmation/replacement signal alongside them.

**Delegate**: CPU only, unconditionally — GPU delegate support inside
Tauri's embedded webview (WebView2 on Windows, WebKitGTK on Linux, WKWebView
on macOS) hasn't been verified cross-platform, so it isn't defaulted on.

**Asset serving**: same self-hosted, no-CDN policy as the legacy backend
(enforced by `tests/static/no-remote-mediapipe.test.mjs`). A
`mediapipeTasksVisionAssets()` Vite plugin (`vite.config.ts`, mirroring the
existing `mediapipeHandsAssets()` plugin) serves the `@mediapipe/tasks-vision`
WASM loader files plus the vendored model at `/mediapipe/tasks-vision/*`,
both in dev (middleware) and in the production build (`writeBundle`). CSP
was verified empirically (dev server + a live browser check) to need **no
changes** — `HandLandmarker.createFromOptions()` and its WASM graph start
successfully under the existing `script-src 'self' 'unsafe-eval'` policy,
with no `worker-src`/`blob:` relaxation required.

**Model provenance**: `vendor/mediapipe/hand_landmarker.task` (float16,
~7.8MB) was downloaded from Google's official public MediaPipe model
distribution URL
(`storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`)
and its MD5 checksum verified against the source's `x-goog-hash` response
header before committing. **License note**: Google's MediaPipe docs
explicitly license the *code samples* Apache-2.0 but do not explicitly
state a redistribution license for the model *weights* themselves. This
file is vendored anyway on the judgment that it's Google's own official
public distribution URL, used as-is (not fine-tuned or modified) by many
other open-source projects — flagged here rather than silently assumed. If
you need to re-fetch or update it, use the same URL and re-verify the MD5
against `x-goog-hash` before committing.

**Tests**: `worldModel.test.ts` (wrist re-anchoring, metric pinch/hand-size,
push/pull thresholds, and a cosine-similarity check that `predictAhead()`
extrapolates along the true 3D direction of a diagonal motion) and an
extension to `spatialModel.test.ts` (non-zero-z fixtures — every fixture
before this addition used `z: 0` everywhere, so the existing `dist3d()` z
term had zero real regression coverage — plus a numeric-pinning test
confirming the 2D static-pose path stayed bit-identical through this
migration).

**Not verified in this pass** (no physical webcam in the environment this
was built in): the `PUSH_PULL_DEPTH_METERS`/pinch-ratio-tolerance constants'
real-camera accuracy, real-world tracking-quality comparison between
backends, and GPU-delegate behavior on an actual Windows/macOS/Linux Tauri
build. The wiring itself (which functions get called, when, and with what
gating) is code-reviewable and unit-tested per the module's own tests, but
the specific threshold values are approximate until tuned against real
camera data — same caveat the 2D thresholds carried when first introduced.

---

## Gaze Tracking (third input modality)

An **opt-in**, `vision.gaze_tracking_enabled` (default off), third input
modality alongside voice + gesture — coarse, on-device gaze-region
estimation, fused into the multimodal fusion engine as a passive
disambiguating signal, never a standalone command trigger on its own.

**Accuracy expectations, stated plainly**: a consumer webcam has no IR
eye-tracker hardware, so this is deliberately **not** pixel-precise
pointing — that would need a real per-user calibration routine this
feature doesn't attempt. Instead, `lib/gesture/gazeTracking.ts`'s
`assessGazeFrame()` reports one of five coarse regions
(`center`/`left`/`right`/`up`/`down`) from where the iris sits within each
eye socket (MediaPipe's 478-point face mesh with iris refinement,
indices 468/473), averaged across both eyes. Before a region is accepted,
Heliox rejects non-finite/incomplete landmarks, closed-eye or blink frames,
extreme projected eye asymmetry from a turned head, implausible iris positions,
and eye readings that disagree. Quality can only reduce confidence. A dead
zone absorbs normal center jitter, and a two-sample hysteresis gate is required
before a region change enters fusion. MediaPipe is the learned face/landmark
model; Heliox's quality, discretization, and hysteresis layers are deterministic
and unit-tested against synthetic fixtures. Their thresholds still require
physical validation across users and cameras.

**On-device processing, privacy-first**: `GestureControl.svelte` loads
`@mediapipe/tasks-vision`'s `FaceLandmarker` as a separate model on the
same webcam stream. A gaze-enabled session also uses Tasks-Vision's
`HandLandmarker` even when the stored hand preference is `legacy`: the
legacy `@mediapipe/hands` WASM and Tasks-Vision WASM both install an
Emscripten `Module` global and cannot safely coexist in one page. Keeping
both models in the Tasks-Vision runtime prevents that initialization
collision. Face tracking runs
at a reduced sampling rate (at most once every 500ms — a coarse "which rough
direction" signal doesn't need 30fps, and running two ML inference
passes every single frame on the CPU delegate is a real cost worth
avoiding). A confirmed region change is sent immediately and an unchanged reading is
refreshed every 750ms so it remains inside the fusion engine's 1.5-second
correlation window without producing per-frame RPC traffic. **Only the
coarse region label + a confidence float are ever sent to the backend** —
never raw face landmarks or video frames, the
same minimal-data-sent principle the existing gesture pipeline already
follows (gesture events send just a gesture name, not landmark
coordinates).

**How to activate and verify it**: turn on Settings → Gaze Tracking, then
use **Open Command** and click the visible **Gaze ready → start camera**
control. Camera access remains a separate, explicit user action. The Command
control and camera preview show model loading, face scanning, the live coarse
region/confidence, and local/daemon errors; the Settings section distinguishes
an enabled preference from a genuinely active camera/model session. Changing
the preference while an existing camera session is running hot-enables or
hot-disables FaceLandmarker without restarting hand tracking.

**Fusion**: `pilot.multimodal.fusion.MultimodalFusionEngine` gained a
third `ModalityType.GAZE` and its own buffer (`on_gaze_event()`), but
gaze is architecturally different from voice/gesture — it only ever
*ingests* into the buffer for lookup by the other two modalities; it
never independently produces or emits a `FusedIntent` (there's no
`GESTURE_STANDALONE_COMMANDS`-style table for gaze, because "looking
somewhere" isn't a command on its own the way a gesture can be). When a
recent gaze reading exists within the fusion time window, it's attached
as `metadata["gaze_region"]`/`metadata["gaze_confidence"]` on whatever
intent voice/gesture produce, and gives a small, capped confidence bonus
specifically when the resolved gesture modifier is `target` or `select`
(the cases where "where is the user looking" is actually relevant
context) — it never lowers a confidence voice/gesture already
established, and it never rewrites the command text itself via string
matching (deliberately: the well-tested `MULTIMODAL_TEMPLATES` design
stays untouched; gaze is passed through as structured metadata for a
downstream consumer to use, not spliced into natural language here).

**Model provenance**: `vendor/mediapipe/face_landmarker.task` (~3.76MB),
same source/verification/license-ambiguity note as
`hand_landmarker.task` above — downloaded from Google's official public
MediaPipe model distribution URL, MD5 verified against the source's
`x-goog-hash` header before committing. The existing
`mediapipeTasksVisionAssets()` Vite plugin already serves any file placed
in the vendor directory, so no build-tooling changes were needed to add
this second model.

**Not verified in this pass**: real-camera accuracy of the coarse region
estimation and whether the confidence, quality, deadzone, and hysteresis
constants hold up across actual users, eyewear, skin tones, lighting, cameras,
and head poses — the same caveat as the 3D world-model layer above.

---

## Neural-intent fusion and cancellation

The optional neural research path does not replace gesture/gaze fusion.
Immediately before creating a neural preview and again before committing it,
the daemon captures one bounded multimodal snapshot containing only recent
voice text/confidence, gesture label/confidence, coarse gaze region/confidence,
and timing metadata. Raw camera/audio/landmark data is excluded.

An open-palm, palm-pull, stop phrase, or other registered cancellation signal
always dominates a simultaneous neural selection and disarms neural control.
Gaze remains passive context and cannot raise neural authority or create a
command. Gesture/voice agreement may add context to the audit and preview, but
it cannot lower the neural dwell, quality, confidence, margin, freshness,
world-model, permission, or approval requirements.

Neural acquisition continues while gesture/camera processing is active, and
all ordinary side effects coordinate through the shared execution lease so
enabling every modality does not silently suppress another feature. See
[Neural Intent Research Controls](docs/NEURAL_INTENT.md).

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

## Selection study pointer and gesture approval

### Study pointer mode

The selection study (Settings → Assistance & co-adaptation → Run a selection
session) can use the hand as a pointer inside Heliox's own window. While a
hand-pointer session runs and is not paused, the study requests the pointer
(`studyPointer.requested` in `stores/studyPointer.ts`), and with the camera on
`GestureControl.svelte` works differently:

- Each verified frame publishes the filtered index fingertip (no prediction),
  mirrored and scaled like Cursor Mode but mapped to the study area, together
  with a measured pinch. The OS cursor is never moved or clicked:
  `move_gesture_cursor`, `click_gesture_cursor`, `cursor_move` and
  `cursor_click` are not called, and Cursor Mode is switched off and cannot be
  turned on until the study releases the pointer.
- The pinch counts as closed after 2 consecutive frames below the pinch
  threshold and as open again only above 1.3 times the threshold.
- A hand loss cancels. When no verified hand has been seen for more than
  200 ms, the study is told once, the published pinch is forced open and a
  press or hold in progress ends. After the hand is found again it must be
  verified for 8 frames, and then the pinch must be seen open on 8 consecutive
  fresh frames before any press can start. The same rule applies after every
  study pause, so a pinch held closed through a tracking loss or a pause never
  selects anything.
- An open palm pauses the study. It is checked first on every frame, as in
  Cursor Mode. Every other gesture is ignored while the study holds the
  pointer.
- While a study session is open, the camera preview shrinks to a 64 × 48 px
  thumbnail above the camera button, and its badges and the gesture readouts
  are hidden, so the bottom-right camera controls take as little of the study
  area as possible.

### After the study pauses or ends

When the study releases the pointer (Pause, Esc, an open palm, an approval or
any other dialog, leaving the Study tab, or the end of the session), every
gesture is disarmed: approve, deny, tab navigation and gestures bound to
workflows. Gestures work again only after the hand has been out of view for at
least 200 ms and is then verified on 8 frames, or after no hand has been seen
for 500 ms. A gesture must also begin after that point: one whose recognition
began earlier is dropped even if it completes later. A pose held when the
study paused can therefore never start a workflow.

### Approving and denying by gesture

Thumbs up and palm push approve; thumbs down and palm pull deny
(`gesture/actionPolicy.ts`). Approval by gesture is narrower than the dialog's
own Approve button:

- It sends only the actions checked in the dialog. The dialog reports its
  checked boxes when it opens and on every change. If nothing is checked, or
  the reported boxes belong to an earlier approval, nothing is sent and Heliox
  says "Nothing is checked. Check the actions you want to approve, or deny."
- The dialog must be on screen: the Command tab is open, the approval is the
  active dialog, the window is visible, and it has been visible for at least
  1 second. Otherwise Heliox says "Open the Command tab to approve. Gestures
  can approve only a dialog you have had time to see."
- Each time a dialog appears or shows a new approval, approve gestures are
  disarmed until the hand leaves the camera view and comes back, under the
  same 200 ms, 8-frame and 500 ms rule as above. A thumbs up held while the
  dialog appears does nothing, and Heliox says "Lower your hand out of view,
  then raise it to approve by gesture."
- Deny by gesture works whenever gestures are armed, whatever the dialog's
  state. The dialog's Deny button and a spoken or typed denial work at once in
  every case.
- The daemon records which channel answered (`input_channel` is `gesture` or
  `dialog`), never whether a pointer or a key was used.

### Gaze during a study

While a selection study session is open, in any input mode and whether or not
it is paused, gaze readings stay on the device. `gazeRuntime` is still
updated, and the study reads it locally, but no `gaze_event` RPC is sent, so
no gaze reading reaches the daemon's experience ledger. Sending resumes when
the session ends.

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
Only stop and pending-approval control signals have built-in effects. Every
other row below is a suggested binding and cannot create a task or change the
OS unless the user explicitly enables Gesture Workflow Bindings and assigns
that gesture in Settings.

| # | Gesture | Emoji | How To Do It | Suggested Binding |
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

These detect **hand movement over time** using position history buffers. As
above, only pending-approval control signals have built-in effects; the other
actions shown are suggestions for explicit bindings.

| # | Gesture | Emoji | How To Do It | Suggested Binding |
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

## Research corpus and offline benchmark contract

The implementation was reviewed against current model documentation and public
research corpora. These sources are **evaluation candidates, not bundled
training data**. Their licenses and access terms must be checked before any
download or commercial use, and Heliox does not commit raw face/hand media.

| Source | What it contributes | Heliox use now |
|--------|---------------------|----------------|
| [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js) and [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js) | Official VIDEO-mode confidence, tracking, world-landmark, face-blendshape, and synchronous-web constraints | Runtime contract and threshold semantics |
| [HaGRIDv2](https://github.com/hukenovs/hagrid) / [WACV paper](https://openaccess.thecvf.com/content/WACV2024/html/Kapitanov_HaGRID_--_HAnd_Gesture_Recognition_Image_Dataset_WACV_2024_paper.html) | Diverse subjects, lighting, distance, explicit `no_gesture` negatives, static/control poses | Candidate for licensed offline false-positive and coverage evaluation; not vendored |
| [EgoGesture](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html) | 83 static/dynamic egocentric gesture classes with train/validation/test sequences | Candidate dynamic-gesture stress corpus; not equivalent to Heliox's frontal webcam |
| [MPIIGaze](https://collaborative-ai.org/research/datasets/MPIIGaze/) | 213,659 everyday laptop-use frames across months, with illumination and appearance variation | Research-only coarse-region evaluation candidate; non-commercial dataset terms apply |
| [GazeCapture](https://openaccess.thecvf.com/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html) | Roughly 2.4M mobile frames from 1,474 subjects and calibration/no-calibration evidence | Scale/calibration reference; official release is research-use only |
| [ETH-XGaze](https://ait.ethz.ch/xgaze) and [Gaze360](https://openaccess.thecvf.com/content_ICCV_2019/html/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.html) | Extreme head-pose, illumination, temporal, uncertainty, and cross-dataset protocols | Future robustness references; access/redistribution restrictions prevent bundling |

`cameraSignalBenchmark.ts` evaluates already-recorded, timestamped detector
outputs without opening a camera or invoking any cursor, workflow, daemon, or
OS action. It reports:

- hand-presence precision/recall/F1;
- gesture-event precision/recall/F1 and false activations per minute;
- coarse-gaze coverage, accuracy, rejection specificity, and excess region
  transitions per minute.

The test suite validates the metric math and synthetic detector edge cases. It
does **not** establish real-camera accuracy. A release-quality physical corpus
still needs consented users, multiple cameras, lighting/background/skin-tone/
eyewear/head-pose coverage, explicit no-hand/no-gesture negatives, and a frozen
person-disjoint test split. Keep raw media outside Git; only consented,
de-identified detector outputs should enter local benchmark fixtures.

---

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `REQUIRED_FRAMES` | 5 | Consecutive frames needed to confirm gesture |
| `GESTURE_COOLDOWN_MS` | 1200 | Milliseconds between gesture triggers |
| `MOTION_BUFFER_SIZE` | 20 | Max frames in wrist/index history buffers |
| `MAX_TRAIL_LENGTH` | 60 | Max points in air-drawing trail |
| `modelComplexity` | 0 | Legacy MediaPipe Hands compatibility model (0=lite, 1=full) |
| `minDetectionConfidence` | 0.7 | Minimum hand detection/handedness confidence accepted by either backend |
| `minHandPresenceConfidence` | 0.7 | Tasks-Vision threshold before it re-runs palm detection |
| `minTrackingConfidence` | 0.65 | Hand tracking confidence / bounding-box IoU threshold |
| `minFaceDetectionConfidence` | 0.6 | Gaze face detection threshold |
| `minFacePresenceConfidence` | 0.6 | Gaze face-presence threshold |
| `face minTrackingConfidence` | 0.6 | Gaze face tracking threshold |
| `GAZE_INFERENCE_INTERVAL_MS` | 500 | Maximum gaze sampling rate while sharing the UI thread with hand tracking |
| `GAZE_HEARTBEAT_MS` | 750 | Refresh interval for a stable accepted gaze region |
| `HAND_ACQUIRE_FRAMES` | 8 | Verified frames before a hand counts as present; also the re-arm and study press gates |
| Gesture re-arm (`createGestureArm`) | 200 / 500 ms | Hand out of view for 200 ms then 8 verified frames, or 500 ms with no hand |
| `GESTURE_APPROVAL_MIN_VISIBLE_MS` | 1000 | How long the approval dialog must be visible before a gesture may approve it |
| Study pinch hysteresis | 2 frames / 1.3× | Closed after 2 frames below the pinch threshold; open above 1.3 times it |

---

## Adding New Gestures

To add a new gesture:

1. **Define the finger pattern** in `classifyGesture()` inside `GestureControl.svelte` — place it in priority order
2. **Add an emoji** to the `GESTURE_EMOJIS` map
3. **Keep built-in actions limited to safety/navigation controls.** Task actions
   should use a user-visible gesture workflow binding rather than a hidden
   hardcoded command.
4. **Add/update the workflow binding contract** in Settings and test
   submit/pause/resume/cancel through the daemon's durable workflow path.
5. **Handle in `App.svelte`** only when the gesture is a local UI navigation
   control.
6. **Add temporal/no-hand tests.** MediaPipe must prove a hand exists; the
   temporal verifier may only reduce or reject the candidate.

---

## Source Files

| File | Role |
|------|------|
| `tauri-app/ui/src/lib/components/GestureControl.svelte` | Core gesture engine + gesture-cursor bridge |
| `tauri-app/ui/src/lib/gesture/spatialModel.ts` | Spatial/world-model layer — temporal filtering, thumb-extension check, hand quality scoring, kinematic prediction |
| `tauri-app/ui/src/lib/gesture/spatialModel.test.ts` | Unit tests for the spatial model's pure functions |
| `tauri-app/ui/src/lib/gesture/worldModel.ts` | Real-metric-scale 3D world-model layer (wrist-relative worldLandmarks, metric hand-size/pinch, push-pull, coupled 3D temporal filtering) — "tasks" backend only |
| `tauri-app/ui/src/lib/gesture/worldModel.test.ts` | Unit tests for the 3D world-model layer |
| `tauri-app/ui/src/lib/gesture/temporalGestureVerifier.ts` | Privacy-preserving temporal hand continuity verifier; supporting evidence only |
| `tauri-app/ui/src/lib/gesture/temporalGestureVerifier.test.ts` | No-hand, warm-up, continuity, rejection, and reset tests |
| `tauri-app/ui/vendor/mediapipe/hand_landmarker.task` | Vendored HandLandmarker model (float16) — see provenance/license note above |
| `tauri-app/ui/vendor/mediapipe/face_landmarker.task` | Vendored FaceLandmarker model (float16, iris refinement) — see gaze tracking provenance note above |
| `tauri-app/ui/vite.config.ts` | `mediapipeTasksVisionAssets()` plugin — self-hosts the Tasks-vision WASM loader + vendored models (hand + face) at `/mediapipe/tasks-vision/*` |
| `tauri-app/ui/src/lib/gesture/gazeTracking.ts` | Pure coarse gaze-region estimation from iris/eye-socket landmarks — third fusion modality, no calibration |
| `tauri-app/ui/src/lib/gesture/gazeTracking.test.ts` | Unit tests for gaze-region discretization against synthetic face-mesh fixtures |
| `tauri-app/ui/src/lib/gesture/cameraSignalBenchmark.ts` | Non-executing metrics for recorded hand, gesture-event, and coarse-gaze outputs |
| `tauri-app/ui/src/lib/gesture/cameraSignalBenchmark.test.ts` | Metric arithmetic, timestamp validation, false-activation, and gaze-flicker tests |
| `tauri-app/ui/src/lib/gesture/calibration.ts` | On-device gesture calibration — EMA, reversal detection, localStorage store |
| `tauri-app/ui/src/lib/gesture/calibration.test.ts` | Unit tests for calibration EMA/clamping/reversal-pairing logic |
| `tauri-app/ui/src/lib/gesture/studyPointerTracker.ts` | Study pinch hysteresis, normalized study pointer, gesture re-arm gate, gaze send rule during a study |
| `tauri-app/ui/src/lib/gesture/approvalGesture.ts` | What an approve or deny gesture may do with a pending approval (visible 1 s, re-armed hand, checked actions only) |
| `tauri-app/ui/src/lib/stores/studyPointer.ts` | In-window study pointer state and its press gate (`ready`, `pressArmed`) |
| `tauri-app/ui/tests/static/gesture-gaze-study.test.mjs` | Static checks: no gaze RPC during a study, every gesture dispatch behind the re-arm gate |
| `tauri-app/ui/src/lib/utils/runtime.ts` | `isTauriRuntime()` — used to pick the native vs. daemon-RPC cursor path |
| `tauri-app/ui/src/lib/stores/settings.ts` | `gesture_cursor`, `adaptive_calibration`, `vision.mediapipe_backend`, `vision.gaze_tracking_enabled` settings sections |
| `tauri-app/ui/src/lib/components/SettingsPanel.svelte` | Gesture Cursor Control + Gesture/Voice Calibration settings UI |
| `tauri-app/src-tauri/src/commands.rs` | `move_gesture_cursor`/`click_gesture_cursor` Tauri commands (enigo) |
| `daemon/pilot/server.py` | `cursor_move`/`cursor_click`, `reset_wake_calibration`/`list_wake_variants`, `gaze_event` RPC handlers |
| `daemon/pilot/config.py` | `GestureCursorConfig`, `AdaptiveCalibrationConfig`, `VisionConfig.mediapipe_backend`/`gaze_tracking_enabled` |
| `daemon/pilot/multimodal/fusion.py` | `MultimodalFusionEngine` — voice+gesture fusion, plus gaze as a passive third modality (buffer-only, confidence-bonus metadata) |
| `daemon/pilot/system/voice_calibration.py` | On-device wake-word calibration — Levenshtein near-miss detection, promotion, JSON store |
| `tauri-app/ui/src/App.svelte` | Gesture to UI navigation handler |
| `tauri-app/src-tauri/tauri.conf.json` | CSP for the self-hosted UI and WASM runtime; MediaPipe models are vendored, not CDN-loaded |
