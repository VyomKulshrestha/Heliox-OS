# Recorded workflows (local preview)

Settings → Recorded workflows lets a user explicitly start recording the
actions **Heliox executes** during a task. A separate consent can additionally
enable a temporary desktop-wide hook for the user's manual mouse and keyboard
activity. Neither source is passively active. Recording is off at daemon startup,
and stopping without saving discards the session. The user supplies a name
and goal, reviews the resulting ordered steps, then may run or schedule a
complete recipe. Manual text entry becomes an editable placeholder; the user
provides its value immediately before a run. The value is not saved in the
workflow recipe, but it is sent to the configured model and may appear in
normal task history. Do not enter passwords in a placeholder. Workflows with
text placeholders cannot be scheduled unattended. Replay is an adaptive
re-plan through Heliox's existing
autonomous planner, permission gate, approval prompts, executor, and verifier;
it is **not** a literal macro or a guarantee that a changed UI will work.
After a submitted run finishes, the user may optionally rate whether the
**goal** was achieved, partly achieved, or not achieved. This explicit label
is stored encrypted with the workflow and kept separate from the executor's
step status; an executor success does not imply goal success. Only a completed
job launched for that workflow can be rated, and a run can be rated once.
Deleting the workflow removes its ratings and visual samples. The outcome
label itself is not used by the live planner, world model, or trainer. If the
user separately opts into replay visual capture for a run, its eligible
before/after transitions are added to the encrypted offline visual-training
pool only when that run receives explicit feedback.

The Heliox-action recorder retains an allowlist of navigation, browser targeting,
and mouse actions only after Heliox reports that the action actually executed
successfully. It omits Heliox-entered text, form contents, clipboard data, scripts,
shell commands, and parallel batches. A recording with omitted actions is
marked incomplete and cannot be replayed or scheduled; re-recording the same
unsupported action will not make it complete. Coordinate-based actions need
explicit user review before replay or scheduling because screen layouts can
change. No saved recipe grants additional execution authority. A schedule is
checked while the daemon is running; an overdue job can dispatch when the
daemon next starts. Required approvals still appear in Heliox, and an
unattended job can fail or expire rather than bypass them.

The manual hook records mouse clicks, straight-line drag endpoints, scroll
coordinates, and whether a text-entry burst occurred, but **never stores typed
characters**. Enter, Tab,
and Escape are recorded as key steps. Modifier shortcuts and unknown
keys mark the recording incomplete rather than guessing an unsafe replay.
Drag paths are not retained, so only start/end coordinates and duration are
available as a reference to the guarded planner.
Input received while Heliox executes an action or awaits approval is
excluded from the manual stream, so user actions at exactly those times may
also be missed. Such mixed sessions carry a visible overlap warning and need
review; on Windows, the hook also excludes, as a best-effort safeguard, a
foreground window whose whole title is “Heliox-OS” (compared with case, hyphens,
underscores and spacing normalised) and, when Windows names the process that owns
the window, only when that process is the Heliox executable. A browser page,
editor or folder whose title merely mentions Heliox-OS is recorded. Hook health
is shown while recording. The hook stops when the recording stops or the daemon
exits. Global hooks may miss elevated/UAC screens or fail silently; a manual
workflow is therefore a draft, not a faithful or safe replay guarantee.

The separate “16×16 before/after screen features” option is off by default
even when action recording is on. It captures a fresh in-memory feature around
a single eligible **Heliox-executed** mouse action, after the normal action
gates. If both manual and visual recording are explicitly enabled, it also
keeps only the latest quantized feature in memory while the hook runs, using
it as the pre-action state for a manual click when it is fresh and predates
the press. Unused frames are discarded, not written to disk. A short settle
interval follows each action; rapid intervening actions, text entry, stale
pre-frames, changed screen bounds, and clicks outside the captured primary
monitor yield no manual visual pair and are counted as visual gaps. Manual
drags, scrolls, and keyboard actions are recipe steps but not visual training
pairs. Neither path writes raw screenshots. Derived features **can still reveal screen
content**; they are encrypted with AES-256-GCM in the local recorded-workflow
store, with the key held in the OS keyring. Without a working keyring, recording
fails closed. The normal list/get RPCs and workflow prompts never expose the
features. Values are quantized to 8-bit grayscale before storage. There is no
upload or automatic training/promotion path.

For an individual workflow replay, a second per-run checkbox can opt in to the
same local 16×16 features. This captures eligible Heliox mouse actions from
that replay task only; it does not capture manual input, text, or raw
screenshots, and scheduled runs never opt in. Features remain in a bounded
in-memory buffer until the user rates the completed run, then the transitions
and separate goal-outcome label are written to the encrypted workflow store.
Unrated features expire 30 minutes after the first captured transition or on
daemon restart. A replay contributes at most 50 pairs, and a workflow at most
200 replay pairs. These transitions are observations, not per-action credit
assignments; the goal-outcome label is preserved separately and is not used as
a transition target.

An explicit offline command can
decrypt them in memory, train on CUDA or CPU, and write encrypted weights plus
a metrics-only report. The optional linear 16×16 export matches the live
predictor contract; if explicitly written to the Heliox data directory, the
daemon loads it after restart in shadow mode only. It is never marked
validated for gating. Other architectures remain research checkpoints and
are not runtime-loaded. Training keeps entire recordings in one partition
and merges recordings with identical quantized pre-action features before
splitting, so the same coarse screen cannot appear on both sides of the holdout.
It requires at least four independent merged groups and ten pairs in each
held-out partition:

```bash
cd daemon
python -m scripts.train_world_visual_transition --recorded-store <DATA_DIR/recorded_workflows.enc> --architecture linear --device cuda --runtime-weights <DATA_DIR/ui_jepa_weights.enc> --weights benchmarks/.bench-state/local_visual_candidate.pt.enc --report benchmarks/.bench-state/local_visual_candidate.json
```

`DATA_DIR` is resolved by `pilot.config` (and can be overridden by
`XDG_DATA_HOME`). Keep the output files local; weights are derived from private
screens even though the artifact itself is encrypted. The runtime candidate is
loaded only by the optional shadow predictor. Its predictions cannot approve,
execute, or reduce deterministic warnings.
Samples are observations of before/after state, not independently verified
task-success labels. Matching 16×16 features can still hide different app
state; the grouping prevents an obvious leakage path but does not prove
counterfactual equivalence. The visual feature store is capped at 200
demonstration pairs per recording, 200 replay pairs per workflow, and the
store accepts at most 50 saved workflows. Deleting a
workflow also deletes its local samples.

Current limitations: manual actions have no robust app/element identity and
the 16×16 grayscale feature is too coarse to establish real desktop world
model quality. The published inverse probe in
[INTELLIGENCE_EVALUATION.md](INTELLIGENCE_EVALUATION.md) shows that public
desktop transitions still lack convincing action information. Do not claim
reference-model parity, reliable arbitrary-app replay, or autonomous approval from this
preview.
