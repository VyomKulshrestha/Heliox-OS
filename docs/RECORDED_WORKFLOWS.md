# Recorded workflows (local preview)

Settings → Recorded workflows lets a user explicitly start recording the
actions **Heliox executes** during a task. It does not passively record the
user's manual mouse or keyboard activity. Recording is off at daemon startup,
and stopping without saving discards the session. The user supplies a name
and goal, reviews the resulting ordered steps, then may run or schedule a
complete recipe. Replay is an adaptive re-plan through Heliox's existing
autonomous planner, permission gate, approval prompts, executor, and verifier;
it is **not** a literal macro or a guarantee that a changed UI will work.

The recorder retains an allowlist of navigation, browser targeting, and
mouse actions only after Heliox reports that the action actually executed
successfully. It omits typed text, form contents, clipboard data, scripts,
shell commands, and parallel batches. A recording with omitted actions is
marked incomplete and cannot be replayed or scheduled; re-recording the same
unsupported action will not make it complete. Coordinate-based actions need
explicit user review before replay or scheduling because screen layouts can
change. No saved recipe grants additional execution authority. A schedule is
checked while the daemon is running; an overdue job can dispatch when the
daemon next starts. Required approvals still appear in Heliox, and an
unattended job can fail or expire rather than bypass them.

The separate “16×16 before/after screen features” option is off by default
even when action recording is on. It captures a fresh in-memory feature only
around a single eligible mouse action, after the normal action gates. It does
not write raw screenshots. Derived features **can still reveal screen
content**; they are encrypted with AES-256-GCM in the local recorded-workflow
store, with the key held in the OS keyring. Without a working keyring, recording
fails closed. The normal list/get RPCs and workflow prompts never expose the
features. Values are quantized to 8-bit grayscale before storage. There is no
upload path or automatic training/promotion path.
Samples are observations of before/after state, not independently verified
task-success labels. The visual feature store is capped at 200 pairs per
recording and the store accepts at most 50 saved workflows. Deleting a
workflow also deletes its local samples.

Current limitations: manual desktop actions are not observed; text-entry
workflows need a separate parameterization and secret-handling design; and
the 16×16 grayscale feature is too coarse to establish real desktop world
model quality. The published inverse probe in
[INTELLIGENCE_EVALUATION.md](INTELLIGENCE_EVALUATION.md) shows that public
desktop transitions still lack convincing action information. Do not claim
Jev parity, reliable arbitrary-app replay, or autonomous approval from this
preview.
