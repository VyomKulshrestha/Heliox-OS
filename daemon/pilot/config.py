"""Runtime configuration loader and manager."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from pilot.actions import PermissionTier
from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, ActionFamily, SourceProfile


def _xdg(env_var: str, fallback: str) -> Path:
    """The XDG folder named by ``env_var``, or ``~/<fallback>``. As the XDG specification says, an unset, empty or
    relative value is ignored: used as it was, an empty one put Heliox's folder under the daemon's working directory,
    where "Remove Heliox's files" (uninstall.rs, which follows the same rule) does not look."""
    value = os.environ.get(env_var, "").strip()
    if value and Path(value).is_absolute():
        return Path(value)
    return Path.home() / fallback


# The test suite's own folder (tests/folder_guard.py), never a real install's. It is used only when the folder named
# by HELIOX_TEST_ROOT carries TEST_ROOT_MARKER, a file the suite writes there, so setting the variable by hand, or a
# leftover in a shell, is ignored: setup, removal and the app never see Heliox anywhere but its usual folders.
TEST_ROOT_VARIABLE = "HELIOX_TEST_ROOT"
TEST_ROOT_MARKER = ".heliox-test-root"


def _test_root() -> Path | None:
    named = os.environ.get(TEST_ROOT_VARIABLE, "").strip()
    if not named:
        return None
    root = Path(named)
    return root if (root / TEST_ROOT_MARKER).is_file() else None


_TEST_ROOT = _test_root()


def _heliox_dir(env_var: str, fallback: str) -> Path:
    """Heliox's own folder under the XDG one; under the test root, laid out as in a home folder, during tests."""
    if _TEST_ROOT is not None:
        return _TEST_ROOT / fallback / "heliox-os"
    return _xdg(env_var, fallback) / "heliox-os"


def _default_runtime_dir() -> Path:
    """Resolve runtime dir when XDG_RUNTIME_DIR is unset (macOS, Windows, minimal Linux); the test root's during tests."""
    if _TEST_ROOT is not None:
        return _TEST_ROOT / "runtime" / "heliox-os"
    xdg = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg:
        return Path(xdg) / "heliox-os"
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "heliox-os" / "runtime"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "heliox-os" / "runtime"
    run_user = Path(f"/run/user/{uid}")
    if run_user.is_dir() and os.access(run_user, os.W_OK):
        return run_user / "heliox-os"
    tmp = os.environ.get("TMPDIR", "/tmp")
    return Path(tmp) / f"heliox-os-runtime-{uid}"


CONFIG_DIR = _heliox_dir("XDG_CONFIG_HOME", ".config")
DATA_DIR = _heliox_dir("XDG_DATA_HOME", ".local/share")
STATE_DIR = _heliox_dir("XDG_STATE_HOME", ".local/state")
RUNTIME_DIR = _default_runtime_dir()


# Two places under the home folder that do not follow the XDG variables, as the app (folders.rs) knows them: learned
# data some parts keep (voice calibration, persona, biometric, cloud hand-off) and the folder setup and the app use
# (~/.config/heliox-os: the environment, tools such as the macOS screen-reading helper). Every folder here is in
# daemon/tests/heliox_folders_cases.json, which the app's Rust test reads too.
def _cache_dir() -> Path:
    return (_TEST_ROOT or Path.home()) / ".cache" / "heliox"


def _setup_dir() -> Path:
    return (_TEST_ROOT or Path.home()) / ".config" / "heliox-os"


CACHE_DIR = _cache_dir()
SETUP_DIR = _setup_dir()
CONFIG_FILE = CONFIG_DIR / "config.toml"
RESTRICTIONS_FILE = CONFIG_DIR / "restrictions.toml"
DB_FILE = DATA_DIR / "pilot.db"
EXPERIENCE_DB_FILE = DATA_DIR / "experience.db"
DURABLE_TASK_DB_FILE = DATA_DIR / "durable_tasks.db"
TEMPORAL_MEMORY_DB_FILE = DATA_DIR / "temporal_memory.db"
AUDIT_FILE = DATA_DIR / "audit.jsonl"
PERMISSION_AUDIT_DB_FILE = DATA_DIR / "permission_audit.db"
PERMISSION_AUDIT_KEY_FILE = DATA_DIR / "permission_audit.key"
AGENT_GATEWAY_AUDIT_DB_FILE = DATA_DIR / "agent_gateway_audit.db"
AGENT_GATEWAY_AUDIT_KEY_FILE = DATA_DIR / "agent_gateway_audit.key"
CO_ADAPTATION_DB_FILE = DATA_DIR / "co_adaptation.db"
CO_ADAPTATION_PREFERENCES_FILE = DATA_DIR / "co_adaptation_preferences.json"
CO_ADAPTATION_EXPORTS_DIR = DATA_DIR / "co_adaptation_exports"
LOG_FILE = STATE_DIR / "pilot.log"

# Derived directories shared across modules — use these instead of hardcoded paths
PLUGINS_DIR = CONFIG_DIR / "plugins"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
PERSONA_FILE = DATA_DIR / "persona.md"


@dataclass
class ModelConfig:
    provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    mode: str = "lightweight"  # "lightweight" | "full"
    gpu_memory_limit_mb: int = 0  # 0 = no limit
    idle_unload_seconds: int = 60
    cloud_provider: str = ""  # "openai" | "openrouter" | "claude" | "gemini" | "meta"
    cloud_model: str = ""
    # Subscription-backed official coding-agent CLIs.  Heliox never reads or
    # stores their OAuth credentials; the selected CLI owns authentication.
    subscription_provider: str = "codex"  # "codex" | "claude"
    subscription_model: str = ""  # blank uses the CLI/account default
    subscription_timeout_seconds: int = 120
    # Keep the Heliox payload bounded before the coding-agent harness adds its
    # own context.  Refuse oversize calls instead of silently spending quota.
    subscription_max_prompt_chars: int = 48000
    # Rate limiting — applied to every LLM call via ModelRouter
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60  # sustained requests per minute
    rate_limit_burst: int = 5  # token bucket burst capacity
    # Budget tracking — cumulative monthly spend limit
    budget_enabled: bool = True
    budget_monthly_limit_usd: float = 10.0
    # Per-action and per-task budget enforcement
    # These limit single LLM calls and per-task cumulative spend, complementing
    # the monthly cap. Useful for halting runaway autonomous loops.
    # The planner's current static policy/tool prompt is slightly above 4k
    # tokens before the user's command is added, so a 4k default blocks
    # ordinary requests before planning starts.
    max_tokens_per_action: int = 12000  # cap on tokens for a single LLM call
    max_tokens_per_task: int = 100000  # cumulative token cap per orchestrator task
    max_usd_per_task: float = 0.10  # cumulative USD cap per task
    max_consecutive_failures: int = 3  # circuit breaker threshold (Phase 4)


@dataclass
class SecurityConfig:
    root_enabled: bool = False
    dry_run: bool = False
    snapshot_on_destructive: bool = True
    snapshot_backend: str = "auto"  # "auto" | "btrfs" | "timeshift" | "none"
    snapshot_retention_count: int = 10
    snapshot_retention_days: int = 7
    unrestricted_shell: bool = False  # Allow ANY shell command (bypass whitelist)
    # Code execution sandbox — isolates agent-generated code from the host OS
    sandbox_mode: str = "auto"  # "auto" | "firecracker" | "docker" | "restricted" | "none"
    sandbox_memory_mb: int = 128  # memory cap applied inside the sandbox (MB)
    sandbox_timeout: int = 30  # max wall-clock seconds for sandboxed execution
    sandbox_network: bool = False  # allow outbound network inside the sandbox
    sandbox_kernel_guard: bool = True  # Linux seccomp-BPF syscall denylist for restricted mode
    sandbox_blocked_syscalls: list[str] = field(default_factory=lambda: ["unlink", "unlinkat"])
    sandbox_firecracker_binary: str = "firecracker"  # executable path or command name
    sandbox_firecracker_kernel_image: str = ""  # vmlinux path used by strict microVM mode
    sandbox_firecracker_rootfs_path: str = ""  # rootfs image path used by strict microVM mode
    sandbox_firecracker_fallback: bool = True  # fall back to Docker/restricted if microVM mode is unavailable


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8785
    auth_token: str = ""


def _default_whisper_model() -> str:
    try:
        from pilot.system.speech_models import default_whisper_model

        return default_whisper_model()
    except Exception:  # the manifest unreadable: the English recommendation
        return "base"


@dataclass
class VoiceConfig:
    language: str = "auto"  # auto detect or manual language code
    # Faster-Whisper is preferred for substantially lower local latency and
    # memory use. "auto" retains the proven OpenAI Whisper fallback. "streaming" decodes while you speak
    # (sherpa-onnx, English only, pilot.system.streaming_asr) and falls back to Whisper when it has no text.
    transcription_engine: str = "auto"
    # What the recommended download brings for the language of this computer: "base" for English (with
    # Parakeet), "small" otherwise (pilot.system.speech_models.default_whisper_model). A size that is not on the
    # computer falls back to one that is (model_presence.usable_whisper_model).
    whisper_model: str = field(default_factory=_default_whisper_model)
    # Stable "host API::device name" identifier returned by
    # list_audio_input_devices. "auto" preserves automatic selection.
    input_device: str = "auto"
    # Continuous VAD-based endpointing (see pilot.system.vad) — replaces
    # blind fixed-duration recording windows with natural start/stop
    # boundaries. Approximate defaults, not tuned against real microphone
    # hardware; revisit if real usage shows false starts/premature cutoffs.
    vad_energy_threshold: float = 0.02
    vad_adaptive_threshold_enabled: bool = True
    # A short conversational pause should complete the utterance quickly.
    # 450 ms still tolerates natural spacing between words while avoiding the
    # 1+ second dead-air penalty that makes always-listening control feel laggy.
    vad_silence_ms: float = 450.0
    vad_max_utterance_seconds: float = 20.0
    # Semantic end of turn (pilot.system.turn_detector, Pipecat Smart Turn v3.2, about 8.7 MB, CPU). After 200 ms
    # of silence the model says whether the sentence sounds finished: finished ends the turn at once, unfinished
    # keeps listening for up to turn_hold_ms so a pause to think is not cut off. "silence" uses vad_silence_ms only.
    turn_detection: str = "smart_turn"  # "smart_turn" | "silence"
    # The recogniser whose words are acted on. "whisper" (Whisper, whisper_model) works out of the box. "parakeet"
    # uses NVIDIA Parakeet TDT 0.6b v2 (pilot.system.parakeet_asr; about 640 MB, downloaded by hand), which on a
    # Bluetooth headset kept "Monday, the twelfth of November" where Whisper base did not, in about 0.7 s; Whisper
    # still runs beside it for the wake phrase. Without the model files it falls back to "whisper".
    final_pass: str = "whisper"  # "whisper" | "parakeet"
    # While the person is in a call (Teams, Zoom, Meet, WhatsApp, Slack, ...; pilot.system.call_detection),
    # Heliox does not speak and does not take what is said as addressed to it. Windows, macOS 14+ (and older with a
    # call app open) and Linux with PulseAudio or PipeWire.
    quiet_in_calls: bool = True
    parakeet_model_dir: str = ""  # "" = <Heliox data>/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
    turn_hold_ms: float = 2500.0
    # Interrupt Heliox's own TTS playback the instant the user starts
    # talking, instead of waiting for it to finish. A pure responsiveness/
    # UX improvement (doesn't expand what the system can do, unlike e.g.
    # gesture_cursor), so on by default like adaptive_calibration.
    barge_in_enabled: bool = True
    # No longer used. It made Heliox take any sentence that sounded like a command as meant for it, without the
    # wake phrase, so it acted on what other people in the room or on a video said (264 dispatches in an hour of
    # meetings and videos on the Conversation Bench). Speech is now taken as meant for Heliox only after the wake
    # phrase or the microphone button, and in the short follow-up window below (pilot.system.addressing). Kept so
    # older config files still load.
    continuous_conversation_enabled: bool = False
    # After Heliox answers someone who addressed it, a reply within this many seconds needs no wake phrase, if it is
    # plausibly meant for Heliox (pilot.system.addressing). 8 s: people answer within about a second; 30 s let a
    # room in.
    follow_up_window_seconds: float = 8.0
    # Local model-backed speech or the existing OS-native TTS (Windows
    # SAPI/macOS say/Linux espeak). Kokoro is the more natural built-in
    # option; Pocket remains available for existing installations. Both
    # model engines fall back to os_native on import/model/audio failure.
    tts_engine: str = "kokoro_tts"  # "kokoro_tts" | "pocket_tts" | "os_native"
    tts_voice: str = "af_heart"  # model voice preset; ignored when tts_engine == "os_native"
    # The system voice by name when tts_engine == "os_native" ("" = the system's default): a voice of the person's
    # language when Kokoro has none (pilot.system.voice_by_language).
    system_voice: str = ""
    # The local voice runs in its own process: about 1.6-1.9 GB of RAM (plus about 840 MB of GPU memory on CUDA)
    # while loaded, and 20-25 s to load from cold. It stays loaded until this many seconds pass with nobody
    # talking to Heliox and Heliox saying nothing, and loads again as soon as the person is heard.
    tts_keep_loaded_seconds: float = 300.0
    # On a computer with less than 10 GB of memory in all, speak with the system voice instead of loading the local
    # one (about 1.6-1.9 GB), and say so once (pilot.system.speech_budget). Off: the local voice on every computer,
    # still only when 2 GB are free at load time.
    light_on_small_memory: bool = True
    # Where the local voice runs: "auto" (the GPU when CUDA works, else the CPU), "cpu" or "cuda". On the CPU a
    # long sentence takes about 3.5 s instead of 0.25 s, but no GPU memory is used.
    tts_device: str = "auto"
    # Per headset, the microphone to listen with instead (input_device "auto" only): a Bluetooth headset in call
    # mode sends narrowband sound that recognisers mishear (pilot.system.mic_quality). Filled only when the person
    # presses "Use <microphone> for input" in the voice bar; removed from Settings. {headset name: input id}.
    headset_input_choices: dict[str, str] = field(default_factory=dict)


@dataclass
class ScreenVisionConfig:
    capture_interval_seconds: float = 3.0
    capture_timeout_seconds: float = 10.0
    max_consecutive_timeouts: int = 3
    auto_resume_after_seconds: float = 30.0


@dataclass
class VisionConfig:
    # "legacy" keeps the existing @mediapipe/hands normalized-2D pipeline;
    # "tasks" switches to @mediapipe/tasks-vision's HandLandmarker, which
    # additionally exposes real-metric-scale worldLandmarks (see
    # GESTURES.md's "3D World-Model Layer" section and worldModel.ts).
    # New installs default to "tasks" so orientation-independent 3D finger
    # geometry is available for the full static-pose classifier. Existing
    # users may retain "legacy" as an explicit compatibility preference.
    # Flipping this restarts the gesture engine rather than hot-swapping.
    mediapipe_backend: str = "tasks"
    # Coarse, on-device gaze-region estimation (see GESTURES.md's gaze
    # section and tauri-app/ui/src/lib/gesture/gazeTracking.ts) — a THIRD
    # input modality alongside voice/gesture, fed into the multimodal
    # fusion engine (pilot.multimodal.fusion) as a passive disambiguating
    # signal, never a standalone command trigger on its own. Off by
    # default: a new capability (another camera-derived signal sent to the
    # backend, however coarse), same opt-in-first treatment as
    # gesture_cursor.
    gaze_tracking_enabled: bool = False


@dataclass
class GestureCursorConfig:
    """Continuous gesture-to-cursor bridge (see GESTURES.md). Off by default
    — this drives the real OS mouse cursor, so it must be an explicit opt-in,
    never silently enabled by a gesture or a config migration."""

    enabled: bool = False
    sensitivity: float = 1.0
    # How far ahead (ms) the kinematic predictor in spatialModel.ts
    # extrapolates hand trajectory — see predictAhead()/predictCursorTarget().
    prediction_ms: float = 80.0
    # Blend between the current filtered position (0.0) and the predicted
    # position (1.0) fed to the cursor. Kept modest by default: the
    # predictor's velocity estimate is empirically amplified for a sustained
    # motion (see spatialModel.test.ts), so a small blend avoids overshoot
    # until this is tuned against real camera data.
    blend: float = 0.3


@dataclass
class AdaptiveCalibrationConfig:
    """On-device continual-learning/personalization loop for voice+gesture
    recognition (see GESTURES.md's "Gesture Calibration" section and
    pilot.system.voice_calibration). Both toggles default on: unlike
    gesture_cursor (which grants a NEW capability, real cursor control),
    this only nudges existing recognition thresholds within a bounded,
    reversible range from implicit signals already present in normal usage
    - there's no new capability being granted, so an opt-in default isn't
    warranted the way it is for gesture_cursor."""

    gesture_enabled: bool = True
    voice_wake_word_enabled: bool = True


@dataclass
class GatewayConfig:
    """Agent Gateway (see pilot.security.gateway) — source-scoped permission
    floors for shell/browsing/system-control actions, layered alongside the
    existing tier-based PermissionChecker. Enabled by default: unlike
    gesture_cursor, this only ever *restricts* what a plan may do relative
    to today's tier system, never grants anything new, so there's no
    opt-in-only reason to default it off."""

    enabled: bool = True
    source_profiles: dict[str, SourceProfile] = field(default_factory=lambda: dict(DEFAULT_SOURCE_PROFILES))
    # Learned Risk Gate (see pilot.security.risk_gate) — a small MLP
    # trained on real sandboxed telemetry that predicts concrete outcome
    # fields (disk usage, process-count delta) for a proposed action,
    # scored by hardcoded rules (risk_safety.py) that can only ADD
    # caution on top of the existing heuristic_risk() signal (see
    # destructive_critic.py's risk_score()), never remove it. The v2 gate
    # is enabled for new installs because learned output is scored beside
    # (never instead of) deterministic rules, so it cannot weaken the
    # established safety baseline. Existing explicit user settings remain
    # unchanged when loading a saved config.
    risk_gate_enabled: bool = True


@dataclass
class GestureWorkflowBinding:
    """One user-authored gesture -> multi-step-goal binding (see
    pilot.agents.voice_gesture_workflow.VoiceGestureWorkflowEngine). Users
    define these themselves in Settings — there is no built-in fixed set,
    since a gesture is just a fixed pose/motion name and can't carry
    open-ended goal text the way a voice command can."""

    gesture_name: str = ""
    goal_template: str = ""
    enabled: bool = True


SUPPORTED_GESTURE_WORKFLOW_GESTURES: tuple[str, ...] = (
    "palm",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "fist",
    "point_up",
    "rock",
    "ok",
    "call_me",
    "finger_gun",
    "pinch",
    "middle_finger",
    "pinky_up",
    "vulcan",
    "crossed_fingers",
    "snap_ready",
    "devil_horns",
    "palm_down",
    "palm_up",
    "three_up",
    "four_up",
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "circular_cw",
    "circular_ccw",
    "palm_push",
    "palm_pull",
    "two_finger_swipe_left",
    "two_finger_swipe_right",
)


@dataclass
class GestureWorkflowConfig:
    """Binds specific gestures to multi-step workflow goals. Off by default:
    unlike the Agent Gateway (which only restricts), binding a gesture to an
    open-ended goal is a NEW capability — the gesture now drives an
    autonomous multi-step plan instead of a single fixed action, so it needs
    an explicit opt-in the same way gesture_cursor does."""

    enabled: bool = False
    bindings: list[GestureWorkflowBinding] = field(default_factory=list)
    pending_trigger_window_seconds: float = 90.0
    paused_window_seconds: float = 1800.0


@dataclass
class SelfHealingConfig:
    """Autonomous Healing Engine (see pilot.agents.autonomous_healing) —
    passive system-health monitoring (CPU/memory/disk, via the existing
    BackgroundTaskManager) that plans a remediation goal when a threshold
    is crossed. Off by default: unlike the Agent Gateway, generating and
    potentially auto-executing a remediation plan without being asked is a
    NEW autonomous capability, not a restriction, so it needs the same
    explicit opt-in as gesture_cursor/gesture_workflows.

    Autonomy is tiered even once enabled: `auto_execute_max_tier` bounds
    only the plans that may run WITHOUT a confirmation prompt (an
    int(PermissionTier) value — default 1 = USER_WRITE); anything above
    that, or containing an irreversible action, is proposed and held
    pending explicit user confirmation instead of auto-executed. This
    mirrors risk_gate_enabled's own reasoning: an unproven autonomous
    capability defaults to the more conservative of its two behaviors."""

    enabled: bool = False
    auto_execute_max_tier: int = 1  # PermissionTier.USER_WRITE
    cooldown_seconds: float = 600.0
    confirm_timeout_seconds: float = 300.0
    watched_metrics: list[str] = field(default_factory=lambda: ["cpu", "memory", "disk"])
    goal_templates: dict[str, str] = field(default_factory=dict)


@dataclass
class NarrationConfig:
    """Live Execution Narrator (see pilot.agents.narrator) — narrates plan
    execution as it happens and can pre-emptively pause a plan/action on a
    risk signal this codebase already computes (the Agent Gateway's critic
    WARN verdict, this session's dom_diff.assess_target pre-execution
    check) rather than silently discarding it. Off by default: spoken
    interruptions and pausing execution are new user-facing behavior, not
    a pure restriction, so this gets the same opt-in-first treatment as
    gesture_cursor/gesture_workflows/self_healing.

    `interrupt_on_risk` only ever gates a plan/action BEFORE it runs — it
    never attempts to cancel something already executing."""

    enabled: bool = False
    narrate_steps: bool = True
    interrupt_on_risk: bool = True
    proactive_review_enabled: bool = True
    # Direct user corrections are out-of-band: they can stop an in-flight
    # action and revise the plan even when spoken narration is disabled.
    live_corrections_enabled: bool = True
    follow_up_enabled: bool = True
    max_auto_revisions: int = 2
    confirm_timeout_seconds: float = 120.0
    advisory_timeout_seconds: float = 5.0


@dataclass
class PreviewConfig:
    """ "Simulate before executing" preview (see pilot.system.action_preview
    and pilot.agents.narrator.ExecutionNarrator.on_action_preview) — before
    the JARVIS Autonomy orchestrator (AutonomousExecutor) commits to an
    action, pauses and shows a real screenshot with the target UI element
    highlighted (plus, for browser actions, a real measured DOM diff from an
    isolated dry run) and waits for confirm/deny. Off by default: this adds
    real latency (a screenshot + VLM call, and for browser actions a real
    dry-run browser tab) before every gated action, the same opt-in-first
    treatment as gesture_cursor/gesture_workflows/self_healing/narration.

    Only ever applies to autonomous/unattended invocations — interactive
    voice/text/gesture commands already have the user watching in real
    time, so this would only add friction there without the corresponding
    benefit (catching a mistake nobody was there to see)."""

    enabled: bool = False
    confirm_timeout_seconds: float = 120.0


@dataclass
class UndoConfig:
    """Per-task undo (see pilot.workflows.undo_journal). Before Heliox writes, moves,
    copies or deletes a file or replaces the clipboard, it keeps what it needs to put
    the change back, so a finished task can be undone item by item. Copies stay on
    this computer under DATA_DIR/undo and are deleted after ``retention_days`` or
    when the total passes ``max_total_mb``. Changes larger than ``max_task_mb`` for
    one task are still made, but are listed as not undoable."""

    enabled: bool = True
    retention_days: int = 7
    max_task_mb: int = 512
    max_total_mb: int = 2048


@dataclass
class ReceiptsConfig:
    """Signed task receipts (see pilot.workflows.task_receipts). After a task that ran
    actions, Heliox writes a receipt listing each action, its approval and its result,
    signed with this computer's key and chained to the previous receipt. Parameters
    and outputs are stored only as hashes. Receipts older than ``retention_days`` are
    deleted, except the newest."""

    enabled: bool = True
    retention_days: int = 90


@dataclass
class StorageConfig:
    """Heliox's data folder (see pilot.system.data_size). The audit log, the durable task tables, the checkpoints
    and the experience ledger are never pruned on their own; their sizes are shown in Settings, and Heliox says once
    when the whole data folder passes ``data_folder_warn_gb``."""

    data_folder_warn_gb: int = 5


@dataclass
class BriefConfig:
    """Daily brief (see pilot.workflows.daily_brief). Off by default. At ``time`` (24-hour
    HH:MM, local) Heliox sends one short summary of the day: what it ran, where your
    time went if the activity timeline is on, and what can still be undone. With
    ``speak`` it is also read aloud."""

    enabled: bool = False
    time: str = "18:00"
    speak: bool = False


def _default_timeline_excluded_apps() -> list[str]:
    return ["1password", "bitwarden", "keepass", "lastpass", "dashlane", "keychain access"]


@dataclass
class TimelineConfig:
    """Activity timeline (see pilot.workflows.activity_timeline). Off by default. When on,
    Heliox records which app and window title were in front, and when, so you can ask
    "what was I working on yesterday afternoon?". Never pixels or screen text. Windows
    whose app or title contains an ``excluded_apps`` entry, and private browsing
    windows, are never recorded."""

    enabled: bool = False
    retention_days: int = 7
    excluded_apps: list[str] = field(default_factory=_default_timeline_excluded_apps)


# An Ollama model name such as "qwen3.5:4b" or "library/llama3.2:3b-instruct-q4_K_M".
_OLLAMA_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,80}(?::[A-Za-z0-9._-]{1,40})?")


@dataclass
class LiveConfig:
    """Heliox Live (see pilot.live): the always-on companion that notices mistakes as you work and says so.
    Off by default. When on, it reads the text field you are typing in (never a password field), listens to
    what you say to Heliox, and points out specific slips: a weekday that doesn't match its date, a wrong sum,
    a risky command before you press Enter, a key pasted into a chat. Everything stays on this computer and in
    memory; it never acts on its own."""

    enabled: bool = False
    level: str = "balanced"  # "quiet", "balanced" or "proactive"
    read_focused_text: bool = True
    ocr_fallback: bool = False  # read the screen when no text field is readable; heavier, so off by default
    listen_to_speech: bool = True
    # Learn how long this person pauses mid-sentence, and wait for a real break (pilot.live.timing).
    personal_timing: bool = True
    # Check what they write against their own calendar, to-dos and reminders (pilot.live.grounded). Off by
    # default; clashes and named events are checked in code, and grounded_model (in the local Ollama) links
    # paraphrases. If the model is missing, the code checks still run.
    grounded_checks: bool = False
    grounded_model: str = "qwen3.5:4b"
    muted_kinds: list[str] = field(default_factory=list)


@dataclass
class ConversationConfig:
    """The fast conversational layer (see pilot.conversation): questions, remarks and follow-ups said to Heliox are
    answered by a small fast model, streamed into speech sentence by sentence; requests to act go to the planner
    with a short spoken acknowledgement. On by default; it uses the provider set under [model].

    Each provider's fast model is its own setting. Empty means: Claude's fast tier (claude-haiku-5-5), or for any
    other provider the model already configured for it. share_screen_text decides whether the end of the text
    field you are typing in (read by Heliox Live, never a password field) goes with a question: "auto" shares it
    with a model on this computer and not with a cloud model, "on" always, "off" never. The focused app's name
    is sent in every case except for apps Live never reads.

    provisional_answers: Heliox starts answering a question you said after "Hey Heliox" (or with the microphone
    button) before it is sure you have finished, and says nothing unless your final words are the same; a start it
    drops costs about what an answer costs, and dropped starts may cost at most $0.05 an hour. "auto" turns it on
    for a cloud model once its measured accuracy earns it (docs/INTERACTION_MODEL.md), "on" for every model,
    "off" never."""

    enabled: bool = True
    share_screen_text: str = "auto"  # "auto" | "on" | "off"
    max_reply_tokens: int = 400
    claude_model: str = ""
    openai_model: str = ""
    openrouter_model: str = ""
    gemini_model: str = ""
    meta_model: str = ""
    ollama_model: str = ""
    # The web search the quick road opens for "search the web for ...": an http(s) address with {query}.
    search_url: str = "https://www.google.com/search?q={query}"
    provisional_answers: str = "auto"  # "auto" | "on" | "off"
    # A follow-up without the wake phrase loses its reach for the person's own data once the listener heard this many
    # utterances it did not take since Heliox last spoke (pilot.conversation.reach). Off until it is measured on
    # recordings of a person alone; the count is logged on every follow-up either way.
    # What a model that is not on this computer is told about the person (pilot.conversation.adapters.
    # personal_in_prompt): "auto" sends the name, the answer style and the parts of the profile that bear on what is
    # asked; "off" sends nothing about them. A model on this computer is not affected. never_sent holds under both.
    share_profile: str = "auto"  # "auto" | "off"
    second_voice_guard: bool = False
    second_voice_threshold: int = 1  # 2 caught no single second voice; at 1 the cost alone is unmeasured


@dataclass
class SuggestionsConfig:
    """Proactive suggestions (see pilot.workflows.suggestions). Off by default. When on,
    Heliox offers one gentle suggestion now and then, computed in code from your goals,
    habits and to-do list, held during a focus session. It never acts on a suggestion.

    ``offers`` (pilot.personal.offers): Heliox offers what you usually do at about this time ("You usually open
    Outlook about now. Want me to open it?") and picks from your habits, goals and to-dos, at most one an hour and
    three a day, at a quiet moment, all computed on this computer. "auto" (default) turns them on once your profile
    has something in it; "on" and "off" override."""

    enabled: bool = False
    offers: str = "auto"  # "auto" | "on" | "off"


@dataclass
class ClipboardConfig:
    """Clipboard history (see pilot.workflows.clipboard_history). Off by default. When on,
    Heliox remembers the last 25 pieces of text you copied, for up to eight hours, in
    memory only, so you can ask "what did I copy earlier?". Nothing is kept from the
    activity timeline's excluded apps or private windows, and text that looks like a
    password, key or card number is never kept."""

    history: bool = False


@dataclass
class SupervisionConfig:
    """User Manual Supervision (see pilot.agents.user_supervision) — watches
    the user's OWN independent screen/keyboard/mouse activity, never anything
    Heliox itself executes. The single biggest privacy surface in this
    codebase; off by default, with `keyboard_mouse_hook_enabled` as a
    SEPARATE, starker opt-in from `enabled` (screen/OCR-only supervision) —
    a user can turn on cognitive coaching + OCR-based risk warnings without
    installing the global keyboard/mouse hook at all.

    Advisory only — see user_supervision.on_risk_pattern_detected's
    docstring for why this never blocks, unlike ExecutionNarrator: Heliox
    has no way to intercept or block the user's own OS-level input, it only
    observes a copy via the hook."""

    enabled: bool = False
    keyboard_mouse_hook_enabled: bool = False
    cognitive_coaching_enabled: bool = True
    risk_pattern_detection_enabled: bool = True
    tick_interval_seconds: float = 1.5
    ocr_interval_seconds: float = 8.0
    stress_coaching_threshold: float = 0.75
    cognitive_load_coaching_threshold: float = 0.8
    coaching_cooldown_seconds: float = 900.0
    risk_cooldown_seconds: float = 30.0
    keystroke_buffer_max_chars: int = 256
    ocr_snippet_max_chars: int = 400


@dataclass
class CoAdaptationConfig:
    """Assistance co-adaptation (see pilot.coadaptation and docs/COADAPTATION.md). Off by default: it observes how
    the person selects things inside Heliox's own study surface and changes presentation only. Anything stored also
    needs an active, purpose-bound consent record. It never changes approvals, permissions or execution.

    No key needs a restart. A change reaches the study at the next session_begin, which stores the effective
    settings in the new session's row; an open session keeps the settings it started with. Only enabled = False
    acts at once: it ends open sessions.

    Study sessions follow the study protocol: they pin exploration_floor and switching_penalty to the protocol's
    values whatever is set here, and take ambiguity_threshold, unintended_activation_tolerance,
    noninferiority_margin and upper_bound_credibility from here only where the value is stricter than the
    protocol's (a lower threshold, tolerance or margin, a higher credibility). The session row lists the keys whose
    value here differs from the protocol."""

    enabled: bool = False
    adaptive_enabled: bool = False  # Phase B learned choices; also needs purpose adaptive_assistance
    learn_across_sessions: bool = False  # also needs purpose cross_session_learning
    gaze_prior_enabled: bool = False  # also needs vision.gaze_tracking_enabled and purpose gaze_prior
    hand_pointer_enabled: bool = False  # also needs purpose hand_pointer and a running camera
    ambiguity_threshold: float = 0.25  # [0.05, 0.25]: the person may only make τ stricter; never learned
    unintended_activation_tolerance: float = 0.05  # [0.0, 0.2]; a study session uses it only when stricter
    noninferiority_margin: float = 0.01  # [0.0, 0.05]; a study session uses it only when stricter
    upper_bound_credibility: float = 0.95  # [0.8, 0.99]; a study session uses it only when stricter (higher)
    exploration_floor: float = 0.1  # [0.0, 0.3]; a study session uses the protocol's value
    switching_penalty: float = 0.1  # [0.0, 1.0]; a study session uses the protocol's value
    burden_prompt_every_blocks: int = 1  # [1, 8]
    day_boundary_hour: int = 4  # [0, 23]


# Inclusive ranges of the numeric co_adaptation keys. The loader clamps a hand-edited value into its range;
# co_adaptation_config_update rejects a value outside it.
CO_ADAPTATION_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "ambiguity_threshold": (0.05, 0.25),
    "unintended_activation_tolerance": (0.0, 0.2),
    "noninferiority_margin": (0.0, 0.05),
    "upper_bound_credibility": (0.8, 0.99),
    "exploration_floor": (0.0, 0.3),
    "switching_penalty": (0.0, 1.0),
}
CO_ADAPTATION_INT_RANGES: dict[str, tuple[int, int]] = {
    "burden_prompt_every_blocks": (1, 8),
    "day_boundary_hour": (0, 23),
}
CO_ADAPTATION_BOOL_KEYS = frozenset(
    {"enabled", "adaptive_enabled", "learn_across_sessions", "gaze_prior_enabled", "hand_pointer_enabled"}
)


@dataclass
class ProxyConfig:
    http: str | None = None
    https: str | None = None
    no_proxy: str | None = None


@dataclass
class MemoryConfig:
    checkpoint_interval_seconds: int = 300
    pruning_interval_seconds: int = 3600
    pruning_min_memories: int = 10
    max_context_tokens: int = 8000
    max_recent_messages: int = 10
    # Keep things said to Heliox in passing ("my flight is on Friday") as short-lived notes (pilot.personal.passing).
    # Never health, money, secrets or other people's private details. Off: only "remember that ..." is kept.
    passing_notes: bool = True
    # Learn routines from what the person asks Heliox for (pilot.personal.routines): the kind of request and its
    # target, the weekday and the hour, kept 60 days on this computer, never in a model's prompt. Off: nothing kept.
    learn_routines: bool = True


@dataclass
class RSSConfig:
    enabled: bool = False
    feeds: list[str] = field(default_factory=list)
    poll_interval_hours: float = 24.0
    max_items_per_feed: int = 10


@dataclass
class CalendarConfig:
    enabled: bool = False
    caldav_url: str = ""
    caldav_username: str = ""
    caldav_password_provider: str = ""  # Vault provider for password
    ics_files: list[str] = field(default_factory=list)


@dataclass
class EmailConfig:
    enabled: bool = False
    imap_host: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password_provider: str = "email"


@dataclass
class SemanticSearchConfig:
    enabled: bool = False
    folders: list[str] = field(default_factory=list)  # List of directories to index
    index_dir: str = ""  # Custom directory to store index (defaults to DATA_DIR/semantic_index)


@dataclass
class CognitiveConfig:
    enabled: bool = True


@dataclass
class TypedDecisionConfig:
    """Opt-in local research scorer; never a source of execution authority."""

    enabled: bool = False
    scoring_mode: str = "trained_head"
    artifact_dir: str = ""
    report_path: str = ""
    model_id: str = "Qwen/Qwen3-0.6B"
    max_length: int = 2048
    temperature: float = 1.0
    # With Gnomon on, also review the plans the person asks for directly (chat and voice), not only autonomous
    # work. It can only add a confirmation, never approve; when it is not loaded yet, the request does not wait.
    # On by default since Gnomon v7 passed the everyday-plan gate G6 on 10 November 2026: 1 false alarm on 27
    # correct plans and no miss on 12 bad ones (v4.1: 6 false alarms). See docs/GNOMON_TRAINING.md.
    review_requests: bool = True
    # Load and run Gnomon in a separate process the daemon supervises. A CUDA fault, an out-of-memory error or a
    # kill there answers the pending review as "not reviewed" and restarts the worker; in-process, the same fault
    # ended the whole daemon on 10 November 2026. Costs one Python process.
    worker_process: bool = True


@dataclass
class GpuConfig:
    """Limits for Heliox's own GPU work (Gnomon, the local voice, on-demand OCR); see pilot.system.gpu_arbiter."""

    # No model starts on the card at or above this temperature; the voice and OCR use the CPU instead.
    max_temperature_c: float = 83.0
    # Free graphics memory a model needs beyond its own size before it may load.
    vram_headroom_mb: int = 512
    # This many GPU faults (worker crashes, CUDA errors, hangs) within the window switch the GPU off until restart.
    fault_limit: int = 3
    fault_window_seconds: float = 600.0


@dataclass
class RedisConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    key_prefix: str = "pilot:"
    default_ttl: int = 300
    max_memory_cache_size: int = 512


@dataclass
class NetworkConfig:
    """LAN mesh network configuration for multi-instance collaboration."""

    enabled: bool = False
    port: int = 8786  # peer-to-peer WebSocket port (separate from client port)
    peer_timeout_s: int = 30  # seconds before a silent peer is considered gone
    skill_sync_enabled: bool = False  # broadcast/receive plugins from authenticated peers
    collab_exec_enabled: bool = False  # distribute actions only to authenticated peers


@dataclass
class AirHandoffConfig:
    """Opt-in LAN receiver for encrypted, one-target phone handoffs.

    This is intentionally separate from the peer-compute mesh. The receiver
    exposes no execution tools and starts only after the user enables it.
    """

    enabled: bool = False
    port: int = 8787
    max_transfer_mb: int = 25
    # A paired phone may send a task request; it waits on this computer until you press Run.
    accept_requests: bool = False
    # Also start requests at once as background jobs limited to PHONE_READ_ONLY_ACTIONS.
    run_read_only_requests: bool = False


@dataclass
class SshHostConfig:
    """One allowed SSH destination (referenced by alias in ssh_* actions)."""

    name: str = ""
    hostname: str = ""
    port: int = 22
    username: str = ""
    private_key_provider: str = ""  # KeyVault provider name containing the private key (PEM)
    passphrase_provider: str = ""  # Optional KeyVault provider name for key passphrase
    strict_host_key_checking: bool = True


@dataclass
class SshConfig:
    enabled: bool = False
    connect_timeout_seconds: int = 10
    allowed_hosts: list[SshHostConfig] = field(default_factory=list)


@dataclass
class Restrictions:
    protected_folders: list[str] = field(default_factory=list)
    protected_packages: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    sandbox_allowed_commands: list[str] = field(
        default_factory=lambda: ["echo", "ls", "dir", "cat", "type", "ping", "whoami", "pwd", "grep", "find"]
    )


@dataclass
class PilotConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    screen_vision: ScreenVisionConfig = field(default_factory=ScreenVisionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gesture_cursor: GestureCursorConfig = field(default_factory=GestureCursorConfig)
    adaptive_calibration: AdaptiveCalibrationConfig = field(default_factory=AdaptiveCalibrationConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    gesture_workflows: GestureWorkflowConfig = field(default_factory=GestureWorkflowConfig)
    self_healing: SelfHealingConfig = field(default_factory=SelfHealingConfig)
    narration: NarrationConfig = field(default_factory=NarrationConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    undo: UndoConfig = field(default_factory=UndoConfig)
    receipts: ReceiptsConfig = field(default_factory=ReceiptsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    brief: BriefConfig = field(default_factory=BriefConfig)
    clipboard: ClipboardConfig = field(default_factory=ClipboardConfig)
    suggestions: SuggestionsConfig = field(default_factory=SuggestionsConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    co_adaptation: CoAdaptationConfig = field(default_factory=CoAdaptationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rss: RSSConfig = field(default_factory=RSSConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    semantic_search: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    air_handoff: AirHandoffConfig = field(default_factory=AirHandoffConfig)
    ssh: SshConfig = field(default_factory=SshConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    restrictions: Restrictions = field(default_factory=Restrictions)
    first_run_complete: bool = False
    redis: RedisConfig = field(default_factory=RedisConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    typed_decision: TypedDecisionConfig = field(default_factory=TypedDecisionConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)

    @classmethod
    def load(cls) -> PilotConfig:
        """Load config from disk, creating defaults if missing.

        A file that cannot be read is copied aside and reported through
        ``load_errors``; Heliox then runs on defaults but refuses to save, so one
        typo can no longer turn into a silent overwrite of the person's settings.
        """
        config = cls()
        errors: list[str] = []

        if CONFIG_FILE.exists():
            try:
                raw = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                _validate_config_types(raw)
                # Merge into a fresh object so a failure part-way leaves the defaults intact.
                config = _merge_config(cls(), raw)
            except Exception as e:
                backup = _keep_unreadable_copy(CONFIG_FILE)
                logger.error(f"Failed to load config.toml: {e}. Falling back to safe defaults.")
                errors.append(f"config.toml could not be read ({e}). A copy was kept at {backup}.")

        if RESTRICTIONS_FILE.exists():
            try:
                raw = tomllib.loads(RESTRICTIONS_FILE.read_text(encoding="utf-8"))
                config.restrictions = _parse_restrictions(raw)
            except Exception as e:
                backup = _keep_unreadable_copy(RESTRICTIONS_FILE)
                logger.error(
                    "Failed to load restrictions.toml: %s. Falling back to built-in safety restrictions.",
                    e,
                )
                errors.append(f"restrictions.toml could not be read ({e}). A copy was kept at {backup}.")

        config._load_errors = errors
        config._disk_signature = _disk_signature()
        return config

    @property
    def load_errors(self) -> list[str]:
        return list(getattr(self, "_load_errors", []))

    def clear_load_errors(self) -> None:
        """The person chose to start over (Reset to defaults); saving is allowed again."""
        self._load_errors = []

    def set_runtime_override(self, section: str, key: str, value: Any) -> None:
        """Change a value for this run only; save() keeps writing the person's own value.

        Used for the --dry-run flag and for a fallback model picked because the configured
        one is missing, so neither becomes a saved setting behind the person's back.
        """
        overrides = getattr(self, "_runtime_overrides", None)
        if overrides is None:
            overrides = self._runtime_overrides = {}
        target = getattr(self, section)
        overrides.setdefault((section, key), getattr(target, key))
        if getattr(self, "_runtime_override_values", None) is None:
            self._runtime_override_values = {}
        self._runtime_override_values[(section, key)] = value
        setattr(target, key, value)

    def reload_if_edited(self) -> bool:
        """Adopt edits made to the config files while Heliox runs; True when something was reloaded.

        Only configs that were loaded from or saved to disk take part, so test configs
        built in memory never pick up the person's real files.
        """
        known = getattr(self, "_disk_signature", None)
        if known is None or known == _disk_signature():
            return False
        fresh = type(self).load()
        if fresh.load_errors:
            self._load_errors = fresh.load_errors
            return False
        for item in fields(self):
            setattr(self, item.name, getattr(fresh, item.name))
        for (section, key), value in getattr(self, "_runtime_override_values", {}).items():
            # The file now holds the person's value; keep this run's override on top of it.
            target = getattr(self, section)
            self._runtime_overrides[(section, key)] = getattr(target, key)
            setattr(target, key, value)
        self._disk_signature = fresh._disk_signature
        logger.info("Picked up changes made to the Heliox config files")
        return True

    def save(self) -> None:
        """Persist current config to disk."""
        if self.load_errors:
            raise ConfigWriteBlocked(
                "Heliox could not read your config file, so it will not overwrite it: "
                + " ".join(self.load_errors)
                + " Fix the file and restart Heliox, or use Reset to defaults."
            )
        from pilot.real_folders import refuse_real_folders

        # Only the daemon saves the person's settings; a bench, eval or script on the real folders is refused here.
        refuse_real_folders("Saving Heliox's settings outside the daemon")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        data = _config_to_dict(self)
        for (section, key), original in getattr(self, "_runtime_overrides", {}).items():
            if section in data and isinstance(data[section], dict):
                data[section][key] = original
        restrictions = data.pop("restrictions", {})

        try:
            _save_with_backup(CONFIG_FILE, _carry_comments(_read_text(CONFIG_FILE), tomli_w.dumps(data)))

            if restrictions:
                _save_with_backup(
                    RESTRICTIONS_FILE, _carry_comments(_read_text(RESTRICTIONS_FILE), tomli_w.dumps(restrictions))
                )
        except OSError as error:
            # A full disk or read-only folder is said once in plain words; the caller still sees the failure.
            from pilot.system.storage_health import storage

            storage().report("config", error, "your settings")
            raise
        self._disk_signature = _disk_signature()


class ConfigWriteBlocked(RuntimeError):
    """The config file on disk could not be read, so saving would destroy it."""


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _disk_signature() -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    return (_file_signature(CONFIG_FILE), _file_signature(RESTRICTIONS_FILE))


def _keep_unreadable_copy(path: Path) -> Path:
    """Copy an unreadable config file aside before anything can replace it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.unreadable-{stamp}")
    try:
        shutil.copy2(path, backup)
    except OSError:
        logger.warning("Could not keep a copy of %s", path, exc_info=True)
    return backup


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _comment_anchor(line: str, section: str) -> tuple[str, str | None] | None:
    stripped = line.strip()
    if stripped.startswith("["):
        return (stripped, None)
    if stripped and "=" in stripped and not stripped.startswith("#"):
        return (section, stripped.split("=", 1)[0].strip().strip('"').strip("'"))
    return None


def _carry_comments(old_text: str, new_text: str) -> str:
    """Keep the person's comment lines when the config is rewritten.

    tomli_w writes values only, so every settings save used to delete notes the
    person wrote in config.toml. A comment block directly above a section header
    or a key is put back above the same header or key; comments whose section or
    key no longer exists, and comments at the end of a value line, are dropped.
    """
    if "#" not in old_text:
        return new_text
    comments: dict[tuple[str, str | None], list[str]] = {}
    pending: list[str] = []
    section = ""
    for line in old_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            pending.append(line.rstrip())
            continue
        anchor = _comment_anchor(line, section)
        if anchor is None:
            continue
        if anchor[1] is None:
            section = anchor[0]
        if pending:
            comments[anchor] = pending
            pending = []
    output: list[str] = []
    section = ""
    for line in new_text.splitlines():
        anchor = _comment_anchor(line, section)
        if anchor is not None:
            if anchor[1] is None:
                section = anchor[0]
            output.extend(comments.pop(anchor, []))
        output.append(line)
    output.extend(pending)  # a comment block at the very end of the file
    return "\n".join(output) + "\n"


# Every save that changes a settings file keeps the file it replaces beside it, as <name>.saved-<YYYYmmdd-HHMMSS>, in
# the settings folder (so removal keeps them with "keep my data" and deletes them with everything else). Kept: the
# newest SAVED_COPIES, and beyond those the earliest copy of each day for SAVED_DAYS days. Spoken settings ("stop
# interrupting me", a Live mute) save the file, so a voice-first person can make many saves a day; the daily copies
# still reach back a month. A save that wipes the person's values (a tool writing a fresh config over theirs, as
# happened on 12 November 2026) can then be undone by copying a copy back.
SAVED_COPIES = 10
SAVED_DAYS = 30


_SAVED_STAMP = re.compile(r"\d{8}-\d{6}(-\d{2})?")


def _saved_copies(path: Path) -> list[Path]:
    """The copies this module made (a file named otherwise, by the person say, is not one and is never removed)."""
    return sorted(
        copy
        for copy in path.parent.glob(f"{path.name}.saved-*")
        if _SAVED_STAMP.fullmatch(copy.name.rsplit(".saved-", 1)[1])
    )


def _copies_to_keep(copies: list[Path], today: str) -> set[Path]:
    """The newest SAVED_COPIES, and the earliest copy of each of the last SAVED_DAYS days (names sort by time)."""
    keep = set(copies[-SAVED_COPIES:]) if SAVED_COPIES else set()
    first_of_day: dict[str, Path] = {}
    for copy in copies:  # oldest first: the first seen for a day is its earliest
        first_of_day.setdefault(copy.name.rsplit(".saved-", 1)[1][:8], copy)
    newest = time.mktime(time.strptime(today, "%Y%m%d"))
    for day, copy in first_of_day.items():
        age_days = (newest - time.mktime(time.strptime(day, "%Y%m%d"))) / 86400
        if age_days < SAVED_DAYS:
            keep.add(copy)
    return keep


def _save_with_backup(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, first keeping the file it replaces when that differs."""
    previous = _read_text(path) if path.exists() else None
    if previous is not None and previous != content:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        copy = path.with_name(f"{path.name}.saved-{stamp}")
        number = 1
        while copy.exists():  # two saves in one second
            number += 1
            copy = path.with_name(f"{path.name}.saved-{stamp}-{number:02d}")
        try:
            shutil.copy2(path, copy)
        except OSError:
            logger.warning("Could not keep a copy of %s before saving", path, exc_info=True)
        copies = _saved_copies(path)
        keep = _copies_to_keep(copies, time.strftime("%Y%m%d"))
        for old in copies:
            if old not in keep:
                with contextlib.suppress(OSError):
                    old.unlink()
    _atomic_write_text(path, content)


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a UTF-8 config file only after its full contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows another program (antivirus, an indexer, an editor) can hold the file for a
        # moment; retry briefly instead of failing the setting change.
        for attempt in range(5):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


logger = logging.getLogger("pilot.config")


def _validate_config_types(raw: dict) -> None:
    """Validate that the user's config has no typos and uses correct types."""
    expected_types = {
        "model": {
            "provider": str,
            "ollama_base_url": str,
            "ollama_model": str,
            "mode": str,
            "gpu_memory_limit_mb": int,
            "idle_unload_seconds": int,
            "cloud_provider": str,
            "cloud_model": str,
            "subscription_provider": str,
            "subscription_model": str,
            "subscription_timeout_seconds": int,
            "subscription_max_prompt_chars": int,
            "rate_limit_enabled": bool,
            "rate_limit_rpm": int,
            "rate_limit_burst": int,
            "budget_enabled": bool,
            "budget_monthly_limit_usd": (int, float),
            "max_tokens_per_action": int,
            "max_tokens_per_task": int,
            "max_usd_per_task": (int, float),
            "max_consecutive_failures": int,
        },
        "security": {
            "root_enabled": bool,
            "dry_run": bool,
            "snapshot_on_destructive": bool,
            "snapshot_backend": str,
            "snapshot_retention_count": int,
            "snapshot_retention_days": int,
            "unrestricted_shell": bool,
            "sandbox_mode": str,
            "sandbox_memory_mb": int,
            "sandbox_timeout": int,
            "sandbox_network": bool,
            "sandbox_kernel_guard": bool,
            "sandbox_blocked_syscalls": list,
            "sandbox_firecracker_binary": str,
            "sandbox_firecracker_kernel_image": str,
            "sandbox_firecracker_rootfs_path": str,
            "sandbox_firecracker_fallback": bool,
        },
        "server": {
            "host": str,
            "port": int,
            "auth_token": str,
        },
        "voice": {
            "language": str,
            "transcription_engine": str,
            "whisper_model": str,
            "input_device": str,
            "vad_energy_threshold": (int, float),
            "vad_adaptive_threshold_enabled": bool,
            "vad_silence_ms": (int, float),
            "vad_max_utterance_seconds": (int, float),
            "turn_detection": str,
            "final_pass": str,
            "quiet_in_calls": bool,
            "parakeet_model_dir": str,
            "turn_hold_ms": (int, float),
            "barge_in_enabled": bool,
            "continuous_conversation_enabled": bool,
            "follow_up_window_seconds": (int, float),
            "tts_engine": str,
            "tts_voice": str,
            "system_voice": str,
            "tts_keep_loaded_seconds": (int, float),
            "light_on_small_memory": bool,
            "headset_input_choices": dict,
            "tts_device": str,
        },
        "screen_vision": {
            "capture_interval_seconds": (int, float),
            "capture_timeout_seconds": (int, float),
            "max_consecutive_timeouts": int,
            "auto_resume_after_seconds": (int, float),
        },
        "vision": {
            "mediapipe_backend": str,
            "gaze_tracking_enabled": bool,
        },
        "gesture_cursor": {
            "enabled": bool,
            "sensitivity": (int, float),
            "prediction_ms": (int, float),
            "blend": (int, float),
        },
        "adaptive_calibration": {
            "gesture_enabled": bool,
            "voice_wake_word_enabled": bool,
        },
        "gateway": {
            "enabled": bool,
            "source_profiles": dict,
            "risk_gate_enabled": bool,
        },
        "gesture_workflows": {
            "enabled": bool,
            "bindings": list,
            "pending_trigger_window_seconds": (int, float),
            "paused_window_seconds": (int, float),
        },
        "self_healing": {
            "enabled": bool,
            "auto_execute_max_tier": int,
            "cooldown_seconds": (int, float),
            "confirm_timeout_seconds": (int, float),
            "watched_metrics": list,
            "goal_templates": dict,
        },
        "narration": {
            "enabled": bool,
            "narrate_steps": bool,
            "interrupt_on_risk": bool,
            "proactive_review_enabled": bool,
            "live_corrections_enabled": bool,
            "follow_up_enabled": bool,
            "max_auto_revisions": int,
            "confirm_timeout_seconds": (int, float),
            "advisory_timeout_seconds": (int, float),
        },
        "preview": {
            "enabled": bool,
            "confirm_timeout_seconds": (int, float),
        },
        "undo": {
            "enabled": bool,
            "retention_days": int,
            "max_task_mb": int,
            "max_total_mb": int,
        },
        "receipts": {
            "enabled": bool,
            "retention_days": int,
        },
        "storage": {
            "data_folder_warn_gb": int,
        },
        "timeline": {
            "enabled": bool,
            "retention_days": int,
            "excluded_apps": list,
        },
        "brief": {
            "enabled": bool,
            "time": str,
            "speak": bool,
        },
        "clipboard": {
            "history": bool,
        },
        "suggestions": {
            "enabled": bool,
            "offers": str,
        },
        "live": {
            "enabled": bool,
            "level": str,
            "read_focused_text": bool,
            "ocr_fallback": bool,
            "listen_to_speech": bool,
            "personal_timing": bool,
            "grounded_checks": bool,
            "grounded_model": str,
            "muted_kinds": list,
        },
        "conversation": {
            "enabled": bool,
            "share_screen_text": (str, bool),
            "max_reply_tokens": int,
            "claude_model": str,
            "openai_model": str,
            "openrouter_model": str,
            "gemini_model": str,
            "meta_model": str,
            "ollama_model": str,
            "search_url": str,
            "provisional_answers": str,
            "share_profile": str,
            "second_voice_guard": bool,
            "second_voice_threshold": int,
        },
        "supervision": {
            "enabled": bool,
            "keyboard_mouse_hook_enabled": bool,
            "cognitive_coaching_enabled": bool,
            "risk_pattern_detection_enabled": bool,
            "tick_interval_seconds": (int, float),
            "ocr_interval_seconds": (int, float),
            "stress_coaching_threshold": (int, float),
            "cognitive_load_coaching_threshold": (int, float),
            "coaching_cooldown_seconds": (int, float),
            "risk_cooldown_seconds": (int, float),
            "keystroke_buffer_max_chars": int,
            "ocr_snippet_max_chars": int,
        },
        "co_adaptation": {
            "enabled": bool,
            "adaptive_enabled": bool,
            "learn_across_sessions": bool,
            "gaze_prior_enabled": bool,
            "hand_pointer_enabled": bool,
            "ambiguity_threshold": (int, float),
            "unintended_activation_tolerance": (int, float),
            "noninferiority_margin": (int, float),
            "upper_bound_credibility": (int, float),
            "exploration_floor": (int, float),
            "switching_penalty": (int, float),
            "burden_prompt_every_blocks": int,
            "day_boundary_hour": int,
        },
        "memory": {
            "checkpoint_interval_seconds": int,
            "pruning_interval_seconds": int,
            "pruning_min_memories": int,
            "max_context_tokens": int,
            "max_recent_messages": int,
            "passing_notes": bool,
            "learn_routines": bool,
        },
        "rss": {
            "enabled": bool,
            "feeds": list,
            "poll_interval_hours": (int, float),
            "max_items_per_feed": int,
        },
        "calendar": {
            "enabled": bool,
            "caldav_url": str,
            "caldav_username": str,
            "caldav_password_provider": str,
            "ics_files": list,
        },
        "email": {
            "enabled": bool,
            "imap_host": str,
            "smtp_host": str,
            "smtp_port": int,
            "username": str,
            "password_provider": str,
        },
        "redis": {
            "enabled": bool,
            "host": str,
            "port": int,
            "db": int,
            "password": str,
            "ssl": bool,
            "key_prefix": str,
            "default_ttl": int,
            "max_memory_cache_size": int,
        },
        "network": {
            "enabled": bool,
            "port": int,
            "peer_timeout_s": int,
            "skill_sync_enabled": bool,
            "collab_exec_enabled": bool,
        },
        "air_handoff": {
            "enabled": bool,
            "port": int,
            "max_transfer_mb": int,
            "accept_requests": bool,
            "run_read_only_requests": bool,
        },
        "ssh": {
            "enabled": bool,
            "connect_timeout_seconds": int,
            "allowed_hosts": list,
        },
        "semantic_search": {
            "enabled": bool,
            "folders": list,
            "index_dir": str,
        },
        "cognitive": {
            "enabled": bool,
        },
        "typed_decision": {
            "enabled": bool,
            "scoring_mode": str,
            "artifact_dir": str,
            "report_path": str,
            "model_id": str,
            "max_length": int,
            "temperature": (int, float),
            "review_requests": bool,
            "worker_process": bool,
        },
        "gpu": {
            "max_temperature_c": (int, float),
            "vram_headroom_mb": int,
            "fault_limit": int,
            "fault_window_seconds": (int, float),
        },
        "proxy": {
            "http": str,
            "https": str,
            "no_proxy": str,
        },
    }

    allowed_top_level = set(expected_types) | {"first_run_complete"}
    unknown_sections = sorted(set(raw) - allowed_top_level)
    if unknown_sections:
        error_msg = f"Invalid config section found: '{unknown_sections[0]}'. Please check for typos."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if "first_run_complete" in raw and not isinstance(raw["first_run_complete"], bool):
        raise ValueError("Invalid type: 'first_run_complete' must be bool.")

    for section, expected_keys in expected_types.items():
        if section in raw and isinstance(raw[section], dict):
            for actual_key, actual_value in raw[section].items():
                # Catch invalid keys
                if actual_key not in expected_keys:
                    error_msg = f"Invalid config key found: '{section}.{actual_key}'. Please check for typos."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Catch invalid types
                expected_type = expected_keys[actual_key]
                numeric_type = expected_type in {int, float} or (
                    isinstance(expected_type, tuple) and any(item in {int, float} for item in expected_type)
                )
                if not isinstance(actual_value, expected_type) or (numeric_type and isinstance(actual_value, bool)):
                    error_msg = (
                        f"Invalid type: '{section}.{actual_key}' must be "
                        f"{_format_type_name(expected_type)}, got "
                        f"{type(actual_value).__name__}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

    if "proxy" in raw and isinstance(raw["proxy"], dict):
        _validate_proxy_section(raw["proxy"])


def _validate_proxy_section(raw: dict[str, Any]) -> None:
    for key in ("http", "https"):
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid proxy configuration: proxy.{key} must be a string.")
        _validate_proxy_url(value, key)

    no_proxy_value = raw.get("no_proxy")
    if no_proxy_value is not None and not isinstance(no_proxy_value, str):
        raise ValueError("Invalid proxy configuration: proxy.no_proxy must be a string.")


def _format_type_name(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(t.__name__ for t in expected_type)
    return expected_type.__name__


def _merge_config(config: PilotConfig, raw: dict[str, Any]) -> PilotConfig:
    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)

    if "security" in raw:
        for k, v in raw["security"].items():
            if hasattr(config.security, k):
                setattr(config.security, k, v)

    if "server" in raw:
        for k, v in raw["server"].items():
            if hasattr(config.server, k):
                setattr(config.server, k, v)

    if "voice" in raw:
        for k, v in raw["voice"].items():
            if hasattr(config.voice, k):
                setattr(config.voice, k, v)

    if "screen_vision" in raw:
        for k, v in raw["screen_vision"].items():
            if hasattr(config.screen_vision, k):
                if k == "max_consecutive_timeouts":
                    setattr(config.screen_vision, k, int(v))
                else:
                    setattr(config.screen_vision, k, float(v))

    if "vision" in raw:
        for k, v in raw["vision"].items():
            if hasattr(config.vision, k):
                setattr(config.vision, k, v)

    if "gesture_cursor" in raw:
        for k, v in raw["gesture_cursor"].items():
            if hasattr(config.gesture_cursor, k):
                if k == "enabled":
                    setattr(config.gesture_cursor, k, bool(v))
                else:
                    setattr(config.gesture_cursor, k, float(v))

    if "adaptive_calibration" in raw:
        for k, v in raw["adaptive_calibration"].items():
            if hasattr(config.adaptive_calibration, k):
                setattr(config.adaptive_calibration, k, bool(v))

    if "gateway" in raw and isinstance(raw["gateway"], dict):
        gateway_raw = raw["gateway"]
        config.gateway.enabled = bool(gateway_raw.get("enabled", config.gateway.enabled))
        config.gateway.risk_gate_enabled = bool(gateway_raw.get("risk_gate_enabled", config.gateway.risk_gate_enabled))

        profiles_raw = gateway_raw.get("source_profiles", {})
        if isinstance(profiles_raw, dict):
            parsed_profiles: dict[str, SourceProfile] = dict(config.gateway.source_profiles)
            for name, profile_raw in profiles_raw.items():
                if not isinstance(profile_raw, dict):
                    continue
                default = parsed_profiles.get(name, SourceProfile())
                if "max_tier" in profile_raw and isinstance(profile_raw["max_tier"], dict):
                    # A family left out of a written max_tier table is read-only, not unlimited.
                    max_tier = {family.value: int(PermissionTier.READ_ONLY) for family in ActionFamily}
                    max_tier.update({str(k): int(v) for k, v in profile_raw["max_tier"].items()})
                else:
                    max_tier = dict(default.max_tier)
                parsed_profiles[name] = SourceProfile(
                    max_tier=max_tier,
                    deny_action_types=[str(a) for a in profile_raw.get("deny_action_types", default.deny_action_types)],
                    allow_root=bool(profile_raw.get("allow_root", default.allow_root)),
                )
            config.gateway.source_profiles = parsed_profiles

    if "gesture_workflows" in raw and isinstance(raw["gesture_workflows"], dict):
        gw_raw = raw["gesture_workflows"]
        config.gesture_workflows.enabled = bool(gw_raw.get("enabled", config.gesture_workflows.enabled))
        if "pending_trigger_window_seconds" in gw_raw:
            config.gesture_workflows.pending_trigger_window_seconds = float(gw_raw["pending_trigger_window_seconds"])
        if "paused_window_seconds" in gw_raw:
            config.gesture_workflows.paused_window_seconds = float(gw_raw["paused_window_seconds"])

        bindings_raw = gw_raw.get("bindings", [])
        if isinstance(bindings_raw, list):
            parsed_bindings: list[GestureWorkflowBinding] = []
            seen_gestures: set[str] = set()
            for item in bindings_raw:
                if not isinstance(item, dict):
                    continue
                gesture_name = str(item.get("gesture_name", "")).strip().lower()
                goal_template = str(item.get("goal_template", "")).strip()
                if (
                    gesture_name not in SUPPORTED_GESTURE_WORKFLOW_GESTURES
                    or not goal_template
                    or gesture_name in seen_gestures
                ):
                    continue
                seen_gestures.add(gesture_name)
                parsed_bindings.append(
                    GestureWorkflowBinding(
                        gesture_name=gesture_name,
                        goal_template=goal_template,
                        enabled=bool(item.get("enabled", True)),
                    )
                )
            config.gesture_workflows.bindings = parsed_bindings

    if "self_healing" in raw and isinstance(raw["self_healing"], dict):
        sh_raw = raw["self_healing"]
        config.self_healing.enabled = bool(sh_raw.get("enabled", config.self_healing.enabled))
        if "auto_execute_max_tier" in sh_raw:
            config.self_healing.auto_execute_max_tier = int(sh_raw["auto_execute_max_tier"])
        if "cooldown_seconds" in sh_raw:
            config.self_healing.cooldown_seconds = float(sh_raw["cooldown_seconds"])
        if "confirm_timeout_seconds" in sh_raw:
            config.self_healing.confirm_timeout_seconds = float(sh_raw["confirm_timeout_seconds"])
        if "watched_metrics" in sh_raw and isinstance(sh_raw["watched_metrics"], list):
            config.self_healing.watched_metrics = [str(m) for m in sh_raw["watched_metrics"]]
        if "goal_templates" in sh_raw and isinstance(sh_raw["goal_templates"], dict):
            config.self_healing.goal_templates = {str(k): str(v) for k, v in sh_raw["goal_templates"].items()}

    if "narration" in raw and isinstance(raw["narration"], dict):
        n_raw = raw["narration"]
        config.narration.enabled = bool(n_raw.get("enabled", config.narration.enabled))
        if "narrate_steps" in n_raw:
            config.narration.narrate_steps = bool(n_raw["narrate_steps"])
        if "interrupt_on_risk" in n_raw:
            config.narration.interrupt_on_risk = bool(n_raw["interrupt_on_risk"])
        if "proactive_review_enabled" in n_raw:
            config.narration.proactive_review_enabled = bool(n_raw["proactive_review_enabled"])
        if "live_corrections_enabled" in n_raw:
            config.narration.live_corrections_enabled = bool(n_raw["live_corrections_enabled"])
        if "follow_up_enabled" in n_raw:
            config.narration.follow_up_enabled = bool(n_raw["follow_up_enabled"])
        if "advisory_timeout_seconds" in n_raw:
            config.narration.advisory_timeout_seconds = max(
                1.0,
                min(30.0, float(n_raw["advisory_timeout_seconds"])),
            )
        if "max_auto_revisions" in n_raw:
            config.narration.max_auto_revisions = int(n_raw["max_auto_revisions"])
        if "confirm_timeout_seconds" in n_raw:
            config.narration.confirm_timeout_seconds = float(n_raw["confirm_timeout_seconds"])

    if "preview" in raw and isinstance(raw["preview"], dict):
        pv_raw = raw["preview"]
        config.preview.enabled = bool(pv_raw.get("enabled", config.preview.enabled))
        if "confirm_timeout_seconds" in pv_raw:
            config.preview.confirm_timeout_seconds = float(pv_raw["confirm_timeout_seconds"])

    if "undo" in raw and isinstance(raw["undo"], dict):
        u_raw = raw["undo"]
        config.undo.enabled = bool(u_raw.get("enabled", config.undo.enabled))
        if "retention_days" in u_raw:
            config.undo.retention_days = min(max(int(u_raw["retention_days"]), 1), 90)
        if "max_task_mb" in u_raw:
            config.undo.max_task_mb = min(max(int(u_raw["max_task_mb"]), 1), 10240)
        if "max_total_mb" in u_raw:
            config.undo.max_total_mb = min(max(int(u_raw["max_total_mb"]), 16), 102400)

    if "receipts" in raw and isinstance(raw["receipts"], dict):
        r_raw = raw["receipts"]
        config.receipts.enabled = bool(r_raw.get("enabled", config.receipts.enabled))
        if "retention_days" in r_raw:
            config.receipts.retention_days = min(max(int(r_raw["retention_days"]), 1), 3650)

    if "storage" in raw and isinstance(raw["storage"], dict) and "data_folder_warn_gb" in raw["storage"]:
        config.storage.data_folder_warn_gb = min(max(int(raw["storage"]["data_folder_warn_gb"]), 1), 100000)

    if "timeline" in raw and isinstance(raw["timeline"], dict):
        t_raw = raw["timeline"]
        config.timeline.enabled = bool(t_raw.get("enabled", config.timeline.enabled))
        if "retention_days" in t_raw:
            config.timeline.retention_days = min(max(int(t_raw["retention_days"]), 1), 90)
        if isinstance(t_raw.get("excluded_apps"), list):
            config.timeline.excluded_apps = [str(name)[:80] for name in t_raw["excluded_apps"][:50]]

    if "brief" in raw and isinstance(raw["brief"], dict):
        b_raw = raw["brief"]
        config.brief.enabled = bool(b_raw.get("enabled", config.brief.enabled))
        config.brief.speak = bool(b_raw.get("speak", config.brief.speak))
        if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(b_raw.get("time", ""))):
            config.brief.time = str(b_raw["time"])

    if "clipboard" in raw and isinstance(raw["clipboard"], dict):
        config.clipboard.history = bool(raw["clipboard"].get("history", config.clipboard.history))

    if "suggestions" in raw and isinstance(raw["suggestions"], dict):
        config.suggestions.enabled = bool(raw["suggestions"].get("enabled", config.suggestions.enabled))
        offers = raw["suggestions"].get("offers")
        if isinstance(offers, str) and offers.strip().casefold() in ("auto", "on", "off"):
            config.suggestions.offers = offers.strip().casefold()

    if "live" in raw and isinstance(raw["live"], dict):
        l_raw = raw["live"]
        for key in (
            "enabled",
            "read_focused_text",
            "ocr_fallback",
            "listen_to_speech",
            "personal_timing",
            "grounded_checks",
        ):
            setattr(config.live, key, bool(l_raw.get(key, getattr(config.live, key))))
        model = l_raw.get("grounded_model")
        if isinstance(model, str) and _OLLAMA_MODEL_NAME.fullmatch(model.strip()):
            config.live.grounded_model = model.strip()
        if l_raw.get("level") in ("quiet", "balanced", "proactive"):
            config.live.level = str(l_raw["level"])
        if isinstance(l_raw.get("muted_kinds"), list):
            from pilot.live.detectors import KINDS

            config.live.muted_kinds = [str(k) for k in l_raw["muted_kinds"] if str(k) in KINDS]

    if "conversation" in raw and isinstance(raw["conversation"], dict):
        c_raw = raw["conversation"]
        config.conversation.enabled = bool(c_raw.get("enabled", config.conversation.enabled))
        share = c_raw.get("share_screen_text")
        if share is True or share is False:
            config.conversation.share_screen_text = "on" if share else "off"
        elif isinstance(share, str) and share.strip().casefold() in ("auto", "on", "off"):
            config.conversation.share_screen_text = share.strip().casefold()
        tokens = c_raw.get("max_reply_tokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            config.conversation.max_reply_tokens = max(60, min(tokens, 2000))
        for key in ("claude", "openai", "openrouter", "gemini", "meta", "ollama"):
            name = c_raw.get(f"{key}_model")
            if isinstance(name, str) and len(name.strip()) <= 120:
                setattr(config.conversation, f"{key}_model", name.strip())
        search = c_raw.get("search_url")
        if (
            isinstance(search, str)
            and "{query}" in search
            and re.match(r"^https?://[^\s/$.?#][^\s]*$", search.strip().replace("{query}", "q"), re.IGNORECASE)
            and len(search) <= 300
        ):
            config.conversation.search_url = search.strip()
        early = c_raw.get("provisional_answers")
        if isinstance(early, str) and early.strip().casefold() in ("auto", "on", "off"):
            config.conversation.provisional_answers = early.strip().casefold()
        share = c_raw.get("share_profile")
        if isinstance(share, str) and share.strip().casefold() in ("auto", "off"):
            config.conversation.share_profile = share.strip().casefold()
        guard = c_raw.get("second_voice_guard")
        if isinstance(guard, bool):
            config.conversation.second_voice_guard = guard
        threshold = c_raw.get("second_voice_threshold")
        if isinstance(threshold, int) and not isinstance(threshold, bool) and 1 <= threshold <= 10:
            config.conversation.second_voice_threshold = threshold

    if "supervision" in raw and isinstance(raw["supervision"], dict):
        s_raw = raw["supervision"]
        config.supervision.enabled = bool(s_raw.get("enabled", config.supervision.enabled))
        if "keyboard_mouse_hook_enabled" in s_raw:
            config.supervision.keyboard_mouse_hook_enabled = bool(s_raw["keyboard_mouse_hook_enabled"])
        if "cognitive_coaching_enabled" in s_raw:
            config.supervision.cognitive_coaching_enabled = bool(s_raw["cognitive_coaching_enabled"])
        if "risk_pattern_detection_enabled" in s_raw:
            config.supervision.risk_pattern_detection_enabled = bool(s_raw["risk_pattern_detection_enabled"])
        if "tick_interval_seconds" in s_raw:
            config.supervision.tick_interval_seconds = float(s_raw["tick_interval_seconds"])
        if "ocr_interval_seconds" in s_raw:
            config.supervision.ocr_interval_seconds = float(s_raw["ocr_interval_seconds"])
        if "stress_coaching_threshold" in s_raw:
            config.supervision.stress_coaching_threshold = float(s_raw["stress_coaching_threshold"])
        if "cognitive_load_coaching_threshold" in s_raw:
            config.supervision.cognitive_load_coaching_threshold = float(s_raw["cognitive_load_coaching_threshold"])
        if "coaching_cooldown_seconds" in s_raw:
            config.supervision.coaching_cooldown_seconds = float(s_raw["coaching_cooldown_seconds"])
        if "risk_cooldown_seconds" in s_raw:
            config.supervision.risk_cooldown_seconds = float(s_raw["risk_cooldown_seconds"])
        if "keystroke_buffer_max_chars" in s_raw:
            config.supervision.keystroke_buffer_max_chars = int(s_raw["keystroke_buffer_max_chars"])
        if "ocr_snippet_max_chars" in s_raw:
            config.supervision.ocr_snippet_max_chars = int(s_raw["ocr_snippet_max_chars"])

    if "co_adaptation" in raw and isinstance(raw["co_adaptation"], dict):
        ca_raw = raw["co_adaptation"]
        for key in sorted(CO_ADAPTATION_BOOL_KEYS):
            if key in ca_raw:
                setattr(config.co_adaptation, key, bool(ca_raw[key]))
        for key, (low, high) in CO_ADAPTATION_FLOAT_RANGES.items():
            # min/max would turn a NaN into the upper bound, so a non-finite value is ignored.
            if key in ca_raw and math.isfinite(float(ca_raw[key])):
                setattr(config.co_adaptation, key, max(low, min(high, float(ca_raw[key]))))
        for key, (low, high) in CO_ADAPTATION_INT_RANGES.items():
            if key in ca_raw:
                setattr(config.co_adaptation, key, max(low, min(high, int(ca_raw[key]))))

    if "memory" in raw:
        for k, v in raw["memory"].items():
            if hasattr(config.memory, k):
                if k in (
                    "max_context_tokens",
                    "max_recent_messages",
                    "checkpoint_interval_seconds",
                    "pruning_interval_seconds",
                    "pruning_min_memories",
                ):
                    setattr(config.memory, k, int(v))
                else:
                    setattr(config.memory, k, v)

    if "rss" in raw:
        for k, v in raw["rss"].items():
            if hasattr(config.rss, k):
                setattr(config.rss, k, v)

    if "calendar" in raw:
        for k, v in raw["calendar"].items():
            if hasattr(config.calendar, k):
                setattr(config.calendar, k, v)

    if "email" in raw:
        for k, v in raw["email"].items():
            if hasattr(config.email, k):
                setattr(config.email, k, v)

    if "network" in raw:
        for k, v in raw["network"].items():
            if hasattr(config.network, k):
                setattr(config.network, k, v)

    if "air_handoff" in raw:
        for k, v in raw["air_handoff"].items():
            if hasattr(config.air_handoff, k):
                if k == "port":
                    setattr(config.air_handoff, k, max(1024, min(65535, int(v))))
                elif k == "max_transfer_mb":
                    setattr(config.air_handoff, k, max(1, min(250, int(v))))
                else:
                    setattr(config.air_handoff, k, bool(v))

    if "ssh" in raw and isinstance(raw["ssh"], dict):
        ssh_raw = raw["ssh"]
        config.ssh.enabled = bool(ssh_raw.get("enabled", config.ssh.enabled))
        if "connect_timeout_seconds" in ssh_raw:
            config.ssh.connect_timeout_seconds = int(ssh_raw["connect_timeout_seconds"])

        hosts_raw = ssh_raw.get("allowed_hosts", [])
        if isinstance(hosts_raw, list):
            parsed_hosts: list[SshHostConfig] = []
            for item in hosts_raw:
                if not isinstance(item, dict):
                    continue
                parsed_hosts.append(
                    SshHostConfig(
                        name=str(item.get("name", "")),
                        hostname=str(item.get("hostname", "")),
                        port=int(item.get("port", 22)),
                        username=str(item.get("username", "")),
                        private_key_provider=str(item.get("private_key_provider", "")),
                        passphrase_provider=str(item.get("passphrase_provider", "")),
                        strict_host_key_checking=bool(item.get("strict_host_key_checking", True)),
                    )
                )
            config.ssh.allowed_hosts = parsed_hosts
    if "redis" in raw:
        for k, v in raw["redis"].items():
            if hasattr(config.redis, k):
                setattr(config.redis, k, v)

    if "semantic_search" in raw:
        for k, v in raw["semantic_search"].items():
            if hasattr(config.semantic_search, k):
                setattr(config.semantic_search, k, v)

    if "cognitive" in raw:
        for k, v in raw["cognitive"].items():
            if hasattr(config.cognitive, k):
                setattr(config.cognitive, k, v)

    if "typed_decision" in raw:
        for k, v in raw["typed_decision"].items():
            if hasattr(config.typed_decision, k):
                setattr(config.typed_decision, k, v)

    if "gpu" in raw and isinstance(raw["gpu"], dict):
        for k, v in raw["gpu"].items():
            if hasattr(config.gpu, k):
                setattr(config.gpu, k, v)

    if "proxy" in raw and isinstance(raw["proxy"], dict):
        for k, v in raw["proxy"].items():
            if hasattr(config.proxy, k):
                if k in ("http", "https", "no_proxy"):
                    if not isinstance(v, str):
                        error_msg = f"Invalid type: 'proxy.{k}' must be str, got {type(v).__name__}."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                if k in ("http", "https") and v:
                    _validate_proxy_url(v, k)
                setattr(config.proxy, k, v)

    config.first_run_complete = raw.get(
        "first_run_complete",
        config.first_run_complete,
    )

    return config


def _validate_proxy_url(url: str, key: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid proxy URL for proxy.{key}: {url}")


def _parse_restrictions(raw: dict[str, Any]) -> Restrictions:
    return Restrictions(
        protected_folders=raw.get("protected_folders", []),
        protected_packages=raw.get("protected_packages", []),
        blocked_commands=raw.get("blocked_commands", []),
        sandbox_allowed_commands=raw.get(
            "sandbox_allowed_commands", ["echo", "ls", "dir", "cat", "type", "ping", "whoami", "pwd", "grep", "find"]
        ),
    )


def _config_to_dict(config: PilotConfig) -> dict[str, Any]:
    from dataclasses import asdict

    def strip_none(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip_none(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [strip_none(item) for item in value]
        return value

    return strip_none(asdict(config))


def _probe_writable(folder: Path) -> None:
    """Write and remove a file of this process's own in ``folder``; raises if the folder cannot be written.

    The name is unique to the process and the call: two copies of Heliox starting at the same moment (a double-click
    on the icon) used to share one ".write_test", so one removed the file the other was about to remove, and the
    loser reported the data folder as not writable and ended with a traceback instead of the sentence about the
    copy already running. A file someone else removed does not make the folder unwritable."""
    import uuid

    probe = folder / f".write_test-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    probe.write_text("test")
    probe.unlink(missing_ok=True)


def ensure_dirs() -> None:
    """Create all required XDG directories and validate write access.

    A data folder that cannot be written because the disk is full or the drive or folder is read-only does not stop
    the daemon here: start-up still listens and answers, says why, and checks again every 30 s
    (pilot.system.storage_health.preflight). Before, this raised, and the daemon ended before it listened, so the
    window could not connect and nobody was told why. Anything else (a file where the folder should be) still raises.
    """
    from pilot.system.storage_health import classify

    data_folders = (DATA_DIR, SCREENSHOTS_DIR)
    blocked = False
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, RUNTIME_DIR, PLUGINS_DIR, SCREENSHOTS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            if d not in data_folders or classify(error) is None:
                raise
            if not blocked:
                logger.warning("The data folder cannot be written (%s); start-up says so and waits for it", error)
            blocked = True  # the other folders are still made: the log, the tokens and the settings need them
    if blocked:
        return

    try:
        _probe_writable(DATA_DIR)
    except Exception as e:
        if isinstance(e, OSError) and classify(e) is not None:
            logger.warning("The data folder cannot be written (%s); start-up says so and waits for it", e)
            return
        logger.error(f"DATA_DIR is not writable: {DATA_DIR}")
        raise RuntimeError(f"DATA_DIR is not writable: {DATA_DIR}") from e


def migrate_old_paths() -> None:
    """Migrate data from old hardcoded path conventions to the new canonical layout.

    Old paths migrated:
      ~/.heliox/plugins/       → PLUGINS_DIR
      ~/.heliox/screenshots/   → SCREENSHOTS_DIR
      ~/.heliox/persona.md     → PERSONA_FILE
      ~/.config/pilot/         → CONFIG_DIR (config only, non-destructive)
    """
    home = Path.home()
    old_pairs: list[tuple[Path, Path]] = []

    old_heliox_plugins = home / ".heliox" / "plugins"
    if old_heliox_plugins.exists() and old_heliox_plugins.is_dir():
        old_pairs.append((old_heliox_plugins, PLUGINS_DIR))

    old_heliox_screenshots = home / ".heliox" / "screenshots"
    if old_heliox_screenshots.exists() and old_heliox_screenshots.is_dir():
        old_pairs.append((old_heliox_screenshots, SCREENSHOTS_DIR))

    old_persona = home / ".heliox" / "persona.md"
    if old_persona.exists() and old_persona.is_file():
        old_pairs.append((old_persona, PERSONA_FILE))

    old_config_pilot = home / ".config" / "pilot"
    if old_config_pilot.exists() and old_config_pilot.is_dir():
        old_pairs.append((old_config_pilot, CONFIG_DIR))

    for src, dst in old_pairs:
        if dst.exists():
            logger.info("Migration: skipping %s → %s (destination already exists)", src, dst)
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                src.rename(dst)
            else:
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    target = dst / item.name
                    if not target.exists():
                        item.rename(target)
                src.rmdir()
            logger.info("Migration: moved %s → %s", src, dst)
        except OSError as exc:
            logger.warning("Migration: could not move %s → %s: %s", src, dst, exc)
