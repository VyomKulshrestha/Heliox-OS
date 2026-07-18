"""Runtime configuration loader and manager."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
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

from pilot.security.gateway import DEFAULT_SOURCE_PROFILES, SourceProfile


def _xdg(env_var: str, fallback: str) -> Path:
    return Path(os.environ.get(env_var, Path.home() / fallback))


def _default_runtime_dir() -> Path:
    """Resolve runtime dir when XDG_RUNTIME_DIR is unset (macOS, Windows, minimal Linux)."""
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


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / "heliox-os"
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share") / "heliox-os"
STATE_DIR = _xdg("XDG_STATE_HOME", ".local/state") / "heliox-os"
RUNTIME_DIR = _default_runtime_dir()
CONFIG_FILE = CONFIG_DIR / "config.toml"
RESTRICTIONS_FILE = CONFIG_DIR / "restrictions.toml"
DB_FILE = DATA_DIR / "pilot.db"
AUDIT_FILE = DATA_DIR / "audit.jsonl"
PERMISSION_AUDIT_DB_FILE = DATA_DIR / "permission_audit.db"
PERMISSION_AUDIT_KEY_FILE = DATA_DIR / "permission_audit.key"
AGENT_GATEWAY_AUDIT_DB_FILE = DATA_DIR / "agent_gateway_audit.db"
AGENT_GATEWAY_AUDIT_KEY_FILE = DATA_DIR / "agent_gateway_audit.key"
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
    cloud_provider: str = ""  # "openai" | "claude" | "gemini"
    cloud_model: str = ""
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
    max_tokens_per_action: int = 4000  # cap on tokens for a single LLM call
    max_tokens_per_task: int = 50000  # cumulative token cap per orchestrator task
    max_usd_per_task: float = 0.10  # cumulative USD cap per task
    max_consecutive_failures: int = 3  # circuit breaker threshold (Phase 4)


@dataclass
class SecurityConfig:
    root_enabled: bool = False
    confirm_tier2: bool = True
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


@dataclass
class VoiceConfig:
    language: str = "auto"  # auto detect or manual language code
    whisper_model: str = "base"


@dataclass
class ScreenVisionConfig:
    capture_interval_seconds: float = 3.0
    capture_timeout_seconds: float = 10.0
    max_consecutive_timeouts: int = 3
    auto_resume_after_seconds: float = 30.0


@dataclass
class VisionConfig:
    camera_index: int = 0
    # "legacy" keeps the existing @mediapipe/hands normalized-2D pipeline;
    # "tasks" switches to @mediapipe/tasks-vision's HandLandmarker, which
    # additionally exposes real-metric-scale worldLandmarks (see
    # GESTURES.md's "3D World-Model Layer" section and worldModel.ts).
    # Defaults to "legacy" — flipping this restarts the gesture engine
    # rather than hot-swapping mid-session.
    mediapipe_backend: str = "legacy"


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
class SemanticSearchConfig:
    enabled: bool = False
    folders: list[str] = field(default_factory=list)  # List of directories to index
    index_dir: str = ""  # Custom directory to store index (defaults to DATA_DIR/semantic_index)


@dataclass
class CognitiveConfig:
    enabled: bool = True


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
    skill_sync_enabled: bool = True  # broadcast/receive plugins from peers
    collab_exec_enabled: bool = True  # distribute parallelizable action batches


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
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rss: RSSConfig = field(default_factory=RSSConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    semantic_search: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ssh: SshConfig = field(default_factory=SshConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    restrictions: Restrictions = field(default_factory=Restrictions)
    first_run_complete: bool = False
    redis: RedisConfig = field(default_factory=RedisConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)

    @classmethod
    def load(cls) -> PilotConfig:
        """Load config from disk, creating defaults if missing."""
        config = cls()

        if CONFIG_FILE.exists():
            try:
                raw = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                _validate_config_types(raw)
                config = _merge_config(config, raw)
            except Exception as e:
                logger.error(f"Failed to load config.toml: {e}. Falling back to safe defaults.")

        if RESTRICTIONS_FILE.exists():
            raw = tomllib.loads(RESTRICTIONS_FILE.read_text(encoding="utf-8"))
            config.restrictions = _parse_restrictions(raw)

        return config

    def save(self) -> None:
        """Persist current config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        data = _config_to_dict(self)
        restrictions = data.pop("restrictions", {})

        CONFIG_FILE.write_text(
            tomli_w.dumps(data),
            encoding="utf-8",
        )

        if restrictions:
            RESTRICTIONS_FILE.write_text(
                tomli_w.dumps(restrictions),
                encoding="utf-8",
            )


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
            "rate_limit_enabled": bool,
            "rate_limit_rpm": int,
            "rate_limit_burst": int,
            "budget_enabled": bool,
            "budget_monthly_limit_usd": float,
            "max_tokens_per_action": int,
            "max_tokens_per_task": int,
            "max_usd_per_task": float,
            "max_consecutive_failures": int,
        },
        "security": {
            "root_enabled": bool,
            "confirm_tier2": bool,
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
            "whisper_model": str,
        },
        "screen_vision": {
            "capture_interval_seconds": (int, float),
            "capture_timeout_seconds": (int, float),
            "max_consecutive_timeouts": int,
            "auto_resume_after_seconds": (int, float),
        },
        "vision": {
            "camera_index": int,
            "mediapipe_backend": str,
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
        },
        "gesture_workflows": {
            "enabled": bool,
            "bindings": list,
            "pending_trigger_window_seconds": (int, float),
            "paused_window_seconds": (int, float),
        },
        "memory": {
            "checkpoint_interval_seconds": int,
            "pruning_interval_seconds": int,
            "pruning_min_memories": int,
            "max_context_tokens": int,
            "max_recent_messages": int,
        },
        "rss": {
            "enabled": bool,
            "feeds": list,
            "poll_interval_hours": (int, float),
            "max_items_per_feed": int,
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
        "proxy": {
            "http": str,
            "https": str,
            "no_proxy": str,
        },
    }

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
                if not isinstance(actual_value, expected_type):
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

        profiles_raw = gateway_raw.get("source_profiles", {})
        if isinstance(profiles_raw, dict):
            parsed_profiles: dict[str, SourceProfile] = dict(config.gateway.source_profiles)
            for name, profile_raw in profiles_raw.items():
                if not isinstance(profile_raw, dict):
                    continue
                default = parsed_profiles.get(name, SourceProfile())
                parsed_profiles[name] = SourceProfile(
                    max_tier={str(k): int(v) for k, v in profile_raw.get("max_tier", default.max_tier).items()},
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
            for item in bindings_raw:
                if not isinstance(item, dict):
                    continue
                parsed_bindings.append(
                    GestureWorkflowBinding(
                        gesture_name=str(item.get("gesture_name", "")),
                        goal_template=str(item.get("goal_template", "")),
                        enabled=bool(item.get("enabled", True)),
                    )
                )
            config.gesture_workflows.bindings = parsed_bindings

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

    if "network" in raw:
        for k, v in raw["network"].items():
            if hasattr(config.network, k):
                setattr(config.network, k, v)

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


def ensure_dirs() -> None:
    """Create all required XDG directories and validate write access."""
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, RUNTIME_DIR, PLUGINS_DIR, SCREENSHOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    test_file = DATA_DIR / ".write_test"

    try:
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
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
