"""Runtime configuration loader and manager."""
# Issue #194: Startup DATA_DIR write-permission validation.

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


def _xdg(env_var: str, fallback: str) -> Path:
    return Path(os.environ.get(env_var, Path.home() / fallback))


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / "pilot"
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share") / "pilot"
STATE_DIR = _xdg("XDG_STATE_HOME", ".local/state") / "pilot"
RUNTIME_DIR = (
    Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid() if hasattr(os, 'getuid') else 1000}")) / "pilot"
)

CONFIG_FILE = CONFIG_DIR / "config.toml"
RESTRICTIONS_FILE = CONFIG_DIR / "restrictions.toml"
DB_FILE = DATA_DIR / "pilot.db"
AUDIT_FILE = DATA_DIR / "audit.jsonl"
PERMISSION_AUDIT_DB_FILE = DATA_DIR / "permission_audit.db"
PERMISSION_AUDIT_KEY_FILE = DATA_DIR / "permission_audit.key"
LOG_FILE = STATE_DIR / "pilot.log"


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
    sandbox_mode: str = "auto"  # "auto" | "docker" | "restricted" | "none"
    sandbox_memory_mb: int = 128  # memory cap applied inside the sandbox (MB)
    sandbox_timeout: int = 30  # max wall-clock seconds for sandboxed execution
    sandbox_network: bool = False  # allow outbound network inside the sandbox


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
    restrictions: Restrictions = field(default_factory=Restrictions)
    first_run_complete: bool = False

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
                setattr(config.screen_vision, k, float(v))

    config.first_run_complete = raw.get(
        "first_run_complete",
        config.first_run_complete,
    )

    return config


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

    return asdict(config)


def validate_data_dir_writable() -> None:
    """Fast-fail check: verify that every critical data directory is writable.

    Called at daemon startup (via :func:`ensure_dirs`) before any SQLite
    connections or audit-log writes are attempted.  If *any* directory is not
    writable, a clear human-readable error is emitted and a
    ``PermissionError`` is raised so the process exits with a non-zero code
    instead of crashing mid-execution with a cryptic I/O failure.

    Raises:
        PermissionError: If one or more critical directories are not writable
            by the current process.
    """
    _critical_dirs = {
        "CONFIG_DIR": CONFIG_DIR,
        "DATA_DIR": DATA_DIR,
        "STATE_DIR": STATE_DIR,
    }
    failures: list[str] = []

    for name, directory in _critical_dirs.items():
        try:
            # Use a temporary file inside the directory to probe write access.
            # NamedTemporaryFile with delete=True cleans up automatically.
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".pilot_write_probe_", delete=True):
                pass
        except OSError as exc:
            failures.append(
                f"  • {name} ({directory}): {exc.strerror} [errno {exc.errno}]"
            )
            logger.error(
                "Startup permission check FAILED for %s (%s): %s",
                name,
                directory,
                exc.strerror,
            )

    if failures:
        hint = (
            f"Run: sudo chown -R $USER {DATA_DIR.parent}  "
            f"(or set XDG_DATA_HOME to a writable path)"
        )
        msg = (
            "Pilot daemon cannot start — the following directories are not writable:\n"
            + "\n".join(failures)
            + "\n"
            + hint
        )
        logger.critical(msg)
        raise PermissionError(msg)

    logger.debug(
        "Startup write-permission check passed for: %s",
        ", ".join(_critical_dirs.keys()),
    )


def ensure_dirs() -> None:
    """Create all required XDG directories and verify they are writable.

    This is the single call-site for all directory bootstrap logic.  It must
    be invoked before any agent subsystem or database is initialised so that
    permission failures surface at startup rather than mid-execution.
    """
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, RUNTIME_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Fast-fail: ensure the daemon can actually write to its own directories
    # before SQLite checkpoints (PR #165) or audit logs attempt to do so.
    validate_data_dir_writable()
