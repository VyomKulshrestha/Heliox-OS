"""Runtime configuration loader and manager."""

from __future__ import annotations

import os
import sys
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


@dataclass
class SecurityConfig:
    root_enabled: bool = False
    confirm_tier2: bool = True
    snapshot_on_destructive: bool = True
    snapshot_backend: str = "auto"  # "auto" | "btrfs" | "timeshift" | "none"
    snapshot_retention_count: int = 10
    snapshot_retention_days: int = 7
    unrestricted_shell: bool = False  # Allow ANY shell command (bypass whitelist)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8785
    auth_token: str = ""


@dataclass
class Restrictions:
    protected_folders: list[str] = field(default_factory=list)
    protected_packages: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)


@dataclass
class PilotConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    restrictions: Restrictions = field(default_factory=Restrictions)
    first_run_complete: bool = False

    @classmethod
    def load(cls) -> PilotConfig:
        """Load config from disk, creating defaults if missing."""
        config = cls()
        if CONFIG_FILE.exists():
            raw = tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            config = _merge_config(config, raw)
        if RESTRICTIONS_FILE.exists():
            raw = tomllib.loads(RESTRICTIONS_FILE.read_text(encoding="utf-8"))
            config.restrictions = _parse_restrictions(raw)
        return config

    def save(self) -> None:
        """Persist current config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = _config_to_dict(self)
        restrictions = data.pop("restrictions", {})
        CONFIG_FILE.write_text(tomli_w.dumps(data), encoding="utf-8")
        if restrictions:
            RESTRICTIONS_FILE.write_text(tomli_w.dumps(restrictions), encoding="utf-8")


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
    config.first_run_complete = raw.get("first_run_complete", config.first_run_complete)
    return config


def _parse_restrictions(raw: dict[str, Any]) -> Restrictions:
    return Restrictions(
        protected_folders=raw.get("protected_folders", []),
        protected_packages=raw.get("protected_packages", []),
        blocked_commands=raw.get("blocked_commands", []),
    )


def _config_to_dict(config: PilotConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(config)


def ensure_dirs() -> None:
    """Create all required XDG directories."""
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, RUNTIME_DIR):
        d.mkdir(parents=True, exist_ok=True)
