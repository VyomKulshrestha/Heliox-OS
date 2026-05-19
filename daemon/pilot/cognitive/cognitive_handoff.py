"""Cross-Device Cognitive Handoff — cognitive state that follows the user across devices.

This module implements the sixth revolutionary feature:
- Detects when user context should transfer to another device
- Syncs cognitive state to a "cognitive cloud"
- Provides seamless handoff between desktop, mobile, tablet
- Maintains context continuity across sessions
- Device-aware suggestions (e.g., "Continue on mobile?")

Architecture:
  Device Monitor → Cognitive Cloud Sync → Handoff Manager → Context Transfer → Device Resumption
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pilot.utils.logger import get_logger

logger = get_logger("pilot.cognitive.cognitive_handoff")

# ── Configuration ──

SYNC_INTERVAL_SECONDS = 30.0  # Sync to cloud every 30s
CLOUD_STORAGE_PATH = Path.home() / ".cache" / "heliox" / "cloud"
MAX_HANDOVERS = 20  # Keep last 20 handoffs
CONTEXT_RETENTION_HOURS = 24

DEVICE_TYPES = StrEnum("DeviceType", "desktop mobile tablet unknown")


# ── Data Structures ──


@dataclass
class CognitiveSnapshot:
    """A snapshot of cognitive state for transfer."""

    snapshot_id: str
    timestamp: float

    # Core cognitive metrics
    attention: float
    stress: float
    load: float

    # Context
    active_app: str = ""
    active_task: str = ""
    recent_actions: list[str] = field(default_factory=list)
    current_workflow: str = ""
    workflow_step: int = 0

    # Workspace state
    workspace_data: dict[str, Any] = field(default_factory=dict)

    # Recommendations
    suggested_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "attention": round(self.attention, 2),
            "stress": round(self.stress, 2),
            "load": round(self.load, 2),
            "active_app": self.active_app,
            "active_task": self.active_task,
            "recent_actions": self.recent_actions[-5:],
            "current_workflow": self.current_workflow,
            "workspace_data": self.workspace_data,
        }


@dataclass
class DeviceSession:
    """A session on a specific device."""

    session_id: str
    device_type: str
    device_name: str
    started_at: float
    last_active: float
    cognitive_snapshots: list[CognitiveSnapshot] = field(default_factory=list)
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "last_active": datetime.fromtimestamp(self.last_active).isoformat(),
            "is_active": self.is_active,
            "snapshot_count": len(self.cognitive_snapshots),
        }


@dataclass
class Handoff:
    """A completed handoff between devices."""

    handoff_id: str
    from_device: DeviceSession
    to_device: DeviceSession
    timestamp: float
    context_synced: bool = True
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "from_device": self.from_device.device_name,
            "to_device": self.to_device.device_name,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "context_synced": self.context_synced,
        }


# ── Storage ──


class CloudStore:
    """Cloud storage for cross-device synchronization."""

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or CLOUD_STORAGE_PATH
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _get_cloud_path(self) -> Path:
        return self._data_dir / "cognitive_cloud.json"

    def push_snapshot(self, snapshot: CognitiveSnapshot) -> None:
        """Push a snapshot to the cloud."""
        path = self._get_cloud_path()

        # Load existing
        data = {"snapshots": [], "last_update": 0.0}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Add snapshot
        data["snapshots"].append(snapshot.to_dict())
        data["last_update"] = time.time()

        # Keep last 10 snapshots
        data["snapshots"] = data["snapshots"][-10:]

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        """Get the latest snapshot from the cloud."""
        path = self._get_cloud_path()

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            snapshots = data.get("snapshots", [])
            return snapshots[-1] if snapshots else None
        except Exception:
            return None

    def save_context(self, context_id: str, context_data: dict[str, Any]) -> None:
        """Save context for handoff."""
        path = self._data_dir / f"context_{context_id}.json"
        data = {
            "context_id": context_id,
            "data": context_data,
            "saved_at": time.time(),
            "expires_at": time.time() + (CONTEXT_RETENTION_HOURS * 3600),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_context(self, context_id: str) -> dict[str, Any] | None:
        """Load context for handoff."""
        path = self._data_dir / f"context_{context_id}.json"

        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() > data.get("expires_at", 0):
                path.unlink()
                return None
            return data.get("data")
        except Exception:
            return None


# ── Core Engine ──


class CognitiveHandoffEngine:
    """Engine that manages cross-device cognitive state."""

    def __init__(self, device_name: str = "desktop", store: CloudStore | None = None):
        self._device_name = device_name
        self._device_type = self._detect_device_type()
        self._store = store or CloudStore()

        # Session management
        self._session = DeviceSession(
            session_id=str(uuid.uuid4()),
            device_type=self._device_type,
            device_name=self._device_name,
            started_at=time.time(),
            last_active=time.time(),
        )

        # Known devices
        self._known_devices: list[DeviceSession] = []

        # Handoff history
        self._handoffs: list[Handoff] = []

        # Sync state
        self._last_sync_time: float = 0.0
        self._pending_sync: bool = False

    def _detect_device_type(self) -> str:
        """Detect the current device type."""
        import platform

        system = platform.system().lower()

        if system == "windows" or system == "darwin" or system == "linux":
            return "desktop"  # Assume desktop for PC
        return "unknown"

    # ── Session Management ──

    def register_activity(self) -> None:
        """Register activity to keep session active."""
        self._session.last_active = time.time()
        self._session.is_active = True

    def pause_session(self) -> None:
        """Pause the current session (user went idle)."""
        self._session.is_active = False

    def end_session(self) -> None:
        """End the current session."""
        self._session.is_active = False

        # Add to known devices
        self._known_devices.append(self._session)

        # Keep only last 5 devices
        if len(self._known_devices) > 5:
            self._known_devices = self._known_devices[-5:]

        # Create new session
        self._session = DeviceSession(
            session_id=str(uuid.uuid4()),
            device_type=self._device_type,
            device_name=self._device_name,
            started_at=time.time(),
            last_active=time.time(),
            is_active=True,
        )

    # ── Cognitive Snapshots ──

    def capture_snapshot(
        self,
        attention: float,
        stress: float,
        load: float,
        active_app: str = "",
        active_task: str = "",
        workflow: str = "",
    ) -> CognitiveSnapshot:
        """Capture current cognitive state as a snapshot."""
        now = time.time()

        # Get recent actions from the session
        recent = [s.to_dict() for s in self._session.cognitive_snapshots[-5:]]

        snapshot = CognitiveSnapshot(
            snapshot_id=str(uuid.uuid4())[:8],
            timestamp=now,
            attention=attention,
            stress=stress,
            load=load,
            active_app=active_app,
            active_task=active_task,
            recent_actions=[a.get("active_task", "") for a in recent],
            current_workflow=workflow,
            workflow_step=0,
        )

        self._session.cognitive_snapshots.append(snapshot)

        # Keep last 10 snapshots in session
        if len(self._session.cognitive_snapshots) > 10:
            self._session.cognitive_snapshots = self._session.cognitive_snapshots[-10:]

        # Sync to cloud
        self._store.push_snapshot(snapshot)

        return snapshot

    def sync_to_cloud(self) -> None:
        """Force sync to cloud."""
        now = time.time()

        if self._session.cognitive_snapshots:
            latest = self._session.cognitive_snapshots[-1]
            self._store.push_snapshot(latest)

        self._last_sync_time = now
        logger.info("Synced cognitive state to cloud")

    # ── Handoff Management ──

    def initiate_handoff(self, target_device: str) -> Handoff | None:
        """Initiate a handoff to another device."""
        # Find target device
        target = None
        for device in self._known_devices:
            if device.device_name == target_device and device.is_active:
                target = device
                break

        if not target:
            logger.warning("Target device not found: %s", target_device)
            return None

        # Get latest context
        context = self._store.get_latest_snapshot()

        # Create handoff
        handoff = Handoff(
            handoff_id=str(uuid.uuid4())[:8],
            from_device=self._session,
            to_device=target,
            timestamp=time.time(),
            context_synced=context is not None,
        )

        self._handoffs.append(handoff)

        # Keep last MAX_HANDOVERS
        if len(self._handoffs) > MAX_HANDOVERS:
            self._handoffs = self._handoffs[-MAX_HANDOVERS:]

        logger.info(
            "Handoff initiated: %s -> %s",
            self._session.device_name,
            target_device,
        )

        return handoff

    def receive_handoff(self) -> dict[str, Any] | None:
        """Receive a handoff from another device."""
        context = self._store.get_latest_snapshot()

        if not context:
            return None

        # Acknowledge the handoff
        if self._handoffs:
            self._handoffs[-1].acknowledged = True

        return context

    def get_handoff_suggestion(
        self,
        load: float,
        stress: float,
    ) -> str | None:
        """Get a suggestion for device handoff based on cognitive state."""
        # High load on desktop → suggest mobile
        if self._device_type == "desktop" and load > 0.8:
            # Check if mobile has been used recently
            for device in self._known_devices:
                if device.device_type == "mobile" and device.is_active:
                    return f"Your cognitive load is high ({int(load * 100)}%). Want to switch to {device.device_name}?"

        # High stress → suggest break on mobile
        if stress > 0.7:
            return "You seem stressed. Want to take this to a different device for a fresh start?"

        return None

    # ── Context Transfer ──

    def save_context_for_handoff(
        self,
        context_id: str,
        context_data: dict[str, Any],
    ) -> None:
        """Save context for future handoff."""
        self._store.save_context(context_id, context_data)
        logger.info("Saved context: %s", context_id)

    def load_handoff_context(self, context_id: str) -> dict[str, Any] | None:
        """Load context from a handoff."""
        return self._store.load_context(context_id)

    # ── Device Discovery ──

    def register_device(self, device_name: str, device_type: str) -> None:
        """Register another device."""
        # Check if already registered
        for device in self._known_devices:
            if device.device_name == device_name:
                device.last_active = time.time()
                device.is_active = True
                return

        # Add new device
        new_device = DeviceSession(
            session_id=str(uuid.uuid4()),
            device_type=device_type,
            device_name=device_name,
            started_at=time.time(),
            last_active=time.time(),
            is_active=True,
        )
        self._known_devices.append(new_device)

        logger.info("Registered device: %s (%s)", device_name, device_type)

    def get_active_devices(self) -> list[dict[str, Any]]:
        """Get all active devices."""
        devices = [d.to_dict() for d in self._known_devices if d.is_active]

        # Add current device
        devices.append(self._session.to_dict())

        return devices

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        return {
            "current_device": {
                "name": self._session.device_name,
                "type": self._session.device_type,
                "session_id": self._session.session_id[:8],
            },
            "known_devices": len(self._known_devices),
            "active_devices": len([d for d in self._known_devices if d.is_active]),
            "handoffs": len(self._handoffs),
            "last_sync": datetime.fromtimestamp(self._last_sync_time).isoformat() if self._last_sync_time else None,
            "cloud_snapshot": self._store.get_latest_snapshot() is not None,
        }
