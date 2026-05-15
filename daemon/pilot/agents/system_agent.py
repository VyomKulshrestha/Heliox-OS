"""System Agent — handles OS operations like files, processes, services.

This is the workhorse agent for all direct system interactions:
filesystem, packages, services, power, environment, window management,
volume, brightness, WiFi, disk, user/group ops, and registry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.system_agent")

# All action types owned by the System Agent
SYSTEM_ACTION_TYPES: set[ActionType] = {
    # File operations
    ActionType.FILE_READ,
    ActionType.FILE_WRITE,
    ActionType.FILE_DELETE,
    ActionType.FILE_MOVE,
    ActionType.FILE_COPY,
    ActionType.FILE_LIST,
    ActionType.FILE_SEARCH,
    ActionType.DIRECTORY_SUMMARY,
    ActionType.FILE_PERMISSIONS,
    # Package management
    ActionType.PACKAGE_INSTALL,
    ActionType.PACKAGE_REMOVE,
    ActionType.PACKAGE_UPDATE,
    ActionType.PACKAGE_SEARCH,
    # Service management
    ActionType.SERVICE_START,
    ActionType.SERVICE_STOP,
    ActionType.SERVICE_RESTART,
    ActionType.SERVICE_ENABLE,
    ActionType.SERVICE_DISABLE,
    ActionType.SERVICE_STATUS,
    # Desktop / GNOME
    ActionType.GNOME_SETTING_READ,
    ActionType.GNOME_SETTING_WRITE,
    ActionType.DBUS_CALL,
    # Shell
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    # Process management
    ActionType.PROCESS_LIST,
    ActionType.PROCESS_KILL,
    ActionType.PROCESS_INFO,
    # Clipboard
    ActionType.CLIPBOARD_READ,
    ActionType.CLIPBOARD_WRITE,
    # System info
    ActionType.SYSTEM_INFO,
    ActionType.DISK_USAGE,
    ActionType.MEMORY_USAGE,
    ActionType.CPU_USAGE,
    ActionType.NETWORK_INFO,
    ActionType.BATTERY_INFO,
    # Power management
    ActionType.POWER_SHUTDOWN,
    ActionType.POWER_RESTART,
    ActionType.POWER_SLEEP,
    ActionType.POWER_LOCK,
    ActionType.POWER_LOGOUT,
    # Scheduled tasks
    ActionType.SCHEDULE_CREATE,
    ActionType.SCHEDULE_LIST,
    ActionType.SCHEDULE_DELETE,
    # Environment
    ActionType.ENV_GET,
    ActionType.ENV_SET,
    ActionType.ENV_LIST,
    # Window management
    ActionType.WINDOW_LIST,
    ActionType.WINDOW_FOCUS,
    ActionType.WINDOW_CLOSE,
    ActionType.WINDOW_MINIMIZE,
    ActionType.WINDOW_MAXIMIZE,
    # Volume / audio
    ActionType.VOLUME_GET,
    ActionType.VOLUME_SET,
    ActionType.VOLUME_MUTE,
    # Display
    ActionType.BRIGHTNESS_GET,
    ActionType.BRIGHTNESS_SET,
    ActionType.SCREENSHOT,
    # Network
    ActionType.WIFI_LIST,
    ActionType.WIFI_CONNECT,
    ActionType.WIFI_DISCONNECT,
    # Disk
    ActionType.DISK_LIST,
    ActionType.DISK_MOUNT,
    ActionType.DISK_UNMOUNT,
    # User
    ActionType.USER_LIST,
    ActionType.USER_INFO,
    # Download
    ActionType.DOWNLOAD_FILE,
    # Registry (Windows)
    ActionType.REGISTRY_READ,
    ActionType.REGISTRY_WRITE,
    # Open
    ActionType.OPEN_APPLICATION,
    ActionType.OPEN_URL,
    ActionType.NOTIFY,
    # Input control
    ActionType.MOUSE_CLICK,
    ActionType.MOUSE_DOUBLE_CLICK,
    ActionType.MOUSE_RIGHT_CLICK,
    ActionType.MOUSE_MOVE,
    ActionType.MOUSE_DRAG,
    ActionType.MOUSE_SCROLL,
    ActionType.MOUSE_POSITION,
    ActionType.KEYBOARD_TYPE,
    ActionType.KEYBOARD_PRESS,
    ActionType.KEYBOARD_HOTKEY,
    ActionType.KEYBOARD_HOLD,
    # Screen vision
    ActionType.SCREEN_OCR,
    ActionType.SCREEN_FIND_TEXT,
    ActionType.SCREEN_ANALYZE,
    ActionType.SCREEN_ELEMENT_MAP,
    # Triggers
    ActionType.TRIGGER_CREATE,
    ActionType.TRIGGER_LIST,
    ActionType.TRIGGER_DELETE,
    ActionType.TRIGGER_START,
    ActionType.TRIGGER_STOP,
    # File intelligence
    ActionType.FILE_PARSE,
    ActionType.FILE_SEARCH_CONTENT,
}


class SystemAgent(BaseAgent):
    """Specialist agent for all OS-level operations."""

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.SYSTEM, model_router=model_router)
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=at,
                description=f"System operation: {at.value}",
                requires_confirmation=at
                in {
                    ActionType.POWER_SHUTDOWN,
                    ActionType.POWER_RESTART,
                    ActionType.FILE_DELETE,
                    ActionType.PACKAGE_REMOVE,
                    ActionType.PROCESS_KILL,
                },
            )
            for at in SYSTEM_ACTION_TYPES
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the SYSTEM AGENT for Heliox OS. "
            "You are the primary handler for all operating system operations: "
            "file management, process control, package management, services, "
            "power management, display/audio settings, network configuration, "
            "input control (mouse/keyboard), screen analysis, and scheduled tasks. "
            "You have direct access to the OS and can execute any system-level command. "
            "Always verify the current state before making changes. "
            "For destructive operations, flag them for user confirmation."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in SYSTEM_ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute system-related actions from the plan."""
        import time

        start = time.time()
        self.status = AgentStatus.BUSY

        # Filter to only actions this agent owns
        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        # Build a sub-plan with only our actions
        sub_plan = ActionPlan(
            actions=my_actions,
            explanation=f"System Agent executing {len(my_actions)} action(s)",
            raw_input=user_input,
        )

        results = await self._executor.execute(sub_plan)
        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results
