"""Narrow executor-backed specialists for high-value operating domains.

These agents do not introduce alternate execution paths. They filter plans to
their reviewed capability set and send the resulting sub-plan through the same
Executor, Agent Gateway, permission checker, world model, and audit ledger as
the SystemAgent.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.agent_mesh import AgentBudgetPolicy
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride


class _ExecutorDomainAgent(BaseAgent):
    """Common bounded delegation path for narrow built-in specialists."""

    ACTION_TYPES: ClassVar[frozenset[ActionType]] = frozenset()
    CONFIRMATION_ACTIONS: ClassVar[frozenset[ActionType]] = frozenset()
    DOMAIN_LABEL: ClassVar[str] = "domain"
    mesh_keywords: ClassVar[tuple[str, ...]] = ()
    mesh_budget: ClassVar[AgentBudgetPolicy] = AgentBudgetPolicy(
        max_tokens_per_task=6_000,
        max_actions_per_task=12,
        max_latency_ms_per_action=90_000,
        max_concurrency=1,
    )
    mesh_filesystem_read: ClassVar[tuple[str, ...]] = ()
    mesh_filesystem_write: ClassVar[tuple[str, ...]] = ()
    mesh_network_domains: ClassVar[tuple[str, ...]] = ()
    mesh_process_names: ClassVar[tuple[str, ...]] = ()
    mesh_credential_names: ClassVar[tuple[str, ...]] = ()
    mesh_clipboard: ClassVar[tuple[str, ...]] = ()
    mesh_devices: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        role: AgentRole,
        model_router: ModelRouter,
        executor: Executor,
    ) -> None:
        super().__init__(role=role, model_router=model_router)
        self._executor = executor

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=action_type,
                description=f"{self.DOMAIN_LABEL} operation: {action_type.value}",
                requires_confirmation=action_type in self.CONFIRMATION_ACTIONS,
            )
            for action_type in sorted(self.ACTION_TYPES, key=lambda item: item.value)
        ]

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialist-specific operating contract."""

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in self.ACTION_TYPES

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        started_at = time.monotonic()
        self.status = AgentStatus.BUSY
        actions = [action for action in plan.actions if self.can_handle(action.action_type)]
        if not actions:
            self.status = AgentStatus.IDLE
            return []
        sub_plan = ActionPlan(
            actions=actions,
            explanation=f"{self.__class__.__name__} executing {len(actions)} action(s)",
            raw_input=user_input,
        )
        try:
            results = await self._executor.execute(
                sub_plan,
                initial_last_output=str((context or {}).get("initial_last_output", "")),
                initial_largest_output=str((context or {}).get("initial_largest_output", "")),
                invocation_source=self.get_invocation_source(),
                scope_override=scope_override,
                user_confirmed=bool((context or {}).get("user_confirmed", False)),
            )
            self._record_task(
                int((time.monotonic() - started_at) * 1000),
                bool(results) and all(result.success for result in results),
            )
            return results
        finally:
            self.status = AgentStatus.IDLE


class FileOperationsAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.FILE_READ,
            ActionType.FILE_WRITE,
            ActionType.FILE_DELETE,
            ActionType.FILE_MOVE,
            ActionType.FILE_COPY,
            ActionType.FILE_LIST,
            ActionType.FILE_SEARCH,
            ActionType.DIRECTORY_SUMMARY,
            ActionType.DIRECTORY_SIZE,
            ActionType.FILE_HASH,
            ActionType.FILE_COMPARE,
            ActionType.FILE_PERMISSIONS,
            ActionType.FILE_PARSE,
            ActionType.FILE_SEARCH_CONTENT,
        }
    )
    CONFIRMATION_ACTIONS = frozenset({ActionType.FILE_DELETE, ActionType.FILE_PERMISSIONS})
    DOMAIN_LABEL = "File and content"
    mesh_keywords = (
        "file",
        "folder",
        "directory",
        "hash",
        "compare",
        "content",
        "permissions",
    )
    mesh_filesystem_read = ("user_selected_paths",)
    mesh_filesystem_write = ("user_selected_paths",)

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.FILES, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the FILE OPERATIONS AGENT. Inspect and transform user-selected files, "
            "compare content and hashes, and keep every path inside the validated scope."
        )


class PackageManagementAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.PACKAGE_INSTALL,
            ActionType.PACKAGE_REMOVE,
            ActionType.PACKAGE_UPDATE,
            ActionType.PACKAGE_SEARCH,
        }
    )
    CONFIRMATION_ACTIONS = frozenset(
        {
            ActionType.PACKAGE_INSTALL,
            ActionType.PACKAGE_REMOVE,
            ActionType.PACKAGE_UPDATE,
        }
    )
    DOMAIN_LABEL = "Package management"
    mesh_keywords = ("package", "dependency", "install", "update", "uninstall")
    mesh_process_names = ("reviewed_package_manager",)

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.PACKAGES, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the PACKAGE MANAGEMENT AGENT. Search, install, update, and remove "
            "packages through the validated platform adapter with confirmation for mutations."
        )


class ServiceManagementAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.SERVICE_START,
            ActionType.SERVICE_STOP,
            ActionType.SERVICE_RESTART,
            ActionType.SERVICE_ENABLE,
            ActionType.SERVICE_DISABLE,
            ActionType.SERVICE_STATUS,
        }
    )
    CONFIRMATION_ACTIONS = frozenset(ACTION_TYPES - {ActionType.SERVICE_STATUS})
    DOMAIN_LABEL = "Service management"
    mesh_keywords = ("service", "daemon", "start", "stop", "restart", "status")
    mesh_process_names = ("validated_service_name",)

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.SERVICES, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the SERVICE MANAGEMENT AGENT. Diagnose service state before applying "
            "a bounded lifecycle change and report the observed post-condition."
        )


class DesktopAutomationAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.OPEN_APPLICATION,
            ActionType.NOTIFY,
            ActionType.CLIPBOARD_READ,
            ActionType.CLIPBOARD_WRITE,
            ActionType.WINDOW_LIST,
            ActionType.WINDOW_FOCUS,
            ActionType.WINDOW_CLOSE,
            ActionType.WINDOW_MINIMIZE,
            ActionType.WINDOW_MAXIMIZE,
            ActionType.VOLUME_GET,
            ActionType.VOLUME_SET,
            ActionType.VOLUME_MUTE,
            ActionType.BRIGHTNESS_GET,
            ActionType.BRIGHTNESS_SET,
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
            ActionType.GNOME_SETTING_READ,
            ActionType.GNOME_SETTING_WRITE,
            ActionType.DBUS_CALL,
        }
    )
    CONFIRMATION_ACTIONS = frozenset(
        {
            ActionType.WINDOW_CLOSE,
            ActionType.MOUSE_CLICK,
            ActionType.MOUSE_DOUBLE_CLICK,
            ActionType.MOUSE_RIGHT_CLICK,
            ActionType.MOUSE_DRAG,
            ActionType.KEYBOARD_TYPE,
            ActionType.KEYBOARD_PRESS,
            ActionType.KEYBOARD_HOTKEY,
            ActionType.KEYBOARD_HOLD,
            ActionType.GNOME_SETTING_WRITE,
            ActionType.DBUS_CALL,
        }
    )
    DOMAIN_LABEL = "Desktop automation"
    mesh_keywords = (
        "desktop",
        "window",
        "mouse",
        "keyboard",
        "clipboard",
        "volume",
        "brightness",
    )
    mesh_clipboard = ("read", "write")
    mesh_devices = ("screen", "mouse", "keyboard", "audio")

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.DESKTOP, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the DESKTOP AUTOMATION AGENT. Use visible UI state and bounded input "
            "operations, preferring deterministic controls and never inventing coordinates."
        )


class WorkflowAutomationAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.SCHEDULE_CREATE,
            ActionType.SCHEDULE_LIST,
            ActionType.SCHEDULE_DELETE,
            ActionType.TRIGGER_CREATE,
            ActionType.TRIGGER_LIST,
            ActionType.TRIGGER_DELETE,
            ActionType.TRIGGER_START,
            ActionType.TRIGGER_STOP,
        }
    )
    CONFIRMATION_ACTIONS = frozenset(
        {
            ActionType.SCHEDULE_CREATE,
            ActionType.SCHEDULE_DELETE,
            ActionType.TRIGGER_DELETE,
        }
    )
    DOMAIN_LABEL = "Workflow automation"
    mesh_keywords = ("schedule", "trigger", "automation", "monitor", "recurring", "workflow")
    mesh_process_names = ("scheduler", "trigger_engine")

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.AUTOMATION, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the WORKFLOW AUTOMATION AGENT. Create explicit schedules and reactive "
            "triggers with bounded firing, cooldown, and user-visible lifecycle controls."
        )


class IntegrationAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.API_REQUEST,
            ActionType.API_GITHUB,
            ActionType.API_SEND_EMAIL,
            ActionType.API_WEBHOOK,
            ActionType.API_SLACK,
            ActionType.API_DISCORD,
            ActionType.API_SCRAPE,
        }
    )
    CONFIRMATION_ACTIONS = frozenset(
        {
            ActionType.API_SEND_EMAIL,
            ActionType.API_WEBHOOK,
            ActionType.API_SLACK,
            ActionType.API_DISCORD,
        }
    )
    DOMAIN_LABEL = "API integration"
    mesh_keywords = ("api", "github", "webhook", "slack", "discord", "integration", "request")
    mesh_network_domains = ("user_requested_domains",)
    mesh_credential_names = ("explicit_integration_credentials",)

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.INTEGRATIONS, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the INTEGRATION AGENT. Use explicit endpoints and credentials, validate "
            "responses, and require confirmation before sending data to third parties."
        )


class VisionAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.SCREENSHOT,
            ActionType.SCREEN_OCR,
            ActionType.SCREEN_FIND_TEXT,
            ActionType.SCREEN_ANALYZE,
            ActionType.SCREEN_ELEMENT_MAP,
            ActionType.SCREEN_ACCESSIBILITY_SNAPSHOT,
            ActionType.SCREEN_DETECT_ELEMENTS,
        }
    )
    DOMAIN_LABEL = "Screen vision"
    mesh_keywords = ("screen", "vision", "ocr", "visual", "element", "detect", "screenshot")
    mesh_devices = ("screen",)
    mesh_budget = AgentBudgetPolicy(
        max_tokens_per_task=8_000,
        max_actions_per_task=10,
        max_latency_ms_per_action=120_000,
        max_concurrency=1,
    )

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.VISION, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the VISION AGENT. Ground every observation in a real screenshot or OCR "
            "result and return measured element coordinates with confidence."
        )


class PluginRuntimeAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.WASM_CALL,
            ActionType.PLUGIN_CALL,
            ActionType.SKILL_RUN,
        }
    )
    CONFIRMATION_ACTIONS = frozenset({ActionType.PLUGIN_CALL})
    DOMAIN_LABEL = "Plugin runtime"
    mesh_keywords = ("plugin", "wasm", "skill", "marketplace", "extension", "tool")
    mesh_filesystem_read = ("manifest_declared_only",)
    mesh_filesystem_write = ("manifest_declared_only",)
    mesh_network_domains = ("manifest_declared_only",)
    mesh_credential_names = ("manifest_declared_only",)
    mesh_budget = AgentBudgetPolicy(
        max_tokens_per_task=4_000,
        max_actions_per_task=8,
        max_latency_ms_per_action=30_000,
        max_concurrency=1,
    )

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.PLUGIN_RUNTIME, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the PLUGIN RUNTIME AGENT. Invoke only installed, approved tools through "
            "their capability broker; never import marketplace code into the daemon."
        )


class NetworkAgent(_ExecutorDomainAgent):
    ACTION_TYPES = frozenset(
        {
            ActionType.NETWORK_INFO,
            ActionType.WIFI_LIST,
            ActionType.WIFI_CONNECT,
            ActionType.WIFI_DISCONNECT,
            ActionType.DOWNLOAD_FILE,
        }
    )
    CONFIRMATION_ACTIONS = frozenset({ActionType.WIFI_CONNECT, ActionType.WIFI_DISCONNECT})
    DOMAIN_LABEL = "Network"
    mesh_keywords = ("network", "wifi", "download", "connection", "internet", "wireless")
    mesh_network_domains = ("user_requested_domains",)
    mesh_devices = ("network_adapter",)

    def __init__(self, model_router: ModelRouter, executor: Executor) -> None:
        super().__init__(role=AgentRole.NETWORK, model_router=model_router, executor=executor)

    def get_system_prompt(self) -> str:
        return (
            "You are the NETWORK AGENT. Inspect connectivity and perform only explicit "
            "network changes or downloads through validated platform adapters."
        )
