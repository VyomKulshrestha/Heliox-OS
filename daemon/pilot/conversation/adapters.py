"""Glue between the layer and the rest of Heliox, kept thin so the layer itself depends on nothing else.

Everything that may not exist on a platform or in a test (the focused-app lookup, Live, the voice listener, the
profile, the budget) is reached through ``getattr`` and a ``try``; when a piece is missing the layer gets "not
available here" and carries on without it.

What the fast model is told about the screen (``conversation.share_screen_text``):

=====================  =================================  ====================================================
setting                model on this machine (Ollama)      cloud model (Claude, Gemini, OpenAI-compatible)
=====================  =================================  ====================================================
``auto`` (default)     app name + end of the focused field  app name only
``on``                 app name + end of the focused field  app name + end of the focused field
``off``                app name only                        app name only
=====================  =================================  ====================================================

"App name" is the display name ("VS Code", "Windows Terminal"), never the window title. The text is the last
400 characters of the focused editable field as Heliox Live read it (Live must be on; never a password field),
and it is left out whole when it looks like it holds a key or password. For Heliox's own window, password
managers and apps excluded from the timeline nothing is sent at all, in every row.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pilot.conversation.context import SCREEN_CHARS, ContextSources, Screen

logger = logging.getLogger("pilot.conversation.adapters")

SHARE_VALUES = ("auto", "on", "off")


@dataclass(frozen=True, slots=True)
class Focus:
    """The focused window as an OS backend reports it. ``text`` is only filled from Live's gated reader."""

    app: str
    title: str = ""
    text: str = ""
    editable: bool = False
    source: str = ""


def share_setting(value: Any) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    text = str(value or "auto").strip().casefold()
    return text if text in SHARE_VALUES else "auto"


def _never_read(app: str, title: str, excluded: list[str]) -> bool:
    from pilot.live.detectors import _PASSWORD_MANAGERS

    where = f"{app} {title}".casefold()
    try:
        from pilot.workflows.manual_workflow_hook import _is_heliox_window_title

        heliox = _is_heliox_window_title(title)
    except Exception:
        heliox = "heliox" in where
    return (
        heliox
        or any(name in where for name in _PASSWORD_MANAGERS)
        or any(name.casefold() in where for name in excluded if name.strip())
    )


def _holds_secret(text: str) -> bool:
    from pilot.live.detectors import _SECRETS

    return any(pattern.search(text) for _label, pattern in _SECRETS)


def screen_for_model(
    focus: Focus | None,
    *,
    setting: str,
    model_local: bool,
    excluded_apps: list[str],
    label: Callable[[str], str],
) -> Screen | None:
    """What may be said about the screen to this model. Pure: tested on every OS."""
    if focus is None or not focus.app or focus.app == "Unknown":
        return None  # not available on this platform, or nothing focused
    if _never_read(focus.app, focus.title, excluded_apps):
        return None
    try:
        name = label(focus.app) or ""
    except Exception:
        name = ""
    if not name:
        return None
    share = setting == "on" or (setting == "auto" and model_local)
    excerpt = ""
    if share and focus.source == "focused_field" and focus.editable and focus.text.strip():
        tail = " ".join(focus.text.split())[-SCREEN_CHARS:]
        excerpt = "" if _holds_secret(focus.text) else tail
    return Screen(app=name, excerpt=excerpt)


class FocusSource:
    """The focused app and, through Live, the focused field's text. Each part answers None when unavailable.

    Live's observation is the only place text comes from, so Live's own gates (password fields, Heliox's window,
    password managers, excluded apps) apply before this layer adds its own. The app name without Live comes
    from ``pilot.agents.screen_vision._get_active_window``, which has a backend for Windows, macOS and Linux and
    reports "Unknown" where it cannot tell.
    """

    def __init__(self, server: Any, *, timeout: float = 0.4) -> None:
        self._server = server
        self._timeout = timeout

    def _live_observation(self) -> Any:
        live = getattr(self._server, "_live", None)
        config = getattr(getattr(self._server, "config", None), "live", None)
        if live is None or not getattr(config, "enabled", False):
            return None
        # Needed from Live: a public ``current_observation()``; until then its last look is read directly.
        getter = getattr(live, "current_observation", None)
        return getter() if callable(getter) else getattr(live, "_last_obs", None)

    async def focus(self) -> Focus | None:
        obs = self._live_observation()
        if obs is not None:
            return Focus(
                app=str(getattr(obs, "app", "") or ""),
                title=str(getattr(obs, "title", "") or ""),
                text=str(getattr(obs, "text", "") or ""),
                editable=bool(getattr(obs, "editable", False)),
                source=str(getattr(obs, "source", "") or ""),
            )
        try:
            from pilot.agents.screen_vision import _get_active_window

            app, title = await asyncio.wait_for(asyncio.to_thread(_get_active_window), self._timeout)
        except Exception:
            return None
        return Focus(app=str(app or ""), title=str(title or "")) if app else None


def _app_label(process: str) -> str:
    from pilot.system.app_names import app_label

    return app_label(process)


def _task_state(server: Any, layer: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    runtime = getattr(server, "_interaction_runtime", None)
    snapshot = runtime.status() if runtime is not None else {}
    pending = getattr(server, "_pending_confirms", None) or {}
    if pending:
        state["approval_waiting"] = True
    if getattr(server, "_interactive_request_active", False) and snapshot.get("active"):
        state["running"] = str(snapshot.get("user_input") or "")[:160]
        state["phase"] = str(snapshot.get("message") or "")[:80]
    if layer is not None and layer.last_result:
        state["last_result"] = layer.last_result[:240]
    return state


def _profile_lines(server: Any) -> list[str]:
    profile = getattr(server, "_user_profile", None)
    if profile is None:
        return []
    return list(profile.context_lines())


def _status(server: Any) -> dict[str, Any]:
    runtime = getattr(server, "_interaction_runtime", None)
    return runtime.status() if runtime is not None else {}


def layer_for(server: Any) -> Any:
    """The server's conversation layer, built on first use from what the server already has."""
    existing = getattr(server, "_conversation", None)
    if existing is not None:
        return existing
    from pilot.conversation.fast_model import build_fast_model, describe_setup
    from pilot.conversation.layer import ConversationLayer

    cache: dict[str, Any] = {}

    def settings() -> Any:
        return getattr(server.config, "conversation", None)

    def model() -> Any:
        config = server.config
        model_cfg = config.model
        conv = settings()
        signature = (
            model_cfg.provider,
            model_cfg.cloud_provider,
            model_cfg.cloud_model,
            model_cfg.ollama_model,
            tuple(sorted(vars(conv).items())) if conv is not None else (),
        )
        if cache.get("signature") != signature:
            cache["signature"] = signature
            cache["model"] = build_fast_model(
                config,
                vault=getattr(server, "_vault", None),
                router=getattr(server, "_model_router", None),
                budget=getattr(server, "_budget_tracker", None),
            )
            built = cache["model"]
            logger.info(
                "Conversation model: %s (%s, %s)",
                getattr(built, "name", "none"),
                describe_setup(config),
                "streamed" if getattr(built, "streams", False) else "one piece",
            )
        return cache["model"]

    async def sink(text: str, channel: str, key: str) -> str:
        outcome = await server._speak_companion_text(text, channel=channel, dedupe_key=key)
        return str(getattr(outcome, "status", "error"))

    def user_speaking() -> bool:
        listener = getattr(server, "_voice_listener", None)
        return bool(listener is not None and getattr(listener, "user_is_speaking", False))

    focus = FocusSource(server)

    async def screen(model_local: bool) -> Screen | None:
        conv = settings()
        return screen_for_model(
            await focus.focus(),
            setting=share_setting(getattr(conv, "share_screen_text", "auto")),
            model_local=model_local,
            excluded_apps=list(getattr(getattr(server.config, "timeline", None), "excluded_apps", []) or []),
            label=_app_label,
        )

    layer: Any = None
    sources = ContextSources(
        profile_lines=lambda: _profile_lines(server),
        task_state=lambda: _task_state(server, layer),
        screen_async=screen,
    )
    layer = ConversationLayer(
        sink=sink,
        model=model,
        sources=sources,
        # Active once the server's model stack is up (set in start()); a server built without it keeps the old path.
        enabled=lambda: (
            bool(getattr(settings(), "enabled", True)) and getattr(server, "_model_router", None) is not None
        ),
        user_speaking=user_speaking,
        task_status=lambda: _status(server),
    )
    server._conversation = layer
    return layer
