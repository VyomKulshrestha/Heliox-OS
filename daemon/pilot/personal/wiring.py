"""The thin glue between the server and the companion pieces (the floor, the Live link, personal context, memory
talk and offers), so the server needs only one-line hooks. Every function tolerates a server built without the piece
it reads (tests build partial servers), and nothing here imports audio, speech or model packages.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("pilot.personal.wiring")

LOCAL_PROVIDERS = frozenset({"ollama", "local"})
TYPING_QUIET_SECONDS = 5.0


def floor_for(server: Any) -> Any:
    floor = getattr(server, "_live_floor", None)
    if floor is None:
        from pilot.live.floor import Floor

        floor = Floor()
        server._live_floor = floor
    return floor


def call_watch_for(server: Any) -> Any:
    """Whether the person is in a call, for Live: the listener's own answer, else polled detection."""
    watch = getattr(server, "_live_calls", None)
    if watch is None:
        from pilot.live.calls import CallWatch

        watch = CallWatch(
            listener=lambda: getattr(server, "_voice_listener", None),
            enabled=lambda: bool(getattr(getattr(server.config, "voice", None), "quiet_in_calls", True)),
        )
        server._live_calls = watch
    return watch


def model_is_local(server: Any) -> bool:
    model = getattr(getattr(server, "config", None), "model", None)
    return str(getattr(model, "provider", "") or "").casefold() in LOCAL_PROVIDERS


def _share_setting(server: Any) -> str:
    from pilot.conversation.adapters import share_setting

    conv = getattr(getattr(server, "config", None), "conversation", None)
    return share_setting(getattr(conv, "share_screen_text", "auto"))


def _layer(server: Any) -> Any:
    return getattr(server, "_conversation", None)


def link_for(server: Any) -> Any:
    link = getattr(server, "_live_link", None)
    if link is None:
        from pilot.live.conversation_link import LiveConversationLink

        link = LiveConversationLink(
            live=lambda: getattr(server, "_live", None),
            layer=lambda: _layer(server),
            setting=lambda: _share_setting(server),
            model_local=lambda: model_is_local(server),
            clock=time.monotonic,
        )
        server._live_link = link
    return link


def personal_context_for(server: Any) -> Any:
    context = getattr(server, "_personal_context", None)
    if context is None:
        from pilot.personal.context import PersonalContext

        goals = getattr(server, "_goals", None)
        habits = getattr(server, "_habits", None)
        context = PersonalContext(
            profile=lambda: getattr(server, "_user_profile", None),
            goals=goals.list if goals is not None else list,
            habits=habits.list if habits is not None else list,
        )
        server._personal_context = context
    return context


def install(server: Any, layer: Any) -> None:
    """Give the conversation layer the Live line and the personal line, once."""
    if layer is None or getattr(layer, "_companion_installed", False):
        return
    sources = getattr(layer, "sources", None)
    extra = getattr(sources, "extra", None)
    if not isinstance(extra, list):
        return
    extra.append(link_for(server).context_line)
    extra.append(personal_context_for(server).extra)
    layer._companion_installed = True


def begin_turn(server: Any, text: str) -> int:
    """An addressed utterance reached dispatch: it holds the floor, and its words select the personal context."""
    personal_context_for(server).note(text)
    try:
        from pilot.conversation import layer_for

        install(server, layer_for(server))
    except Exception:
        logger.debug("Could not attach the companion context to the conversation layer", exc_info=True)
    return int(floor_for(server).open_turn())


def end_turn(server: Any, token: int) -> None:
    floor_for(server).close_turn(token)


def note_heliox_speech(server: Any, channel: Any, dedupe_key: str) -> None:
    """Called first thing in the server's speech path: conversation speech (not Live's own) has started."""
    if str(dedupe_key).startswith(("live:", "offer:")):
        return
    floor = getattr(server, "_live_floor", None)
    if floor is not None and floor.turn_open:
        floor.heliox_spoke()


async def live_speak(server: Any, text: str, urgent: bool) -> str:
    """Say a Live remark if the floor still allows it after the person has finished, and record it as a Heliox turn.

    Returns the speech outcome status, or "held" when a conversation turn took the floor in the meantime (the card
    is already on screen, so nothing is lost)."""
    from pilot.system.companion_speech import SpeechChannel

    listener = getattr(server, "_voice_listener", None)
    if listener is not None and getattr(listener, "is_running", False):
        wait = getattr(listener, "wait_for_floor", None)
        if callable(wait):
            try:
                await wait()
            except Exception:
                logger.debug("wait_for_floor failed", exc_info=True)
    live = getattr(server, "_live", None)
    floor = getattr(server, "_live_floor", None)
    if floor is not None and live is not None:
        entries = live.last_spoken_entries()
        about = bool(entries) and all(
            e.get("source") == "speech" and floor.about_this_turn(int(e.get("noted_seq") or 0)) for e in entries
        )
        if not floor.live_may_speak(about_this_turn=about):
            logger.info("Live remark held as a card: a conversation turn has the floor")
            return "held"
    channel = SpeechChannel.LIVE_CORRECTION if urgent else SpeechChannel.PROACTIVE_SUGGESTION
    outcome = await server._speak_companion_text(text, channel=channel, dedupe_key=f"live:{text}")
    status = str(getattr(outcome, "status", "error"))
    try:
        link_for(server).record_spoken(text, status)
    except Exception:
        logger.debug("Could not record the Live remark in the conversation", exc_info=True)
    return status


# ── memory talk ──


def memory_talk_for(server: Any) -> Any:
    talk = getattr(server, "_memory_talk", None)
    if talk is None:
        from pilot.personal.memory_talk import MemoryTalk

        goals = getattr(server, "_goals", None)
        talk = MemoryTalk(
            profile=lambda: getattr(server, "_user_profile", None),
            goals=goals.list if goals is not None else list,
            notes_enabled=lambda: bool(getattr(getattr(server.config, "memory", None), "passing_notes", True)),
        )
        server._memory_talk = talk
    return talk


# ── offers ──


def offers_for(server: Any, data_dir: Any = None) -> Any:
    offers = getattr(server, "_offers", None)
    if offers is not None:
        return offers
    from pathlib import Path

    from pilot.personal.offers import Offers
    from pilot.system.app_names import app_label

    config = server.config
    timeline = getattr(server, "_activity_timeline", None)
    suggester = getattr(server, "_suggester", None)
    profile = getattr(server, "_user_profile", None)

    def spans(start: float, end: float) -> list[dict[str, Any]]:
        if timeline is None or not getattr(timeline, "enabled", False):
            return []
        return list(timeline.spans(start, end, limit=5000))

    offers = Offers(
        mode=lambda: str(getattr(config.suggestions, "offers", "auto")),
        profile_empty=(profile.is_empty if profile is not None else (lambda: True)),
        suggestion=(suggester.suggest if suggester is not None else (lambda: None)),
        legacy_suggestions_on=lambda: bool(getattr(config.suggestions, "enabled", False)),
        spans=spans,
        label=app_label,
        state_path=Path(data_dir) / "offers.json" if data_dir is not None else None,
    )
    server._offers = offers
    return offers


def quiet_moment(server: Any) -> bool:
    """A moment a person sitting next to you would choose: nobody talking, no conversation, no focus session, no
    call, not typing."""
    floor = getattr(server, "_live_floor", None)
    if floor is not None and floor.held():
        return False
    listener = getattr(server, "_voice_listener", None)
    if listener is not None and getattr(listener, "user_is_speaking", False):
        return False
    if getattr(server, "config", None) is not None and call_watch_for(server)():
        return False  # in a call: no offer, by voice or on screen
    focus = getattr(server, "_focus", None)
    if focus is not None and getattr(focus, "active", False):
        return False
    if getattr(server, "_interactive_request_active", False) or getattr(server, "_pending_confirms", None):
        return False
    coordinator = getattr(server, "_speech_coordinator", None)
    if coordinator is not None and coordinator.status().get("active"):
        return False
    live = getattr(server, "_live", None)
    if live is not None and live.current_observation() is not None:
        if time.monotonic() - float(getattr(live, "_last_text_change", float("-inf"))) < TYPING_QUIET_SECONDS:
            return False
    return True


async def deliver_offer(server: Any, offer: Any, offer_id: str = "") -> None:
    """A card always (the app's offer card: Yes, Not today, Don't offer this, Why); spoken only when the person is
    using voice (the listener runs), so they can answer by voice too."""
    from pilot.system.companion_speech import SpeechChannel

    await server._broadcast_notification(
        "offer",
        {
            "id": offer_id or uuid.uuid4().hex[:10],
            "key": offer.key,
            "kind": offer.kind,
            "text": offer.text,
            "request": offer.request,
        },
    )
    layer = _layer(server)
    if layer is not None:
        layer.memory.add_heliox(offer.text)
    listener = getattr(server, "_voice_listener", None)
    if listener is not None and getattr(listener, "is_running", False):
        await server._speak_companion_text(
            offer.text, channel=SpeechChannel.PROACTIVE_SUGGESTION, dedupe_key=f"offer:{offer.key}"
        )


def start_offers(server: Any, data_dir: Any) -> None:
    from pilot.personal.offers import OfferLoop

    offers = offers_for(server, data_dir)

    async def deliver(offer: Any, offer_id: str) -> None:
        await deliver_offer(server, offer, offer_id)

    loop = OfferLoop(offers, quiet_moment=lambda: quiet_moment(server), deliver=deliver)
    server._offer_loop = loop
    loop.start()


async def offer_reply(server: Any, text: str) -> str | None:
    """The person's answer to the latest offer. Only when the offer is newer than Live's latest card."""
    offers = getattr(server, "_offers", None)
    if offers is None:
        return None
    from pilot.workflows.routines import normalize_phrase

    offered = offers.last_offered_seq()
    live = getattr(server, "_live", None)
    raised = live.last_raised() if live is not None else None
    if offered is not None and raised and int(raised.get("seq") or 0) > offered:
        return None  # Live spoke since: "why?" and "thanks" are about that (ordered by sequence, not clock)
    latest = offers.latest()
    answer = offers.reply(normalize_phrase(text))
    if answer is None:
        return None
    await _act_on(server, answer)
    if latest is not None and latest.answered:
        await server._broadcast_notification("offer_withdrawn", {"id": latest.id, "reason": "answered"})
    return answer.reply


async def _act_on(server: Any, answer: Any, *, from_card: bool = False) -> None:
    from pilot.personal.offers import INTRO

    if answer.silence_all:
        await server._handle_update_config({"section": "suggestions", "values": {"offers": "off"}}, None)
    if answer.request == INTRO.request and from_card:
        await start_getting_to_know(server)  # in the chat, and aloud only when they use voice
        return
    if answer.request:
        task = asyncio.ensure_future(server._voice_turn(answer.request))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())


async def offers_rpc(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """The app's side of offers. ``action``: "status"; "set" with ``mode`` (auto, on, off); "unsilence" with ``key``;
    "answer" with ``id`` and ``answer`` (yes, not_today, silence, why) from an offer card."""
    from pilot.personal.offers import MODES

    offers = offers_for(server)
    action = str(params.get("action") or "status")
    if action == "set":
        mode = str(params.get("mode") or "").casefold()
        if mode not in MODES:
            return {"status": "error", "message": "mode must be auto, on or off"}
        result = await server._handle_update_config({"section": "suggestions", "values": {"offers": mode}}, None)
        if result.get("status") != "ok":
            return result
    elif action == "intro":
        from pilot.personal.offers import INTRO

        if params.get("on"):
            offers.unsilence(INTRO.key)
        else:
            offers.silence(INTRO)
    elif action == "unsilence":
        if not offers.unsilence(str(params.get("key") or "")):
            return {"status": "error", "message": "That offer is not silenced."}
    elif action == "answer":
        answer = offers.answer(str(params.get("id") or ""), str(params.get("answer") or ""))
        if answer is None:
            return {"status": "error", "message": "That offer is no longer available."}
        await _act_on(server, answer, from_card=True)
        if params.get("answer") != "why":
            await server._broadcast_notification(
                "offer_withdrawn", {"id": params.get("id"), "reason": params.get("answer")}
            )
        return {"status": "ok", "reply": answer.reply, **offers.status()}
    elif action != "status":
        return {"status": "error", "message": "action must be status, set, intro, unsilence or answer"}
    return {"status": "ok", **offers.status()}


# ── getting to know the person (pilot.personal.getting_to_know) ──


def getting_to_know_for(server: Any) -> Any:
    gtk = getattr(server, "_getting_to_know", None)
    if gtk is None:
        from pilot.personal.getting_to_know import GettingToKnow

        async def set_quiet(quiet: bool) -> None:
            if not quiet:
                return
            await server._handle_update_config({"section": "suggestions", "values": {"offers": "off"}}, None)
            await server._handle_update_config({"section": "live", "values": {"level": "quiet"}}, None)

        gtk = GettingToKnow(profile=lambda: getattr(server, "_user_profile", None), set_quiet=set_quiet)
        server._getting_to_know = gtk
    return gtk


async def getting_to_know_reply(server: Any, text: str) -> str | None:
    """An answer while the questions are open, or the phrase that starts them ("get to know me")."""
    if getattr(server, "_user_profile", None) is None:
        return None
    return await getting_to_know_for(server).answer(text)


async def start_getting_to_know(server: Any) -> str:
    """Start from the app (the intro card or the form's "ask me instead"): the question goes to the chat, and is
    said aloud only when the person is using voice."""
    from pilot.system.companion_speech import SpeechChannel

    question = getting_to_know_for(server).start()
    await server._broadcast_notification(
        "voice_result", {"command": "", "status": "answered", "message": question, "source": "getting_to_know"}
    )
    listener = getattr(server, "_voice_listener", None)
    if listener is not None and getattr(listener, "is_running", False):
        await server._speak_companion_text(question, channel=SpeechChannel.FINAL_ANSWER, dedupe_key="gtk:start")
    return question


async def getting_to_know_rpc(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """The app's side: ``action`` "status", "start", "stop", "save" (the form: name, use, length, speaking) or
    "remove" with ``id`` (a stored line)."""
    profile = getattr(server, "_user_profile", None)
    if profile is None:
        return {"status": "error", "message": "Your profile is not ready yet."}
    gtk = getting_to_know_for(server)
    action = str(params.get("action") or "status")
    if action == "start":
        if gtk.active:  # already asking: a second start (the first-run screen, a second click) changes nothing
            return {"status": "ok", "question": gtk.current_question(), "already_active": True}
        # The first-run screen asks once after setup. It must not ask someone who already has a profile or who
        # turned the questions off in Settings; a person who asks ("get to know me") goes through the chat instead.
        if not profile.is_empty():
            return {"status": "ok", "question": "", "skipped": "profile"}
        if not offers_for(server).status().get("intro", True):
            return {"status": "ok", "question": "", "skipped": "intro_off"}
        return {"status": "ok", "question": await start_getting_to_know(server)}
    if action == "stop":
        return {"status": "ok", "reply": gtk.stop()}
    if action == "save":
        result = await gtk.save_form(
            name=str(params.get("name") or ""),
            use=str(params.get("use") or ""),
            length=str(params.get("length") or ""),
            speaking=str(params.get("speaking") or ""),
        )
        return {"status": "ok", **result}
    if action == "remove":
        if not await profile.remove_item(str(params.get("id") or "")):
            return {"status": "error", "message": "That is no longer stored."}
        return {"status": "ok", "profile_empty": profile.is_empty()}
    if action != "status":
        return {"status": "error", "message": "action must be status, start, stop, save or remove"}
    return {
        "status": "ok",
        "active": gtk.active,
        "profile_empty": profile.is_empty(),
        "intro": bool(offers_for(server).status().get("intro", True)),
    }


# ── Live's terminal check (pilot.live.shell_line) ──


def shell_server_for(server: Any) -> Any:
    found = getattr(server, "_shell_lines_server", None)
    if found is None:
        from pilot.live.shell_line import ShellLines, ShellLineServer

        found = ShellLineServer(
            ShellLines(), enabled=lambda: bool(getattr(getattr(server.config, "live", None), "enabled", False))
        )
        server._shell_lines_server = found
    return found


def start_shell_lines(server: Any) -> None:
    shell_server_for(server).start()  # a no-op on Windows, where consoles are read directly


def shell_observation(server: Any, app: str, title: str) -> Any:
    """Live's look at a terminal from the zsh integration: the line being typed, while it is fresh."""
    found = getattr(server, "_shell_lines_server", None)
    if found is None or not found.listening:
        return None
    latest = found.lines.latest()
    if latest is None:
        return None
    key, seen = latest
    from pilot.live.engine import Observation

    # The shell's name in the title makes the detectors read it as a terminal line, which it is.
    return Observation(
        text=seen.line, app=app or seen.term, title=f"{title} ({seen.shell})", source="focused_field", field=key
    )


async def live_shell_rpc(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """``action``: "status" (what Add would do, for each shell), "install" or "uninstall" with ``shell``."""
    import os
    import sys
    from pathlib import Path

    from pilot.live import shell_line

    if sys.platform == "win32":
        return {
            "status": "ok",
            "supported": False,
            "message": "On Windows Heliox reads the terminal itself, so nothing needs adding.",
        }
    found = shell_server_for(server)
    home, env = Path.home(), dict(os.environ)
    action = str(params.get("action") or "status")
    shell = str(params.get("shell") or "")
    try:
        if action == "install":
            shell_line.install(shell, home, env, found.path)
        elif action == "uninstall":
            shell_line.uninstall(shell, home, env)
        elif action != "status":
            return {"status": "error", "message": "action must be status, install or uninstall"}
    except (ValueError, OSError) as error:
        return {"status": "error", "message": str(error)}
    shells = {}
    for name in shell_line.SHELLS:
        chosen = shell_line.plan(name, home, env)
        shells[name] = {
            "installed": shell_line.installed(name, home, env),
            "script": str(chosen.script_path),
            "startup_file": str(chosen.startup_file) if chosen.startup_file else "",
            "line": chosen.line,
            "remove": chosen.remove,
        }
    return {
        "status": "ok",
        "supported": True,
        "listening": found.listening,
        "problem": found.problem,
        "socket": str(found.path),
        "shells": shells,
        "bash": "bash only offers hooks after Enter; with ble.sh it could work like zsh, which is not built yet.",
    }


__all__ = [
    "begin_turn",
    "deliver_offer",
    "end_turn",
    "floor_for",
    "getting_to_know_for",
    "getting_to_know_reply",
    "getting_to_know_rpc",
    "install",
    "link_for",
    "live_speak",
    "memory_talk_for",
    "model_is_local",
    "note_heliox_speech",
    "offer_reply",
    "offers_for",
    "offers_rpc",
    "personal_context_for",
    "quiet_moment",
    "start_offers",
]
