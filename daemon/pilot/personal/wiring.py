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
            e.get("source") == "speech" and floor.about_this_turn(float(e.get("at", 0.0))) for e in entries
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
    if listener is not None and (getattr(listener, "user_is_speaking", False) or getattr(listener, "in_call", "")):
        return False
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


async def deliver_offer(server: Any, offer: Any) -> None:
    """A card always; spoken only when the person is using voice (the listener runs), so they can answer."""
    from pilot.system.companion_speech import SpeechChannel

    await server._broadcast_notification(
        "nudge", {"id": f"offer-{uuid.uuid4().hex[:8]}", "text": offer.text, "kind": "offer", "offer": offer.kind}
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

    async def deliver(offer: Any) -> None:
        await deliver_offer(server, offer)

    loop = OfferLoop(offers, quiet_moment=lambda: quiet_moment(server), deliver=deliver)
    server._offer_loop = loop
    loop.start()


async def offer_reply(server: Any, text: str) -> str | None:
    """The person's answer to the latest offer. Only when the offer is newer than Live's latest card."""
    offers = getattr(server, "_offers", None)
    if offers is None:
        return None
    from pilot.workflows.routines import normalize_phrase

    offered_at = offers.last_offered_at()
    live = getattr(server, "_live", None)
    raised = live.last_raised() if live is not None else None
    if offered_at is not None and raised and float(raised.get("at", 0.0)) > offered_at:
        return None  # Live spoke since: "why?" and "thanks" are about that
    answer = offers.reply(normalize_phrase(text))
    if answer is None:
        return None
    if answer.silence_all:
        await server._handle_update_config({"section": "suggestions", "values": {"offers": "off"}}, None)
    if answer.request:
        task = asyncio.ensure_future(server._voice_turn(answer.request))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return answer.reply


__all__ = [
    "begin_turn",
    "deliver_offer",
    "end_turn",
    "floor_for",
    "install",
    "link_for",
    "live_speak",
    "memory_talk_for",
    "model_is_local",
    "note_heliox_speech",
    "offer_reply",
    "offers_for",
    "personal_context_for",
    "quiet_moment",
    "start_offers",
]
