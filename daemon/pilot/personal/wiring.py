"""The thin glue between the server and the companion pieces (the floor, the Live link, personal context, memory
talk and offers), so the server needs only one-line hooks. Every function tolerates a server built without the piece
it reads (tests build partial servers), and nothing here imports audio, speech or model packages.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from datetime import datetime, timedelta
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
    if status in ("spoken", "interrupted") and listener is not None:
        # Live spoke to the person: for a few seconds its own replies ("why?", "not now") need no wake phrase.
        # The listener decides what counts (pilot.live.engine.is_live_reply); before it has this, nothing changes.
        note = getattr(listener, "note_live_spoke", None)
        if callable(note):
            try:
                note()
            except Exception:
                logger.debug("note_live_spoke failed", exc_info=True)
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
        suggestions=(suggester.picks if suggester is not None else None),
        legacy_suggestions_on=lambda: bool(getattr(config.suggestions, "enabled", False)),
        timeline_off=lambda: not bool(getattr(getattr(config, "timeline", None), "enabled", False)),
        spans=spans,
        moments=lambda now: _day_moments(server, now) + _learned(server, now),
        label=app_label,
        state_path=Path(data_dir) / "offers.json" if data_dir is not None else None,
    )
    server._offers = offers
    return offers


def routines_for(server: Any) -> Any:
    """The person's routine record (pilot.personal.routines), made once."""
    record = getattr(server, "_routines", None)
    if record is not None:
        return record
    from pathlib import Path

    from pilot.config import DATA_DIR
    from pilot.personal.routines import RoutineRecord

    def learning() -> bool:
        return bool(getattr(getattr(server.config, "memory", None), "learn_routines", True))

    record = RoutineRecord(Path(DATA_DIR) / "routines.json", enabled=learning)

    async def stop() -> None:
        await server._handle_update_config({"section": "memory", "values": {"learn_routines": False}}, None)
        await server._broadcast_notification("config_updated", {"method": "voice", "section": "memory"})

    record.stop = stop
    server._routines = record
    return record


def note_request(server: Any, text: str, *, planned: bool) -> None:
    """A request the person addressed to Heliox and Heliox took: counted as a day of use (dates only, for the one
    timeline offer) and kept for routines when it is one (never raises)."""
    offers = getattr(server, "_offers", None)
    if offers is not None:
        try:
            offers.note_used(offers.now().date().isoformat())
        except Exception:
            logger.debug("Could not count the day of use", exc_info=True)
    record = getattr(server, "_routines", None)
    if record is None:
        return
    try:
        record.note(str(text), planned=planned)
    except Exception:
        logger.debug("Could not note the request for routines", exc_info=True)


def note_task_done(server: Any, request: str) -> None:
    """A request finished well: when it matches one of the person's active goals, offer to note it as progress at
    the next quiet moment (pilot.personal.offers.goal_progress). Never raises."""
    goals = getattr(server, "_goals", None)
    if goals is None or not str(request or "").strip():
        return
    try:
        from pilot.personal.offers import goal_progress

        offer = goal_progress(str(request), goals.active())
        if offer is not None:
            offers_for(server).offer_soon(offer)
    except Exception:
        logger.debug("Could not check the request against the goals", exc_info=True)


def _learned(server: Any, now: datetime) -> list[Any]:
    record = getattr(server, "_routines", None)
    if record is None:
        return []
    try:
        return record.offers(now)
    except Exception:
        logger.debug("Could not read the learned routines", exc_info=True)
        return []


def _day_moments(server: Any, now: datetime) -> list[Any]:
    """The start and end of the working day the person told Heliox, as offers (pilot.personal.offers.day_moments)."""
    from pilot.personal.offers import day_moments

    profile = getattr(server, "_user_profile", None)
    if profile is None:
        return []
    try:
        return day_moments(now, start=profile.work_start_minute(), end=profile.work_end_minute())
    except Exception:
        logger.debug("Could not read the working day", exc_info=True)
        return []


# Until this hour the evening's quiet holds after the end of the working day ("I stop work at six").
WORK_QUIET_UNTIL_HOUR = 5


def quiet_now(server: Any, now: datetime | None = None) -> str:
    """Why Heliox holds back offers, suggestions and Live's non-urgent catches right now, or "": a focus session,
    "stay quiet unless I ask", or after the end of the working day the person gave ("I stop work at six"), until
    5 in the morning. The first time the working day's end applies on a day, one sentence goes to the chat."""
    focus = getattr(server, "_focus", None)
    if focus is not None and getattr(focus, "active", False):
        return "focus"
    profile = getattr(server, "_user_profile", None)
    if profile is None:
        return ""
    try:
        if profile.quiet():
            return "asked"
        end = profile.work_end_minute()
    except Exception:
        logger.debug("Could not read the quiet preferences", exc_info=True)
        return ""
    if end is None:
        return ""
    now = now or datetime.now()
    minute = now.hour * 60 + now.minute
    if not (minute >= end or now.hour < WORK_QUIET_UNTIL_HOUR):
        return ""
    day = (now - timedelta(hours=WORK_QUIET_UNTIL_HOUR)).date()  # the evening and the small hours are one evening
    if getattr(server, "_work_end_said_on", None) != day:
        server._work_end_said_on = day
        sentence = (
            f"It's after {end // 60:02d}:{end % 60:02d}, when you stop work, so offers and suggestions wait until "
            "morning. Risky things still get flagged."
        )
        broadcast = getattr(server, "_broadcast_notification", None)
        if callable(broadcast):
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(
                    broadcast(
                        "voice_result",
                        {"command": "", "status": "answered", "message": sentence, "source": "working_day"},
                    )
                )
    return "after_work"


def held_for_quiet(server: Any) -> bool:
    """For Live and suggestions (their ``focus_active``): hold everything that is not urgent."""
    try:
        return bool(quiet_now(server))
    except Exception:
        logger.debug("Quiet check failed", exc_info=True)
        return False


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
    if quiet_now(server):
        return False  # a focus session, "stay quiet unless I ask", or after the working day
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


def offer_in_reply(server: Any, kind: str, text: str, request: str, why: str) -> str:
    """One concrete offer made in reply to something the person said ("I never clean anything up." -> "Want me to
    find what's taking up space in Downloads?"), under the same rules as every other offer. Returns the offer's id
    when it may be made (it is then counted: the hourly and daily caps, "not today", "don't offer that", "why?" and
    "stop suggesting things" all apply), or "" when it may not, and then nothing is offered and nothing is said about
    it:

    * offers are off ("stop suggesting things", "no more offers", Settings);
    * "auto" (the default) and Heliox knows nothing about the person yet: the first week offers nothing but the
      getting-to-know questions;
    * quiet: a focus session, "stay quiet unless I ask", after the end of the working day, or a call;
    * the caps (one an hour, three a day, a minimum gap), this offer or its kind silenced, or declined today, or
      already offered today.

    ``kind`` names the kind of offer for silencing ("tidy_downloads"); ``request`` is what "yes" sends down the
    normal road; ``why`` is the answer to "why?"."""
    from pilot.personal.offers import Offer

    offers = offers_for(server)
    if not offers.active():
        return ""
    if quiet_now(server, offers.now()) or (getattr(server, "config", None) is not None and call_watch_for(server)()):
        return ""
    offer = Offer(key=f"reply:{kind}", kind=kind, text=text, request=request, why=why)
    today = offers.now().date().isoformat()
    if (
        offer.key in offers.silenced
        or f"kind:{kind}" in offers.silenced
        or offers.declined.get(offer.key) == today
        or any(m.offer.key == offer.key and m.day == today for m in offers.made)
        or not offers.room()
    ):
        return ""
    return offers.mark_offered(offer)


# What an offer's words say about the person, by kind, for the marker that stands in for them in a model's history
# when personal lines may not go to that model (conversation.share_profile). The words name the kind and nothing else.
PROFILE_OFFER_MARKERS = {
    "habit": "(made an offer on this computer, about one of your habits)",
    "stale_goal": "(made an offer on this computer, about one of your goals)",
    "goal_step": "(made an offer on this computer, about one of your goals)",
    "goal_progress": "(made an offer on this computer, about one of your goals)",
    "todos": "(made an offer on this computer, about your to-do list)",
    "routine": "(made an offer on this computer, about your working day)",
    "usual_app": "(made an offer on this computer, about an app you usually open)",
}
# Offers built from the routine record (pilot.personal.routines) go to a model on this computer only, under every
# value of the switch.
PRIVATE_OFFER_KINDS = frozenset({"learned_routine"})
OFFERED_HERE = "(made an offer on this computer, from your routines)"


def offer_in_history(offer: Any, layer: Any, server: Any = None) -> str:
    """What the conversation's history keeps of an offer, the history a model is later sent. An offer built from
    what Heliox knows about the person keeps its words only when personal lines may go to that model
    (pilot.conversation.adapters.personal_in_prompt, the one place that decides it); otherwise the marker naming its
    kind. One built from the routine record keeps its words only for a model on this computer. Offers that hold
    nothing about the person (getting to know them, the timeline) keep their words."""
    kind = str(getattr(offer, "kind", ""))
    try:
        model = layer._model() if layer is not None else None
    except Exception:
        model = None
    if kind in PRIVATE_OFFER_KINDS:
        return str(offer.text) if bool(getattr(model, "local", False)) else OFFERED_HERE
    if kind not in PROFILE_OFFER_MARKERS:
        return str(offer.text)
    from pilot.conversation.adapters import personal_in_prompt

    try:
        allowed = personal_in_prompt(server, model)
    except Exception:
        allowed = False  # when in doubt, nothing about the person
    return str(offer.text) if allowed else PROFILE_OFFER_MARKERS[kind]


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
        layer.memory.add_heliox(offer_in_history(offer, layer, server))
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

        # No set_quiet: the preference getting to know stores is the one quiet_now reads (UserProfile.quiet), the
        # same one "stay quiet unless I ask" said on its own sets, undone by "you can speak up again". The config's
        # offers and Live level stay what the person chose in Settings.
        gtk = GettingToKnow(profile=lambda: getattr(server, "_user_profile", None))
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


# ── a slip in something said TO Heliox (pilot.live.held) ──


async def live_hold(server: Any, text: str) -> Any:
    """The hook for every addressed turn, typed or spoken, before any road takes it (it replaces the plain
    ``live.offer_speech`` call at the top of dispatch). None: go on. Otherwise ``say`` is said or shown now, and
    ``proceed`` is the request to go on with, or None: the turn waits for the person's answer, or was dropped."""
    live = getattr(server, "_live", None)
    if live is None:
        return None
    try:
        return live.hold(text)
    except Exception:
        logger.warning("Live could not check the turn; it goes on unchecked", exc_info=True)
        return None


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


def live_blind_note(server: Any, app: str, focused: Any) -> None:
    """Tell Live whether the reader gave anything for the window in front, so it can say once, quietly, when it
    cannot see into an app (pilot.live.blind_spots). A terminal Live reads directly counts as seen into."""
    live = getattr(server, "_live", None)
    note = getattr(live, "note_foreground", None)
    if not callable(note):
        return
    readable = focused is not None
    if not readable:
        from pilot.live.terminal_text import which_terminal

        terminal = which_terminal(app)
        if terminal:
            ready = terminal_text_for(server).status().get(terminal, {})
            readable = bool(ready.get("ready", ready.get("found", False)) and not ready.get("needs"))
    state, message = "ready", ""
    reader_status = getattr(getattr(server, "_live_reader", None), "status", None)
    if callable(reader_status):
        try:
            current = reader_status()
            state, message = str(current.state), str(current.message)
        except Exception:
            logger.debug("Reader status failed", exc_info=True)
    try:
        note(app, readable, state, message)
    except Exception:
        logger.debug("Live could not note the window in front", exc_info=True)


def terminal_text_for(server: Any) -> Any:
    found = getattr(server, "_terminal_text", None)
    if found is None:
        from pilot.live.terminal_text import TerminalText

        found = server._terminal_text = TerminalText()
    return found


def shell_observation(server: Any, app: str, title: str) -> Any:
    """Live's look at a terminal that shows accessibility nothing: the line being typed, while it is fresh, from
    the zsh or fish integration, or else from kitty or WezTerm themselves (pilot.live.terminal_text)."""
    from pilot.live.engine import Observation

    found = getattr(server, "_shell_lines_server", None)
    latest = found.lines.latest() if found is not None and found.listening else None
    if latest is not None:
        key, seen = latest
        # The shell's name in the title makes the detectors read it as a terminal line, which it is.
        return Observation(
            text=seen.line, app=app or seen.term, title=f"{title} ({seen.shell})", source="focused_field", field=key
        )
    from pilot.live.terminal_text import which_terminal

    terminal = which_terminal(app, title)
    live = getattr(getattr(server, "config", None), "live", None)
    if not terminal or not getattr(live, "read_focused_text", True):
        return None
    reader = terminal_text_for(server)
    reader.poke(terminal)  # in the background: this look uses what the last one found
    seen = reader.latest(terminal)
    if seen is None:
        return None
    return Observation(
        text=seen.line, app=app, title=f"{title} ({terminal} terminal)", source="focused_field", field=seen.pane
    )


async def live_shell_rpc(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """``action``: "status" (what Add would do, for each shell), "test" with ``shell`` (the dry run: would the hook
    work here), "install" (refused unless the dry run passes: a silent hook is never added) or "uninstall"."""
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
    tested: dict[str, Any] = {}
    try:
        if action in ("test", "install"):
            ready = await shell_line.dry_run(shell, found.path if found.listening else None)
            tested = {shell: ready.as_dict()}
            if action == "install" and not ready.ok:
                message = f"Not added: {ready.problem}" + (f" {ready.install}" if ready.install else "")
                return {"status": "error", "message": message, "ready": tested}
            if action == "install":
                pinned = (ready.nc, ready.nc_flags) if shell == "fish" and ready.nc else None
                shell_line.install(shell, home, env, found.path, pinned)
        elif action == "uninstall":
            shell_line.uninstall(shell, home, env)
        elif action != "status":
            return {"status": "error", "message": "action must be status, test, install or uninstall"}
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
        "ready": tested,
        "fish_bound_ms": int(shell_line.BOUND_SECONDS * 1000),
        "terminals": terminal_text_for(server).status(),
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
