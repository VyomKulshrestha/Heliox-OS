"""A setting change that takes protection away waits for the person's approval in the daemon's own dialog.

Narrowing a guard is always free. Widening one (or sending requests to a computer that is not this one, or turning
a Live capture on) takes effect only once the person approves it; denied or unanswered, the setting stays as it was
and the reply is "cancelled". Until 13 November 2026 any holder of the window's token changed these with no
approval. Each case goes through the real handler.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.security import guard_changes as guards
from pilot.server import PilotServer


class Person:
    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed
        self.asked: list[dict[str, Any]] = []

    async def __call__(self, plan_id: str, plan: Any, ws: Any, **kwargs: Any):
        self.asked.append({"plan_id": plan_id, **kwargs})
        everything = set(range(len(plan.actions)))
        return self.confirmed, (everything if self.confirmed else set()), everything


class Receipts:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.finished: list[dict[str, Any]] = []

    def record_action(self, task_key, plan_id, action, result, **kwargs) -> None:
        self.actions.append(action.target)

    async def finalize(self, task_key, **kwargs):
        self.finished.append(kwargs)

    async def approved_settings(self) -> set[str]:
        return set(self.actions)


def _server(person: Person, monkeypatch) -> PilotServer:
    config = PilotConfig()
    monkeypatch.setattr(config, "save", lambda *a, **k: None)
    server = PilotServer(config)
    server._wait_for_confirmation = person  # type: ignore[method-assign]
    server._task_receipts = Receipts()
    server._broadcast_notification = AsyncMock()
    return server


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


def _lines(person: Person) -> list[str]:
    return [line["target"] for line in person.asked[0]["shown"]] if person.asked else []


@pytest.mark.asyncio
async def test_turning_on_unrestricted_shell_waits_and_a_denial_leaves_it_off(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    reply = await server._handle_update_config({"section": "security", "values": {"unrestricted_shell": True}}, _ws())

    assert reply["status"] == "cancelled" and server.config.security.unrestricted_shell is False
    assert _lines(person) == ["Let Heliox run any shell command"]
    assert person.asked[0]["force_all_actions"] is True and server._task_receipts.actions == []


@pytest.mark.asyncio
async def test_an_approved_widening_takes_effect_and_is_on_the_receipts(monkeypatch):
    person = Person(confirmed=True)
    server = _server(person, monkeypatch)

    reply = await server._handle_update_config({"section": "security", "values": {"root_enabled": True}}, _ws())

    assert reply["status"] != "cancelled" and server.config.security.root_enabled is True
    assert server._task_receipts.actions == ["Let Heliox act as administrator (security.root_enabled)"]
    assert server._task_receipts.finished[0]["source"] == "settings"


@pytest.mark.asyncio
async def test_narrowing_is_free(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.security.root_enabled = True
    server.config.security.dry_run = False

    await server._handle_update_config(
        {"section": "security", "values": {"root_enabled": False, "dry_run": True}}, _ws()
    )
    await server._handle_update_config(
        {"section": "restrictions", "values": {"protected_folders": ["/a", "/b"]}}, _ws()
    )

    assert person.asked == []
    assert server.config.security.root_enabled is False and server.config.security.dry_run is True


@pytest.mark.parametrize(
    ("section", "before", "values", "line"),
    [
        ("security", {"dry_run": True}, {"dry_run": False}, "Let Heliox's actions really run"),
        (
            "security",
            {"snapshot_on_destructive": True},
            {"snapshot_on_destructive": False},
            "Stop taking a snapshot before destructive actions",
        ),
        (
            "security",
            {"sandbox_mode": "docker"},
            {"sandbox_mode": "none"},
            "Run commands in a weaker sandbox (none instead of docker)",
        ),
        (
            "security",
            {},
            {"sandbox_firecracker_binary": "/tmp/fc"},
            "Use /tmp/fc as the sandbox program",
        ),
        ("preview", {"enabled": True}, {"enabled": False}, "Stop previewing plans before they run"),
        (
            "proxy",
            {},
            {"https": "http://proxy.example.com:3128"},
            "Send Heliox's web requests through proxy.example.com",
        ),
        (
            "conversation",
            {},
            {"search_url": "https://search.example.com/?q={query}"},
            "Search the web with search.example.com",
        ),
        ("email", {}, {"imap_host": "imap.example.com"}, "Read mail from imap.example.com"),
        ("calendar", {}, {"caldav_url": "https://cal.example.com/dav"}, "Use the calendar at cal.example.com"),
    ],
)
@pytest.mark.asyncio
async def test_each_widening_through_update_config_asks(monkeypatch, section, before, values, line):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    for key, value in before.items():
        setattr(getattr(server.config, section), key, value)
    previous = {key: getattr(getattr(server.config, section), key) for key in values}

    reply = await server._handle_update_config({"section": section, "values": values}, _ws())

    assert line in _lines(person)
    assert reply["status"] == "cancelled"
    assert {key: getattr(getattr(server.config, section), key) for key in values} == previous


@pytest.mark.asyncio
async def test_removing_a_protected_folder_asks_and_names_it(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.restrictions.protected_folders = ["/home/sam/private", "/home/sam/keys"]

    reply = await server._handle_update_config(
        {"section": "restrictions", "values": {"protected_folders": ["/home/sam/keys"]}}, _ws()
    )

    assert reply["status"] == "cancelled" and _lines(person) == ["Stop protecting /home/sam/private"]
    assert server.config.restrictions.protected_folders == ["/home/sam/private", "/home/sam/keys"]


@pytest.mark.asyncio
async def test_a_cloud_model_names_where_requests_go_and_a_local_one_never_asks(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    await server._handle_update_config(
        {"section": "model", "values": {"provider": "cloud", "cloud_provider": "claude"}}, _ws()
    )
    assert person.asked[0]["reason"] == (
        "Your requests will go to api.anthropic.com, together with the text Heliox reads for them, and your name, your "
        "answer style and the parts of your profile that bear on what you ask (you can stop that part in Settings, "
        "Your profile and the model)."
    )
    assert server.config.model.provider == "ollama"

    person.asked.clear()
    server.config.model.provider, server.config.model.cloud_provider = "cloud", "claude"
    await server._handle_update_config({"section": "model", "values": {"provider": "ollama"}}, _ws())
    assert person.asked == [] and server.config.model.provider == "ollama"


@pytest.mark.asyncio
async def test_turning_live_on_with_its_captures_asks_in_words_about_what_is_captured(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    reply = await server._handle_update_config({"section": "live", "values": {"enabled": True}}, _ws())

    assert reply["status"] == "cancelled" and server.config.live.enabled is False
    described = [line["description"] for line in person.asked[0]["shown"]]
    assert "Heliox will read the text in the field you are typing in." in described


@pytest.mark.asyncio
async def test_the_risk_gate_off_asks(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.gateway.risk_gate_enabled = True

    reply = await server._handle_risk_gate_config_update({"enabled": False}, _ws())

    assert reply["status"] == "cancelled" and server.config.gateway.risk_gate_enabled is True
    assert _lines(person) == ["Turn off the risk check"]


@pytest.mark.asyncio
async def test_self_healing_and_the_narrator_ask_before_they_let_more_run(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    tier = server.config.self_healing.auto_execute_max_tier

    healing = await server._handle_self_healing_config_update({"auto_execute_max_tier": tier + 1}, _ws())
    narration = await server._handle_narration_config_update({"interrupt_on_risk": False}, _ws())
    shorter = await server._handle_narration_config_update({"confirm_timeout_seconds": 1.0}, _ws())

    assert healing["status"] == narration["status"] == shorter["status"] == "cancelled"
    assert server.config.self_healing.auto_execute_max_tier == tier
    assert server.config.narration.interrupt_on_risk is True


@pytest.mark.asyncio
async def test_a_gateway_floor_raised_asks_and_lowered_is_free(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    profile, floor = next(iter(server.config.gateway.source_profiles.items()))
    family = next(iter(floor.max_tier), "file")
    tier = int(floor.max_tier.get(family, 0))

    raised = await server._handle_gateway_policy_update({"profile": profile, "max_tier": {family: tier + 1}}, _ws())
    assert raised["status"] == "cancelled" and person.asked

    person.asked.clear()
    if tier > 0:
        await server._handle_gateway_policy_update({"profile": profile, "max_tier": {family: tier - 1}}, _ws())
        assert person.asked == []


@pytest.mark.asyncio
async def test_a_reset_that_would_widen_asks_first(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.security.dry_run = True  # the default is off: a reset would let actions really run

    reply = await server._handle_reset_config({}, _ws())

    assert reply["status"] == "cancelled" and server.config.security.dry_run is True
    assert "Let Heliox's actions really run" in _lines(person)


def test_a_change_of_destination_to_this_computer_is_not_a_widening():
    model = {"provider": "cloud", "cloud_provider": "openai"}
    assert guards.section_widenings("model", model, {"provider": "ollama"}) == []
    assert guards.section_widenings("proxy", {"https": None}, {"https": "http://127.0.0.1:8080"}) == []


@pytest.mark.asyncio
async def test_a_spoken_request_to_turn_live_on_asks_in_the_window_and_voice_cannot_answer_it(monkeypatch):
    person = Person(confirmed=False)  # the dialog went unanswered, or the person said no
    server = _server(person, monkeypatch)

    reply = await server._control_reply("watch over me")

    # No window is open: nothing is asked yet, and the request is kept for the next window.
    assert reply == "Open Heliox and approve it there."  # spoken_settings.OPEN_TO_APPROVE
    assert server.config.live.enabled is False and person.asked == []
    assert server._spoken_widening is not None


@pytest.mark.asyncio
async def test_the_listener_never_takes_wake_words_from_the_caller(monkeypatch):
    from pilot.server import WAKE_WORDS

    server = _server(Person(confirmed=False), monkeypatch)
    started: dict[str, Any] = {}

    class Listener:
        is_running = False

        def __init__(self, *, wake_words, **kwargs):
            started["wake_words"] = wake_words

        async def start(self, *args, **kwargs):
            return None

        def __getattr__(self, name):
            return lambda *a, **k: None

    import pilot.system.voice as voice

    monkeypatch.setattr(voice, "ContinuousVoiceListener", Listener)
    try:
        await server._handle_voice_listener_start({"wake_words": ["okay"]}, _ws())
    except Exception:
        pass  # what happens after the listener is made is not this test's business

    assert started["wake_words"] == list(WAKE_WORDS) and all("heliox" in w or "pilot" in w for w in WAKE_WORDS)


def _dispatching(server: PilotServer) -> PilotServer:
    server._handlers = {"update_config": server._handle_update_config}
    return server


async def _call(server: PilotServer, section: str, values: dict[str, Any]) -> dict[str, Any]:
    import json

    from pilot.server import JsonRpcRequest

    response = await server._dispatch(
        JsonRpcRequest(method="update_config", params={"section": section, "values": values}, id=1), _ws()
    )
    return json.loads(response)["result"]


class Waiting(Person):
    """A person who leaves the dialog open until ``answer`` is set."""

    def __init__(self, confirmed: bool) -> None:
        import asyncio

        super().__init__(confirmed)
        self.answer = asyncio.Event()
        self.open = 0
        self.most_open = 0

    async def __call__(self, plan_id: str, plan: Any, ws: Any, **kwargs: Any):
        self.open += 1
        self.most_open = max(self.most_open, self.open)
        try:
            await self.answer.wait()
            return await super().__call__(plan_id, plan, ws, **kwargs)
        finally:
            self.open -= 1


async def _spoken_tasks(server: PilotServer) -> None:
    import asyncio

    await asyncio.gather(*list(server.__dict__.get("_spoken_widening_tasks", set())))


@pytest.mark.asyncio
async def test_a_narrowing_applies_at_once_while_a_widening_waits_for_its_answer(monkeypatch):
    import asyncio

    person = Waiting(confirmed=True)
    server = _dispatching(_server(person, monkeypatch))
    widening = asyncio.create_task(_call(server, "security", {"unrestricted_shell": True}))
    await asyncio.sleep(0.05)  # its dialog is open now

    narrowing = await asyncio.wait_for(_call(server, "live", {"listen_to_speech": False}), 2)

    assert narrowing["status"] != "cancelled" and server.config.live.listen_to_speech is False
    person.answer.set()
    assert (await widening)["status"] != "cancelled" and server.config.security.unrestricted_shell is True


@pytest.mark.asyncio
async def test_a_second_widening_waits_for_the_first_dialog_then_asks_its_own(monkeypatch):
    import asyncio

    person = Waiting(confirmed=True)
    server = _dispatching(_server(person, monkeypatch))
    first = asyncio.create_task(_call(server, "security", {"unrestricted_shell": True}))
    second = asyncio.create_task(_call(server, "security", {"root_enabled": True}))
    await asyncio.sleep(0.05)
    assert person.open == 1  # one dialog at a time
    person.answer.set()
    await first
    await second

    assert person.most_open == 1 and len(person.asked) == 2
    assert server.config.security.unrestricted_shell is True and server.config.security.root_enabled is True


@pytest.mark.asyncio
async def test_a_setting_changed_while_the_dialog_was_open_is_not_written(monkeypatch):
    import asyncio

    person = Waiting(confirmed=True)
    server = _dispatching(_server(person, monkeypatch))
    server.config.restrictions.protected_folders = ["/a", "/b"]
    widening = asyncio.create_task(_call(server, "restrictions", {"protected_folders": ["/b"]}))
    await asyncio.sleep(0.05)
    server.config.restrictions.protected_folders = ["/a", "/b", "/c"]  # changed underneath: now /c goes too
    person.answer.set()

    reply = await widening

    assert reply["status"] == "cancelled" and "changed while you were deciding" in reply["message"]
    assert server.config.restrictions.protected_folders == ["/a", "/b", "/c"]


@pytest.mark.asyncio
async def test_a_spoken_widening_with_a_window_open_is_put_there_and_applied_once_approved(monkeypatch):
    person = Person(confirmed=True)
    server = _dispatching(_server(person, monkeypatch))
    server._clients.add(object())

    reply = await server._control_reply("watch over me")
    await _spoken_tasks(server)

    assert reply == "I've put that in Heliox's window for you to approve."  # spoken_settings.APPROVE_IN_WINDOW
    assert person.asked and server.config.live.enabled is True


@pytest.mark.asyncio
async def test_off_by_voice_is_immediate_and_back_on_to_what_was_approved_does_not_ask(monkeypatch):
    person = Person(confirmed=True)
    server = _dispatching(_server(person, monkeypatch))
    server._clients.add(object())
    await server._control_reply("watch over me")
    await _spoken_tasks(server)
    asked_once = len(person.asked)

    await server._control_reply("stop correcting me")
    assert server.config.live.enabled is False and len(person.asked) == asked_once  # off: never asks
    reply = await server._control_reply("watch over me")

    assert server.config.live.enabled is True and len(person.asked) == asked_once  # approved before: not again
    assert "approve" not in reply


@pytest.mark.asyncio
async def test_a_spoken_widening_kept_for_the_next_window_is_asked_once_and_not_after_it_expires(monkeypatch):
    import pilot.server as server_module

    person = Person(confirmed=False)
    server = _dispatching(_server(person, monkeypatch))
    await server._control_reply("watch over me")
    server._offer_kept_spoken_widening()
    await _spoken_tasks(server)
    server._offer_kept_spoken_widening()  # a second window: not asked again
    assert len(person.asked) == 1

    await server._control_reply("watch over me")
    monkeypatch.setattr(server_module.time, "monotonic", lambda: 10**12)
    server._offer_kept_spoken_widening()
    await _spoken_tasks(server)
    assert len(person.asked) == 1  # expired: not asked


@pytest.mark.asyncio
async def test_clipboard_history_on_asks_in_words_and_off_is_free(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    on = await server._handle_update_config({"section": "clipboard", "values": {"history": True}}, _ws())
    assert on["status"] == "cancelled" and server.config.clipboard.history is False
    line = person.asked[0]["shown"][0]
    assert line["target"] == "Let Heliox remember what you copy" and "in memory on this computer" in line["description"]

    person.asked.clear()
    server.config.clipboard.history = True
    await server._handle_update_config({"section": "clipboard", "values": {"history": False}}, _ws())
    assert person.asked == [] and server.config.clipboard.history is False


@pytest.mark.asyncio
async def test_the_activity_timeline_turned_back_on_by_voice_after_one_approval_in_the_window(tmp_path):
    from pilot.workflows.task_receipts import TaskReceipts
    from tests.settings_approval import approve_setting_change

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._task_receipts = TaskReceipts(tmp_path / "receipts", key_dir=tmp_path / "key")

    # The first time, voice cannot turn it on: the window asks.
    assert await server._control_reply("turn on the activity timeline") == "Open Heliox and approve it there."
    assert server.config.timeline.enabled is False
    await approve_setting_change(server, "timeline", {"enabled": True})  # the person's yes in the dialog
    # Turning on exactly what the person approved on this computer does not ask again, nor after it was off.
    assert await server._control_reply("turn on the activity timeline") == "The activity timeline is on."
    assert server.config.timeline.enabled is True
    assert await server._control_reply("turn off the activity timeline") == "The activity timeline is off."
    assert await server._control_reply("turn on the activity timeline") == "The activity timeline is on."
    assert server.config.timeline.enabled is True


@pytest.mark.asyncio
async def test_supervision_watching_the_screen_and_the_keyboard_asks_and_a_no_leaves_it_off(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    reply = await server._handle_supervision_config_update(
        {"enabled": True, "keyboard_mouse_hook_enabled": True}, _ws()
    )

    assert reply["status"] == "cancelled"
    assert server.config.supervision.enabled is False and server.config.supervision.keyboard_mouse_hook_enabled is False
    lines = {line["target"]: line["description"] for line in person.asked[0]["shown"]}
    assert "every key you press and every mouse click" in lines["Let Heliox watch your keyboard and mouse"]
    assert "Let Heliox watch your screen" in lines


@pytest.mark.asyncio
async def test_turning_supervision_or_its_hook_off_is_free(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.supervision.enabled = True
    server.config.supervision.keyboard_mouse_hook_enabled = True
    server._background = None
    server._supervision_hook = None

    await server._handle_supervision_config_update({"keyboard_mouse_hook_enabled": False}, _ws())

    assert person.asked == [] and server.config.supervision.keyboard_mouse_hook_enabled is False


@pytest.mark.asyncio
async def test_sharing_screen_text_asks_with_a_cloud_model_naming_it_and_is_free_with_a_local_one(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    local = await server._handle_update_config(
        {"section": "conversation", "values": {"share_screen_text": "on"}}, _ws()
    )
    assert person.asked == [] and local["status"] != "cancelled"

    server.config.conversation.share_screen_text = "auto"
    server.config.model.provider, server.config.model.cloud_provider = "cloud", "claude"
    cloud = await server._handle_update_config(
        {"section": "conversation", "values": {"share_screen_text": "on"}}, _ws()
    )
    assert cloud["status"] == "cancelled" and server.config.conversation.share_screen_text == "auto"
    assert _lines(person) == ["Send the text on your screen to api.anthropic.com"]


@pytest.mark.asyncio
async def test_moving_to_a_cloud_model_with_screen_text_shared_says_the_screen_text_goes_too(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.conversation.share_screen_text = "on"

    await server._handle_update_config(
        {"section": "model", "values": {"provider": "cloud", "cloud_provider": "openai"}}, _ws()
    )

    (line,) = person.asked[0]["shown"]
    assert line["target"] == "Use the language model at api.openai.com"
    assert line["description"] == (
        "Your requests will go to api.openai.com, together with the text Heliox reads for them, the end of the text "
        "you are typing, and your name, your answer style and the parts of your profile that bear on what you ask "
        "(you can stop that part in Settings, Your profile and the model)."
    )


@pytest.mark.asyncio
async def test_the_cloud_model_sentence_leaves_out_what_the_person_does_not_share(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.conversation.share_profile = "off"

    await server._handle_update_config(
        {"section": "model", "values": {"provider": "cloud", "cloud_provider": "openai"}}, _ws()
    )

    assert person.asked[0]["shown"][0]["description"] == (
        "Your requests will go to api.openai.com, together with the text Heliox reads for them."
    )


@pytest.mark.asyncio
async def test_turning_the_profile_back_on_is_free_with_a_model_on_this_computer(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.conversation.share_profile = "off"

    local = await server._handle_update_config({"section": "conversation", "values": {"share_profile": "auto"}}, _ws())
    assert person.asked == [] and local["status"] != "cancelled"

    server.config.conversation.share_profile = "off"
    server.config.model.provider, server.config.model.cloud_provider = "cloud", "claude"
    cloud = await server._handle_update_config({"section": "conversation", "values": {"share_profile": "auto"}}, _ws())
    assert cloud["status"] == "cancelled" and server.config.conversation.share_profile == "off"
    assert _lines(person) == ["Send what Heliox knows about you to api.anthropic.com"]


@pytest.mark.asyncio
async def test_lives_screen_reading_asks_and_gaze_tracking_asks(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.live.enabled = True

    await server._handle_update_config({"section": "live", "values": {"ocr_fallback": True}}, _ws())
    await server._handle_update_config({"section": "vision", "values": {"gaze_tracking_enabled": True}}, _ws())

    assert _lines(person)[:1] == ["Let Live read your screen"]
    assert person.asked[1]["shown"][0]["target"] == "Let Heliox use the camera to see where you look"
    assert server.config.live.ocr_fallback is False and server.config.vision.gaze_tracking_enabled is False


@pytest.mark.asyncio
async def test_gaze_tracking_is_not_turned_back_on_by_an_old_approval(monkeypatch):
    from pilot.server import CAPTURE_SETTINGS

    assert not any("vision.".startswith(prefix) for prefix in CAPTURE_SETTINGS)  # the camera asks every time
    assert "live." in CAPTURE_SETTINGS  # Live's screen reading may come back by voice once approved


@pytest.mark.asyncio
async def test_adding_a_folder_to_index_asks_naming_it_and_removing_one_says_its_index_stays(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.semantic_search.folders = ["/home/sam/notes"]

    added = await server._handle_update_config(
        {"section": "semantic_search", "values": {"folders": ["/home/sam/notes", "/home/sam/mail"]}}, _ws()
    )
    assert added["status"] == "cancelled" and server.config.semantic_search.folders == ["/home/sam/notes"]
    line = person.asked[0]["shown"][0]
    assert (
        line["target"] == "Let Heliox index /home/sam/mail" and "search index on this computer" in line["description"]
    )

    person.asked.clear()
    removed = await server._handle_update_config({"section": "semantic_search", "values": {"folders": []}}, _ws())
    assert person.asked == [] and server.config.semantic_search.folders == []
    assert "/home/sam/notes" in removed["note"] and "stays in the search index" in removed["note"]


def _ssh_server(person: Person, monkeypatch) -> PilotServer:
    server = _server(person, monkeypatch)
    server._vault = MagicMock()
    server._vault.get_key = AsyncMock(return_value=None)
    server._vault.store_key = AsyncMock()
    return server


SSH_HOST = {
    "name": "build-box",
    "hostname": "10.0.0.7",
    "port": 2222,
    "username": "builder",
    "private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\nnot a real key\n-----END OPENSSH PRIVATE KEY-----",
    "strict_host_key_checking": True,
    "enabled": False,
}


@pytest.mark.asyncio
async def test_a_new_ssh_host_asks_naming_the_account_and_a_no_stores_no_key(monkeypatch):
    person = Person(confirmed=False)
    server = _ssh_server(person, monkeypatch)

    reply = await server._handle_ssh_save_host(dict(SSH_HOST), _ws())

    assert reply["status"] == "cancelled" and server.config.ssh.allowed_hosts == []
    server._vault.store_key.assert_not_awaited()  # nothing stored before the yes, the key included
    (line,) = person.asked[0]["shown"]
    assert line["target"] == "Let Heliox sign in to builder@10.0.0.7:2222"
    assert "run commands there" in line["description"]


@pytest.mark.asyncio
async def test_an_ssh_host_without_its_host_key_check_and_ssh_on_say_so(monkeypatch):
    person = Person(confirmed=False)
    server = _ssh_server(person, monkeypatch)

    await server._handle_ssh_save_host({**SSH_HOST, "strict_host_key_checking": False, "enabled": True}, _ws())

    assert _lines(person) == [
        "Let Heliox sign in to builder@10.0.0.7:2222",
        "Skip checking that 10.0.0.7 is the computer it was",
        "Let Heliox run commands on your SSH hosts",
    ]
    assert server.config.ssh.enabled is False


@pytest.mark.asyncio
async def test_an_approved_ssh_host_is_saved_and_a_new_key_for_the_same_place_asks_nothing(monkeypatch):
    person = Person(confirmed=True)
    server = _ssh_server(person, monkeypatch)

    saved = await server._handle_ssh_save_host(dict(SSH_HOST), _ws())
    assert saved["status"] == "ok" and [h.hostname for h in server.config.ssh.allowed_hosts] == ["10.0.0.7"]
    server._vault.store_key.assert_awaited()

    person.asked.clear()
    again = await server._handle_ssh_save_host({**SSH_HOST, "private_key": "another key"}, _ws())
    assert again["status"] == "ok" and person.asked == []

    moved = await server._handle_ssh_save_host({**SSH_HOST, "hostname": "10.0.0.8"}, _ws())
    assert moved["status"] == "ok" and _lines(person) == ["Let Heliox sign in to builder@10.0.0.8:2222"]


@pytest.mark.asyncio
async def test_a_malformed_ssh_host_is_refused_before_anything_is_asked(monkeypatch):
    person = Person(confirmed=True)
    server = _ssh_server(person, monkeypatch)

    no_key = await server._handle_ssh_save_host({**SSH_HOST, "private_key": ""}, _ws())
    bad_port = await server._handle_ssh_save_host({**SSH_HOST, "port": 70000}, _ws())

    assert no_key["status"] == bad_port["status"] == "error" and person.asked == []


@pytest.mark.asyncio
async def test_turning_ssh_on_asks_and_off_is_free(monkeypatch):
    person = Person(confirmed=False)
    server = _ssh_server(person, monkeypatch)

    on = await server._handle_ssh_set_enabled({"enabled": True}, _ws())
    assert on["status"] == "cancelled" and server.config.ssh.enabled is False
    assert _lines(person) == ["Let Heliox run commands on your SSH hosts"]

    person.asked.clear()
    server.config.ssh.enabled = True
    off = await server._handle_ssh_set_enabled({"enabled": False}, _ws())
    assert off["status"] == "ok" and person.asked == [] and server.config.ssh.enabled is False


@pytest.mark.asyncio
async def test_letting_phones_send_requests_and_start_them_asks_and_a_no_changes_nothing(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)

    reply = await server._handle_air_handoff_requests_configure(
        {"accept_requests": True, "run_read_only_requests": True}, _ws()
    )

    assert reply["status"] == "cancelled"
    assert server.config.air_handoff.accept_requests is False
    assert server.config.air_handoff.run_read_only_requests is False
    lines = {line["target"]: line["description"] for line in person.asked[0]["shown"]}
    assert "waits on this computer until you press Run" in lines["Let your paired phones send Heliox requests"]
    assert "without you pressing Run" in lines["Let your paired phones start status checks without you"]


@pytest.mark.asyncio
async def test_phone_switches_turned_off_are_free_and_a_malformed_request_asks_nothing(monkeypatch):
    person = Person(confirmed=False)
    server = _server(person, monkeypatch)
    server.config.air_handoff.accept_requests = True
    server.config.air_handoff.run_read_only_requests = True
    server._publish_air_handoff_state = AsyncMock(return_value={})

    bad = await server._handle_air_handoff_requests_configure(
        {"accept_requests": False, "run_read_only_requests": True}, _ws()
    )
    off = await server._handle_air_handoff_requests_configure(
        {"accept_requests": False, "run_read_only_requests": False}, _ws()
    )

    assert bad["status"] == "error" and off["status"] == "ok" and person.asked == []
    assert server.config.air_handoff.accept_requests is False
