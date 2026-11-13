"""What a model that is not on this computer is told about the person (conversation.share_profile), decided in one
place: pilot.conversation.adapters.personal_in_prompt. "auto" (the default) is today's behaviour; "off" keeps every
personal line out; a model on this computer gets them all under either value.

Each thing about the person that can reach a model has a test named after it below. A new one added later belongs
next to them: it must ask personal_in_prompt, and "off" must keep it out."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from pilot.config import PilotConfig
from pilot.conversation import layer_for
from pilot.conversation.adapters import personal_in_prompt
from pilot.conversation.context import build_messages
from pilot.server import PilotServer
from pilot.workflows.user_profile import UserProfile


async def _nothing(*_args, **_kwargs):
    return None


def _server(tmp_path, *, cloud: bool, share: str = "auto") -> PilotServer:
    config = PilotConfig()
    if cloud:
        config.model.provider, config.model.cloud_provider = "cloud", "claude"
    else:
        config.model.provider = "ollama"
    config.conversation.share_profile = share
    server = PilotServer(config)
    profile = UserProfile(tmp_path / "profile.json", remember=_nothing, forget=_nothing)
    server._user_profile = profile
    asyncio.run(profile.set_name("Sam"))
    asyncio.run(profile.set_style("length", "short"))
    asyncio.run(profile.answer_async("my manager is Priya"))
    asyncio.run(profile.add("about", "I'm training for a marathon"))
    layer = layer_for(server)
    layer._model = lambda: SimpleNamespace(local=not cloud)
    return server


def _sent(server: PilotServer, words: str) -> str:
    layer = layer_for(server)
    messages = build_messages(words, layer.sources, layer.memory, screen=None)
    return " ".join(str(m.get("content", "")) for m in messages)


CASES = [  # (cloud, share, personal lines go)
    (True, "auto", True),
    (True, "off", False),
    (False, "auto", True),
    (False, "off", True),  # a model on this computer: unaffected
]


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_one_function_decides(tmp_path, cloud, share, goes):
    assert personal_in_prompt(_server(tmp_path, cloud=cloud, share=share)) is goes


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_name_line(tmp_path, cloud, share, goes):
    assert ("Call them Sam" in _sent(_server(tmp_path, cloud=cloud, share=share), "what time is it")) is goes


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_answer_style_line(tmp_path, cloud, share, goes):
    server = _server(tmp_path, cloud=cloud, share=share)
    layer = layer_for(server)
    lines = layer.sources.profile_lines()  # a turn with no words: the name and the style
    assert any(line.startswith("Answer style") for line in lines) is goes


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_answer_length_line(tmp_path, cloud, share, goes):
    assert ("They asked for short answers" in _sent(_server(tmp_path, cloud=cloud, share=share), "why")) is goes


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_profile_lines_that_bear_on_the_words(tmp_path, cloud, share, goes):
    sent = _sent(_server(tmp_path, cloud=cloud, share=share), "how should I train for my marathon this week")
    assert ("marathon" in sent.split("[The person said]")[0]) is goes


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_personal_context_line(tmp_path, cloud, share, goes):
    """Goals, habits and profile lines matched to the words (pilot.personal.context.PersonalContext.extra)."""
    from pilot.conversation.context import _personal_extra
    from pilot.personal import wiring

    server = _server(tmp_path, cloud=cloud, share=share)
    server._goals = SimpleNamespace(
        list=lambda: [{"title": "Run the Berlin half", "next_step": "a long run on Sunday"}]
    )
    server._personal_context = None
    wiring.install(server, layer_for(server))
    context = wiring.personal_context_for(server)
    words = "what about my Berlin half this Sunday"
    line = context.extra(words)
    assert line and _personal_extra(context.extra)
    assert (line in _sent(server, words)) is goes


@pytest.mark.parametrize(("cloud", "share", "goes"), CASES)
def test_the_people_in_the_planners_hand_over(tmp_path, cloud, share, goes):
    server = _server(tmp_path, cloud=cloud, share=share)
    people = layer_for(server).sources.people()
    assert (people.get("manager") == "Priya") is goes


def test_an_existing_config_without_the_key_keeps_todays_behaviour(tmp_path, monkeypatch):
    import pilot.config as config_module

    path = tmp_path / "config.toml"
    path.write_text('[model]\nprovider = "cloud"\ncloud_provider = "claude"\n[conversation]\nenabled = true\n')
    monkeypatch.setattr(config_module, "CONFIG_FILE", path)
    loaded = config_module.PilotConfig.load()
    assert loaded.conversation.share_profile == "auto"


def test_turning_it_off_is_free_and_turning_it_back_on_asks(tmp_path):
    server = _server(tmp_path, cloud=True, share="auto")
    assert server._widenings_for("update_config", {"section": "conversation", "values": {"share_profile": "off"}}) == []
    server.config.conversation.share_profile = "off"
    found = server._widenings_for("update_config", {"section": "conversation", "values": {"share_profile": "auto"}})
    assert [w.key for w in found] == ["conversation.share_profile"]


def test_choosing_a_cloud_model_says_the_profile_goes_too(tmp_path):
    server = _server(tmp_path, cloud=False)
    found = server._widenings_for("update_config", {"section": "model", "values": {"provider": "cloud"}})
    assert any("your name, your answer style" in w.stops for w in found)


def test_by_voice_off_is_at_once_and_on_goes_to_the_window(tmp_path, monkeypatch):
    from pilot.personal import spoken_settings

    server = _server(tmp_path, cloud=True, share="auto")
    saved: list[dict] = []

    async def update(params, ws):
        saved.append(params)
        server.config.conversation.share_profile = params["values"]["share_profile"]
        return {"status": "ok"}

    async def nowhere(*_a, **_k):
        return None

    monkeypatch.setattr(server, "_handle_update_config", update)
    monkeypatch.setattr(server, "_broadcast_notification", nowhere)
    said = asyncio.run(server._settings_request("Don't send my profile to the cloud."))
    assert said == spoken_settings.PROFILE_OFF and server.config.conversation.share_profile == "off"
    assert saved == [{"section": "conversation", "values": {"share_profile": "off"}}]
    said = asyncio.run(server._settings_request("You can send my profile again."))
    assert said in (spoken_settings.OPEN_TO_APPROVE, spoken_settings.APPROVE_IN_WINDOW)
    assert server.config.conversation.share_profile == "off"  # nothing saved until the window approves it


@pytest.mark.parametrize(
    ("cloud", "share", "ends"),
    [
        (True, "auto", "Your name, style and what bears on a question go to api.anthropic.com."),
        (True, "off", "None of it goes to api.anthropic.com."),
        (False, "auto", "None of it leaves this computer."),
    ],
)
def test_what_do_you_know_about_me_says_whether_it_is_sent(tmp_path, cloud, share, ends):
    server = _server(tmp_path, cloud=cloud, share=share)
    answer = asyncio.run(server._profile_request("What do you know about me?"))
    assert answer.endswith(ends)
