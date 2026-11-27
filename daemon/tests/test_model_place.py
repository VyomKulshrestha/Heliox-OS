"""A model counts as on this computer only when its requests cannot leave it (pilot.models.model_place): Ollama or the
local backend at a loopback address, ``localhost``, a Unix socket, or a host name that resolved to loopback only.
Ollama anywhere else gets exactly what a cloud model gets, under the same settings.

Every test here uses a fake resolver: no real lookup and no real network. The request bodies are captured by an
``httpx.MockTransport`` on the conversation model the server itself builds from its config."""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import httpx
import pytest

from pilot.config import PilotConfig
from pilot.models import model_place
from pilot.models.model_place import Places, literal_place, model_elsewhere, model_on_this_computer

LOOKUPS = {
    "my-ollama": ["127.0.0.1", "::1"],
    "models.example.com": ["203.0.113.7"],
    "host.docker.internal": ["192.168.65.254"],
    "split.example": ["127.0.0.1", "10.0.0.9"],
}


class FakeResolver:
    def __init__(self, answers: dict[str, list[str]] | None = None) -> None:
        self.answers = dict(LOOKUPS if answers is None else answers)
        self.asked: list[str] = []

    def __call__(self, host: str) -> list[str]:
        self.asked.append(host)
        if host not in self.answers:
            raise OSError("no such host")
        return self.answers[host]


@pytest.fixture
def resolver(monkeypatch) -> FakeResolver:
    fake = FakeResolver()
    monkeypatch.setattr(model_place, "PLACES", Places(resolve=fake, run=lambda work: work()))
    return fake


# ── where an address points ──


@pytest.mark.parametrize(("url", "here"), [
    ("", True),  # the default address
    ("http://127.0.0.1:11434", True),
    ("http://127.8.9.10:11434", True),
    ("http://localhost:11434", True),
    ("http://LOCALHOST.:11434", True),
    ("http://ollama.localhost:11434", True),
    ("http://[::1]:11434", True),
    ("http://[::ffff:127.0.0.1]:11434", True),
    ("http://0.0.0.0:11434", False),  # not a loopback address: no standard says where it goes
    ("http://[::]:11434", False),
    ("unix:///run/ollama.sock", True),
    ("http+unix://%2Frun%2Follama.sock", True),
    ("http://192.168.1.20:11434", False),
    ("http://10.0.0.2:11434", False),
    ("http://172.17.0.1:11434", False),  # a container bridge
    ("http://172.20.0.1:11434", False),  # a WSL bridge
    ("http://[fe80::1%25eth0]:11434", False),
    ("http://[::ffff:192.168.1.20]:11434", False),
    ("http://100.101.102.103:11434", False),  # a tunnel or overlay network address
    ("http://user:secret@127.0.0.1:11434", True),
    ("http://user:secret@192.168.1.20:11434", False),
    ("not a url at all://", False),
])  # fmt: skip
def test_addresses_judged_without_a_lookup(url, here):
    assert literal_place(url) is here


@pytest.mark.parametrize("url", ["http://my-ollama:11434", "https://models.example.com", "http://host.docker.internal"])
def test_a_host_name_needs_its_lookup(url):
    assert literal_place(url) is None


@pytest.mark.parametrize(("url", "here"), [
    ("http://my-ollama:11434", True),  # resolves to loopback only
    ("https://models.example.com/ollama", False),  # a public host
    ("http://host.docker.internal:11434", False),  # Docker's way back to the host machine
    ("http://split.example:11434", False),  # one address of two is elsewhere
    ("http://gone.example:11434", False),  # the lookup failed
])  # fmt: skip
def test_a_host_name_is_judged_by_what_it_resolves_to(resolver, url, here):
    assert model_place.endpoint_on_this_computer(url) is here
    assert model_place.endpoint_on_this_computer(url) is here
    assert len(resolver.asked) == 1  # looked up once, then remembered


def test_a_name_counts_as_another_computer_until_its_lookup_is_in_and_the_caller_never_waits():
    started: list = []
    places = Places(resolve=FakeResolver(), run=started.append)
    assert places.on_this_computer("http://my-ollama:11434") is False
    assert places.on_this_computer("http://my-ollama:11434") is False
    assert len(started) == 1  # one lookup, started once, run elsewhere
    started[0]()
    assert places.on_this_computer("http://my-ollama:11434") is True


def test_saving_the_model_settings_looks_names_up_again():
    fake = FakeResolver()
    places = Places(resolve=fake, run=lambda work: work())
    assert places.on_this_computer("http://my-ollama:11434") is True
    fake.answers["my-ollama"] = ["192.168.1.20"]
    assert places.on_this_computer("http://my-ollama:11434") is True  # remembered
    places.forget()
    assert places.on_this_computer("http://my-ollama:11434") is False
    assert fake.asked == ["my-ollama", "my-ollama"]


def test_a_failed_lookup_logs_no_address(caplog):
    places = Places(resolve=FakeResolver({}), run=lambda work: work())
    with caplog.at_level("DEBUG", logger="pilot.models.model_place"):
        assert places.on_this_computer("http://user:secret@gone.example:11434") is False
    assert "secret" not in caplog.text and "gone.example" not in caplog.text


@pytest.mark.parametrize(("provider", "url", "here"), [
    ("ollama", "http://127.0.0.1:11434", True),
    ("local", "http://127.0.0.1:11434", True),
    ("ollama", "http://192.168.1.20:11434", False),
    ("local", "http://192.168.1.20:11434", False),  # the local backend asks Ollama first
    ("cloud", "http://127.0.0.1:11434", False),
    ("subscription", "http://127.0.0.1:11434", False),
])  # fmt: skip
def test_the_configured_model(resolver, provider, url, here):
    model = SimpleNamespace(provider=provider, ollama_base_url=url)
    assert model_on_this_computer(model) is here
    assert model_on_this_computer({"provider": provider, "ollama_base_url": url}) is here


def test_settings_is_told_only_the_host_of_a_model_elsewhere(resolver):
    elsewhere = SimpleNamespace(provider="ollama", ollama_base_url="http://user:secret@192.168.1.20:11434/x")
    assert model_elsewhere(elsewhere) == "192.168.1.20"
    assert model_elsewhere(SimpleNamespace(provider="ollama", ollama_base_url="http://127.0.0.1:11434")) == ""
    assert model_elsewhere(SimpleNamespace(provider="cloud", ollama_base_url="http://192.168.1.20:11434")) == ""


def test_the_settings_guard_names_ollama_elsewhere_like_a_cloud_provider(resolver):
    from pilot.security.guard_changes import model_destination, section_widenings

    assert model_destination({"provider": "ollama", "ollama_base_url": "http://[::1]:11434"}) == ""
    assert model_destination({"provider": "local", "ollama_base_url": "http://192.168.1.20:11434"}) == "192.168.1.20"
    assert model_destination({"provider": "ollama", "ollama_base_url": "http://my-ollama:11434"}) == ""
    found = section_widenings(
        "model",
        {"provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434"},
        {"ollama_base_url": "http://192.168.1.20:11434"},
    )
    assert [w.key for w in found] == ["model.provider"] and "192.168.1.20" in found[0].stops
    # With Ollama elsewhere, turning the profile back on is a widening, as it is for a cloud model.
    shares = section_widenings(
        "conversation", {"share_profile": "off"}, {"share_profile": "auto"}, destination="192.168.1.20"
    )
    assert [w.key for w in shares] == ["conversation.share_profile"]


# ── the request body, for each endpoint and each sharing rule ──

ENDPOINTS = [  # (address, on this computer)
    ("http://127.0.0.1:11434", True),
    ("http://192.168.1.20:11434", False),
    ("https://models.example.com", False),
    ("http://my-ollama:11434", True),
    ("http://gone.example:11434", False),
    ("http://[::1]:11434", True),
]
NOW = datetime(2026, 11, 24, 18, 30)
DECISION = "I'm going to watch a series tonight and do the report in the morning"


def _server(tmp_path, url: str, share: str = "auto"):
    from pilot.conversation import layer_for
    from pilot.conversation.fast_model import HttpFastModel
    from pilot.server import PilotServer
    from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
    from pilot.workflows.reminders import ReminderStore
    from pilot.workflows.todos import TodoStore
    from pilot.workflows.user_profile import UserProfile

    config = PilotConfig()
    config.model.provider = "ollama"
    config.model.ollama_base_url = url
    config.conversation.share_profile = share
    server = PilotServer(config)
    sent: list[str] = []

    async def speak(text, channel, dedupe_key="", recorder=None):
        return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

    async def nowhere(*_args, **_kwargs):
        return None

    async def execute(params, ws):
        raise AssertionError(f"planned: {params.get('input')}")

    server._speech_coordinator.speak = speak
    server._broadcast_notification = nowhere
    server._handle_execute = execute
    server._model_router = object()
    server._wall_clock = lambda: NOW
    server._todos = TodoStore(tmp_path / "todos.json")
    server._reminders = ReminderStore(tmp_path / "reminders.json", clock=lambda: NOW.timestamp())
    server._user_profile = UserProfile(tmp_path / "profile.json", remember=nowhere, forget=nowhere)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(url.rstrip("/"))  # the address the person set, and no other
        sent.append(request.content.decode("utf-8"))
        body = json.dumps({"message": {"content": "Okay."}, "done": True}) + "\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})

    model = HttpFastModel(
        provider="ollama",
        name="m",
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url=url,
    )
    layer_for(server)._model = lambda: model
    return server, sent, model


async def _know_sam(server) -> None:
    await server._user_profile.set_name("Sam")
    await server._user_profile.add("about", "I'm training for a marathon")


async def _say(server, words: str) -> None:
    from pilot.system.voice import voice_turn_addressed

    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch(words)


def _context(body: str) -> str:
    return body.split("[The person said]")[0]


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
def test_the_model_object_says_where_it_runs(resolver, tmp_path, url, here):
    _server_, _sent, model = _server(tmp_path, url)
    assert model.local is here
    assert model.ollama  # spoken to as Ollama wherever it runs


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
async def test_to_dos_go_in_their_own_words_only_to_this_computer(resolver, tmp_path, url, here):
    server, sent, _model = _server(tmp_path, url)
    server._todos.add("Finish the quarterly report by 9 am tomorrow (about four hours of work left)")
    await _say(server, DECISION)
    context = _context(sent[-1])
    if here:
        assert "four hours of work left" in context
    else:
        assert 'a to-do about \\"report\\", due tomorrow at 9:00 in the morning' in context
        assert "quarterly" not in context and "four hours" not in context


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
async def test_share_profile_off_keeps_the_person_out_of_every_model_but_one_here(resolver, tmp_path, url, here):
    server, sent, _model = _server(tmp_path, url, share="off")
    await _know_sam(server)
    await _say(server, "what's the capital of Australia?")
    assert ("Call them Sam" in _context(sent[-1])) is here


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
async def test_the_whole_profile_goes_only_to_this_computer(resolver, tmp_path, url, here):
    server, sent, _model = _server(tmp_path, url, share="auto")
    await _know_sam(server)
    await _say(server, "what's the capital of Australia?")
    context = _context(sent[-1])
    assert "Call them Sam" in context  # "auto": the name goes to every model
    assert ("marathon" in context) is here  # the rest only where the words bear on it, unless it stays here


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
def test_the_text_being_typed_goes_only_to_this_computer_under_auto(resolver, tmp_path, url, here):
    from pilot.conversation.adapters import screen_for_model
    from pilot.conversation.planner_context import screen_text_ok_for_planner

    server, _sent, model = _server(tmp_path, url)
    focus = SimpleNamespace(
        app="Notes", title="Plans", source="focused_field", editable=True, text="Dear Priya, the offsite moves"
    )
    for setting, goes in (("auto", here), ("on", True), ("off", False)):
        screen = screen_for_model(focus, setting=setting, model_local=model.local, excluded_apps=[], label=str)
        assert bool(screen.excerpt) is goes
        server.config.conversation.share_screen_text = setting
        assert screen_text_ok_for_planner(server.config) is goes


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
def test_live_shares_what_it_read_only_with_this_computer_under_auto(resolver, tmp_path, url, here):
    from pilot.live.conversation_link import shareable
    from pilot.personal.wiring import model_is_local

    server, _sent, _model = _server(tmp_path, url)
    assert model_is_local(server) is here
    entry = {"kind": "typing", "text": "the offsite moves to Thursday"}
    assert shareable(entry, setting="auto", model_local=model_is_local(server)) is here


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
def test_what_heliox_says_about_the_profile(resolver, tmp_path, url, here):
    server, _sent, _model = _server(tmp_path, url)
    sentence = server._profile_sent_sentence()
    if here:
        assert sentence == "None of it leaves this computer."
    else:
        assert "leaves this computer" not in sentence and "go to" in sentence


@pytest.mark.parametrize(("url", "here"), ENDPOINTS)
async def test_live_asks_the_model_about_personal_facts_only_on_this_computer(resolver, tmp_path, url, here):
    from pilot.live.grounded import Fact, GroundedChecker

    asked: list[str] = []

    async def complete(system: str, prompt: str) -> str:
        asked.append(prompt)
        return "{}"

    server, _sent, _model = _server(tmp_path, url)
    server._live_events = []
    checker = server._build_live_grounded()
    checker._complete = complete
    flight = Fact(id="event:1", kind="event", text="Flight to Berlin", title="Flight to Berlin", day=date(2026, 11, 26))
    checker._facts = lambda: [flight]
    checker._today = lambda: date(2026, 11, 24)
    await checker.check("The airport trip is on Friday morning")
    assert isinstance(checker, GroundedChecker)
    assert bool(asked) is here
    if not here:
        assert checker.last_verdict == "left to code: the model is on another computer" and checker.model_calls == 0


# ── the lookup starts with the daemon ──


def test_the_daemon_start_looks_the_model_host_up_in_the_background_without_waiting():
    started: list = []
    fake = FakeResolver()
    places = Places(resolve=fake, run=started.append)
    model = {"provider": "ollama", "ollama_base_url": "http://my-ollama:11434"}

    assert model_place.look_up_ahead(model, places) is True
    assert fake.asked == []  # nothing was looked up on the caller's thread
    assert model_on_this_computer(model, places) is False  # until the answer is in

    started[0]()  # the background lookup finishes

    assert fake.asked == ["my-ollama"]
    assert model_on_this_computer(model, places) is True
    assert len(started) == 1  # asking again starts no second lookup


@pytest.mark.parametrize("model", [
    {"provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434"},  # needs no lookup
    {"provider": "ollama", "ollama_base_url": ""},
    {"provider": "cloud", "ollama_base_url": "http://my-ollama:11434"},  # not a local model
])  # fmt: skip
def test_nothing_is_looked_up_ahead_when_nothing_needs_it(model):
    started: list = []
    assert model_place.look_up_ahead(model, Places(resolve=FakeResolver(), run=started.append)) is False
    assert started == []


def test_a_failed_lookup_at_start_stays_another_computer():
    places = Places(resolve=FakeResolver({}), run=lambda work: work())
    model = {"provider": "local", "ollama_base_url": "http://ollama:11434"}

    model_place.look_up_ahead(model, places)

    assert model_on_this_computer(model, places) is False


def test_the_daemon_starts_the_lookup_when_it_initializes():
    import inspect

    from pilot.server import PilotServer

    assert "look_up_ahead(self.config.model)" in inspect.getsource(PilotServer.initialize)
