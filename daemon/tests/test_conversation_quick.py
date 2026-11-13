"""The quick road: which single actions may skip the planner call, how they are built per platform, and that the
server runs them through the normal pipeline and falls back to the planner whenever anything would ask."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, VerificationResult
from pilot.config import PilotConfig
from pilot.conversation import quick
from pilot.conversation.handover import HandoverTarget
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.router import normalize, route
from pilot.security.permissions import PermissionChecker


def action(kind: str, target: str = "", **parameters: Any) -> Action:
    from pilot.agents.planner import Planner

    return Action(
        action_type=ActionType(kind), target=target, parameters=Planner._parse_parameters(ActionType(kind), parameters)
    )


ADMITTED = [
    action("open_application", "notepad", name="notepad"),
    action("open_url", "https://example.com", url="https://example.com"),
    action("window_focus", "chrome", title="chrome"),
    action("volume_set", level=30),
    action("volume_mute", mute=True),
    action("keyboard_press", "nexttrack", key="nexttrack"),
    action("battery_info"),
    action("disk_usage"),
    action("memory_usage"),
    action("cpu_usage"),
]

REFUSED = [
    action("open_application", "cmd", name="cmd"),
    action("open_application", "powershell", name="powershell", args=["-c", "rm -r ~"]),
    action("open_application", "notepad", name="notepad", args=["C:/Windows/system.ini"]),
    action("open_application", "explorer", name="explorer", args=["C:/Windows"]),
    action("open_application", "x", name="C:/tools/run.bat"),
    action("open_url", "file:///etc/passwd", url="file:///etc/passwd"),
    action("open_url", "javascript:alert(1)", url="javascript:alert(1)"),
    action("keyboard_press", "enter", key="enter"),
    action("keyboard_press", "", key="playpause", text="hello"),
    action("keyboard_type", "", text="rm -rf /"),
    action("keyboard_hotkey", "", keys=["alt", "f4"]),
    action("volume_set", level=150),
    action("file_write", "C:/notes.txt", path="C:/notes.txt", content="x"),
    action("file_delete", "C:/notes.txt", path="C:/notes.txt"),
    action("file_move", "C:/a", path="C:/a", destination="C:/b"),
    action("shell_command", "", command="dir"),
    action("package_install", "", name="vlc"),
    action("api_send_email"),
    action("schedule_create"),
    action("process_kill", "", name="chrome"),
    action("window_close", "", title="chrome"),
]


@pytest.mark.parametrize("candidate", ADMITTED, ids=lambda a: a.action_type.value)
def test_allow_listed_actions_are_admitted(candidate):
    assert quick.admissible(candidate, platform="win32") == (True, "")
    assert not PermissionChecker(PilotConfig()).check_action(candidate).requires_confirmation


@pytest.mark.parametrize("candidate", REFUSED, ids=lambda a: f"{a.action_type.value}:{a.target}")
def test_everything_else_is_refused(candidate):
    ok, reason = quick.admissible(candidate, platform="win32", home=Path("C:/Users/sam"))
    assert not ok and reason


def test_the_allow_list_never_needs_approval_by_the_permission_tiers():
    for kind in quick.ALLOWED:
        assert not action(kind.value).requires_confirmation, kind


def plan_for(text: str, platform: str = "win32", home: Path = Path("C:/Users/sam")):
    decided = route(text)
    built = quick.from_words(normalize(text), decided.verb, decided.target, platform=platform, home=home)
    return None if built is None else built.plan.actions[0]


@pytest.mark.parametrize(
    ("text", "kind", "params"),
    [
        ("Open Notepad", "open_application", {"name": "notepad"}),
        ("could you launch spotify", "open_application", {"name": "spotify"}),
        ("open youtube.com", "open_url", {"url": "https://youtube.com"}),
        (
            "search the web for flights to Lisbon",
            "open_url",
            {"url": "https://www.google.com/search?q=flights+to+lisbon"},
        ),
        ("switch to chrome", "window_focus", {"title": "chrome"}),
        ("mute", "volume_mute", {"mute": True}),
        ("unmute", "volume_mute", {"mute": False}),
        ("set the volume to 30", "volume_set", {"level": 30}),
        ("next song", "keyboard_press", {"key": "nexttrack"}),
        ("pause the music", "keyboard_press", {"key": "playpause"}),
        ("turn the volume up", "volume_set", {"step": 10}),  # the system's volume call, not five key presses
        ("quieter", "volume_set", {"step": -10}),
        ("How much disk space do I have left?", "disk_usage", {}),
        ("what's the battery level", "battery_info", {}),
    ],
)
def test_plain_requests_are_built_in_code(text, kind, params):
    built = plan_for(text)
    assert built is not None and built.action_type.value == kind
    assert built.parameters.model_dump(exclude_defaults=True) == params


@pytest.mark.parametrize(
    "text",
    [
        "open the report from yesterday",  # a file the planner has to find
        "open cmd",  # runs commands
        "start the backup",  # "start" is not "open"
        "take a screenshot",
        "delete the old backups",
        "move the screenshots into a new folder",
        "send an email to Sam",
        "search my documents for the invoice",  # a file search, not the web
        "That's enough for the backup folder, now zip it",
        "install the latest Python",
        "open it",
        "Is Spotify running?",  # the planner says it in words
    ],
)
def test_anything_else_goes_to_the_planner(text):
    assert plan_for(text) is None


def test_opening_a_folder_per_platform():
    windows = plan_for("open my downloads folder", "win32", Path("C:/Users/sam"))
    assert (windows.parameters.name, windows.parameters.args) == ("explorer", [str(Path("C:/Users/sam") / "Downloads")])
    linux = plan_for("open my downloads folder", "linux", Path("/home/sam"))
    assert (linux.parameters.name, linux.parameters.args) == ("xdg-open", [str(Path("/home/sam") / "Downloads")])
    assert plan_for("open my downloads folder", "darwin", Path("/Users/sam")) is None  # the planner opens it there


def test_a_model_proposal_is_used_only_on_the_allow_list_and_never_with_arguments():
    good = quick.from_model(
        {"action_type": "open_application", "target": "notepad", "parameters": {"name": "notepad"}}, words="open it"
    )
    assert good is not None and good.source == "model"
    for bad in (
        {"action_type": "shell_command", "parameters": {"command": "dir"}},
        {
            "action_type": "open_application",
            "parameters": {"name": "explorer", "args": [str(Path.home() / "Downloads")]},
        },
        {"action_type": "file_delete", "parameters": {"path": "C:/x"}},
        {"action_type": "keyboard_press", "parameters": {"key": "enter"}},
        {"action_type": "nonsense"},
        "open_application",
    ):
        assert quick.from_model(bad, words="x") is None


def test_state_readings_are_said_in_words():
    assert quick.speakable_state("=== Battery ===\n  Charge: 84%\n  Status: Discharging") == (
        "The battery is at 84 percent, discharging."
    )
    disk = "=== Disk ===\n  C:\\ (C:\\) - 952.6 GB total, 754.8 GB used (79.2%), fstype=NTFS"
    assert quick.speakable_state(disk) == "You have 198 gigabytes free on drive C."
    assert quick.speakable_state("Opened Notepad") == ""


# ── the layer ──


class Sink:
    def __init__(self):
        self.said = []

    async def __call__(self, text, channel, key):
        self.said.append(text)
        return "spoken"


@pytest.mark.asyncio
async def test_the_layer_picks_the_quick_road_for_a_plain_request():
    layer = ConversationLayer(sink=Sink(), model=lambda: None)
    result = await layer.take("Open Notepad")
    assert result.target is HandoverTarget.QUICK
    plan = result.execute_params()["_quick_plan"]
    assert isinstance(plan, ActionPlan) and plan.actions[0].action_type is ActionType.OPEN_APPLICATION
    complex_result = await layer.take("Delete the old backups in my downloads")
    assert complex_result.target is HandoverTarget.FULL_PLANNER
    assert "_quick_plan" not in complex_result.execute_params()


@pytest.mark.asyncio
async def test_the_layer_takes_a_model_proposal_only_from_the_allow_list():
    class Model:
        local, name, provider, streams, calls = False, "fake", "fake", True, []

        def __init__(self, line):
            self.line = line

        async def stream(self, system, messages):
            yield self.line

    ok = 'ACT {"verb":"open","object":"it","request":"Open Notepad","say":"Opening Notepad.","action":{"action_type":"open_application","parameters":{"name":"notepad"}}}'
    bad = 'ACT {"verb":"delete","object":"it","request":"Delete notes.txt","say":"Deleting it.","action":{"action_type":"file_delete","parameters":{"path":"C:/notes.txt"}}}'
    layer = ConversationLayer(sink=Sink(), model=lambda: Model(ok))
    layer.memory.add_user("I was editing notes in Notepad")
    assert (await layer.take("open it")).target is HandoverTarget.QUICK
    layer = ConversationLayer(sink=Sink(), model=lambda: Model(bad))
    layer.memory.add_user("there is a file notes.txt")
    refused = await layer.take("delete it")
    assert refused.target is HandoverTarget.FULL_PLANNER and refused.planner_input == "delete it"
    assert "Delete notes.txt" in refused.context.for_planner()


@pytest.mark.asyncio
async def test_routing_and_building_a_quick_plan_takes_milliseconds():
    texts = ("Open Notepad", "mute", "next song", "how much disk space do I have", "switch to chrome")
    best = float("inf")
    # The best of four passes: the first pays for imports, and a busy CI runner can stall any single pass (the mean
    # of one pass was 81 ms on a macOS runner on 12 November with nothing in the code slower).
    for _ in range(4):
        layer = ConversationLayer(sink=Sink(), model=lambda: None)
        started = time.perf_counter()
        for text in texts:
            result = await layer.take(text)
            assert result.target is HandoverTarget.QUICK
        best = min(best, (time.perf_counter() - started) / len(texts))
    assert best < 0.05


# ── the server: the same pipeline, never an approval, the planner whenever anything would ask ──


def _server(seen: dict[str, Any], *, planner_plan: ActionPlan | None = None):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.security.dry_run = False
    server = PilotServer(config)
    server._planner = AsyncMock()
    server._planner.plan.return_value = planner_plan or ActionPlan(
        actions=[action("disk_usage")], explanation="checked", raw_input="x"
    )
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = PermissionChecker(config)  # the real tiers
    server._memory = AsyncMock()
    server._broadcast_notification = AsyncMock()
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    gnomon = SimpleNamespace(review_plan=AsyncMock(return_value=None), status=lambda: {}, warm=lambda: None)
    server._gnomon_reviewer = gnomon
    seen["gnomon"] = gnomon

    async def confirm(plan_id, approval_plan, ws, **kwargs):
        seen.setdefault("confirmations", []).append([a.action_type.value for a in approval_plan.actions])
        return False, set(), set()

    server._wait_for_confirmation = confirm

    async def execute(run_plan, **kwargs):
        seen.setdefault("ran", []).append([a.action_type.value for a in run_plan.actions])
        return [ActionResult(action=a, success=True, output="Launched Notepad.") for a in run_plan.actions]

    server._executor = SimpleNamespace(execute=execute)
    server._orchestrator = None
    return server


def _ws():
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


def quick_plan(kind="open_application", **parameters):
    return ActionPlan(actions=[action(kind, parameters.get("name", ""), **parameters)], explanation="", raw_input="x")


@pytest.mark.asyncio
async def test_a_quick_plan_runs_without_the_planner_call_or_any_approval():
    seen: dict[str, Any] = {}
    server = _server(seen)
    response = await server._handle_execute(
        {"input": "Open Notepad", "source": "voice", "_quick_plan": quick_plan(name="notepad")}, _ws()
    )
    assert response["status"] == "success"
    server._planner.plan.assert_not_awaited()
    assert seen["ran"] == [["open_application"]] and "confirmations" not in seen
    seen["gnomon"].review_plan.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_world_model_warning_sends_the_request_to_the_planner_instead(monkeypatch):
    import pilot.agents.destructive_critic as critic

    real = critic.assess_plan_risk

    def worried(plan, config, source=""):
        import dataclasses

        assessment = real(plan, config, source=source)
        return dataclasses.replace(assessment, world_model_score=0.99)  # the world model asks for approval

    monkeypatch.setattr(critic, "assess_plan_risk", worried)
    seen: dict[str, Any] = {}
    server = _server(seen)
    await server._handle_execute({"input": "Open Notepad", "_quick_plan": quick_plan(name="notepad")}, _ws())
    server._planner.plan.assert_awaited()  # fell back, never forward
    assert ["open_application"] not in seen.get("ran", [])  # the quick plan never ran
    assert all(c != ["open_application"] for c in seen.get("confirmations", []))  # and never asked for approval


@pytest.mark.asyncio
async def test_a_plan_the_tiers_would_ask_about_is_never_run_quickly():
    seen: dict[str, Any] = {}
    server = _server(seen)
    smuggled = ActionPlan(actions=[action("shell_command", "", command="dir")], explanation="", raw_input="x")
    await server._handle_execute({"input": "list the folder", "_quick_plan": smuggled}, _ws())
    server._planner.plan.assert_awaited()
    assert ["shell_command"] not in seen.get("ran", [])


@pytest.mark.asyncio
async def test_a_client_cannot_send_a_quick_plan():
    seen: dict[str, Any] = {}
    server = _server(seen)
    as_json = {"actions": [{"action_type": "open_application", "parameters": {"name": "notepad"}}]}
    await server._handle_execute({"input": "Open Notepad", "_quick_plan": as_json}, _ws())
    server._planner.plan.assert_awaited()  # a dict is not a plan; the planner plans as always


def test_the_quick_road_never_imports_a_platform_module_at_build_time():
    # Building plans is pure: it runs the same on ubuntu, macOS and Windows CI.
    for platform in ("win32", "darwin", "linux"):
        assert plan_for("Open Notepad", platform) is not None
        assert plan_for("mute", platform) is not None


def test_a_dispatch_hands_the_quick_plan_to_the_pipeline():
    from pilot.server import PilotServer

    async def run():
        server = PilotServer(PilotConfig())
        server._voice_listener = None
        server._broadcast_notification = AsyncMock()
        server._handle_execute = AsyncMock(return_value={"status": "success", "message": "Launched Notepad."})
        server._speak_voice_response = AsyncMock(return_value=False)
        server._model_router = object()
        server._conversation = ConversationLayer(sink=Sink(), model=lambda: None)
        await server._voice_command_dispatch("Open Notepad")
        params = server._handle_execute.await_args.args[0]
        assert isinstance(params["_quick_plan"], ActionPlan)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("text", "ack", "done"),
    [
        ("Open Notepad", "Opening Notepad.", "Notepad is open."),
        ("open youtube.com", "Opening youtube.com.", "It's open in your browser."),
        (
            "search the web for flights to Lisbon",
            "Searching the web for flights to Lisbon.",
            "It's open in your browser.",
        ),
        ("switch to chrome", "Switching to Chrome.", "Done."),
        ("unmute", "Unmuting.", "Done."),
        ("next song", "Next track.", "Done."),
    ],
)
def test_what_the_person_hears_before_and_after(text, ack, done):
    decided = route(text)
    built = quick.from_words(normalize(text), decided.verb, decided.target, platform="win32")
    assert quick.quick_acknowledgement(built, text) == ack
    assert quick.quick_result_sentence(built, "Launched notepad (resolved via App Paths).", "success") == done


def test_a_failure_says_what_failed():
    built = quick.from_words("open notepad", "open", "notepad", platform="win32")
    sentence = quick.quick_result_sentence(built, "Error: Application 'notepad' was not found.", "error")
    assert sentence == "That didn't work: Application 'notepad' was not found."


# ── typed chat through the same layer ──


class ChatModel:
    local, name, provider, streams = False, "fake", "fake", True

    def __init__(self, pieces):
        self.pieces = pieces
        self.calls = []

    async def stream(self, system, messages):
        self.calls.append(system)
        for piece in self.pieces:
            yield piece


@pytest.mark.asyncio
async def test_a_typed_question_is_answered_in_the_chat_with_its_markdown():
    seen: dict[str, Any] = {}
    server = _server(seen)
    server._model_router = object()
    model = ChatModel(
        [
            "Two ways:\n",
            "- **merge** keeps history\n",
            "- `rebase` rewrites it.\n",
            "I've deleted the old branch for you.",
        ]
    )
    server._conversation = ConversationLayer(sink=Sink(), model=lambda: model)
    ws = _ws()
    response = await server._handle_execute({"input": "What's the difference between merge and rebase?"}, ws)
    server._planner.plan.assert_not_awaited()
    assert response["status"] == "success" and response["conversational"]
    assert "- **merge** keeps history\n- `rebase` rewrites it." in response["message"]
    assert "deleted the old branch" not in response["message"]  # never claims an action it cannot take
    tokens = [c.args[0] for c in ws.send.await_args_list if '"token_stream"' in str(c.args[0])]
    assert len(tokens) >= 2  # streamed into the chat view as it arrived
    assert "chat window" in model.calls[0]  # the typed prompt, not the spoken one


@pytest.mark.asyncio
async def test_a_typed_simple_request_takes_the_quick_road():
    seen: dict[str, Any] = {}
    server = _server(seen)
    server._model_router = object()
    server._conversation = ConversationLayer(sink=Sink(), model=lambda: None)
    response = await server._handle_execute({"input": "Open Notepad"}, _ws())
    server._planner.plan.assert_not_awaited()
    assert response["status"] == "success" and seen["ran"] == [["open_application"]]


@pytest.mark.asyncio
async def test_a_typed_complex_request_goes_to_the_planner_and_attachments_skip_the_layer():
    seen: dict[str, Any] = {}
    server = _server(seen)
    server._model_router = object()
    server._conversation = ConversationLayer(sink=Sink(), model=lambda: ChatModel(["should not be asked"]))
    await server._handle_execute({"input": "Delete the old backups in my downloads"}, _ws())
    server._planner.plan.assert_awaited()
    server2 = _server({})
    server2._model_router = object()
    server2._conversation = ConversationLayer(sink=Sink(), model=lambda: ChatModel(["no"]))
    attachment = [{"name": "notes.txt", "content": "hello"}]
    await server2._handle_execute({"input": "What is in this file?", "attachments": attachment}, _ws())
    server2._planner.plan.assert_awaited()  # attached files are the planner's, as before


def test_the_typed_sources_are_pinned():
    from pilot.conversation.adapters import TYPED_SOURCES

    assert frozenset({"interactive", "text", "chat"}) == TYPED_SOURCES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "voice",
        "mcp",
        "autonomous_job",
        "phone",
        "schedule",
        "scheduled",
        "replay",
        "workflow",
        "local",
        "resume",
        "swarm",
    ],
)
async def test_no_automation_schedule_phone_or_mcp_source_reaches_the_typed_layer(source, monkeypatch):
    import pilot.conversation.adapters as adapters

    called = []

    async def spy(*args, **kwargs):
        called.append(args)
        return None

    monkeypatch.setattr(adapters, "typed_turn", spy)
    seen: dict[str, Any] = {}
    server = _server(seen)
    server._model_router = object()
    server._conversation = ConversationLayer(sink=Sink(), model=lambda: None)
    await server._handle_execute({"input": "How are you?", "source": source}, _ws())
    assert called == []


@pytest.mark.asyncio
async def test_a_typed_request_that_needs_approval_still_opens_the_approval_dialog():
    seen: dict[str, Any] = {}
    plan = ActionPlan(actions=[action("shell_command", "", command="dir")], explanation="list", raw_input="x")
    server = _server(seen, planner_plan=plan)
    server._model_router = object()
    server._conversation = ConversationLayer(sink=Sink(), model=lambda: None)
    await server._handle_execute({"input": "Run dir in my projects folder"}, _ws())
    server._planner.plan.assert_awaited()
    assert seen["confirmations"] == [["shell_command"]]  # the dialog, exactly as before; declined here


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("https://duckduckgo.com/?q={query}", "https://duckduckgo.com/?q=flights+to+lisbon"),
        ("https://www.bing.com/search?q={query}", "https://www.bing.com/search?q=flights+to+lisbon"),
        ("javascript:alert({query})", "https://www.google.com/search?q=flights+to+lisbon"),
        ("https://example.com/search", "https://www.google.com/search?q=flights+to+lisbon"),  # no placeholder
        ("", "https://www.google.com/search?q=flights+to+lisbon"),
    ],
)
def test_the_search_engine_is_the_persons_choice(setting, expected):
    built = quick.from_words(
        "search the web for flights to lisbon",
        "search",
        "the web for flights to lisbon",
        platform="win32",
        search_url=quick.valid_search_url(setting),
    )
    assert built.plan.actions[0].parameters.url == expected


def test_the_search_url_setting_is_validated_when_loaded():
    from pilot.config import PilotConfig, _merge_config

    good = _merge_config(PilotConfig(), {"conversation": {"search_url": "https://duckduckgo.com/?q={query}"}})
    assert good.conversation.search_url == "https://duckduckgo.com/?q={query}"
    for bad in ("file:///x?q={query}", "https://duckduckgo.com/", "not a url {query}"):
        kept = _merge_config(PilotConfig(), {"conversation": {"search_url": bad}})
        assert kept.conversation.search_url == "https://www.google.com/search?q={query}"


@pytest.mark.asyncio
async def test_the_layer_uses_the_configured_search_engine():
    layer = ConversationLayer(sink=Sink(), model=lambda: None, search_url=lambda: "https://duckduckgo.com/?q={query}")
    result = await layer.take("search the web for flights to Lisbon")
    assert result.quick.plan.actions[0].parameters.url.startswith("https://duckduckgo.com/?q=flights")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, "interactive", "text", "chat"])
async def test_the_typed_sources_do_reach_the_layer(source, monkeypatch):
    import pilot.conversation.adapters as adapters

    called = []

    async def spy(*args, **kwargs):
        called.append(args)
        return None

    monkeypatch.setattr(adapters, "typed_turn", spy)
    server = _server({})
    server._model_router = object()
    params = {"input": "How are you?", **({"source": source} if source else {})}
    await server._handle_execute(params, _ws())
    assert len(called) == 1
