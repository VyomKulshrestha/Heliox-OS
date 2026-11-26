"""Further handlers that returned a failure as an ordinary string; each now fails the action. Fakes only."""

from __future__ import annotations

import pytest

from pilot.actions import (
    Action,
    ActionResult,
    ActionType,
    BrowserParams,
    CodeExecParams,
    FileIntelParams,
    OpenUrlParams,
    PtyExecParams,
    SkillRunParams,
)
from pilot.agents.executor import Executor
from pilot.config import PilotConfig
from pilot.system import platform_detect
from pilot.system.platform_detect import Platform


async def _run(action_type: ActionType, handler: str, parameters: object, **attributes: object) -> ActionResult:
    executor = object.__new__(Executor)
    executor._stress_gate = None
    for name, value in attributes.items():
        setattr(executor, name, value)
    executor._dispatch_table = {action_type: getattr(executor, handler)}
    action = Action(action_type=action_type, target="test", parameters=parameters)
    return await executor._execute_single(action, snapshot_id=None)


# ── open_url ignored the opener's exit code ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.LINUX, Platform.MACOS])
async def test_open_url_fails_when_the_opener_fails(monkeypatch: pytest.MonkeyPatch, platform: Platform) -> None:
    async def failing_opener(args, **_kwargs):
        return 3, "", "no method available for opening 'https://example.com'"

    async def failing_launcher(args, **_kwargs):
        return 3, "no method available for opening 'https://example.com'"

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(platform_detect, "run_command", failing_opener)
    monkeypatch.setattr(platform_detect, "launch_and_settle", failing_launcher)

    result = await _run(ActionType.OPEN_URL, "_exec_open_url", OpenUrlParams(url="https://example.com"))

    assert result.success is False
    assert "The browser did not open https://example.com" in (result.error or "")


@pytest.mark.asyncio
async def test_open_url_on_windows_fails_when_the_shell_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import command_line

    def refuse(_target: str) -> None:
        raise OSError("No application is associated with the specified file")

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(command_line.os, "startfile", refuse, raising=False)

    result = await _run(ActionType.OPEN_URL, "_exec_open_url", OpenUrlParams(url="https://example.com"))

    assert result.success is False
    assert "No application is associated" in (result.error or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, None])  # None: xdg-open still running, waiting for the browser it started
async def test_open_url_succeeds_when_the_opener_does(monkeypatch: pytest.MonkeyPatch, exit_code: int | None) -> None:
    launched: list[list[str]] = []

    async def launcher(args, **_kwargs):
        launched.append(args)
        return exit_code, ""

    monkeypatch.setattr(platform_detect, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(platform_detect, "launch_and_settle", launcher)

    result = await _run(ActionType.OPEN_URL, "_exec_open_url", OpenUrlParams(url="example.com"))

    assert result.success is True
    assert result.output == "Opened https://example.com in default browser"
    assert launched == [["xdg-open", "https://example.com"]]


# ── skill_run ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_run_without_a_skill_registry_fails() -> None:
    result = await _run(ActionType.SKILL_RUN, "_exec_skill_run", SkillRunParams(skill_id="echo"), _skill_registry=None)

    assert result.success is False
    assert "'echo' did not run" in (result.error or "")


@pytest.mark.asyncio
async def test_skill_run_of_an_unknown_or_failing_skill_fails() -> None:
    from pilot.skills.base import SkillContext
    from pilot.skills.loader import SkillRegistry

    class _Broken:
        description = "breaks"

        async def run(self, arguments, ctx):
            raise RuntimeError("disk full")

    registry = SkillRegistry()
    registry._skills = {"broken": _Broken()}

    with pytest.raises(LookupError, match="Unknown skill_id 'missing'"):
        await registry.run("missing", {}, SkillContext(pilot_config=None))
    with pytest.raises(RuntimeError, match="Skill 'broken' failed: disk full"):
        await registry.run("broken", {}, SkillContext(pilot_config=None))


# ── code_execute and code_generate_and_run ──────────────────────────


@pytest.mark.asyncio
async def test_code_execute_in_an_unknown_language_fails() -> None:
    from pilot.system.code_exec import execute_code

    with pytest.raises(ValueError, match="Unsupported language: cobol"):
        await execute_code("DISPLAY 'hi'", "cobol", 5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_answer", "expected"),
    [
        (RuntimeError("model server refused the request"), "Code generation failed: model server refused"),
        ("ERROR: Script timed out after 5s", "Script timed out after 5s"),
    ],
)
async def test_code_generate_and_run_failures_fail_the_action(
    monkeypatch: pytest.MonkeyPatch, backend_answer: object, expected: str
) -> None:
    import pilot.models.ollama as ollama
    from pilot.system import code_exec

    class _Client:
        async def is_available(self) -> bool:
            if isinstance(backend_answer, Exception):
                raise backend_answer
            return True

        async def list_models(self) -> list[str]:
            return ["small"]

        async def generate(self, *_args, **_kwargs) -> str:
            return "print('hi')"

    async def execute(*_args, **_kwargs) -> str:
        return str(backend_answer)

    monkeypatch.setattr(ollama, "OllamaClient", _Client)
    monkeypatch.setattr(code_exec, "execute_code", execute)

    result = await _run(
        ActionType.CODE_GENERATE_AND_RUN,
        "_exec_code_generate",
        CodeExecParams(task_description="print hi", language="python", timeout=5),
        _config=PilotConfig(),
    )

    assert result.success is False
    assert expected in (result.error or "")


# ── pty_exec timeout ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pty_exec_timeout_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import pty_session

    class _Session:
        async def exec(self, command: str, timeout: int) -> str:
            return f"ERROR: command timed out after {timeout}s"

    monkeypatch.setattr(
        pty_session.PtySessionManager, "get_session", classmethod(lambda cls, _id="default": _Session())
    )

    result = await _run(ActionType.PTY_EXEC, "_exec_pty_exec", PtyExecParams(command="sleep 100", timeout=5))

    assert result.success is False
    assert "Command timed out after 5s" in (result.error or "")


# ── browser reads of an element that is not there ───────────────────


class _Page:
    async def query_selector(self, _selector: str):
        return None

    async def evaluate(self, _script: str, *_args):
        return None

    async def screenshot(self, **_kwargs) -> None:
        raise AssertionError("a missing element must not fall back to a page screenshot")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda browser: browser.browser_extract("#price"), "Element not found: #price"),
        (lambda browser: browser.browser_extract_table("#rates"), "Table not found: #rates"),
    ],
)
async def test_browser_extract_of_a_missing_element_fails(monkeypatch: pytest.MonkeyPatch, call, message) -> None:
    from pilot.system import browser

    async def page(*_args, **_kwargs):
        return _Page()

    monkeypatch.setattr(browser, "_get_page", page)

    with pytest.raises(LookupError, match=message):
        await call(browser)


@pytest.mark.asyncio
async def test_browser_extract_action_of_a_missing_element_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Backend:
        async def extract(self, selector: str, attribute: str, multiple: bool) -> str:
            raise LookupError(f"Element not found: {selector}")

    result = await _run(
        ActionType.BROWSER_EXTRACT,
        "_exec_browser_extract",
        BrowserParams(selector="#price"),
        _get_browser_backend=lambda: _Backend(),
    )

    assert result.success is False
    assert "Element not found: #price" in (result.error or "")


# ── brightness_get and window_list ──────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WINDOWS, Platform.MACOS, Platform.LINUX])
async def test_brightness_read_that_nothing_answers_fails(monkeypatch: pytest.MonkeyPatch, platform: Platform) -> None:
    from pilot.system import screen

    async def nothing(*_args, **_kwargs):
        return 1, "", "not supported"

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", platform)
    monkeypatch.setattr(screen, "run_command", nothing)
    monkeypatch.setattr(screen, "run_powershell", nothing)

    with pytest.raises(RuntimeError, match="This display's brightness cannot be read"):
        await screen.brightness_get()


@pytest.mark.asyncio
async def test_brightness_read_still_answers_when_it_can(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import screen

    async def answer(*_args, **_kwargs):
        return 0, "70\n", ""

    monkeypatch.setattr(screen, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(screen, "run_powershell", answer)

    assert await screen.brightness_get() == "Brightness: 70%"


@pytest.mark.asyncio
async def test_macos_window_list_fails_when_applescript_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from pilot.system import macos_windows, window_mgr

    async def refused(*_args, **_kwargs):
        return 1, "", "Not authorized to send Apple events to System Events."

    monkeypatch.setattr(window_mgr, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(macos_windows, "available", lambda: None)
    monkeypatch.setattr(window_mgr, "run_command", refused)

    with pytest.raises(RuntimeError, match="Window list failed: Not authorized"):
        await window_mgr.window_list()


# ── file_parse and file_search_content ──────────────────────────────


@pytest.mark.asyncio
async def test_file_parse_of_a_missing_file_fails(tmp_path) -> None:
    result = await _run(ActionType.FILE_PARSE, "_exec_file_parse", FileIntelParams(path=str(tmp_path / "missing.pdf")))

    assert result.success is False
    assert "File not found" in (result.error or "")


@pytest.mark.asyncio
async def test_file_parse_that_the_reader_cannot_parse_fails(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from pilot.system import file_intel

    data = tmp_path / "data.json"
    data.write_text("{not json", encoding="utf-8")

    async def broken(_path: str) -> str:
        raise ValueError("Expecting property name enclosed in double quotes")

    monkeypatch.setattr(file_intel, "_parse_json", broken)

    with pytest.raises(RuntimeError, match="Heliox could not parse data.json"):
        await file_intel.parse_file(str(data))


@pytest.mark.asyncio
async def test_search_in_a_missing_folder_fails(tmp_path) -> None:
    from pilot.system.file_intel import search_file_contents

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        await search_file_contents(str(tmp_path / "nowhere"), "needle")
