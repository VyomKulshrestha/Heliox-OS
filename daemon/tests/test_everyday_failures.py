"""Steps that failed a run for no good reason, or failed in a way a repair round would only repeat. Fakes only: no
browser opens, no brightness is read, no address is fetched, no process is started."""

from __future__ import annotations

import subprocess
import threading

import pytest

from pilot.actions import Action, ActionResult, ActionType, ApiRequestParams, EmptyParams, OpenUrlParams
from pilot.system import platform_detect
from tests.test_failed_actions_are_failures import _run, http  # noqa: F401 - http is a fixture used below

# ── open_url: a launcher that stays is a launch that started ─────────


class _Launcher:
    """Stands in for subprocess.Popen: ``waits`` keeps it running past the settle time, as xdg-open does on a window
    manager without a desktop session until the browser it started exits."""

    made: list[_Launcher] = []

    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.returncode: int | None = None
        self.drained = threading.Event()
        self.calls = 0
        _Launcher.made.append(self)

    def communicate(self, timeout: float | None = None):
        self.calls += 1
        if self.waits and timeout is not None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = self.code
        self.drained.set()
        return None, self.err


def _launcher(code: int, err: bytes = b"", waits: bool = False) -> type[_Launcher]:
    return type("Launcher", (_Launcher,), {"code": code, "err": err, "waits": waits})


@pytest.mark.asyncio
async def test_a_launcher_still_running_after_the_settle_time_has_launched(monkeypatch: pytest.MonkeyPatch) -> None:
    _Launcher.made = []
    monkeypatch.setattr(subprocess, "Popen", _launcher(0, waits=True))

    assert await platform_detect.launch_and_settle(["xdg-open", "https://example.com"], settle=3.0) == (None, "")

    [launched] = _Launcher.made
    assert launched.args == ["xdg-open", "https://example.com"]
    assert launched.kwargs["stdin"] is subprocess.DEVNULL
    launched.drained.wait(timeout=10)  # the reaper thread collects its exit without anyone waiting on the reply
    assert launched.calls == 2


@pytest.mark.asyncio
async def test_a_launcher_that_fails_at_once_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", _launcher(3, b"no method available for opening"))

    assert await platform_detect.launch_and_settle(["xdg-open", "x"], settle=3.0) == (
        3,
        "no method available for opening",
    )


@pytest.mark.asyncio
async def test_a_missing_launcher_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("xdg-open")

    monkeypatch.setattr(subprocess, "Popen", missing)

    code, err = await platform_detect.launch_and_settle(["xdg-open", "x"], settle=3.0)
    assert code == 127 and "not found" in err


def _failed(action_type: ActionType, error: str) -> ActionResult:
    parameters = OpenUrlParams(url="https://example.com") if action_type is ActionType.OPEN_URL else EmptyParams()
    action = Action(action_type=action_type, target="x", parameters=parameters)
    return ActionResult(action=action, success=False, error=error)


@pytest.mark.parametrize(
    ("action_type", "error"),
    [
        (ActionType.OPEN_URL, "The browser did not open https://example.com: exit code 3"),
        (ActionType.BRIGHTNESS_GET, "This display's brightness cannot be read (it does not report it to Windows)"),
        (ActionType.BRIGHTNESS_GET, "This display's brightness cannot be read (the brightness tool is not installed)"),
    ],
)
def test_failures_no_other_plan_can_repair_end_the_run(action_type: ActionType, error: str) -> None:
    from pilot.server import _is_terminal_execution_failure

    # No repair round: it would open a second tab, or ask the same display again.
    assert _is_terminal_execution_failure([_failed(action_type, error)]) is True


# ── api_scrape and HTTP 404: the message says what was found ─────────


@pytest.mark.asyncio
async def test_a_page_that_is_not_there_says_asking_again_gets_the_same_answer(http: dict) -> None:  # noqa: F811
    http["status"] = 404
    http["text"] = "<html>Not here</html>"

    result = await _run(ActionType.API_SCRAPE, "_exec_api_scrape", ApiRequestParams(url="https://8.8.8.8/item/42"))

    assert result.success is False
    assert "HTTP 404" in (result.error or "")
    assert "the server has nothing at this address; the same request will get the same answer" in (result.error or "")


@pytest.mark.asyncio
async def test_a_selector_that_matches_nothing_says_what_the_page_holds(http: dict) -> None:  # noqa: F811
    http["status"] = 200
    http["text"] = (
        "<html><head><title>Daily prices</title></head><body>"
        '<div id="main"><span class="price-tag">12</span><span class="price-tag">14</span>'
        '<p class="note">secret words</p></div></body></html>'
    )

    result = await _run(
        ActionType.API_SCRAPE,
        "_exec_api_scrape",
        ApiRequestParams(url="https://8.8.8.8/prices", selector=".price", extract="text"),
    )

    error = result.error or ""
    assert result.success is False
    assert "No element matches the selector '.price' on 8.8.8.8" in error
    assert "title 'Daily prices'" in error
    assert "#main" in error and ".price-tag" in error and ".note" in error
    assert "Asking for the same selector again will find nothing again" in error
    assert "secret words" not in error  # what a page says is never put in the message
