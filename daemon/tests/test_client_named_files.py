"""Files a client names: read only under the rules every file Heliox reads follows, and sent off only when asked.

resolve_git_conflict reads the file the window's "/git-resolve <path>" names and asks the model for resolutions.
It now refuses what the validator refuses for any read (Heliox's own folders, protected folders), a file outside a
git working tree, and, with a model that is not on this computer, sends nothing until the person approves the
lines that will go there (once per file for the run). air_handoff_grab takes a screenshot or text only, and the
daemon has no extract_file_text method (the window reads a file it was given through the app itself).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.server import PilotServer

BLOCK = "<<<<<<< HEAD\nmine\n=======\ntheirs\n>>>>>>> branch\n"


class Person:
    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed
        self.asked: list[dict[str, Any]] = []

    async def __call__(self, plan_id: str, plan: Any, ws: Any, **kwargs: Any):
        self.asked.append(kwargs)
        return self.confirmed, ({0} if self.confirmed else set()), {0}


class Model:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return '{"proposed_resolution_code": "merged\\n"}'


def _server(person: Person, *, cloud: bool) -> tuple[PilotServer, Model]:
    config = PilotConfig()
    if cloud:
        config.model.provider, config.model.cloud_provider = "cloud", "claude"
    server = PilotServer(config)
    model = Model()
    server._model_router = model
    server._wait_for_confirmation = person  # type: ignore[method-assign]
    return server, model


@pytest.fixture
def conflicted(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    path = repo / "app.py"
    path.write_text(f"before\n{BLOCK}after\n", encoding="utf-8")
    return path


def _ws() -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_with_a_local_model_nothing_is_asked(conflicted):
    person = Person(confirmed=False)
    server, model = _server(person, cloud=False)

    reply = await server._handle_resolve_git_conflict({"filepath": str(conflicted)}, _ws())

    assert reply["status"] == "success" and person.asked == [] and len(model.prompts) == 1


@pytest.mark.asyncio
async def test_with_a_cloud_model_nothing_is_sent_until_the_person_approves(conflicted):
    person = Person(confirmed=False)
    server, model = _server(person, cloud=True)

    reply = await server._handle_resolve_git_conflict({"filepath": str(conflicted)}, _ws())

    assert reply["status"] == "cancelled" and model.prompts == []
    line = person.asked[0]["shown"][0]
    assert "api.anthropic.com" in line["target"] and "5 lines" in line["description"]


@pytest.mark.asyncio
async def test_an_approval_covers_that_file_for_the_run(conflicted):
    person = Person(confirmed=True)
    server, model = _server(person, cloud=True)

    first = await server._handle_resolve_git_conflict({"filepath": str(conflicted)}, _ws())
    second = await server._handle_resolve_git_conflict({"filepath": str(conflicted)}, _ws())

    assert first["status"] == second["status"] == "success"
    assert len(person.asked) == 1 and len(model.prompts) == 2


@pytest.mark.asyncio
async def test_a_protected_folder_and_a_file_outside_a_repository_are_refused(conflicted, tmp_path):
    person = Person(confirmed=True)
    server, model = _server(person, cloud=False)
    server.config.restrictions.protected_folders = [str(conflicted.parent)]
    loose = tmp_path / "loose.py"
    loose.write_text(BLOCK, encoding="utf-8")

    protected = await server._handle_resolve_git_conflict({"filepath": str(conflicted)}, _ws())
    server.config.restrictions.protected_folders = []
    outside = await server._handle_resolve_git_conflict({"filepath": str(loose)}, _ws())

    assert protected["status"] == "error" and "protected folder" in protected["message"]
    assert outside["status"] == "error" and "git repository" in outside["message"]
    assert model.prompts == []


@pytest.mark.asyncio
async def test_heliox_s_own_folders_are_refused(monkeypatch, tmp_path):
    import pilot.security.validator as validator

    person = Person(confirmed=True)
    server, model = _server(person, cloud=False)
    secret = tmp_path / "repo" / "config.toml"
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    secret.write_text(BLOCK, encoding="utf-8")
    monkeypatch.setattr(validator, "inside_heliox_private_dirs", lambda path: "config.toml" in str(path))

    reply = await server._handle_resolve_git_conflict({"filepath": str(secret)}, _ws())

    assert reply["status"] == "error" and model.prompts == []


@pytest.mark.asyncio
async def test_air_handoff_grab_does_not_take_a_file():
    server = PilotServer(PilotConfig())
    server.config.air_handoff.enabled = True
    server._air_handoff_server = SimpleNamespace(running=True)
    server._air_handoff_manager = SimpleNamespace(grab_file=AsyncMock())
    server._publish_air_handoff_state = AsyncMock()

    reply = await server._handle_air_handoff_grab({"kind": "file", "path": "C:/Users/sam/.ssh/id_ed25519"}, _ws())

    assert reply == {"status": "error", "message": "Unsupported handoff kind"}
    server._air_handoff_manager.grab_file.assert_not_called()


def test_the_daemon_has_no_extract_file_text_method():
    assert not hasattr(PilotServer, "_handle_extract_file_text")
