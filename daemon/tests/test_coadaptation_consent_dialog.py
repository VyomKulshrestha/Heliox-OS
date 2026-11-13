"""A co-adaptation consent is the person's answer in the daemon's approval dialog, not the window's payload.

The payload (``authorized: true`` included) says what is being asked for. The grant goes through the real handler and
a real CoAdaptationService on tmp_path; only the person's answer in the dialog is played by the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pilot.coadaptation.consent_dialog import PURPOSE_WORDS, WHERE_KEPT
from tests.test_coadaptation_consent_lapse import _Daemon
from tests.test_coadaptation_service import grant_params

REPO = Path(__file__).resolve().parents[2]


class Person:
    def __init__(self, confirmed: bool) -> None:
        self.confirmed = confirmed
        self.asked: list[dict[str, Any]] = []

    async def __call__(self, plan_id, plan, ws, **kwargs):
        self.asked.append({"plan_id": plan_id, **kwargs})
        everything = set(range(len(plan.actions)))
        return self.confirmed, (everything if self.confirmed else set()), everything

    def lines(self, ask: int = -1) -> dict[str, str]:
        return {line["target"]: line["description"] for line in self.asked[ask]["shown"]}


@pytest.fixture
async def daemon(tmp_path):
    instance = _Daemon(tmp_path)
    # The real _ask_about_data, so the lines reach the dialog as the window shows them.
    del instance.server._ask_about_data
    try:
        yield instance
    finally:
        await instance.close()


async def _grant(daemon: _Daemon, person: Person, *purposes: str, **kwargs: Any) -> dict:
    daemon.server._wait_for_confirmation = person  # type: ignore[method-assign]
    return await daemon.server._handle_co_adaptation_consent_grant(grant_params(purposes, **kwargs), None)


def test_the_dialog_uses_the_consent_sheets_own_words():
    sheet = json.loads((REPO / "tauri-app/ui/src/locales/coadaptation.en.json").read_text(encoding="utf-8"))
    consent = sheet["coadaptation"]["consent"]
    for purpose, (label, detail) in PURPOSE_WORDS.items():
        assert consent["purpose"][purpose.value] == {"label": label, "detail": detail}
    assert set(consent["purpose"]) == {purpose.value for purpose in PURPOSE_WORDS}
    heading = "Where it is kept: only"
    assert consent["where"].startswith(heading)
    assert "Only" + consent["where"][len(heading) :] == WHERE_KEPT
    assert consent["retention"].endswith(", or immediately when you withdraw.")


@pytest.mark.asyncio
async def test_a_grant_asks_naming_each_purpose_and_a_no_stores_nothing(daemon):
    service = await daemon.start()
    person = Person(confirmed=False)

    reply = await _grant(daemon, person, "selection_study", "gaze_prior", retention_days=30)

    assert reply["status"] == "cancelled"
    assert not service.store.is_open
    assert not (daemon.tmp_path / "co_adaptation.db").exists()
    assert daemon.server.config.co_adaptation.enabled is False
    assert person.asked[0]["plan_id"].startswith("data-")
    lines = person.lines()
    assert list(lines) == [
        "Run the selection study",
        "Use coarse gaze direction",
        "Keep it on this computer for 30 days",
    ]
    assert lines["Keep it on this computer for 30 days"] == (
        f"{WHERE_KEPT} Your data is erased automatically after 30 days, or immediately when you withdraw."
    )


@pytest.mark.asyncio
async def test_an_approved_grant_stores_the_consent_and_turns_the_feature_on(daemon):
    service = await daemon.start()

    reply = await _grant(daemon, Person(confirmed=True), "selection_study")

    assert reply["status"] == "ok"
    assert await service.store.participants() == [reply["participant_key"]]
    assert daemon.server.config.co_adaptation.enabled is True


@pytest.mark.asyncio
async def test_an_update_asks_only_about_what_it_adds_and_a_no_keeps_the_consent_as_it_was(daemon):
    await daemon.start()
    first = await _grant(daemon, Person(confirmed=True), "selection_study", "research_export")
    key = first["participant_key"]

    person = Person(confirmed=False)
    reply = await _grant(daemon, person, "selection_study", "research_export", "hand_pointer", participant_key=key)

    assert reply["status"] == "cancelled"
    targets = list(person.lines())
    assert targets[0] == "Use the hand pointer" and len(targets) == 2
    assert targets[1].startswith("Keep it with your study data until ")
    status = await daemon.server._handle_co_adaptation_status({}, None)
    (participant,) = status["participants"]
    assert sorted(participant["consent"]["purposes"]) == ["research_export", "selection_study"]


@pytest.mark.asyncio
async def test_an_update_that_only_removes_or_shortens_asks_nothing(daemon):
    await daemon.start()
    first = await _grant(daemon, Person(confirmed=True), "selection_study", "research_export")

    person = Person(confirmed=False)
    reply = await _grant(daemon, person, "selection_study", participant_key=first["participant_key"], retention_days=30)

    assert reply["status"] == "ok" and person.asked == []
    assert reply["removed"]["purposes"] == ["research_export"]


@pytest.mark.asyncio
async def test_a_payload_the_service_refuses_asks_nothing(daemon):
    await daemon.start()
    person = Person(confirmed=True)
    daemon.server._wait_for_confirmation = person  # type: ignore[method-assign]

    reply = await daemon.server._handle_co_adaptation_consent_grant({**grant_params(), "authorized": False}, None)

    assert reply["status"] == "error" and person.asked == []
