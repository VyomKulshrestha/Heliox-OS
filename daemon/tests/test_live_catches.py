"""Heliox Live, part four: community catch packs, admitted only through a precision gate."""

from __future__ import annotations

import json
import os
import time
from datetime import date

import pytest

from pilot.live import catches
from pilot.live.catches import BUILT_IN_DIR, CatchBook, audit, load_corpus
from pilot.live.detectors import TextContext, detect

CORPUS = load_corpus()
MONDAY = date(2026, 11, 9)


def ctx(app: str = "notepad.exe", title: str = "notes.txt - Notepad") -> TextContext:
    return TextContext(today=MONDAY, app=app, title=title)


def rule(**overrides) -> dict:
    base = {
        "id": "could-of",
        "where": ["prose"],
        "pattern": r"\b(could|should) of\b",
        "flags": "i",
        "title": '"{1} of" is "{1} have"',
        "message": '"{0}" is written "{1} have".',
        "fix": "{1} have",
        "positives": ["I could of known.", "You should of asked.", "We could of won."],
        "decoys": ["I could have known.", "Most of it, of course.", "the should-be state of things"],
    }
    base.update(overrides)
    return base


def pack(*rules: dict, name: str = "test-pack") -> dict:
    return {"name": name, "rules": list(rules)}


# ── the gate ──


def test_the_shipped_packs_pass_every_check():
    for path in sorted(BUILT_IN_DIR.glob("*.json")):
        report, admitted = audit(json.loads(path.read_text(encoding="utf-8")), CORPUS, str(path))
        assert report.ok, report.as_dict()
        assert len(admitted) == len(report.rules) >= 5


def test_the_corpus_is_ordinary_text_from_this_project_only():
    assert len(CORPUS["prose"]) >= 1500 and len(CORPUS["code"]) >= 1000 and len(CORPUS["terminal"]) >= 100
    text = catches.CORPUS.read_text(encoding="utf-8").casefold()
    assert "@gmail" not in text and "marcu" not in text and "://" not in text


def test_a_good_rule_is_admitted_and_fills_its_message_from_the_match():
    report, [admitted] = audit(pack(rule()), CORPUS)
    assert report.ok and report.rules[0].corpus_hits == []
    found = admitted.find("Honestly I should of called", ctx())
    assert found.title == '"should of" is "should have"' and found.fix == "should have"
    assert found.evidence == "should of" and found.kind == "community_catch" and not found.trailing
    assert found.key == "community_catch:test-pack:could-of:should of"
    assert admitted.find("I should of called", ctx("WindowsTerminal.exe", "PowerShell")) is None  # prose only


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"pattern": r"(\w+)+ of"}, "repeats a repeated group"),
        ({"pattern": "(unclosed"}, "does not compile"),
        ({"positives": ["I could of known."]}, "at least 3 positives"),
        ({"urgency": "urgent"}, "only terminal rules may be urgent"),
        ({"decoys": ["I could of known.", "a", "b"]}, "fires on its own look-alike"),
        ({"positives": ["I could of known.", "You should of asked.", "We might of won."]}, "misses its own example"),
        (
            {"pattern": r"\bthe\b", "positives": ["the", "the end", "and the"], "decoys": ["x", "y", "z"]},
            "ordinary line",
        ),
        ({"where": ["everywhere"]}, "where must list"),
        ({"id": "Bad Id"}, "id must be"),
    ],
)
def test_a_rule_that_fails_any_check_is_held_back_with_the_reason(overrides, problem):
    report, admitted = audit(pack(rule(**overrides)), CORPUS)
    assert admitted == [] and not report.ok
    assert any(problem in p for p in report.rules[0].problems), report.rules[0].problems


def test_a_slow_rule_is_held_back(monkeypatch):
    monkeypatch.setattr(catches, "MAX_CORPUS_MS", -1.0)
    report, admitted = audit(pack(rule()), CORPUS)
    assert admitted == [] and "too slow" in report.rules[0].problems[-1]


def test_one_bad_rule_does_not_sink_the_pack_and_bad_packs_say_why():
    report, admitted = audit(pack(rule(), rule(id="broken", pattern="(")), CORPUS)
    assert [r.id for r in admitted] == ["could-of"] and report.admitted == 1 and not report.ok
    assert audit({"rules": []}, CORPUS)[0].problems
    assert audit([], CORPUS)[0].problems == ["a pack is a JSON object with a name and rules"]
    duplicate, kept = audit(pack(rule(), rule()), CORPUS)
    assert len(kept) == 1 and "the same id twice" in duplicate.rules[1].problems


# ── the installed book ──


def test_the_book_loads_your_own_packs_rechecks_changes_and_reports_what_it_held_back(tmp_path):
    mine = tmp_path / "live_catches"
    mine.mkdir()
    (mine / "mine.json").write_text(json.dumps(pack(rule(), rule(id="broken", pattern="("), name="mine")))
    (mine / "junk.json").write_text("{not json")
    (mine / "copy.json").write_text(json.dumps(pack(rule(), name="infra-safety")))  # name already taken
    book = CatchBook([BUILT_IN_DIR, mine], corpus=CORPUS)
    book.refresh(force=True)
    summary = book.summary()
    names = {p["name"]: p for p in summary["packs"]}
    assert names["mine"]["admitted"] == 1 and names["mine"]["held_back"][0]["id"] == "broken"
    assert any("another pack has the same name" in p["problems"] for p in summary["packs"])
    assert any("not valid JSON" in problem for p in summary["packs"] for problem in p["problems"])
    assert {c.key.split(":")[1] for c in book.find("We could of shipped it.", ctx())} == {"writing-idioms", "mine"}

    (mine / "mine.json").write_text(json.dumps(pack(rule(id="fixed-now"), name="mine")))
    os.utime(mine / "mine.json", (time.time() + 5, time.time() + 5))
    book.refresh(force=True)
    assert "fixed-now" in {r.id for r in book.rules}


def test_live_detection_includes_the_admitted_catches():
    book = CatchBook([BUILT_IN_DIR], corpus=CORPUS)
    catches.install(book)
    try:
        found = detect("I think we could of shipped it on Friday.", ctx())
        assert [c.kind for c in found] == ["community_catch"] and found[0].fix == "could have"
        terminal = detect("PS C:\\infra> terraform destroy", ctx("WindowsTerminal.exe", "PowerShell"))
        assert [c.urgency for c in terminal] == ["urgent"]
        assert detect("We could of shipped it.", TextContext(today=MONDAY, source="speech")) == []
    finally:
        catches.install(CatchBook())


def test_the_cli_checks_a_pack_and_fails_on_problems(tmp_path, capsys):
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(pack(rule())))
    bad.write_text(json.dumps(pack(rule(pattern="("))))
    assert catches.main(["check", str(good)]) == 0
    assert "1/1 rules admitted" in capsys.readouterr().out
    assert catches.main(["check", str(bad)]) == 1
    assert "does not compile" in capsys.readouterr().out
