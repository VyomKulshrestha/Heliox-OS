"""Conversational Memory Bench: remember, recall in a later turn, forget, and a preference that changes the reply.

The Memory Bench (``pilot.memory.bench``) checks what reaches the *planner* after weeks of use. This one checks the
conversation: sentences said to Heliox go through the same local handlers the server runs first (the profile, then
``MemoryTalk``), and anything they do not take goes through the real ``ConversationLayer`` with the real context
sources (profile lines in the order the layer sends them, plus ``PersonalContext``). A stand-in fast model records
the messages it would be sent; nothing leaves the computer. A case passes when every local reply is the expected
sentence, every ``model_expect`` phrase is in what the model would see, no ``model_forbid`` phrase is, and no
``store_forbid`` phrase is left anywhere in the memory database (forgetting must remove the words, not hide them).

    python -m pilot.personal.bench [--json results.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).with_name("conversation_memory_bench.json")


class RecordingModel:
    """Stands in for the fast model: keeps what it is sent and says one short sentence."""

    name = "recording"
    provider = "recording"
    local = True
    streams = True

    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []

    async def stream(self, system: str, messages: list[dict[str, str]]):
        self.seen = [dict(m) for m in messages]
        yield "Okay."


def _profile_lines(profile: Any) -> list[str]:
    lines = list(profile.context_lines())
    try:
        from pilot.conversation.adapters import prioritise_profile
    except ImportError:
        return lines
    return list(prioritise_profile(lines))


def _stored_text(db_file: Path) -> str:
    if not db_file.exists():
        return ""
    db = sqlite3.connect(db_file)
    try:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        chunks = []
        for table in tables:
            try:
                rows = db.execute(f'SELECT * FROM "{table}"').fetchall()  # noqa: S608 - names from sqlite_master
            except sqlite3.Error:
                continue
            chunks += [" ".join(str(cell) for cell in row) for row in rows]
    finally:
        db.close()
    return "\n".join(chunks).casefold()


async def run_case(case: dict[str, Any], filler: list[str], *, personal_layer: bool = True) -> dict[str, Any]:
    from pilot.conversation.context import ContextSources
    from pilot.conversation.layer import ConversationLayer
    from pilot.memory.store import MemoryStore
    from pilot.memory.temporal import MemoryProvenance, MemoryScope
    from pilot.personal.context import PersonalContext
    from pilot.personal.memory_talk import MemoryTalk
    from pilot.workflows.user_profile import UserProfile

    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="heliox-conv-memory-bench-", ignore_cleanup_errors=True) as folder:
        root = Path(folder)
        db_file = root / "memory.db"
        memory = MemoryStore(
            checkpoint_interval_seconds=0, pruning_interval_seconds=0, db_file=db_file, semantic_search=False
        )
        await memory.initialize()

        async def remember(predicate: str, value: str) -> str | None:
            fact = await memory.remember_fact(
                subject="user",
                predicate=predicate,
                value=value,
                scope=MemoryScope.USER,
                confidence=1.0,
                provenance=MemoryProvenance.EXPLICIT_USER,
                evidence_payload={"source": "user_profile"},
            )
            return fact.fact_id if fact is not None else None

        async def forget(fact_id: str) -> None:
            try:
                await memory.erase_fact(fact_id, reason="Removed from your profile")
            except KeyError:
                pass

        profile = UserProfile(root / "user_profile.json", remember=remember, forget=forget)
        talk = MemoryTalk(profile=lambda: profile)
        personal = PersonalContext(profile=lambda: profile)
        model = RecordingModel()

        async def sink(text: str, channel: str, key: str) -> str:
            return "spoken"

        layer = ConversationLayer(
            sink=sink,
            model=lambda: model,
            sources=ContextSources(
                profile_lines=lambda: _profile_lines(profile), extra=[personal.extra] if personal_layer else []
            ),
        )
        try:
            for turn in case["turns"]:
                if turn.get("filler"):
                    for sentence in filler:
                        await profile.answer_async(sentence)
                    continue
                say = turn["say"]
                reply = await profile.answer_async(say)
                if reply is None and personal_layer:
                    reply = await talk.answer(say)
                if "reply" in turn and reply != turn["reply"]:
                    problems.append(f"{say!r}: replied {reply!r}, expected {turn['reply']!r}")
                if "reply_starts" in turn and not str(reply or "").startswith(turn["reply_starts"]):
                    problems.append(f"{say!r}: replied {reply!r}, expected it to start {turn['reply_starts']!r}")
                if reply is None:
                    model.seen = []
                    personal.note(say)
                    await layer.take(say)
                    seen = "\n".join(m["content"] for m in model.seen).casefold()
                    if not model.seen and (turn.get("model_expect") or turn.get("model_forbid")):
                        problems.append(f"{say!r}: the model was not asked")
                    for phrase in turn.get("model_expect", []):
                        if phrase.casefold() not in seen:
                            problems.append(f"{say!r}: the model did not get {phrase!r}")
                    for phrase in turn.get("model_forbid", []):
                        if phrase.casefold() in seen:
                            problems.append(f"{say!r}: the model got {phrase!r}, which should be gone")
                elif turn.get("model_expect"):
                    problems.append(f"{say!r}: answered locally ({reply!r}), so the model never saw it")
                if turn.get("store_forbid"):
                    profile_file = root / "user_profile.json"
                    stored = "".join(_stored_text(path) for path in sorted(root.glob("*.db"))) + (
                        profile_file.read_text(encoding="utf-8").casefold() if profile_file.exists() else ""
                    )
                    for phrase in turn["store_forbid"]:
                        if phrase.casefold() in stored:
                            problems.append(f"{say!r}: {phrase!r} is still stored")
        finally:
            await memory.close()
    return {"id": case["id"], "why": case.get("why", ""), "pass": not problems, "problems": problems}


async def run(script: dict[str, Any] | None = None, *, personal_layer: bool = True) -> dict[str, Any]:
    """``personal_layer=False`` measures without ``MemoryTalk`` and ``PersonalContext`` (the state before them)."""
    script = script or json.loads(SCRIPT.read_text(encoding="utf-8"))
    rows = [await run_case(case, script.get("filler", []), personal_layer=personal_layer) for case in script["cases"]]
    return {"cases": len(rows), "passed": sum(row["pass"] for row in rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.personal.bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", type=Path, help="write every case's result here")
    parser.add_argument("--without-personal", action="store_true", help="measure without MemoryTalk and context")
    args = parser.parse_args(argv)
    result = asyncio.run(run(personal_layer=not args.without_personal))
    for row in result["rows"]:
        print(f"{'ok  ' if row['pass'] else 'FAIL'} {row['id']:28} {row['why']}")
        for problem in row["problems"]:
            print(f"       {problem}")
    print(f"\nConversational Memory Bench: {result['passed']}/{result['cases']} cases")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
