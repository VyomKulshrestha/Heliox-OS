"""More of the conversation eval: further checks, more cases, and the same eval over other providers' wire formats.

  python -m benchmarks.conversation_eval.more [--replay openai|gemini|claude] [--provider ...] [--model ...]
                                              [--out FILE.json]

Further checks, decided in code like the first ones:
* one question at a time (a spoken reply that asks two questions leaves the person unsure which to answer);
* at most about 12 seconds of speech (30 words at an ordinary 2.5 words a second);
* numbers and dates in a form a voice reads well: no numeric dates (12/11/2026, 2026-11-12), no unit glued to a
  number (16GB), no long digit runs, no version strings;
* a follow-up after Heliox Live spoke ("why?") answered from Live's remark, which the watching side records as a
  Heliox turn, not from the model's guess;
* the fast layer and the planner agreeing: a question about how to do something stays an answer, a request made as
  a question ("could you...") still hands over.

``--replay`` runs the eval through another provider's request and stream format with no network and no cost: a
stub server answers each turn with the reply Claude Haiku 5.5 gave in a recorded run (recorded_replies.json),
streamed in small pieces in that provider's format (OpenAI-compatible server-sent events, Gemini's
streamGenerateContent, or Anthropic's Messages events). So the per-provider streaming, splitting, cleaning and
usage code runs end to end, and its results must match the recorded run's.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from pilot.real_folders import use_scratch_folders

# Every Heliox folder under a scratch root before pilot.config loads, never the person's own. A paid run reads the
# person's model settings and counts what it spent through pilot.real_folders, never through their files.
use_scratch_folders()

from .cases import CONVERSATIONS, NOW  # noqa: E402
from .checks import Outcome, TurnRecord, check_turn, heard, summarise  # noqa: E402

RECORDED = Path(__file__).with_name("recorded_replies.json")
WORDS_PER_SECOND = 2.5
MAX_SPOKEN_SECONDS = 12.0

UNSAYABLE_NUMBERS = (
    ("numeric date", re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}\b")),
    ("unit glued to a number", re.compile(r"\b\d+(?:\.\d+)?(?:gb|mb|kb|tb|ghz|mhz|ms|px)\b", re.IGNORECASE)),
    ("long digit run", re.compile(r"\b\d{6,}\b")),
    ("version string", re.compile(r"\bv?\d+\.\d+\.\d+\b")),
)

MORE_CONVERSATIONS = [
    {
        "name": "a follow-up after Live spoke",
        "heliox_first": "The twelfth of November is a Thursday. You said Monday.",
        "turns": [
            ("Why?", {"expect": "talk", "mentions": ["Thursday"], "never_mentions": ["Wednesday"]}),
            ("Oh, thanks. Which day did I say?", {"expect": "talk", "mentions": ["Monday"]}),
        ],
    },
    {
        "name": "asking how is not asking to act",
        "turns": [
            ("How do I open the registry editor?", {"expect": "talk"}),
            ("What's the shortcut to lock my screen?", {"expect": "talk"}),
            ("How would I uninstall an app on Windows?", {"expect": "talk"}),
        ],
    },
    {
        "name": "a request made as a question still acts",
        "turns": [
            ("Could you open the calculator?", {"expect": "act"}),
            ("Would you mind muting the sound?", {"expect": "act", "quick": True}),
            ("Can you check how much battery I have left?", {"expect": "act"}),
        ],
    },
    {
        "name": "numbers and dates said well",
        "screen": {"app": "Explorer", "excerpt": "backup-2026-11-12.zip  16GB  modified 12/11/2026 09:41"},
        "turns": [
            ("How big is the backup file on my screen?", {"expect": "talk", "mentions": ["16|sixteen"]}),
            ("When was it last modified?", {"expect": "talk"}),
        ],
    },
    {
        "name": "one question at a time",
        "turns": [
            ("I want to plan my week.", {"expect": "talk"}),
            ("Something about the launch.", {"expect": "talk"}),
        ],
    },
]


def more_checks(record: TurnRecord, spec: dict[str, Any]) -> list[Outcome]:
    said = heard(record)
    if not record.done or not said:
        return []
    out = [Outcome("one_question", said.count("?") <= 1, said)]
    words = len(said.split())
    out.append(Outcome("short_enough", words / WORDS_PER_SECOND <= MAX_SPOKEN_SECONDS, f"{words} words: {said!r}"))
    found = [name for name, pattern in UNSAYABLE_NUMBERS if pattern.search(said)]
    out.append(Outcome("numbers_said_well", not found, f"{', '.join(found)} in {said!r}"))
    return out


# ── replaying a recorded run through another provider's wire format ──


def _replay_transport(provider: str, replies: dict[str, str]) -> Any:
    import httpx

    def last_words(body: dict[str, Any]) -> str:
        if provider == "gemini":
            text = body["contents"][-1]["parts"][0]["text"]
        else:
            text = body["messages"][-1]["content"]
        return text.rsplit("[The person said]\n", 1)[-1].strip()

    def pieces(text: str) -> list[str]:
        return [text[i : i + 7] for i in range(0, len(text), 7)]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        reply = replies.get(last_words(body), "I don't know that one.")
        lines: list[str] = []
        if provider == "gemini":
            for piece in pieces(reply):
                lines.append("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": piece}]}}]}))
            lines.append(
                "data: " + json.dumps({"usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 40}})
            )
        elif provider == "claude":
            lines.append("data: " + json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 500}}}))
            for piece in pieces(reply):
                lines.append(
                    "data: "
                    + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": piece}})
                )
            lines.append("data: " + json.dumps({"type": "message_delta", "usage": {"output_tokens": 40}}))
        else:
            for piece in pieces(reply):
                lines.append("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}))
            lines.append(
                "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 40}})
            )
            lines.append("data: [DONE]")
        return httpx.Response(
            200, content=("\n\n".join(lines) + "\n\n").encode(), headers={"content-type": "text/event-stream"}
        )

    return httpx.MockTransport(handler)


class _Budget:
    def __init__(self) -> None:
        self.records: list[tuple] = []

    def check_budget(self, provider: str) -> None:
        return None

    async def record_usage(self, provider, model, input_tokens, output_tokens, **kwargs) -> None:
        self.records.append((provider, model, input_tokens, output_tokens, kwargs.get("usage_source")))


def replay_model(provider: str) -> tuple[Any, _Budget]:
    import httpx

    from pilot.conversation.fast_model import HttpFastModel

    replies = json.loads(RECORDED.read_text(encoding="utf-8"))["replies"]
    budget = _Budget()

    async def key() -> str:
        return "replay-key"

    model = HttpFastModel(
        provider=provider,
        name={"gemini": "gemini-2.5-flash", "claude": "claude-haiku-5-5"}.get(provider, "gpt-4o-mini"),
        client_factory=lambda: httpx.AsyncClient(transport=_replay_transport(provider, replies)),
        api_key=key,
        budget=budget,
    )
    return model, budget


# ── the run ──


async def run_conversations(
    conversations: list[dict[str, Any]], model: Any, *, extra_checks: bool
) -> list[tuple[str, TurnRecord, list[Outcome]]]:
    from pilot.conversation.context import ContextSources, Screen
    from pilot.conversation.layer import ConversationLayer
    from pilot.conversation.planner_context import planner_sees

    results: list[tuple[str, TurnRecord, list[Outcome]]] = []
    for conversation in conversations:
        screen = conversation.get("screen")
        sources = ContextSources(
            profile_lines=lambda c=conversation: list(c.get("profile", [])),
            task_state=lambda c=conversation: dict(c.get("task", {})),
            screen=lambda s=screen: Screen(app=s["app"], excerpt=s.get("excerpt", "")) if s else None,
            now=lambda: NOW,
        )
        spoken: list[str] = []

        async def sink(text: str, channel: str, key: str, spoken: list[str] = spoken) -> str:
            spoken.append(text)
            return "spoken"

        layer = ConversationLayer(sink=sink, model=lambda m=model: m, sources=sources, hold_after=60.0)
        if conversation.get("heliox_first"):
            layer.memory.add_heliox(conversation["heliox_first"], source="live")  # as the watching side records it
        previous: TurnRecord | None = None
        for text, spec in conversation["turns"]:
            spoken.clear()
            result = await layer.take(text)
            record = TurnRecord(
                text=text,
                route=result.route.kind.value if result.route else "",
                done=result.done,
                spoken=list(spoken),
                raw="",
                planner_input="" if result.done else planner_sees(result),
                quick_types=[a.action_type.value for a in result.quick.plan.actions] if result.quick else [],
            )
            outcomes = check_turn(record, spec, previous)
            if extra_checks:
                outcomes += more_checks(record, spec)
            results.append((f"{conversation['name']}: {text}", record, outcomes))
            previous = record
    return results


def _print(summary: dict[str, Any], label: str) -> None:
    print(f"\n{label}: turns {summary['turns']}")
    for name, rate in sorted(summary["checks"].items()):
        print(f"  {name:24} {rate['passed']}/{rate['of']}")
    print("  hand-over precision", summary["handover"]["precision"], "recall", summary["handover"]["recall"])
    for failure in summary["failures"]:
        print(f"  FAIL {failure['check']}: {failure['turn']} | {failure['detail'][:200]}")


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if args.replay:
        model, budget = replay_model(args.replay)
        results = await run_conversations(CONVERSATIONS, model, extra_checks=False)
        summary = summarise(results)
        summary["turns"] = len(results)
        summary["usage_records"] = len(budget.records)
        summary["model"] = f"replay over {args.replay}"
        return summary
    from pilot.config import PilotConfig
    from pilot.conversation.fast_model import build_fast_model
    from pilot.real_folders import apply_owner_settings, real_budget_tracker
    from pilot.security.vault import KeyVault

    from . import OWNER_CONVERSATION_FIELDS

    config = PilotConfig()  # the run's own: its files are under the scratch root
    apply_owner_settings(config, ("model", "conversation"), only=OWNER_CONVERSATION_FIELDS)
    if args.provider == "ollama":
        config.model.provider = "ollama"
    elif args.provider:
        config.model.provider, config.model.cloud_provider = "cloud", args.provider
    if args.model:
        setattr(config.conversation, f"{args.provider or config.model.cloud_provider or 'ollama'}_model", args.model)
    async with real_budget_tracker(config.model, "the more-turns eval") as budget:  # the one write to their folders
        before = budget._monthly_cost
        model = build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)
        if model is None:
            raise SystemExit("No fast model is configured.")
        results = await run_conversations(MORE_CONVERSATIONS, model, extra_checks=True)
        # The extra checks over the first eval's conversations too: one question, length and numbers.
        results += await run_conversations(CONVERSATIONS, model, extra_checks=True)
        await asyncio.sleep(0.5)
        summary = summarise(results)
        summary["turns"] = len(results)
        summary["model"] = f"{getattr(model, 'provider', '')}:{getattr(model, 'name', '')}"
        summary["spent_usd"] = round(budget._monthly_cost - before, 5)
        summary["records"] = [
            {"turn": name, "talk": r.done, "spoken": r.spoken, "planner_input": r.planner_input}
            for name, r, _ in results
        ]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_eval.more")
    parser.add_argument("--replay", choices=["openai", "gemini", "claude"], default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    summary = asyncio.run(main_async(args))
    _print(summary, summary["model"])
    if "spent_usd" in summary:
        print("  cost: $", summary["spent_usd"])
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
