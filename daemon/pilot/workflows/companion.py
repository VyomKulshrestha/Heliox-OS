"""Your companion: the assistant's name and the tone it talks to you in.

"Your name is Nova", "call yourself Nova", "what's your name", "be more playful",
"be warmer", "be strictly professional", "what's your personality", "introduce yourself"
and "go back to normal". The name and a tone preset are kept in ``DATA_DIR/companion.json``
on this computer. The name is used in greetings and the companion refers to itself by it,
and the tone is added to the model's system prompt so replies carry the same voice. This
only changes how Heliox speaks; it never changes what Heliox may do, and the model is
still told to stay truthful and never claim an action it did not take.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

DEFAULT_NAME = "Heliox"
MAX_NAME = 24

# preset -> (what to call it to the user, the system-prompt line fragment)
PERSONAS: dict[str, tuple[str, str]] = {
    "balanced": ("balanced and clear", "a balanced, clear and natural tone"),
    "friendly": ("friendly and warm", "a friendly, warm tone, a little informal"),
    "warm": ("warm and encouraging", "a warm, encouraging tone that notices how things are going for them"),
    "professional": ("professional and precise", "a professional, precise tone with no filler"),
    "concise": ("brief and to the point", "a very concise tone; short sentences, no padding"),
    "playful": ("playful and upbeat", "a playful, upbeat tone with the occasional light remark, never silly"),
    "witty": ("dry and witty", "a dry, lightly witty tone; one wry aside at most, never at the user's expense"),
    "calm": ("calm and steady", "a calm, steady, reassuring tone"),
    "direct": ("blunt and direct", "a blunt, direct tone; say the thing, skip the hedging"),
}
_WORD_TO_PRESET = {
    "balanced": "balanced", "normal": "balanced", "neutral": "balanced", "default": "balanced",
    "friendly": "friendly", "warm": "warm", "warmer": "warm", "kind": "warm", "encouraging": "warm",
    "professional": "professional", "formal": "professional", "businesslike": "professional",
    "concise": "concise", "brief": "concise", "short": "concise", "terse": "concise",
    "playful": "playful", "fun": "playful", "upbeat": "playful", "cheerful": "playful",
    "witty": "witty", "funny": "witty", "humorous": "witty", "dry": "witty",
    "calm": "calm", "steady": "calm", "gentle": "calm", "chill": "calm", "relaxed": "calm",
    "direct": "direct", "blunt": "direct", "straight": "direct", "no nonsense": "direct", "to the point": "concise",
}  # fmt: skip
_PRESET_WORDS = "|".join(sorted((re.escape(w) for w in _WORD_TO_PRESET), key=len, reverse=True))

NAME_REQUEST = re.compile(
    r"^(?:please )?(?:your name(?: s| is| should be)|i(?: ll| will)? call you|call yourself|rename yourself to"
    r"|from now on you(?: re| are)) (?P<name>[\w .'-]{1,24})$"
)
NAME_GET_REQUEST = re.compile(r"^(?:what(?: s| is) your name|who are you|what do i call you|what should i call you)$")
PERSONA_REQUEST = re.compile(
    rf"^(?:please )?(?:be|talk|speak|sound|act|get|stay|become) "
    rf"(?:more |really |very |super |extra |strictly |a (?:bit|little) more |less )?(?P<p>{_PRESET_WORDS})"
    rf"(?: with me| please| from now on| to me)?$"
    rf"|^(?:please )?(?:use a|have a|take a|keep a) (?P<p2>{_PRESET_WORDS}) (?:tone|personality|voice|manner)$"
)
PERSONA_GET_REQUEST = re.compile(
    r"^(?:what(?: s| is) your (?:personality|tone|vibe)|how (?:are you|do you) (?:set to )?(?:talk|sound))$"
)
INTRO_REQUEST = re.compile(r"^(?:please )?(?:introduce yourself|who are you again|tell me about yourself)$")
RESET_REQUEST = re.compile(
    r"^(?:please )?(?:go back to normal|reset your (?:personality|tone|name)|be yourself again)$"
)


def _normalize(text: str) -> str:
    # Standard strip (drops apostrophes too) so runtime matches the shortcut-reservation check.
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _titlecase(name: str) -> str:
    name = " ".join(str(name).split())[:MAX_NAME]
    return " ".join(w if (w.isupper() and len(w) > 1) else w.capitalize() for w in name.split())


class Companion:
    """The assistant's name and tone, kept in a small JSON file."""

    def __init__(self, path: Path, *, on_change: Callable[[], Awaitable[None]] | None = None) -> None:
        self._path = Path(path)
        self._on_change = on_change

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        name = data.get("name") if isinstance(data, dict) else None
        persona = data.get("persona") if isinstance(data, dict) else None
        return {
            "name": name if isinstance(name, str) and name.strip() else DEFAULT_NAME,
            "persona": persona if persona in PERSONAS else "balanced",
        }

    def _save(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self) -> dict[str, str]:
        return self._load()

    def name(self) -> str:
        return self._load()["name"]

    def persona(self) -> str:
        return self._load()["persona"]

    def system_prompt_line(self, user_name: str = "") -> str:
        """The persona line added to the model's system prompt. Empty when nothing is customized."""
        data = self._load()
        if data["name"] == DEFAULT_NAME and data["persona"] == "balanced" and not user_name:
            return ""
        _, tone = PERSONAS[data["persona"]]
        to = f" You are speaking with {user_name}." if user_name else ""
        named = (
            f' The user calls you "{data["name"]}"; refer to yourself as {data["name"]}.'
            if data["name"] != DEFAULT_NAME
            else ""
        )
        return (
            f"Voice and persona: speak in {tone}.{named}{to} This sets your wording only. "
            "Stay truthful, never invent results, and never claim to have done something you did not do."
        )

    async def set_name(self, name: str) -> str:
        name = _titlecase(name)
        if not 1 <= len(name) <= MAX_NAME:
            raise ValueError("Give me a name of 1 to 24 characters")
        data = self._load()
        data["name"] = name
        self._save(data)
        await self._changed()
        return name

    async def set_persona(self, preset: str) -> str:
        if preset not in PERSONAS:
            raise ValueError("I don't have that tone")
        data = self._load()
        data["persona"] = preset
        self._save(data)
        await self._changed()
        return preset

    async def reset(self) -> None:
        self._save({"name": DEFAULT_NAME, "persona": "balanced"})
        await self._changed()

    async def _changed(self) -> None:
        if self._on_change is not None:
            try:
                await self._on_change()
            except Exception:
                pass

    def introduce(self) -> str:
        data = self._load()
        label, _ = PERSONAS[data["persona"]]
        return (
            f"I'm {data['name']}, your Heliox assistant. I run on your computer, keep what I learn here, "
            f'and I try to be {label}. Tell me what you need, or say "what can I say?" to see what I handle.'
        )

    async def answer_async(self, text: str) -> str | None:
        normalized = _normalize(text)
        raw = text.strip()

        if NAME_GET_REQUEST.match(normalized):
            data = self._load()
            if data["name"] == DEFAULT_NAME:
                return 'I\'m Heliox, your assistant. You can give me another name: say "call yourself ...".'
            return f"I'm {data['name']}, your Heliox assistant."
        if INTRO_REQUEST.match(normalized):
            return self.introduce()
        if PERSONA_GET_REQUEST.match(normalized):
            label, _ = PERSONAS[self.persona()]
            return f'Right now I keep a {label} tone. Say "be more ..." to change it.'
        if RESET_REQUEST.match(normalized):
            await self.reset()
            return "Back to normal: I'm Heliox again, with a balanced tone."

        name = NAME_REQUEST.match(normalized)
        if name:
            m = re.search(
                r"(?i)(?:name(?:'s| is| should be)?|call you|call yourself|rename yourself to|you(?:'re| are)) ([\w .'-]{1,24})$",
                raw,
            )
            try:
                saved = await self.set_name(m.group(1) if m else name.group("name"))
            except ValueError as exc:
                return str(exc)
            return f"Done. I'm {saved} from now on."

        persona = PERSONA_REQUEST.match(normalized)
        if persona:
            preset = _WORD_TO_PRESET[persona.group("p") or persona.group("p2")]
            await self.set_persona(preset)
            label, _ = PERSONAS[preset]
            return f"Okay, I'll be {label}."
        return None
