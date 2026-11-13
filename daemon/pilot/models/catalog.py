"""Which cloud models Heliox offers and uses by default, per provider, each list with the page it was read from.

Every model here was checked on 13 November 2026 against the provider's own pages: its models or pricing page for
what is current, and its deprecations page for what is deprecated or retired and the successor it names. The
defaults (planner and talk) are models the provider lists as current that day; where a former default was
deprecated, the provider's named successor took its place. The Settings panel reads this through the
``model_catalog`` call, so the panel never offers a retired model; a saved model that has been retired or
deprecated is shown as such with the current default suggested, never replaced behind the person's back. A model
id can still be typed by hand (newer or private models).

To re-check: read the pages in SOURCES, update the lists and the date. The price of every model offered here is in
pilot.models.prices; a test keeps the two together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pilot.models.prices import Source

READ_ON = "2026-11-13"

SOURCES: dict[str, list[Source]] = {
    "claude": [
        Source("https://platform.claude.com/docs/en/about-claude/model-deprecations", READ_ON),
        Source("https://platform.claude.com/docs/en/about-claude/pricing", READ_ON),
    ],
    "openai": [
        Source("https://developers.openai.com/api/docs/deprecations", READ_ON),
        Source("https://developers.openai.com/api/docs/pricing", READ_ON),
        Source("https://developers.openai.com/api/docs/models/gpt-6-luna", READ_ON),
    ],
    "gemini": [
        Source("https://aistudio.google.com/docs/deprecations", READ_ON),
        Source("https://aistudio.google.com/docs/pricing", READ_ON),
    ],
    "meta": [
        Source("https://dev.meta.ai/docs/models.md", READ_ON),
        Source("https://dev.meta.ai/docs/pricing-rate-limits", READ_ON),
    ],
    "openrouter": [
        Source("https://openrouter.ai/pricing", READ_ON),
        Source("https://openrouter.ai/openai/gpt-6-luna/llms.txt", READ_ON),
    ],
}

# The model that plans, when the person has not chosen one (model.cloud_model blank).
PLANNER_DEFAULTS: dict[str, str] = {
    "claude": "claude-opus-5-5",  # active, not sooner than 22 September 2027
    "openai": "gpt-6-luna",  # OpenAI's named successor to gpt-4o, which shuts down on 14 January 2027
    "gemini": "gemini-3.5-flash-lite",  # Google's named successor to gemini-2.5-flash (rerouted 1 December 2026)
    "openrouter": "openrouter/auto",
    "meta": "muse-spark-1.3",  # Meta's model "for new work"; muse-spark-1.1, the old default, is still listed
}

# The model that talks, when the person has not chosen one (conversation.<provider>_model blank). Never the
# planner's model: talk needs the first word fast.
TALK_DEFAULTS: dict[str, str] = {
    "claude": "claude-haiku-5-5",  # active, not sooner than 7 October 2027
    "openai": "gpt-6-luna",  # successor to gpt-4o-mini; sent with reasoning effort "none" (REASONING_OFF)
    "gemini": "gemini-2.5-flash-lite",  # "not deprecated", thinking off by default
    "openrouter": "openai/gpt-6-luna",  # on OpenRouter; sent with reasoning effort "none"
    "meta": "muse-spark-1.3",  # latency unmeasured
}

# Defaults that have answered through Heliox's own request shape in a paid run (the conversation eval, 12 and 13
# November, Claude key). Every other default was chosen from its provider's published list and is said to be so in
# Settings until a run with that provider's key has answered.
CHECKED_LIVE: frozenset[tuple[str, str]] = frozenset({("claude", "claude-haiku-5-5")})


def unchecked_note(provider: str) -> str:
    """Settings' help line for a default not yet checked against the live service ("" once it has been)."""
    who = PROVIDER_NAMES.get(provider, provider)
    return (
        f"Default chosen from {who}'s published list on {READ_ON}; not yet checked against the live service by "
        "Heliox's tests."
    )


# Talk models whose own page says reasoning can be switched off with effort "none" (GPT-6 Astra and GPT-6.1 Sol
# reject it, so this is per model, not per family). Talk sends it; the planner does not.
REASONING_OFF: frozenset[str] = frozenset({"gpt-6-luna", "openai/gpt-6-luna"})


def openai_reasoning(model: str) -> bool:
    """An OpenAI reasoning model, also through OpenRouter ("openai/gpt-6-luna"): the GPT-5 and GPT-6 families and the
    o-series, which take a reasoning effort (OpenAI's reasoning guide) and get no sampling temperature."""
    key = (model or "").strip().lower().removeprefix("openai/")
    return key.startswith(("gpt-5", "gpt-6")) or bool(re.match(r"^o\d", key))


# What the panel offers, newest first: current on the provider's pages that day, and priced in pilot.models.prices.
OFFERED: dict[str, list[str]] = {
    "claude": [
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-5-5",
        "claude-haiku-4-5",
    ],
    "openai": [
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ],
    "gemini": [
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    ],
    "openrouter": ["openrouter/auto", "openai/gpt-6-luna"],
    "meta": ["muse-spark-1.3", "muse-spark-1.2", "muse-spark-1.1", "muse-ember-1.0"],
}


@dataclass(frozen=True)
class Withdrawn:
    """A model the provider has deprecated (still answers until ``date``) or retired (requests fail)."""

    state: str  # "deprecated" | "retired" | "unlisted" (no longer on the provider's pages)
    successor: str  # the provider's named replacement, or the planner default when it names none
    date: str = ""  # shutdown or retirement date, ISO; "" when the page gives none


WITHDRAWN: dict[str, dict[str, Withdrawn]] = {
    "claude": {
        "claude-sonnet-4-5": Withdrawn("deprecated", "claude-sonnet-5-5", "2026-11-30"),
        "claude-opus-4-1": Withdrawn("retired", "claude-opus-4-8", "2026-08-05"),
        "claude-opus-4": Withdrawn("retired", "claude-opus-4-8", "2026-06-15"),
        "claude-sonnet-4": Withdrawn("retired", "claude-sonnet-5-5", "2026-06-15"),
        "claude-3-7-sonnet": Withdrawn("retired", "claude-sonnet-5-5", "2026-02-19"),
        "claude-3-5-haiku": Withdrawn("retired", "claude-haiku-4-5", "2026-02-19"),
        "claude-3-haiku": Withdrawn("retired", "claude-haiku-4-5", "2026-04-20"),
        "claude-3-opus": Withdrawn("retired", "claude-opus-4-8", "2026-01-05"),
        "claude-3-5-sonnet": Withdrawn("retired", "claude-sonnet-5-5", "2025-10-28"),
        "claude-3-sonnet": Withdrawn("retired", "claude-sonnet-5-5", "2025-07-21"),
    },
    "openai": {
        "gpt-4o-mini": Withdrawn("deprecated", "gpt-6-luna", "2027-01-14"),
        "gpt-4o": Withdrawn("deprecated", "gpt-6-luna", "2027-01-14"),
        "gpt-4.1-mini": Withdrawn("deprecated", "gpt-6-luna", "2027-01-14"),
        "gpt-4.1": Withdrawn("deprecated", "gpt-6-luna", "2027-01-14"),
        "gpt-5-mini": Withdrawn("deprecated", "gpt-6.1-terra", "2026-12-11"),
        "gpt-5": Withdrawn("deprecated", "gpt-6.1-sol", "2026-12-11"),
        "o3": Withdrawn("deprecated", "gpt-6.1-sol", "2026-12-11"),
        "o3-mini": Withdrawn("retired", "gpt-5.6-sol", "2026-10-23"),
        "o1": Withdrawn("retired", "gpt-5.6-sol", "2026-10-23"),
    },
    "gemini": {
        "gemini-2.5-flash": Withdrawn("deprecated", "gemini-3.5-flash-lite", "2026-12-01"),
        "gemini-2.5-pro": Withdrawn("deprecated", "gemini-3.8-flash", "2026-12-01"),
        "gemini-2.0-flash": Withdrawn("retired", "gemini-3.6-flash", "2026-06-01"),
        "gemini-1.5-pro": Withdrawn("unlisted", PLANNER_DEFAULTS["gemini"]),
        "gemini-1.5-flash": Withdrawn("unlisted", PLANNER_DEFAULTS["gemini"]),
    },
    "openrouter": {},
    "meta": {},
}

_SNAPSHOT = re.compile(r"^-(?:\d{8}|\d{4}-\d{2}-\d{2}|latest)$")
PROVIDER_NAMES = {
    "claude": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Google",
    "openrouter": "OpenRouter",
    "meta": "Meta",
}


def withdrawn(provider: str, model: str) -> Withdrawn | None:
    """The withdrawal of ``model`` (a dated snapshot or "-latest" alias counts as its model), or None."""
    key = (model or "").strip().lower()
    if provider == "openrouter":
        author, _, rest = key.partition("/")
        inner = {"openai": "openai", "anthropic": "claude", "google": "gemini", "meta": "meta"}.get(author)
        return withdrawn(inner, rest.replace(".", "-") if inner == "claude" else rest) if inner else None
    table = WITHDRAWN.get(provider, {})
    for known, found in table.items():
        # A dated snapshot or the "-latest" alias is its model: "gpt-4o-2024-08-06", "claude-3-opus-latest".
        # Nothing else may follow, so "gpt-4o-mini" is not "gpt-4o" and "claude-opus-4-8" is not "claude-opus-4".
        if key == known or (key.startswith(known) and _SNAPSHOT.match(key[len(known) :])):
            return found
    return None


def describe(provider: str, model: str) -> str:
    """The panel's line for a withdrawn model ("" for a current one)."""
    found = withdrawn(provider, model)
    if found is None:
        return ""
    who = PROVIDER_NAMES.get(provider, provider)
    if found.state == "unlisted":
        return f"{model} is no longer on {who}'s list of models. The current default is {found.successor}."
    when = f" on {found.date}" if found.date else ""
    if found.state == "deprecated":
        return f"{model} is deprecated by {who} and stops working{when}. {who} suggests {found.successor}."
    return f"{model} was retired by {who}{when}. {who} suggests {found.successor}."


def catalog() -> dict[str, object]:
    """What the Settings panel shows: per provider, the defaults, the offered models and the withdrawn ones."""
    return {
        "read_on": READ_ON,
        "providers": {
            provider: {
                "planner_default": PLANNER_DEFAULTS[provider],
                "talk_default": TALK_DEFAULTS[provider],
                "planner_note": ""
                if (provider, PLANNER_DEFAULTS[provider]) in CHECKED_LIVE
                else unchecked_note(provider),
                "talk_note": "" if (provider, TALK_DEFAULTS[provider]) in CHECKED_LIVE else unchecked_note(provider),
                "offered": list(OFFERED[provider]),
                "withdrawn": {
                    model: {
                        "state": w.state,
                        "successor": w.successor,
                        "date": w.date,
                        "says": describe(provider, model),
                    }
                    for model, w in WITHDRAWN[provider].items()
                },
                "sources": [{"url": s.url, "read_on": s.read_on} for s in SOURCES[provider]],
            }
            for provider in PLANNER_DEFAULTS
        },
    }
