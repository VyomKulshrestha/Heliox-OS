"""What a cloud model's tokens cost, per provider and model, each table with the page it was read from and when.

Every figure here was read from the provider's own pricing page on the date beside its table, never written from
memory. When a page could not be read, the old figure stays but is marked stale. A model with no entry, or a stale
entry, is costed at the highest current rate in its provider's table (or, when that provider has no current entry,
the highest current rate of any provider): never at zero and never at a guess. The budget view names each such model
("estimated high: no current price for <model>").

Rates are USD per million tokens for the standard tier: no batch, flex or priority discount, and no cache-read
discount outside Claude (cached input is costed as ordinary input, which overstates). Claude's own table, with its
cache rates, is ``pilot.models.budget_tracker.CLAUDE_MODEL_PRICES``.

To re-read a table, fetch its page, update the figures and the date. A scheduled price rise on the page goes in
``rises``, with the day it starts, so the budget follows the page without a release.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime


@dataclass(frozen=True)
class Price:
    """USD per million tokens for one model."""

    input: float
    output: float
    # Beyond this many prompt tokens a higher (input, output) rate applies (0: no such tier).
    long_prompt_threshold: int = 0
    long_input: float = 0.0
    long_output: float = 0.0
    # A rise the page announces: (first day, input, output).
    rises: tuple[str, float, float] | None = None
    # The page could not be read when the table was last checked: costed at the provider's highest rate instead.
    stale: bool = False

    def on(self, day: date) -> Price:
        """The price in force on ``day``."""
        if self.rises is None or day < date.fromisoformat(self.rises[0]):
            return self
        return Price(self.rises[1], self.rises[2], self.long_prompt_threshold, self.long_input, self.long_output)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = self.input, self.output
        if self.long_prompt_threshold and input_tokens > self.long_prompt_threshold:
            in_rate, out_rate = self.long_input, self.long_output
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0


@dataclass(frozen=True)
class Source:
    url: str
    read_on: str  # ISO date the figures were read


SOURCES: dict[str, Source] = {
    "openai": Source("https://developers.openai.com/api/docs/pricing", "2026-11-13"),
    "gemini": Source("https://aistudio.google.com/docs/pricing", "2026-11-13"),
    "meta": Source("https://dev.meta.ai/docs/pricing-rate-limits", "2026-11-13"),
    # OpenRouter bills each model at its provider's list price with no markup (its fee is taken when credit is
    # bought, never per request); a model is costed from its provider's table below.
    "openrouter": Source("https://openrouter.ai/pricing", "2026-11-13"),
    "claude": Source("https://platform.claude.com/docs/en/about-claude/pricing", "2026-11-13"),
}

# Standard tier, the short-context price (the page's tier up to 272K input tokens; Heliox's prompts are far below it).
OPENAI: dict[str, Price] = {
    "gpt-6-astra": Price(10.00, 50.00),
    "gpt-6.1-sol": Price(2.00, 10.00),
    "gpt-6.1-terra": Price(1.00, 5.00),
    "gpt-6-luna": Price(0.10, 0.50),
    "gpt-6-sol": Price(2.00, 10.00),
    "gpt-5.6-sol": Price(4.00, 20.00),  # promotional until 30 November 2026 on the page; re-read then
    "gpt-5.6-terra": Price(2.00, 12.00),
    "gpt-5.6-luna": Price(0.20, 1.20),
    "gpt-5.5": Price(5.00, 30.00),
    "gpt-5.5-pro": Price(30.00, 180.00),
    "gpt-5.4": Price(2.50, 15.00),
    "gpt-5.4-pro": Price(30.00, 180.00),
    "gpt-5.4-mini": Price(0.75, 4.50),
    "gpt-5.4-nano": Price(0.20, 1.25),
    "gpt-5.2": Price(1.75, 14.00),
    "gpt-5.2-pro": Price(21.00, 168.00),
    "gpt-5.1": Price(1.25, 10.00),
    "gpt-5": Price(1.25, 10.00),
    "gpt-5-mini": Price(0.25, 2.00),
    "gpt-5-nano": Price(0.05, 0.40),
    "gpt-5-pro": Price(15.00, 120.00),
    "gpt-4.1": Price(2.00, 8.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "o3-pro": Price(20.00, 80.00),
    "o3": Price(2.00, 8.00),
    "chat-latest": Price(5.00, 30.00),
    "gpt-5.3-codex": Price(1.75, 14.00),
}

# Paid tier, standard (not batch or flex), text input. gemini-2.5-flash and gemini-2.5-pro are marked deprecated on
# the page and route to newer models on 1 December 2026.
GEMINI: dict[str, Price] = {
    "gemini-4-carbon": Price(2.00, 10.00, rises=("2026-12-01", 4.00, 20.00)),
    "gemini-4-argon": Price(2.00, 10.00, rises=("2026-12-01", 4.00, 20.00)),
    "gemini-4-hydrogen": Price(1.50, 7.50),
    "gemini-3.8-flash": Price(0.75, 3.75, rises=("2027-01-01", 1.50, 7.50)),
    "gemini-3.6-flash": Price(0.75, 3.75, rises=("2027-01-01", 1.50, 7.50)),
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
    "gemini-3.1-flash-lite": Price(0.25, 1.50),
    "gemini-3.1-pro-preview": Price(2.00, 12.00, 200_000, 4.00, 18.00),
    "gemini-3-flash-preview": Price(0.50, 3.00),
    "gemini-2.5-pro": Price(1.25, 10.00, 200_000, 2.50, 15.00),
    "gemini-2.5-flash": Price(0.30, 2.50),
    "gemini-2.5-flash-lite": Price(0.10, 0.40),
}

# Standard mode (Contributor-mode variants are cheaper and not listed: they cost at most this).
META: dict[str, Price] = {
    "muse-wonder-1.0": Price(4.00, 20.00),
    "muse-spark-1.3": Price(1.25, 4.25),
    "muse-spark-1.2": Price(1.25, 4.25),
    "muse-spark-1.1": Price(1.25, 4.25),
    # Over 200K input tokens the whole request is billed at the higher rate, as Price.cost does.
    "muse-ember-1.0": Price(0.10, 0.50, 200_000, 0.30, 1.50),
}

TABLES: dict[str, dict[str, Price]] = {"openai": OPENAI, "gemini": GEMINI, "meta": META}
# Which table an OpenRouter model id ("openai/gpt-4o-mini") is priced from, by its author.
OPENROUTER_AUTHORS = {"openai": "openai", "google": "gemini", "anthropic": "claude", "meta": "meta"}
FREE_PROVIDERS = frozenset({"ollama", "local"})
_SNAPSHOT = re.compile(r"^-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{3}|latest)$")


def _today() -> date:
    return datetime.now(UTC).date()


def _highest(prices: list[Price]) -> Price | None:
    current = [p for p in prices if not p.stale]
    return max(current, key=lambda p: (p.output, p.input)) if current else None


def highest_known(day: date | None = None) -> Price:
    """The most expensive current rate of any provider: the cost of a model nothing is known about."""
    from pilot.models.budget_tracker import CLAUDE_MODEL_PRICES

    day = day or _today()
    every = [p.on(day) for table in TABLES.values() for p in table.values()]
    every += [Price(c.input, c.output) for c in CLAUDE_MODEL_PRICES.values()]
    found = _highest(every)
    assert found is not None
    return found


def _lookup(table: dict[str, Price], model: str) -> Price | None:
    key = (model or "").strip().lower()
    if key in table:
        return table[key]
    for known in sorted(table, key=len, reverse=True):
        if key.startswith(known) and _SNAPSHOT.match(key[len(known) :]):
            return table[known]
    return None


def price_for(provider: str, model: str, day: date | None = None) -> tuple[Price, bool]:
    """(price, current) for one call: ``current`` is False when the model has no entry or a stale one, and the
    price is then the provider's highest current rate (or any provider's, when its table has none)."""
    day = day or _today()
    if provider in FREE_PROVIDERS:
        return Price(0.0, 0.0), True
    if provider == "openrouter":
        author, _, rest = (model or "").strip().lower().partition("/")
        target = OPENROUTER_AUTHORS.get(author)
        if target == "claude":
            from pilot.models.budget_tracker import claude_price_for

            claude, known = claude_price_for(rest.replace(".", "-"))
            if known:
                return Price(claude.input, claude.output), True
        elif target is not None:
            found = _lookup(TABLES[target], rest)
            if found is not None and not found.stale:
                return found.on(day), True
        # openrouter/auto, or any model it may route to: the most a request can cost.
        return highest_known(day), False
    table = TABLES.get(provider)
    if table is None:
        return highest_known(day), False
    found = _lookup(table, model)
    if found is not None and not found.stale:
        return found.on(day), True
    fallback = _highest([p.on(day) for p in table.values()])
    return (fallback or highest_known(day)), False


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    price, _ = price_for(provider, model)
    return price.cost(input_tokens, output_tokens)


def estimated_high(provider: str, model: str) -> str:
    """The budget view's words for a call costed without a current price ("" when the price is current)."""
    if provider in FREE_PROVIDERS or provider == "claude" or provider.startswith("subscription:"):
        if provider == "claude":
            from pilot.models.budget_tracker import claude_price_for

            if not claude_price_for(model)[1]:
                return f"estimated high: no current price for {model or 'the default model'}"
        return ""
    _, current = price_for(provider, model)
    return "" if current else f"estimated high: no current price for {model or 'the default model'}"
