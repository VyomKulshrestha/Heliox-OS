"""The price tables the budget counts with (pilot.models.prices): every table names its page and the day it was read,
every default model has a current price, and a model with no current price is costed high, never at zero."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from pilot.conversation.fast_model import FAST_DEFAULTS
from pilot.models import catalog, prices
from pilot.models.budget_tracker import COST_PER_1K_TOKENS, BudgetTracker, claude_price_for
from pilot.models.cloud import DEFAULT_MODELS, PROVIDER_ENDPOINTS


def test_every_table_names_its_page_and_the_day_it_was_read():
    for provider in [*prices.TABLES, "openrouter", "claude"]:
        source = prices.SOURCES[provider]
        assert source.url.startswith("https://"), provider
        date.fromisoformat(source.read_on)


@pytest.mark.parametrize("provider", sorted(PROVIDER_ENDPOINTS))
def test_every_default_model_has_a_current_price(provider):
    for name in {DEFAULT_MODELS[provider], FAST_DEFAULTS[provider]}:
        if provider == "claude":
            assert claude_price_for(name)[1], name
        elif name == "openrouter/auto":
            # It routes to any model, so it is costed at the most a request can cost, and said to be.
            price, current = prices.price_for(provider, name)
            assert not current and price == prices.highest_known()
        else:
            assert prices.price_for(provider, name)[1], f"{provider}: no current price for {name}"


def test_gemini_is_priced_per_model_from_the_page():
    flash, current = prices.price_for("gemini", "gemini-2.5-flash")
    assert current and (flash.input, flash.output) == (0.30, 2.50)
    # 1,000 tokens in and 1,000 out: the old flat rate said $0.000375; the page says $0.0028.
    assert prices.cost_usd("gemini", "gemini-2.5-flash", 1000, 1000) == pytest.approx(0.0028)
    pro = prices.price_for("gemini", "gemini-2.5-pro")[0]
    assert pro.cost(300_000, 0) == pytest.approx(300_000 * 2.50 / 1e6)  # the long-prompt tier


def test_a_rise_the_page_announces_starts_on_its_day():
    price = prices.GEMINI["gemini-3.8-flash"]
    assert price.on(date(2026, 12, 31)).input == 0.75
    assert price.on(date(2027, 1, 1)).input == 1.50


def test_a_model_with_no_entry_is_costed_at_its_providers_highest_rate():
    unknown, current = prices.price_for("openai", "gpt-9-preview")
    assert not current
    assert (unknown.input, unknown.output) == (30.00, 180.00)
    assert prices.estimated_high("openai", "gpt-9-preview") == "estimated high: no current price for gpt-9-preview"
    assert prices.estimated_high("openai", "gpt-4o-mini") == ""


def test_a_dated_snapshot_has_its_models_price():
    assert prices.price_for("openai", "gpt-4o-2024-08-06") == (prices.OPENAI["gpt-4o"], True)


def test_a_stale_entry_is_costed_high(monkeypatch):
    stale = dict(prices.META)
    stale["muse-spark-1.3"] = replace(stale["muse-spark-1.3"], stale=True)
    monkeypatch.setitem(prices.TABLES, "meta", stale)
    price, current = prices.price_for("meta", "muse-spark-1.3")
    assert not current and (price.input, price.output) == (4.00, 20.00)


def test_a_provider_with_only_stale_figures_is_costed_at_the_highest_of_all(monkeypatch):
    monkeypatch.setitem(prices.TABLES, "meta", {k: replace(v, stale=True) for k, v in prices.META.items()})
    price, current = prices.price_for("meta", "muse-spark-1.1")
    assert not current and price == prices.highest_known()


def test_openrouter_is_priced_from_the_models_own_provider():
    assert prices.price_for("openrouter", "openai/gpt-4o-mini") == (prices.OPENAI["gpt-4o-mini"], True)
    assert prices.price_for("openrouter", "google/gemini-2.5-flash-lite")[1]
    claude, current = prices.price_for("openrouter", "anthropic/claude-haiku-5.5")
    assert current and (claude.input, claude.output) == (0.10, 0.50)
    assert not prices.price_for("openrouter", "deepseek/deepseek-v4-pro")[1]


def test_nothing_cloud_is_ever_free():
    for provider in PROVIDER_ENDPOINTS:
        assert COST_PER_1K_TOKENS[provider][0] > 0 and COST_PER_1K_TOKENS[provider][1] > 0
        assert prices.cost_usd(provider, "no-such-model", 1000, 1000) > 0
    assert prices.cost_usd("ollama", "llama3.1:8b", 1000, 1000) == 0


async def test_the_budget_view_names_a_model_costed_high(tmp_path):
    from pilot.config import ModelConfig

    tracker = BudgetTracker(ModelConfig(budget_enabled=True, budget_monthly_limit_usd=10.0), str(tmp_path / "b.db"))
    await tracker.initialize()
    try:
        await tracker.record_usage("openai", "gpt-4o-mini", 1000, 1000)
        await tracker.record_usage("openai", "gpt-9-preview", 1000, 1000)
        stats = await tracker.get_stats()
    finally:
        await tracker.close()
    assert stats["estimated_high"] == ["estimated high: no current price for gpt-9-preview"]
    assert stats["cost_usd"] == pytest.approx((150 + 600) / 1e6 + (30_000 + 180_000) / 1e6)


# ── the models offered and used by default (pilot.models.catalog) ──


@pytest.mark.parametrize("provider", sorted(PROVIDER_ENDPOINTS))
def test_every_default_model_is_offered_and_current(provider):
    for name in (catalog.PLANNER_DEFAULTS[provider], catalog.TALK_DEFAULTS[provider]):
        assert name in catalog.OFFERED[provider], f"{provider}: default {name} is not in the offered list"
        assert catalog.withdrawn(provider, name) is None, f"{provider}: default {name} is withdrawn"


@pytest.mark.parametrize("provider", sorted(PROVIDER_ENDPOINTS))
def test_every_offered_model_is_current_and_priced(provider):
    for name in catalog.OFFERED[provider]:
        assert catalog.withdrawn(provider, name) is None, name
        if provider == "claude":
            assert claude_price_for(name)[1], name
        elif name != "openrouter/auto":
            assert prices.price_for(provider, name)[1], f"{provider}: no current price for offered {name}"


def test_every_list_names_its_pages_and_the_day():
    for provider in PROVIDER_ENDPOINTS:
        assert catalog.SOURCES[provider], provider
        for source in catalog.SOURCES[provider]:
            assert source.url.startswith("https://")
            date.fromisoformat(source.read_on)


@pytest.mark.parametrize(
    ("provider", "saved", "state", "successor"),
    [
        ("claude", "claude-3-7-sonnet-latest", "retired", "claude-sonnet-5-5"),
        ("claude", "claude-3-opus-latest", "retired", "claude-opus-4-8"),
        ("openai", "gpt-4o-2024-08-06", "deprecated", "gpt-6-luna"),
        ("openai", "o1", "retired", "gpt-5.6-sol"),
        ("gemini", "gemini-2.5-flash", "deprecated", "gemini-3.5-flash-lite"),
        ("gemini", "gemini-1.5-pro", "unlisted", "gemini-3.5-flash-lite"),
        ("openrouter", "anthropic/claude-3.7-sonnet", "retired", "claude-sonnet-5-5"),
    ],
)
def test_a_saved_withdrawn_model_is_named_with_the_providers_successor(provider, saved, state, successor):
    found = catalog.withdrawn(provider, saved)
    assert found is not None and (found.state, found.successor) == (state, successor)
    assert successor in catalog.describe(provider, saved) and saved in catalog.describe(provider, saved)


@pytest.mark.parametrize(
    ("provider", "current"),
    [
        ("claude", "claude-opus-4-8"),
        ("claude", "claude-sonnet-4-6"),
        ("openai", "gpt-5.6-sol"),
        ("openai", "gpt-6-luna"),
    ],
)
def test_a_current_model_is_never_mistaken_for_a_withdrawn_one(provider, current):
    assert catalog.withdrawn(provider, current) is None


def test_a_default_not_yet_answered_live_says_so_in_settings():
    shown = catalog.catalog()["providers"]
    # Answered through Heliox's own request: Claude's in the paid conversation runs, Gemini's and OpenRouter's talk
    # defaults in the replay with real keys (14 November).
    for provider in ("claude", "gemini", "openrouter"):
        assert shown[provider]["talk_note"] == ""
    unchecked = [("openai", "planner_note"), ("openai", "talk_note"), ("meta", "planner_note"), ("meta", "talk_note")]
    unchecked += [("gemini", "planner_note"), ("openrouter", "planner_note")]
    for provider, field in unchecked:
        assert "not yet checked against the live service by Heliox's tests" in shown[provider][field]
        assert catalog.READ_ON in shown[provider][field]


def test_the_planner_sends_no_temperature_to_a_reasoning_model():
    assert catalog.openai_reasoning("gpt-6-luna") and catalog.openai_reasoning("openai/gpt-5.6-sol")
    assert catalog.openai_reasoning("o3") and not catalog.openai_reasoning("gpt-4o")
    assert not catalog.openai_reasoning("muse-spark-1.3")


def test_the_panel_reads_the_daemons_lists_and_keeps_none_of_its_own():
    ui = Path(__file__).parents[2] / "tauri-app" / "ui" / "src" / "lib"
    assert 'call("model_catalog")' in (ui / "stores" / "modelCatalog.ts").read_text(encoding="utf-8")
    for name in ("TalkModel.svelte", "PlannerModel.svelte", "SettingsPanel.svelte"):
        text = (ui / "components" / name).read_text(encoding="utf-8")
        if name != "SettingsPanel.svelte":
            assert "../stores/modelCatalog" in text, name
        # No model list of the panel's own: every id offered comes from pilot.models.catalog.
        for model_id in ("claude-3", "gpt-4o", "gemini-1.5", "gemini-2.0", '"o1"', "o3-mini"):
            assert model_id not in text, f"{name} names {model_id}"


def test_the_model_catalog_call_gives_the_panel_everything_it_shows():
    import asyncio

    from pilot.server import PilotServer

    answer = asyncio.run(PilotServer._handle_model_catalog(None, {}, None))  # type: ignore[arg-type]
    assert answer["status"] == "ok" and answer["read_on"] == catalog.READ_ON
    for models in answer["providers"].values():
        assert models["planner_default"] in models["offered"] and models["talk_default"] in models["offered"]
        for model, gone in models["withdrawn"].items():
            assert gone["says"] and model not in models["offered"]
