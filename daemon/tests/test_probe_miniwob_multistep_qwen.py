"""Multi-step probe keeps the policy view free of terminal labels."""

import numpy as np

from scripts import probe_miniwob_multistep_qwen as probe
from scripts.probe_miniwob_multistep_qwen import (
    available_choices,
    navigation_baseline,
    parse_action_index,
    rollout_prompt,
)


def test_action_index_accepts_only_bare_or_prefixed_integer() -> None:
    assert parse_action_index("1", 3) == 1
    assert parse_action_index("Index: 1", 3) == 1
    assert parse_action_index("index: 1.", 3) == 1
    assert parse_action_index("1 or 2", 3) is None
    assert parse_action_index("Index: 3", 3) is None


def test_ollama_model_must_be_installed_and_digest_pinned(monkeypatch) -> None:
    monkeypatch.setattr(
        probe,
        "ollama_json",
        lambda _path: {
            "models": [
                {"name": "qwen3:8b", "digest": "a" * 64},
            ]
        },
    )
    assert probe.ollama_model_digest("qwen3:8b") == "a" * 64
    try:
        probe.ollama_model_digest("missing:8b")
    except ValueError:
        pass
    else:
        raise AssertionError("An uninstalled model cannot be selected")


def test_ollama_choice_rejects_wrong_model_and_overlong_prompt(monkeypatch) -> None:
    captured = []

    def reply(_path, payload):
        captured.append(payload)
        return {"model": "qwen3:8b", "response": "Index: 1", "prompt_eval_count": 40}

    monkeypatch.setattr(probe, "ollama_json", reply)
    lengths = []
    assert probe.choose_ollama("Choose", [{"label": "A"}, {"label": "B"}], [], "qwen3:8b", 256, lengths) == "Index: 1"
    assert lengths == [40]
    assert captured[0]["stream"] is False and captured[0]["think"] is False
    monkeypatch.setattr(
        probe, "ollama_json", lambda *_args: {"model": "qwen3:8b", "response": "1", "prompt_eval_count": 257}
    )
    try:
        probe.choose_ollama("Choose", [], [], "qwen3:8b", 256, lengths)
    except ValueError:
        pass
    else:
        raise AssertionError("An overlong prompt must fail closed")


def test_delayed_terminal_reward_is_checked_after_ui_settles(monkeypatch) -> None:
    class Page:
        def __init__(self) -> None:
            self.mouse = self
            self.clicked = False
            self.waited = False

        def locator(self, _selector):
            return self

        def inner_html(self) -> str:
            return "after" if self.waited else "before"

        def click(self, _x: int, _y: int) -> None:
            self.clicked = True

        def evaluate(self, _script: str) -> dict:
            return {"done": self.clicked and self.waited, "reward": 1 if self.clicked and self.waited else None}

        def wait_for_timeout(self, _milliseconds: int) -> None:
            self.waited = True

    class Context:
        closed = False

        def close(self) -> None:
            self.closed = True

    page = Page()
    context = Context()
    monkeypatch.setattr(probe, "_reset", lambda *_args: (context, page, [], b"png", "dom"))
    monkeypatch.setattr(probe, "visual_feature", lambda *_args: np.zeros(256, dtype=np.float32))
    monkeypatch.setattr(probe, "goal_description", lambda *_args: 'Click "target".')
    monkeypatch.setattr(
        probe,
        "candidate_actions",
        lambda *_args: [
            {"label": "target", "tag": "button", "x": 10, "y": 20},
        ],
    )
    result = probe.run_episode(None, None, "task", "seed", 2, lambda *_args: "0")
    assert result["success"] and result["steps"] == 1
    assert result["stop_reason"] == "terminal" and result["invalid_output"] is None
    assert context.closed


def test_navigation_baseline_reveals_target_then_uses_exact_visible_label() -> None:
    goal = 'Click "Menu", then click "Save".'
    menu = {"label": "Menu", "tag": "button", "x": 10, "y": 20, "gesture": "click"}
    save = {"label": "Save", "tag": "li", "x": 30, "y": 40, "gesture": "click"}
    assert navigation_baseline(goal, [menu], []) == "0"
    assert navigation_baseline(goal, [menu, save], []) == "1"
    assert navigation_baseline(goal, [menu, save], [{"label": "Menu", "x": 10, "y": 20, "gesture": "click"}]) == "1"


def test_navigation_baseline_explores_unvisited_tabs() -> None:
    controls = [
        {"label": "1", "tag": "a", "x": 10, "y": 20, "gesture": "click"},
        {"label": "2", "tag": "a", "x": 30, "y": 20, "gesture": "click"},
    ]
    assert navigation_baseline('Find "target".', controls, []) == "0"
    assert (
        navigation_baseline('Find "target".', controls, [{"label": "1", "x": 10, "y": 20, "gesture": "click"}]) == "1"
    )


def test_navigation_baseline_explores_popup_with_hover() -> None:
    controls = [
        {"label": "Menu", "tag": "button", "x": 10, "y": 20, "gesture": "click"},
        {"label": "Playback", "tag": "li", "x": 20, "y": 40, "gesture": "hover", "has_popup": "true"},
    ]
    history = [{"label": "Menu", "x": 10, "y": 20, "gesture": "click"}]
    assert navigation_baseline('Find "Next".', controls, history) == "1"


def test_navigation_baseline_follows_named_controls_before_submit() -> None:
    controls = [
        {"label": "Submit", "tag": "button", "x": 10, "y": 20, "gesture": "click"},
        {"label": "orRz4", "tag": "input", "x": 30, "y": 40, "gesture": "click"},
        {"label": "dVSD4kc", "tag": "input", "x": 50, "y": 60, "gesture": "click"},
    ]
    goal = "Select orRz4, dVSD4kc and click Submit."
    assert navigation_baseline(goal, controls, []) == "1"
    assert (
        navigation_baseline(
            goal,
            controls,
            [
                {"label": "orRz4", "x": 30, "y": 40, "gesture": "click"},
            ],
        )
        == "2"
    )
    assert (
        navigation_baseline(
            goal,
            controls,
            [
                {"label": "orRz4", "x": 30, "y": 40, "gesture": "click"},
                {"label": "dVSD4kc", "x": 50, "y": 60, "gesture": "click"},
            ],
        )
        == "0"
    )


def test_navigation_baseline_matches_visible_icon_hint() -> None:
    controls = [
        {"label": "Save", "tag": "li", "x": 10, "y": 20, "gesture": "click", "icon_hint": "ui-icon ui-icon-disk"},
        {"label": "Stop", "tag": "li", "x": 30, "y": 20, "gesture": "click", "icon_hint": "ui-icon ui-icon-stop"},
    ]
    assert navigation_baseline("Click the [span ui-icon ui-icon-stop] icon.", controls, []) == "1"


def test_model_prompt_contains_visible_controls_not_rewards() -> None:
    prompt = rollout_prompt('Find "target".', [{"label": "target", "tag": "a"}], [])
    assert '"label":"target"' in prompt
    assert "indices 0 through 0" in prompt
    assert "reward" not in prompt


def test_available_choices_prefers_hover_and_blocks_repeated_action() -> None:
    controls = [
        {"label": "Menu", "x": 10, "y": 20},
        {"label": "Playback", "x": 30, "y": 40, "has_popup": "true"},
    ]
    history = [{"label": "Menu", "x": 10, "y": 20, "gesture": "click", "state_sha256": "previous"}]
    choices = available_choices(controls, history, "current")
    assert [(row["label"], row["gesture"]) for row in choices] == [
        ("Playback", "hover"),
        ("Playback", "click"),
    ]
