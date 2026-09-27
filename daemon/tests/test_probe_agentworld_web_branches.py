import json

from scripts.probe_agentworld_web_branches import (
    change_summary,
    complete_page,
    messages,
    parse_prompt,
    select_branches,
)


def test_prompt_parser_stops_at_action_fence() -> None:
    prompt = ("**Current State:**\npage\n**Action:**\n```text\n"
              "browser_click(element='Save')\n```\nPlease ignore this")
    assert parse_prompt(prompt) == ("page", "browser_click(element='Save')")


def test_change_summary_ignores_reference_only_changes() -> None:
    before = "- Page URL: https://example.test\n- button Save [ref=e1]"
    after = "- Page URL: https://example.test\n- button Save [ref=e2]"
    assert change_summary(before, after)[1] == 0
    assert not complete_page(after)


def test_branch_selection_requires_distinct_actions_and_visible_effects(tmp_path) -> None:
    state = ("### Page\n- Page URL: https://example.test\n- Page Title: Form\n"
             "### Snapshot\n- button Save [ref=e1]")
    def prompt(action: str) -> str:
        return f"**Current State:**\n{state}\n**Action:**\n```text\n{action}\n```"
    first = "\n".join(["### Page", "- Page URL: https://example.test/saved",
                       "### Snapshot",
                       "- Page Title: Saved", "- heading Saved", "- text: Complete"])
    second = "\n".join(["### Page", "- Page URL: https://example.test/help",
                        "### Snapshot",
                        "- Page Title: Help", "- heading Help", "- text: FAQ"])
    record = {"task": "web", "id": 1,
              "prompt": [prompt("browser_click('Save')"), prompt("browser_click('Help')")],
              "response": [first, second]}
    source = tmp_path / "web.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    branches, summary = select_branches(source, 1)
    assert summary["eligible_branch_groups"] == len(branches) == 1
    assert branches[0].first_summary != branches[0].second_summary
    assert messages(branches[0], False)[1] != messages(branches[0], True)[1]
