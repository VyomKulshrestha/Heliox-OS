"""Parser and action-binding contracts for the bounded WebWorld sampler."""

from scripts.sample_webworld_transitions import bound_action, parse_trajectory


def test_parse_multistep_trajectory_uses_previous_observed_state() -> None:
    turns = [
        {"from": "human", "value": "Initial Page State:\nRootWebArea 'Start'\n  [17] button 'Open'\n"
         "First Action: 'click('17')'\n\nNext Page State:"},
        {"from": "gpt", "value": "RootWebArea 'Open'\n  [19] textbox 'Search'"},
        {"from": "human", "value": "Continue the trajectory.\nAction: 'fill('19', 'tea')'"
         "\nNext Page State:"},
        {"from": "gpt", "value": "RootWebArea 'Open'\n  [19] textbox 'Search' value='tea'"},
    ]

    first, second = parse_trajectory(turns)

    assert first == ("RootWebArea 'Start'\n  [17] button 'Open'", "click('17')",
                     "RootWebArea 'Open'\n  [19] textbox 'Search'")
    assert second[0] == first[2]
    assert second[1] == "fill('19', 'tea')"


def test_bind_action_uses_pre_action_node_not_a_numeric_id() -> None:
    tree = "RootWebArea 'Shop'\n  [7] button 'Checkout'\n  [8] textbox 'Search'"

    assert bound_action(tree, "click('7')") == "click target button 'Checkout'"
    assert bound_action(tree, "fill([8], 'tea')") == "fill target textbox 'Search' arguments 'tea'"
    assert bound_action(tree, "click('99')") == "click('99')"


def test_malformed_trajectory_is_rejected_without_partial_rows() -> None:
    turns = [{"from": "human", "value": "Initial Page State:\nTree\nFirst Action: 'click(1)'"
              "\nNext Page State:"}, {"from": "gpt", "value": "Next"},
             {"from": "human", "value": "no action marker"}, {"from": "gpt", "value": "Other"}]

    assert parse_trajectory(turns) == []
