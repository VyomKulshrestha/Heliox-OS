"""Independent retail bindings stay grouped and exclude item-specific calls."""

import numpy as np

from scripts.evaluate_world_tau_binding import build_cases, group_metrics


def test_same_status_alternatives_and_multiple_gold_bindings():
    orders = {
        key: {"order_id": key, "user_id": "customer", "status": status, "items": [{"name": name, "options": {}}]}
        for key, status, name in (
            ("p1", "pending", "Charger"),
            ("p2", "pending", "Tablet"),
            ("p3", "pending", "Mouse"),
            ("d1", "delivered", "Desk"),
        )
    }
    users = {"customer": {"orders": list(orders)}}
    tasks = [
        {
            "user_id": "customer",
            "instruction": "Cancel the charger and tablet orders.",
            "actions": [
                {"name": "cancel_pending_order", "arguments": {"order_id": "p1"}},
                {"name": "cancel_pending_order", "arguments": {"order_id": "p2"}},
                {"name": "modify_pending_order_items", "arguments": {"order_id": "p1", "item_ids": ["item1"]}},
            ],
        }
    ]
    cases, groups = build_cases(tasks, orders, users)
    assert [case.id for case in cases] == [
        "task0:cancel_pending_order:p1",
        "task0:cancel_pending_order:p2",
        "task0:cancel_pending_order:p3",
    ]
    assert [case.in_scope for case in cases] == [True, True, False]
    assert groups["task0:cancel_pending_order"]["implicit"]
    assert "d1" not in cases[0].request
    assert group_metrics(cases, groups, np.asarray([0.8, 0.7, 0.2]), 0.5) == {
        "groups": 1,
        "implicit_groups": 1,
        "correct_top_rank": 1,
        "correct_safe_groups": 1,
    }
