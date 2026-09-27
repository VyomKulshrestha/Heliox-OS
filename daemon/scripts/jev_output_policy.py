"""Fail-closed policy for historical workflows that consume TypeSafe Jev outputs."""

from __future__ import annotations


def block_jev_output_use() -> None:
    raise SystemExit(
        "Disabled pending verified written TypeSafe authorization: this historical workflow "
        "uses Jev outputs for provider calls, training, tuning, or model comparisons."
    )
