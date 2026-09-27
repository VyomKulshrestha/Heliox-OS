"""Retired entrypoint for the historical Jev-output scope trainer."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Disabled: this historical trainer consumes TypeSafe Jev outputs. "
        "Do not use Jev outputs for training, tuning, comparison, or product development "
        "without written provider authorization."
    )


if __name__ == "__main__":
    main()
