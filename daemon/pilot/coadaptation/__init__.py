"""Assistance co-adaptation for Heliox's own selection surfaces (opt-in, presentation-only).

The co-adaptation model predicts how one person selects targets and menu items inside surfaces Heliox renders,
and changes only how those targets are presented and what a selection needs from the person (a larger target, a
confirm step, a clarification prompt). It never approves, clicks or runs anything, never feeds the risk gate or
the approval dialog, and never grants or widens authority. Commit rules decide when the person's own input is
accepted as a selection; they never decide what runs. See docs/COADAPTATION.md.

This package imports nothing at import time so that ``import pilot.coadaptation`` stays light. The pure SDK core
lives in ``pilot.coadaptation.core`` (standard library and numpy only).
"""

__all__ = []
