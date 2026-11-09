"""Heliox Live: the always-on interaction layer.

Modeled on the split Thinking Machines describes for its interaction models: a fast layer that stays present
with the person (perceiving, deciding every micro-turn whether to stay quiet, show something or speak) and a
slower background model that does the heavy work. Here the fast layer is ``LiveCompanion`` and the background
model is Heliox's existing planner and executor, reached only through the normal approval path.

Everything the live layer perceives stays on this computer and in memory. It never acts on its own: it can
point something out, explain it, copy a fix, or hand a request to the planner when the person asks.
"""
