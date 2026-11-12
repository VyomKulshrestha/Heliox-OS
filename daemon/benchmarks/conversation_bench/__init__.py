"""Conversation Bench: the program brief's scorecard as a repeatable measurement of Heliox's spoken side.

Scenario files (``scenarios/*.json``) hold only text and expectations; the audio they name is resolved from a
local, untracked folder (``--audio`` or ``CONVERSATION_BENCH_AUDIO``). The pure modules (``scenarios``,
``timeline``, ``scoring``, ``rubric``, ``targets``, ``corpus``, ``audio``) import nothing beyond the standard
library, so their tests run in CI without audio packages or models. ``driver``, ``synth`` and ``soak`` import
the daemon and the audio stack only when a run starts.
"""
