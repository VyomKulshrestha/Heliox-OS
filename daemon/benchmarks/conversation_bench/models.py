"""Kinds of person by model, as stubbed delays (pure). No model is ever called by the bench.

Figures are approximate, from public third-party latency measurements of hosted models and from llama.cpp-class
local inference on a laptop CPU or integrated GPU; they were not measured by this bench. They set how long the
stubbed planner (and, once it exists, the fast spoken layer) takes before answering:

  cloud   fast talk model: first token 0.6 s, 150 tokens/s. Planner model: first token 1.5 s, 50 tokens/s.
  local   fast talk model (a 3B to 8B model in Ollama or similar): first token 1.5 s, 25 tokens/s.
          Planner (the same machine, an 8B model): first token 2.5 s, 12 tokens/s.
  none    no model configured: the planner answers at once that no model is set up.

Pass conditions by kind (written down here and in docs/CONVERSATION_BENCH.md):
  - Control words (stop, wait, no, a refused yes, a correction, undo) and local answers do not depend on the
    model: same result and same time limits under all three kinds.
  - A request that needs a plan: cloud and local are judged by the scorecard's time limits for model answers;
    with none, Heliox must say promptly (within the local-answer limit) and honestly that it needs a model for
    that and how to set one up, and must not claim an action.
  - A remark or question for the fast layer: with none, a short honest answer about what is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

PLAN_TOKENS = 400  # a typical short plan
REPLY_TOKENS = 40  # a spoken sentence or two
NO_MODEL_MESSAGE = (
    "No API key configured for the planner. Configure a healthy provider in Settings to run this request."
)


@dataclass(frozen=True)
class ModelKind:
    name: str
    talk_first_token_s: float
    talk_tokens_per_s: float
    plan_first_token_s: float
    plan_tokens_per_s: float
    available: bool = True

    def plan_seconds(self, tokens: int = PLAN_TOKENS) -> float:
        if not self.available:
            return 0.05
        return self.plan_first_token_s + tokens / self.plan_tokens_per_s

    def reply_seconds(self, tokens: int = REPLY_TOKENS) -> float:
        if not self.available:
            return 0.0
        return self.talk_first_token_s + tokens / self.talk_tokens_per_s


KINDS: dict[str, ModelKind] = {
    "cloud": ModelKind("cloud", 0.6, 150.0, 1.5, 50.0),
    "local": ModelKind("local", 1.5, 25.0, 2.5, 12.0),
    "none": ModelKind("none", 0.0, 1.0, 0.0, 1.0, available=False),
}
