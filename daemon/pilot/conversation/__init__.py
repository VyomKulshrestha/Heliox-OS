"""The fast conversational layer: owns the spoken turn (see docs/INTERACTION_MODEL.md).

Nothing here needs an audio, speech or model package at import.
"""

from pilot.conversation.adapters import layer_for
from pilot.conversation.layer import ConversationLayer, TurnResult
from pilot.conversation.router import Route, TurnKind, route

__all__ = ["ConversationLayer", "Route", "TurnKind", "TurnResult", "layer_for", "route"]
