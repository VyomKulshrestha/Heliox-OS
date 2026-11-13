"""Suggestions, offers and Live, turned on, off, down or back up by voice: one table, one sentence for each.

Every phrase family names the setting it saves and the sentence Heliox says, and a request to stop something is
answered by a sentence that says it stopped. Phrases arrive normalized the way the server's spoken settings are
(casefolded, punctuation as spaces: "don't" is "don t"). "Stay quiet unless I ask" and "stop interrupting me" are
the profile's (pilot.workflows.user_profile.QUIET_REQUEST): they hold offers, suggestions and Live's non-urgent
catches without changing Settings, and "you can speak up again" undoes them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_PLEASE = r"^(?:please )?(?:heliox )?"
_END = r"(?: please)?$"
_AGAIN = r"(?: again| any more| anymore)?"
_THINGS = r"(?: things| stuff| anything)?"

VOICE_SUGGESTIONS_SETTING = re.compile(
    _PLEASE + r"(?:(?P<off>turn off (?:proactive |the )?suggestions"
    r"|(?:stop|quit) (?:suggesting|making suggestions)" + _THINGS + r"(?: to me)?"
    r"|(?:don t|do not|never) (?:suggest|make suggestions)" + _THINGS + _AGAIN + r"|no more suggestions)"
    r"|(?P<on>turn on (?:proactive |the )?suggestions"
    r"|(?:you can )?(?:start )?(?:suggesting|suggest|making suggestions|make suggestions)(?: things| stuff)? again"
    r"|(?:you can )?start (?:suggesting|making suggestions)(?: things| stuff)?))" + _END
)
VOICE_OFFERS_SETTING = re.compile(
    _PLEASE
    + r"(?:(?P<off>turn off (?:the )?offers|no more offers|(?:stop|quit) offering(?: me)?"
    + _THINGS
    + r"|(?:don t|do not|never) offer(?: me)? (?:things|stuff|anything)(?: else)?"
    + _AGAIN
    + r")"
    r"|(?P<on>turn on (?:the )?offers|(?:you can )?(?:start )?(?:offering|offer)(?: me)?(?: things| stuff)? again"
    r"|(?:you can )?start offering(?: me)?(?: things| stuff)?))" + _END
)
_LIVE_NAMES = r"(?:the |my )?(?:live companion|live mode|always on mode|live)"
VOICE_LIVE_SETTING = re.compile(
    _PLEASE + r"(?:(?P<on>turn on " + _LIVE_NAMES + r"|be my live companion|watch over me|start correcting me"
    r"|correct me as i work|(?:you can )?correct me again)"
    r"|(?P<off>turn off " + _LIVE_NAMES + r"|(?:stop|quit) (?:correcting me|watching me|watching my screen)"
    r"|(?:don t|do not|never) (?:correct me|watch me|watch my screen)" + _AGAIN + r"|no more corrections))" + _END
)
_PROFILE = r"my (?:profile|details|personal details)(?: to (?:the |a )?(?:cloud|model|cloud model|internet))?"
VOICE_PROFILE_SETTING = re.compile(
    _PLEASE
    + r"(?:(?P<off>(?:don t|do not|never) send "
    + _PROFILE
    + _AGAIN
    + r"|(?:stop|quit) sending "
    + _PROFILE
    + r"|keep my (?:profile|details) on (?:this|my) computer|turn off sending "
    + _PROFILE
    + r")"
    r"|(?P<on>(?:you can )?send "
    + _PROFILE
    + r" again|(?:you can )?start sending "
    + _PROFILE
    + r"|turn on sending "
    + _PROFILE
    + r"))"
    + _END
)
VOICE_LIVE_LEVEL_SETTING = re.compile(
    _PLEASE + r"(?:(?P<less>(?:correct|interrupt) me less(?: often)?|(?:fewer|less) (?:corrections|interruptions))"
    r"|(?P<more>(?:you can )?correct me more(?: often)?|more corrections))" + _END
)

SUGGESTIONS_OFF = 'Okay, no more suggestions or offers until you ask. Say "you can suggest things again" to undo it.'
SUGGESTIONS_ON = "Okay, I'll suggest things now and then again, held while you focus."
OFFERS_OFF = 'Okay, no more offers. Say "you can offer things again" to undo it.'
OFFERS_ON = "Okay, I'll offer what you usually do now and then, at a quiet moment."
LIVE_ON = "Okay, I'm with you. I'll point out slips as you work, at your pauses, and stay quiet while you talk."
# Turning on a capture by voice when it needs approval (pilot.security.guard_changes): voice cannot widen a
# guard, so the question goes to the window. Nothing else is said until the change has applied.
APPROVE_IN_WINDOW = "I've put that in Heliox's window for you to approve."
OPEN_TO_APPROVE = "Open Heliox and approve it there."
LIVE_OFF = 'Okay, Live is off: no more corrections. Say "be my live companion" to turn it back on.'
LIVE_LESS = "Okay, fewer corrections: I'll only speak up about risky things and show the rest quietly."
LIVE_LESS_WHILE_OFF = "Okay. Live is off now; when you turn it on, it will only speak up about risky things."
LIVE_MORE = "Okay, back to the usual: I'll point out slips at your pauses."
PROFILE_OFF = (
    "Okay, your name, your answer style and what you told me about yourself stay on this computer now. "
    'Say "you can send my profile again" to undo it.'
)
PROFILE_ON = "Okay, a cloud model gets your name, your answer style and what bears on your question again."
LIVE_MORE_WHILE_OFF = "Okay. Live is off now; when you turn it on, it will point out slips at your pauses."


@dataclass(frozen=True)
class SpokenSetting:
    section: str
    values: dict[str, Any]
    reply: str


def _live_enabled(config: Any) -> bool:
    return bool(getattr(getattr(config, "live", None), "enabled", False))


# (pattern, group, section, values, reply). A reply may depend on the config as it was before the change.
_TABLE: tuple[tuple[re.Pattern[str], str, str, dict[str, Any], str | Callable[[Any], str]], ...] = (
    (VOICE_SUGGESTIONS_SETTING, "off", "suggestions", {"enabled": False, "offers": "off"}, SUGGESTIONS_OFF),
    (VOICE_SUGGESTIONS_SETTING, "on", "suggestions", {"enabled": True, "offers": "on"}, SUGGESTIONS_ON),
    (VOICE_OFFERS_SETTING, "off", "suggestions", {"offers": "off"}, OFFERS_OFF),
    (VOICE_OFFERS_SETTING, "on", "suggestions", {"offers": "on"}, OFFERS_ON),
    (VOICE_PROFILE_SETTING, "off", "conversation", {"share_profile": "off"}, PROFILE_OFF),
    (VOICE_PROFILE_SETTING, "on", "conversation", {"share_profile": "auto"}, PROFILE_ON),
    (VOICE_LIVE_SETTING, "on", "live", {"enabled": True}, LIVE_ON),
    (VOICE_LIVE_SETTING, "off", "live", {"enabled": False}, LIVE_OFF),
    (
        VOICE_LIVE_LEVEL_SETTING,
        "less",
        "live",
        {"level": "quiet"},
        lambda config: LIVE_LESS if _live_enabled(config) else LIVE_LESS_WHILE_OFF,
    ),
    (
        VOICE_LIVE_LEVEL_SETTING,
        "more",
        "live",
        {"level": "balanced"},
        lambda config: LIVE_MORE if _live_enabled(config) else LIVE_MORE_WHILE_OFF,
    ),
)
PATTERNS = (
    VOICE_SUGGESTIONS_SETTING,
    VOICE_OFFERS_SETTING,
    VOICE_LIVE_SETTING,
    VOICE_LIVE_LEVEL_SETTING,
    VOICE_PROFILE_SETTING,
)


def spoken_setting(normalized: str, config: Any) -> SpokenSetting | None:
    """The setting a spoken phrase changes and what Heliox says, or None when it is not one of these."""
    for pattern, group, section, values, reply in _TABLE:
        found = pattern.match(normalized)
        if found and found.group(group) is not None:
            return SpokenSetting(section, dict(values), reply(config) if callable(reply) else reply)
    return None


__all__ = [
    "PATTERNS",
    "VOICE_LIVE_LEVEL_SETTING",
    "VOICE_LIVE_SETTING",
    "VOICE_OFFERS_SETTING",
    "VOICE_SUGGESTIONS_SETTING",
    "SpokenSetting",
    "spoken_setting",
]
