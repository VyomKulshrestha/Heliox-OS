"""What Heliox never keeps about a person, whatever the route: one rule for every store.

The profile (``pilot.workflows.user_profile``: "call me", "remember that I ...", people, preferences), memory talk
("remember that ...", notes said in passing), getting to know: everything that stores something about the person goes
through ``UserProfile.add``, which refuses these, says why, and leaves out anything stored before the rule was there.
Never kept: anything that looks like a password, key or code; health (appointments with a doctor or dentist,
treatment, conditions, medication); money (amounts, a salary, the rent); other people's private details beyond a name
and a relation (someone's address, number, email, health or money).
"""

from __future__ import annotations

import re

_HEALTH = re.compile(
    r"\b(?:doctor|doctors|gp|dentist|dental|orthodontist|therapist|therapy|counsell?or|psychiatrist|psychologist"
    r"|physio|physiotherapy|hospital|clinic|surgery|operation|scan|mri|x-?ray|blood test|test results|diagnosis"
    r"|diagnosed|medication|medicine|pills?|prescription|dose|chemo|chemotherapy|radiotherapy|cancer|diabetes"
    r"|hiv|pregnan(?:t|cy)|ivf|miscarriage|depression|anxiety|adhd|autism|mental health|rehab|illness|sick"
    r"|vaccin(?:e|ation)|injection|std|sti|smear)\b",
    re.IGNORECASE,
)
_MONEY = re.compile(
    r"[$\u20ac\u00a3\u20b9\u00a5]\s?\d|\b\d[\d,.]*\s?(?:k|grand|dollars?|euros?|pounds?|quid|rupees?|bucks|usd"
    r"|eur|gbp|inr|lakhs?|crores?)\b|\b(?:salary|wage|wages|pay ?rise|bonus|rent|mortgage|loan|debt|overdraft"
    r"|bank balance|balance)\b",
    re.IGNORECASE,
)
_SECRET = re.compile(
    # A bare "key" or "code" is ordinary ("the spare key is under the pot", "I write code"): only their secret kinds.
    r"\b(?:password|passcode|passphrase|pin|(?:api|secret|private|access|encryption|licen[cs]e|product|wifi|wi-fi"
    r"|recovery|ssh) keys?|(?:security|access|door|alarm|gate|pin|verification|recovery|backup|one-time|2fa) codes?"
    r"|token|login|username|account number|card number|cvv|sort code"
    r"|iban|routing number|social security|ssn|passport number|licen[cs]e number)s?\b",
    re.IGNORECASE,
)
_PRIVATE_OF_OTHERS = re.compile(
    r"\b(?:\w+'s|his|her|their)\s+(?:home |new |old )?(?:address|flat|house|phone|number|mobile|email|e-mail"
    r"|birthday|date of birth|health|condition|salary|pay|income|bank|account|card|religion|divorce|password)\b",
    re.IGNORECASE,
)


def never_kept(text: str) -> str:
    """Why this may not be kept, in a few words ("it is about money"), or "" when it may."""
    if _SECRET.search(text):
        return "it looks like a password, key or code"
    if _PRIVATE_OF_OTHERS.search(text):
        return "it is someone else's private details"
    if _HEALTH.search(text):
        return "it is about health"
    if _MONEY.search(text):
        return "it is about money"
    return ""


# The kind, for saying what was dropped without saying it again: "one about money".
KINDS = {
    "it looks like a password, key or code": "a password or code",
    "it is someone else's private details": "someone else's private details",
    "it is about health": "health",
    "it is about money": "money",
}

__all__ = ["KINDS", "never_kept"]
