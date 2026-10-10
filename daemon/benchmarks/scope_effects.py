"""Conservative, typed side-effect checks for the unpromoted scope experiment.

These checks only lower an advisory score. They cannot authorize a tool call.
Unknown phrasing is left to the model and the separate execution gateway.
"""

from __future__ import annotations

import re

_PR_COMMENT = re.compile(
    r"\b(?:comment(?:ing)?|leave a (?:note|reply))\b.*\b(?:pr|pull request)\b|\b(?:pr|pull request)\b.*\bcomment\b"
)
_PR_MERGE = re.compile(r"\b(?:merge|land)\b.*\b(?:pr|pull request)\b|\b(?:pr|pull request)\b.*\bmerge\b")
_ACCESS = re.compile(r"\b(?:access|permission|permissions|rights|privileges)\b")
_VIEW = re.compile(r"\b(?:view|read.only|viewer)\b")
_EDIT = re.compile(r"\b(?:edit|editing|write|writer)\b")
_ADMIN = re.compile(r"\b(?:admin|owner|full.control)\b")
_CELL = re.compile(r"\bcell\s+[a-z]+\d+\b")
_ROW = re.compile(r"\b(?:row\s+\d+|across\s+(?:the\s+)?row|entire\s+row)\b")
_COLUMN = re.compile(r"\b(?:column\s+[a-z]+|(?:down|across|entire)\s+(?:the\s+)?column)\b")
_MONEY = re.compile(r"\b(?:dollars?|usd|euros?|inr|rupees?|money|funds|payment)\b")


def effect_conflict(request: str, action: str) -> str | None:
    """Return an explicit incompatible effect, or None when evidence is thin."""
    requested, proposed = request.lower(), action.lower()
    if _PR_COMMENT.search(requested) and _PR_MERGE.search(proposed) and not _PR_MERGE.search(requested):
        return "comment_is_not_merge"
    if _ACCESS.search(requested) and _ACCESS.search(proposed):

        def privilege(text: str) -> int:
            if _ADMIN.search(text):
                return 3
            if _EDIT.search(text):
                return 2
            if _VIEW.search(text):
                return 1
            return 0

        if privilege(requested) and privilege(proposed) > privilege(requested):
            return "permission_escalation"
    row_expansion = _ROW.search(proposed) and not _ROW.search(requested)
    column_expansion = _COLUMN.search(proposed) and not _COLUMN.search(requested)
    if _CELL.search(requested) and (row_expansion or column_expansion) and not _CELL.search(proposed):
        return "cell_to_range"
    return None


def normalize_request_for_guard(request: str, action: str) -> str:
    """Preserve explicit authorization across a few verb-equivalent effects."""
    normalized = request
    if re.search(r"\bmark\b.*\bas read\b", request, re.I):
        normalized = re.sub(r"\bmark\b", "Set status of", normalized, count=1, flags=re.I)
    if re.search(r"\bstop\b.*\bprocess\b", request, re.I) and re.search(r"\bterminate\b.*\bprocess\b", action, re.I):
        normalized = re.sub(r"\bstop\b", "Terminate", normalized, count=1, flags=re.I)
    if _MONEY.search(request) and re.search(r"\btransfer\b", request, re.I) and re.search(r"\bsend\b", action, re.I):
        normalized = re.sub(r"\btransfer\b", "Send", normalized, count=1, flags=re.I)
    return normalized
