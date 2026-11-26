"""The last check on a task's reply before it leaves the daemon: it always carries a sentence to show and one to say.

Every way a plan can end (done, failed, cancelled, denied, a repair round that gave up, a repair round that answered
in prose, an error before planning) returns a dict with ``status`` and usually ``message``. ``with_reply`` makes sure
of both forms: ``message`` is never empty and never raw JSON, and ``spoken`` follows the speech rules
(pilot.reply_rules). A reply that has nothing else says the failed step's own sentence, at the least, or a fixed
sentence for its status.
"""

from __future__ import annotations

from typing import Any

from pilot.reply_rules import ear_faults, for_ear, for_screen, looks_like_json, screen_faults

STATUS_SENTENCES = {
    "success": "Done.",
    "cancelled": "Okay, I stopped.",
    "interrupted": "That was interrupted before it finished.",
    "denied": "Okay, I didn't do it.",
    "partial_failure": "That didn't work.",
    "failed": "That didn't work.",
    "error": "Something went wrong, so nothing more was done.",
    "blocked_by_critic": "The safety review stopped that plan before anything ran.",
    "blocked_by_companion": "I stopped that plan before anything ran: it didn't match what you asked.",
    "busy": "I'm still busy with another task.",
}
NO_STATUS = "That ended without a result I can tell you about."


def _failed_step(response: dict[str, Any]) -> tuple[str, str]:
    """(shown, spoken) of the first failed step in the reply's results, or ("", "")."""
    from pilot.actions import ActionResult
    from pilot.failure_sentences import failure_sentence

    for raw in response.get("results") or []:
        if not isinstance(raw, dict) or raw.get("success") or not raw.get("error"):
            continue
        try:
            result = ActionResult.model_validate(raw)
        except Exception:  # noqa: BLE001 - a result that cannot be read back gives no sentence
            continue
        return failure_sentence(result.action, str(result.error))
    return "", ""


def with_reply(response: Any) -> Any:
    """``response`` with a non-empty ``message`` to show and a ``spoken`` sentence; anything not a dict unchanged."""
    if not isinstance(response, dict):
        return response
    from pilot.agents.unconfirmed import spoken_from_window_line

    status = str(response.get("status") or "")
    fixed = STATUS_SENTENCES.get(status, NO_STATUS)
    step_shown, step_spoken = _failed_step(response) if status != "success" else ("", "")
    message = str(response.get("message") or "").strip()
    explanation = str(response.get("explanation") or "").strip()
    if not message or looks_like_json(message):
        usable = explanation if explanation and not looks_like_json(explanation) else ""
        message = usable or step_shown or fixed
    elif screen_faults(message):
        message = for_screen(message)
    spoken = str(response.get("spoken") or "").strip()
    if not spoken or ear_faults(spoken):
        rest, phrase = spoken_from_window_line(message)
        spoken = step_spoken or for_ear(f"{rest} {phrase}." if phrase else rest) or for_ear(explanation) or fixed
    return {**response, "message": message, "spoken": spoken}
