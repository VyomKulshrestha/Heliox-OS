"""Where the time goes in one scenario, stage by stage, on the wall clock and on the audio's own clock (pure).

Reply stages, from the person's last word: endpoint (until voice activity ends), recognition (until the words
arrive), decision (until Heliox has text to say, or until the planner is called), planner (planner call to text),
synthesis (text to the first sound). Barge-in stages, from the person's first word over Heliox: onset (until voice
activity starts), gate (until the barge-in is accepted), cut (until Heliox's voice has stopped).

The audio clock counts seconds of microphone audio fed to the listener (``audio_t`` on each event). On a machine
that keeps up, both clocks agree; where they differ, the listener fell behind real time and the wall clock is what a
person would have felt.
"""

from __future__ import annotations

from typing import Any

from . import timeline as tl

REPLY_STAGES = ("endpoint", "recognition", "decision", "planner", "synthesis")
BARGE_STAGES = ("onset", "gate", "cut")


def _first(events: list[tl.Event], kinds: tuple[str, ...], after: float, own_only: bool = False) -> tl.Event | None:
    for e in events:
        if e.event in kinds and e.t >= after and not (own_only and e.data.get("setup")):
            return e
    return None


def _span(a: tl.Event | None, b: tl.Event | None) -> dict[str, float | None]:
    if a is None or b is None:
        return {"wall": None, "audio": None}
    audio = None
    if "audio_t" in a.data and "audio_t" in b.data:
        audio = round(float(b.data["audio_t"]) - float(a.data["audio_t"]), 3)
    return {"wall": round(b.t - a.t, 3), "audio": audio}


def breakdown(events: list[tl.Event]) -> dict[str, dict[str, float | None]]:
    """Stage name -> {"wall": seconds, "audio": seconds}; a stage that did not happen is absent."""
    out: dict[str, dict[str, float | None]] = {}
    lasts = [e for e in events if e.event == tl.LAST_WORD]
    if lasts:
        last = lasts[-1]
        vad_end = _first(events, ("vad end",), last.t)
        words = _first(events, (tl.TRANSCRIPT,), last.t)
        planner = _first(events, (tl.PLANNER,), last.t)
        speaks = _first(events, (tl.SPEAKS,), last.t, own_only=True)
        sound = _first(events, (tl.AUDIO_STARTS,), last.t, own_only=True)
        stages = {
            "endpoint": _span(last, vad_end),
            "recognition": _span(vad_end, words),
            "decision": _span(words, planner if planner is not None else speaks),
            "planner": _span(planner, speaks) if planner is not None else _span(None, None),
            "synthesis": _span(speaks, sound),
        }
        out.update({k: v for k, v in stages.items() if v["wall"] is not None})
    barge = _first(events, (tl.BARGE_IN,), float("-inf"))
    firsts = [e for e in events if e.event == tl.FIRST_WORD and barge is not None and e.t <= barge.t]
    if barge is not None and firsts:
        first = firsts[-1]
        onset = _first(events, ("vad start",), first.t - 0.5)
        ended = _first(events, (tl.CUT_OFF, tl.DONE), barge.t)
        stages = {"onset": _span(first, onset), "gate": _span(onset, barge), "cut": _span(barge, ended)}
        out.update({k: v for k, v in stages.items() if v["wall"] is not None})
    return out


def label(cores: int | None, where: str = "GitHub runner") -> str:
    """How every number from a CI run is labelled."""
    return f"synthetic voice, {where} ({cores if cores else '?'} cores), CPU only"


def check_line(line: str, value: float | None, target: str, ok: bool | None, note: str) -> str:
    """One scorecard line as the CI log prints it: grep-able, the number and its provenance on one line."""
    shown = "-" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))
    verdict = {True: "pass", False: "FAIL", None: "owed"}[ok]
    return f"CHECK {line} {shown} target {target} {verdict} [{note}]"


def summarise(per_scenario: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Median of each stage over scenarios, per clock (pure)."""
    import statistics

    out: dict[str, dict[str, float | None]] = {}
    for stage in (*REPLY_STAGES, *BARGE_STAGES):
        walls = [s[stage]["wall"] for s in per_scenario if stage in s and s[stage]["wall"] is not None]
        audios = [s[stage]["audio"] for s in per_scenario if stage in s and s[stage]["audio"] is not None]
        if walls:
            out[stage] = {
                "wall": round(statistics.median(walls), 3),
                "audio": round(statistics.median(audios), 3) if audios else None,
                "n": len(walls),
            }
    return out
