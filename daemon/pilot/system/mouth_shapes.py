"""Mouth shapes for a drawn face: phoneme timing to a ready timeline, and a loudness envelope when there is none.

A face pack (docs/FACE_PACKS.md) has one image per shape in SHAPES. The daemon sends the UI a timeline of shapes for
each spoken piece, so the UI only has to draw: it never parses phonemes. The mapping reads the phoneme symbols alone
(IPA, plus the few capital letters Kokoro's readers use for English diphthongs), so a voice in any language whose
reader writes IPA gets the same treatment, and a symbol nobody listed gets a neutral half-open mouth.

Pure functions only: numpy is imported inside ``envelope`` and nothing here needs an audio package.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

# The fifteen shapes (fourteen and silence), the set most lip-sync tools use, so a pack made for one of them fits.
SHAPES: tuple[str, ...] = ("sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U")

_GROUPS: dict[str, str] = {
    # Lips together.
    "PP": "pbmɸβ",
    # Lower lip to the upper teeth.
    "FF": "fvʋ",
    # Tongue between the teeth.
    "TH": "θð",
    # Tongue tip behind the teeth.
    "DD": "tdɾʈɖʦʣTlɫɭʎɬɮ",
    # Back of the tongue; the mouth stays as the vowel around it left it.
    "kk": "kgɡqɢxɣχʁʀhɦʔŋɴcɟɰ",
    # Lips pushed forward, teeth close.
    "CH": "ʃʒʧʤɕʑʨʥʂʐçʝ",
    # Teeth close, lips spread.
    "SS": "szSɧ",
    # Nasals with the tongue up front.
    "nn": "nɲɳ",
    # R sounds.
    "RR": "ɹrɻɽʁɚɝꭧ",
    # Open vowels (and the diphthongs that start open: I = aɪ, W = aʊ).
    "aa": "aɑɐæäɒIW",
    # Mid vowels (A = eɪ starts here).
    "E": "eɛəᵊɜʌɘɵøœɤA",
    # Close front vowels and glides.
    "I": "iɪɨyʏjɥᵻ",
    # Rounded mid vowels (O = oʊ, Q = əʊ, Y = ɔɪ).
    "O": "oɔOQY",
    # Rounded close vowels and w.
    "U": "uʊɯʉwᵝ",
}
SHAPE_OF: dict[str, str] = {symbol: shape for shape, symbols in _GROUPS.items() for symbol in symbols}

# Marks that change the sound before them (length, nasal, aspiration, palatal) or after them (stress): they never
# move the mouth by themselves, so their time goes to the sound they belong to.
_JOINS_BEFORE = frozenset("ːˑ̃ʰʲʷˀ̩̯")
_JOINS_AFTER = frozenset("ˈˌ")
_PAUSES = frozenset('.,;:!?…—–-"“”«»()¡¿→↓↗↘')
UNKNOWN_SHAPE = "E"
# A gap between words shorter than this keeps the mouth as it was: people do not close their lips between words.
WORD_GAP_SECONDS = 0.08
# Shapes shorter than this join the one before: at 30 frames a second nobody sees them, and the cross-fade covers it.
MIN_SHAPE_SECONDS = 0.02


def shape_of(symbol: str) -> str:
    """The mouth shape for one phoneme symbol (SHAPES); a pause or a space is silence."""
    if not symbol or symbol.isspace() or symbol in _PAUSES:
        return "sil"
    return SHAPE_OF.get(symbol, UNKNOWN_SHAPE)


def sounds(symbol: str) -> bool:
    """``symbol`` is a speech sound: not a pause, a space or a mark that only changes the sound next to it."""
    return shape_of(symbol) != "sil" and symbol not in _JOINS_BEFORE and symbol not in _JOINS_AFTER


def timeline(phonemes: Iterable[Sequence[Any]], length: float | None = None) -> list[list[Any]]:
    """``phonemes``: (symbol, start, end) in seconds from the first sample played. Returns [shape, start_ms, end_ms]
    with no gaps and no two neighbours alike, from 0 to the end of the last phoneme (or ``length``), clipped to
    ``length`` when given."""
    raw: list[list[Any]] = []
    for item in phonemes:
        try:
            symbol, start, end = str(item[0]), float(item[1]), float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if end <= start:
            continue
        if length is not None:
            start, end = max(0.0, start), min(float(length), end)
            if end <= start:
                continue
        if symbol in _JOINS_BEFORE and raw:
            raw[-1][2] = max(raw[-1][2], end)
            continue
        if symbol in _JOINS_AFTER:
            raw.append(["+", start, end])  # given to the next sound below
            continue
        if symbol.isspace() and end - start < WORD_GAP_SECONDS and raw and raw[-1][0] != "sil":
            raw[-1][2] = max(raw[-1][2], end)
            continue
        raw.append([shape_of(symbol), start, end])
    for index in range(len(raw) - 1, -1, -1):  # a stress mark's time opens the sound it stresses
        if raw[index][0] == "+":
            if index + 1 < len(raw):
                raw[index + 1][1] = raw[index][1]
            elif index > 0:
                raw[index - 1][2] = raw[index][2]
            del raw[index]
    merged: list[list[Any]] = []
    cursor = 0.0
    for shape, start, end in sorted(raw, key=lambda item: item[1]):
        start = max(start, cursor)
        if end <= start:
            continue
        if start > cursor:
            _append(merged, "sil", cursor, start)
        _append(merged, shape, start, end)
        cursor = end
    if length is not None and cursor < length and merged:
        _append(merged, "sil", cursor, float(length))
    return [[shape, round(start * 1000), round(end * 1000)] for shape, start, end in _drop_slivers(merged)]


def _append(out: list[list[Any]], shape: str, start: float, end: float) -> None:
    if out and out[-1][0] == shape:
        out[-1][2] = end
    else:
        out.append([shape, start, end])


def _drop_slivers(segments: list[list[Any]]) -> list[list[Any]]:
    out: list[list[Any]] = []
    for shape, start, end in segments:
        if out and end - start < MIN_SHAPE_SECONDS:
            out[-1][2] = end
            continue
        if out and out[-1][0] == shape:
            out[-1][2] = end
            continue
        if out and out[-1][2] - out[-1][1] < MIN_SHAPE_SECONDS:
            out[-1] = [shape, out[-1][1], end]
            continue
        out.append([shape, start, end])
    return out


ENVELOPE_RATE = 25  # values a second
ENVELOPE_TOP = 15  # the loudest value
_QUIET_RMS = 0.004  # below this a window is silence


def envelope(audio: Any, rate: int, per_second: int = ENVELOPE_RATE) -> list[int]:
    """How open the mouth may be, ``per_second`` values a second of ``audio``: 0 (silent) to ENVELOPE_TOP, from each
    window's loudness relative to the piece's own loud parts, so a quiet voice still moves the mouth."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    rate = max(1, int(rate))
    step = max(1, rate // max(1, per_second))
    count = int(np.ceil(len(samples) / step)) if len(samples) else 0
    if count == 0:
        return []
    padded = np.zeros(count * step, dtype=np.float32)
    padded[: len(samples)] = samples
    rms = np.sqrt(np.mean(np.square(padded.reshape(count, step)), axis=1))
    loud = rms[rms > _QUIET_RMS]
    if loud.size == 0:
        return [0] * count
    reference = float(np.percentile(loud, 90)) or 1.0
    levels = np.clip(rms / reference, 0.0, 1.0) ** 0.6 * ENVELOPE_TOP
    levels[rms <= _QUIET_RMS] = 0
    return [int(value) for value in np.rint(levels)]
