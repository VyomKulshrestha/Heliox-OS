"""Listening conditions derived from the real recordings, so one recording speaks for several people (numpy).

Every condition other than ``as-recorded`` is labelled derived. They model what other people's set-ups deliver:

  bt-call      a Bluetooth headset in call mode (hands-free profile, narrowband): 8 kHz sampling, nothing above
               about 3.4 kHz, 8-bit companding. Matched to the owner's own headset recordings of 10 November,
               which carry speech up to 3.4 kHz and are 50 to 65 dB down above it.
  room-20db    a quiet room: pink noise 20 dB under the speech.
  fan-10db     a laptop microphone next to its fan: low, rumbling noise with a hum, 10 dB under the speech.
  voice-10db   an open office: another person talking (local synthetic speech), 10 dB under the speech.
  voice-5db    the same, 5 dB under: someone talking close by.
  quiet-12db   the speaker far from the microphone or a low input gain: 12 dB quieter.
  loud+10db    a hot input gain: 10 dB louder, clipped at full scale.
  slow-0.9     a slower speaker: 10 percent slower, pitch kept.
  fast-1.1     a faster speaker: 10 percent faster, pitch kept.

The noise is seeded, so a condition gives the same audio every run. The second voice needs a babble file
(``synthetic/babble.wav`` in the audio folder, rendered by ``synth``); without it the voice conditions are skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

RATE = 16000


@dataclass(frozen=True)
class Condition:
    name: str
    derived: bool
    describes: str
    needs_babble: bool = False


CONDITIONS: tuple[Condition, ...] = (
    Condition("as-recorded", False, "the recording as it was made"),
    Condition("bt-call", True, "Bluetooth headset in call mode (narrowband 8 kHz)"),
    Condition("room-20db", True, "quiet room, noise 20 dB under the voice"),
    Condition("fan-10db", True, "laptop microphone by a fan, 10 dB under"),
    Condition("voice-10db", True, "open office, another voice 10 dB under", needs_babble=True),
    Condition("voice-5db", True, "another voice close by, 5 dB under", needs_babble=True),
    Condition("quiet-12db", True, "far from the microphone, 12 dB quieter"),
    Condition("loud+10db", True, "hot input gain, 10 dB louder, clipped"),
    Condition("slow-0.9", True, "10 percent slower speaker"),
    Condition("fast-1.1", True, "10 percent faster speaker"),
)
BY_NAME = {c.name: c for c in CONDITIONS}


def rms(x: Any) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def speech_rms(x: Any) -> float:
    """Level of the voiced part only (frames above a tenth of the loudest), so pauses do not lower it."""
    import numpy as np

    frame = RATE // 50
    if len(x) < frame:
        return rms(x)
    frames = x[: len(x) // frame * frame].reshape(-1, frame)
    levels = np.sqrt(np.mean(np.square(frames), axis=1))
    voiced = levels[levels > 0.1 * levels.max()]
    return float(np.sqrt(np.mean(np.square(voiced)))) if len(voiced) else rms(x)


def mix_at_snr(speech: Any, noise: Any, snr_db: float) -> Any:
    import numpy as np

    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / max(1, len(noise)))))
    noise = noise[: len(speech)]
    gain = speech_rms(speech) / max(rms(noise), 1e-9) / (10 ** (snr_db / 20))
    return (speech + gain * noise).astype(np.float32)


def pink_noise(n: int, seed: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    spectrum /= np.sqrt(np.maximum(freqs, 20.0))
    return np.fft.irfft(spectrum, n).astype(np.float32)


def fan_noise(n: int, seed: int) -> Any:
    import numpy as np
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    rumble = sosfilt(butter(4, 600, "low", fs=RATE, output="sos"), rng.standard_normal(n))
    t = np.arange(n) / RATE
    hum = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 240 * t)
    return (rumble / max(rms(rumble), 1e-9) + hum).astype(np.float32)


def bluetooth_call(x: Any) -> Any:
    """Narrowband hands-free audio: down to 8 kHz (steep filter at 3.4 kHz), mu-law 8 bit, back up to 16 kHz."""
    import numpy as np
    from scipy.signal import butter, resample_poly, sosfilt

    low = sosfilt(butter(8, 3400, "low", fs=RATE, output="sos"), x)
    narrow = resample_poly(low, 1, 2)
    mu = 255.0
    peak = max(float(np.abs(narrow).max()), 1e-9)
    y = np.sign(narrow / peak) * np.log1p(mu * np.abs(narrow / peak)) / np.log1p(mu)
    y = np.round(y * 127) / 127
    narrow = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu) * peak
    return resample_poly(narrow, 2, 1).astype(np.float32)


def time_stretch(x: Any, rate: float, frame: int = 640, hop: int = 160) -> Any:
    """Waveform-similarity overlap-add (WSOLA): ``rate`` > 1 is faster, pitch unchanged."""
    import numpy as np

    if abs(rate - 1.0) < 1e-6 or len(x) < frame * 2:
        return x.astype(np.float32)
    window = np.hanning(frame).astype(np.float32)
    out_len = int(len(x) / rate)
    out = np.zeros(out_len + frame, dtype=np.float32)
    norm = np.zeros(out_len + frame, dtype=np.float32)
    tolerance = hop
    previous = None
    for out_pos in range(0, out_len, hop):
        centre = int(out_pos * rate)
        best = centre
        if previous is not None:
            lo, hi = max(0, centre - tolerance), min(len(x) - frame, centre + tolerance)
            best_score = -np.inf
            for cand in range(lo, max(lo + 1, hi), 8):
                seg = x[cand : cand + frame]
                if len(seg) < frame:
                    break
                score = float(np.dot(seg[: frame - hop], previous))
                if score > best_score:
                    best, best_score = cand, score
        seg = x[best : best + frame]
        if len(seg) < frame:
            break
        out[out_pos : out_pos + frame] += seg * window
        norm[out_pos : out_pos + frame] += window
        previous = x[best + hop : best + hop + frame - hop]
        if len(previous) < frame - hop:
            break
    norm[norm < 1e-3] = 1.0
    return (out / norm)[:out_len].astype(np.float32)


def apply(name: str, x: Any, *, babble: Any = None, seed: int = 7) -> Any:
    """``x`` (16 kHz float32) heard under condition ``name``."""
    import numpy as np

    if name not in BY_NAME:
        raise KeyError(f"unknown condition {name!r}")
    x = np.asarray(x, dtype=np.float32)
    makers: dict[str, Callable[[], Any]] = {
        "as-recorded": lambda: x,
        "bt-call": lambda: bluetooth_call(x),
        "room-20db": lambda: mix_at_snr(x, pink_noise(len(x), seed), 20),
        "fan-10db": lambda: mix_at_snr(x, fan_noise(len(x), seed), 10),
        "quiet-12db": lambda: x * 10 ** (-12 / 20),
        "loud+10db": lambda: np.clip(x * 10 ** (10 / 20), -1.0, 1.0),
        "slow-0.9": lambda: time_stretch(x, 0.9),
        "fast-1.1": lambda: time_stretch(x, 1.1),
    }
    if name in ("voice-10db", "voice-5db"):
        if babble is None:
            raise ValueError(f"{name} needs a babble recording")
        offset = (seed * 7919) % max(1, len(babble) - len(x)) if len(babble) > len(x) else 0
        return mix_at_snr(x, babble[offset:], 10 if name == "voice-10db" else 5)
    return np.asarray(makers[name](), dtype=np.float32)


def available(babble_present: bool) -> list[str]:
    return [c.name for c in CONDITIONS if babble_present or not c.needs_babble]
