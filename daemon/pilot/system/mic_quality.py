"""What kind of microphone is this, from the sound alone: the same on Windows, macOS and Linux.

Kinds of microphone are kinds of user. A Bluetooth headset in call (hands-free) mode sends narrowband audio:
nothing above about 3.4 kHz (measured on the owner's headset: energy above 3.8 kHz was zero against the speech
band, while a desk microphone gave 0.1 to 1.3 % of it). Recognisers trained on wideband speech mishear it. A
microphone that is very quiet or that clips also costs words. Heliox says so once, with what would be better,
instead of failing mysteriously.

Pure numpy over the utterances the listener has already captured; nothing here touches a device. The verdict is
pooled over several utterances (at least MIN_VOICED_SECONDS of voice), because one short word ("Wait, stop.")
can have no sibilants and look narrowband on any microphone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE = 16000
FRAME = 512
MIN_VOICED_SECONDS = 4.0
# Energy 4.5-7.5 kHz over 0.3-3.4 kHz, pooled: owner's headset in call mode 3.7e-7, desk microphone 5.6e-3.
NARROWBAND_RATIO = 1e-4
QUIET_DBFS = -48.0  # median voiced level; the owner's microphones sat at -35 to -44 dBFS
CLIPPING_SHARE = 0.005  # share of voiced samples at full scale

MESSAGES = {
    "narrowband": (
        "This microphone is in Bluetooth call mode: it cuts the sound above 3.4 kHz, so Heliox will mishear "
        "words. For better recognition, use the computer's own microphone for input and keep the headset for "
        "listening (Settings, Voice, Microphone)."
    ),
    "quiet": (
        "Your voice reaches Heliox very quietly, so it may miss words. Move the microphone closer or raise its "
        "input level in the system sound settings."
    ),
    "clipping": (
        "The microphone is too loud and distorts your voice, so Heliox may mishear words. Lower its input level "
        "in the system sound settings."
    ),
}


@dataclass
class MicQuality:
    """Pools the voiced frames of successive utterances and gives one verdict once there is enough voice."""

    band: float = 0.0
    high: float = 0.0
    levels: list[float] = field(default_factory=list)
    clipped: int = 0
    voiced_samples: int = 0
    verdict: str = ""  # "", "ok", "narrowband", "quiet" or "clipping"

    def observe(self, samples: Any) -> str:
        """Add one utterance (float samples in -1..1, 16 kHz). Returns the verdict when it is first reached."""
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(audio) < FRAME:
            return ""
        frames = audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        voiced = frames[rms > max(1e-4, 0.2 * float(rms.max()))]
        if not len(voiced):
            return ""
        spectrum = np.abs(np.fft.rfft(voiced * np.hanning(FRAME), axis=1)) ** 2
        freqs = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
        self.band += float(spectrum[:, (freqs > 300) & (freqs < 3400)].sum())
        self.high += float(spectrum[:, (freqs > 4500) & (freqs < 7500)].sum())
        self.levels.extend(float(x) for x in np.sqrt(np.mean(voiced**2, axis=1)))
        self.clipped += int(np.count_nonzero(np.abs(voiced) >= 0.999))
        self.voiced_samples += voiced.size
        if self.verdict or self.voiced_samples < MIN_VOICED_SECONDS * SAMPLE_RATE:
            return ""
        self.verdict = self.judge()
        return self.verdict

    def judge(self) -> str:
        import numpy as np

        if self.clipped / max(1, self.voiced_samples) >= CLIPPING_SHARE:
            return "clipping"
        if self.band > 0 and self.high / self.band < NARROWBAND_RATIO:
            return "narrowband"
        level = 20 * np.log10(max(1e-9, float(np.median(self.levels))))
        if level < QUIET_DBFS:
            return "quiet"
        return "ok"

    def reset(self) -> None:
        """A different microphone: judge it afresh."""
        self.band = self.high = 0.0
        self.levels.clear()
        self.clipped = self.voiced_samples = 0
        self.verdict = ""


def hands_free_name(device_name: str) -> bool:
    """The device name says call mode (Windows: "Headset (... Hands-Free AG Audio)"; Linux: "headset_head_unit",
    "handsfree"; macOS shows the plain headset name, so there the sound decides alone)."""
    name = device_name.casefold()
    return any(
        word in name
        for word in ("hands-free", "handsfree", "headset_head_unit", "headset-head-unit", "head-unit", "head unit")
    )


# Names that are not microphones a person speaks into: recordings of what is played, and virtual devices.
NOT_A_MICROPHONE = ("stereo mix", "what u hear", "loopback", "monitor of", ".monitor", "wave out", "virtual",
                    "cable output", "blackhole", "soundflower")  # fmt: skip
BUILT_IN = ("microphone array", "internal", "built-in", "macbook", "realtek", "intel", "digital microphone")
# The host API each system's capture uses, so the id names a device the listener can open.
PREFERRED_HOSTAPI = {"win32": "Windows WASAPI", "darwin": "Core Audio", "linux": "PulseAudio"}


def _product(name: str) -> str:
    """The product inside a device name: "Headset (Rockerz 415 Hands-Free AG Audio)" -> "rockerz 415"."""
    inside = name[name.find("(") + 1 : name.rfind(")")] if "(" in name and ")" in name else name
    for word in ("hands-free ag audio", "hands-free", "handsfree", "stereo", "headset"):
        inside = inside.casefold().replace(word, " ")
    return " ".join(inside.split())


def suggest_input(devices: list[dict[str, Any]], headset: str, platform: str) -> dict[str, str] | None:
    """The best other microphone, as {"id", "name"}, from list_audio_input_devices (pure).

    Not the headset (nor its other entries), not another hands-free input, not a recording of the output.
    A built-in microphone first; the system's main host API is preferred so the id opens.
    """
    product = _product(headset)
    hostapi = PREFERRED_HOSTAPI.get("linux" if platform.startswith("linux") else platform)
    candidates = []
    for device in devices:
        name = str(device.get("name", ""))
        ident = str(device.get("id", ""))
        folded = name.casefold()
        if not name or headset.casefold() in {folded, ident.casefold(), ident.split("::")[-1].casefold()}:
            continue  # the headset itself, by name or (Linux) by its PulseAudio source name
        if hands_free_name(name) or hands_free_name(ident):
            continue
        if any(word in folded or word in ident.casefold() for word in NOT_A_MICROPHONE):
            continue
        if product and product in folded:
            continue
        score = 2 * any(word in folded for word in BUILT_IN) + (device.get("hostapi") == hostapi)
        candidates.append((score, name, device))
    if not candidates:
        return None
    _score, name, device = max(candidates, key=lambda item: item[0])
    return {"id": str(device.get("id", name)), "name": name}
