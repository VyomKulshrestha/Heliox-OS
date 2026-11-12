"""Read and write WAV files with the standard library and numpy only (no OS audio library, no soundfile).

Reads PCM 8/16/24/32-bit and IEEE float 32/64 WAV (RIFF and WAVE_FORMAT_EXTENSIBLE), any channel count (the
first channel is kept), and returns float32 in [-1, 1] at 16 kHz. Writes 16-bit PCM mono.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

RATE = 16000


class WavError(ValueError):
    pass


def _chunks(data: bytes) -> dict[bytes, bytes]:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")
    out: dict[bytes, bytes] = {}
    pos = 12
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        out.setdefault(cid, body)
        pos += 8 + size + (size & 1)
    return out


def decode(data: bytes) -> tuple[Any, int]:
    """Samples (float32, first channel) and the file's rate."""
    import numpy as np

    chunks = _chunks(data)
    if b"fmt " not in chunks or b"data" not in chunks:
        raise WavError("missing fmt or data chunk")
    fmt = chunks[b"fmt "]
    tag, channels, rate = struct.unpack("<HHI", fmt[:8])
    bits = struct.unpack("<H", fmt[14:16])[0]
    if tag == 0xFFFE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    raw = chunks[b"data"]
    width = bits // 8
    frames = len(raw) // (width * channels)
    raw = raw[: frames * width * channels]
    if tag == 3 and bits in (32, 64):
        samples = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    elif tag == 1 and bits == 8:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        value = np.where(value & 0x800000, value - 0x1000000, value)
        samples = value.astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 32:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise WavError(f"unsupported WAV encoding (format {tag}, {bits} bits)")
    return samples.reshape(-1, channels)[:, 0].copy(), int(rate)


def to_rate(samples: Any, rate: int, target: int = RATE) -> Any:
    import numpy as np

    if rate == target:
        return samples.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(rate), int(target))
    return resample_poly(samples, target // g, int(rate) // g).astype(np.float32)


def read(path: str | Path, start: float | None = None, end: float | None = None) -> Any:
    """A file (or a span of it, in seconds) as 16 kHz float32 mono."""
    samples, rate = decode(Path(path).read_bytes())
    if start is not None:
        samples = samples[int(start * rate) : int(end * rate) if end is not None else None]
    return to_rate(samples, rate)


def encode(samples: Any, rate: int = RATE) -> bytes:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write(path: str | Path, samples: Any, rate: int = RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encode(samples, rate))
