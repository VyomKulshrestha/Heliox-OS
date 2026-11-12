"""Heliox's system voice for real, to a file, for the ``platform-readers`` job.

    python speech_check.py speak      # macOS: `say -o`; Linux: espeak-ng (installed by the step first)
    python speech_check.py no-voice   # Linux, before espeak-ng is installed: one log line, text carries on

``speak`` runs the daemon's own ``voice._tts_macos`` or ``voice._tts_linux`` with an output file and checks the
file holds sound: at least half a second of audio with samples that are not all zero (macOS: ``afinfo`` reads the
duration of the AIFF; Linux: the WAV is read with ``wave``).

``no-voice`` runs ``voice._tts_linux`` three times on a runner where neither espeak-ng nor espeak is installed:
each call must return the "no voice available" message, and the log must hold that message exactly once.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import one_loop  # noqa: E402  (next to this script)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0
TEXT = "Heliox is speaking through the system voice."


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def wav_facts(path: Path) -> tuple[float, int]:
    """(seconds, the largest absolute 16-bit sample) of a WAV file."""
    import array

    with wave.open(str(path), "rb") as audio:
        frames = audio.readframes(audio.getnframes())
        seconds = audio.getnframes() / float(audio.getframerate() or 1)
        width = audio.getsampwidth()
    if width != 2:
        return seconds, -1
    samples = array.array("h", frames)
    return seconds, max((abs(s) for s in samples), default=0)


def speak() -> None:
    from pilot.system import voice

    folder = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    if sys.platform == "darwin":
        out = folder / "heliox-say.aiff"
        out.unlink(missing_ok=True)
        result = one_loop.run(voice._tts_macos(TEXT, None, 180, str(out)))
        info = subprocess.run(["afinfo", str(out)], capture_output=True, text=True, check=False).stdout
        duration = next(
            (line.split(":", 1)[1].strip() for line in info.splitlines() if "estimated duration" in line), ""
        )
        seconds = float(duration.split()[0]) if duration else 0.0
        check(
            "speech-macos/say-writes-audio",
            out.exists() and seconds >= 0.5,
            f"{result}; {seconds:.2f} s, {out.stat().st_size if out.exists() else 0} bytes",
        )
        return
    out = folder / "heliox-espeak.wav"
    out.unlink(missing_ok=True)
    result = one_loop.run(voice._tts_linux(TEXT, None, 175, str(out)))
    seconds, peak = wav_facts(out) if out.exists() else (0.0, 0)
    check(
        "speech-linux/espeak-writes-audio",
        out.exists() and seconds >= 0.5 and peak > 100,
        f"{result}; {seconds:.2f} s, peak {peak}, program {shutil.which('espeak-ng') or shutil.which('espeak')}",
    )


def no_voice() -> None:
    from pilot.system import linux_voice, voice

    present = [name for name in linux_voice.PROGRAMS if shutil.which(name)]
    if not check("speech-linux/runner-has-no-espeak", not present, repr(present)):
        return
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Keep()
    logging.getLogger("pilot").addHandler(handler)
    logging.getLogger("pilot").setLevel(logging.INFO)
    results = [one_loop.run(voice._tts_linux(TEXT, None, 175, None)) for _ in range(3)]
    said = [r.getMessage() for r in records if "no voice available" in r.getMessage()]
    check(
        "speech-linux/no-voice-said-once-text-carries-on",
        all(result == linux_voice.NO_VOICE for result in results) and len(said) == 1,
        f"{len(said)} log line(s); results {[result[:40] for result in results]}",
    )


def main(argv: list[str]) -> int:
    if argv and argv[0] == "no-voice":
        no_voice()
    else:
        speak()
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
