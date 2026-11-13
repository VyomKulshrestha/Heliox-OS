"""Heliox's system voice for real, to a file, for the ``platform-readers`` job.

    python speech_check.py speak      # macOS: `say -o`; Linux: espeak-ng (installed by the step first)
    python speech_check.py no-voice   # Linux, before espeak-ng is installed: one log line, text carries on

``speak`` runs the daemon's own ``voice._tts_macos`` or ``voice._tts_linux`` with an output file and checks the
file holds sound: at least half a second of audio with samples that are not all zero (macOS: ``afinfo`` reads the
duration of the AIFF; Linux: the WAV is read with ``wave``).

``no-voice`` runs ``voice._tts_linux`` three times on a runner where neither espeak-ng nor espeak is installed:
each call must return the "no voice available" message, and the log must hold that message exactly once.

macOS, when Heliox's own call writes no sound: the same text goes through a plain ``say -o`` (no rate, no voice,
the system's defaults) to tell the two cases apart. If the plain call writes sound, Heliox's call is wrong and the
check FAILS, as it would on a person's Mac. If the plain call writes none either, this runner cannot make speech
(the macos-26 images wrote a 4332-byte AIFF header and nothing else on 49bd35f and 5a3a9b3, where the macos-15
image of e51aead wrote 2.14 s), and the check SKIPS with that reason and the voices ``say -v '?'`` lists.

Each check prints one ``CHECK`` line (PASS, FAIL, or SKIP with its reason); the exit code is the number of failed
checks, so a FAIL fails the step and a SKIP does not.
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


def skip(name: str, reason: str) -> None:
    print(f"CHECK {name}: SKIP - {reason}", flush=True)


def aiff_seconds(path: Path) -> float:
    """The duration ``afinfo`` reads from an AIFF file, 0 when it reads none."""
    if not path.exists():
        return 0.0
    info = subprocess.run(["afinfo", str(path)], capture_output=True, text=True, check=False).stdout
    duration = next((line.split(":", 1)[1].strip() for line in info.splitlines() if "estimated duration" in line), "")
    try:
        return float(duration.split()[0]) if duration else 0.0
    except ValueError:
        return 0.0


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
        seconds = aiff_seconds(out)
        detail = f"{result}; {seconds:.2f} s, {out.stat().st_size if out.exists() else 0} bytes"
        if seconds >= 0.5:
            check("speech-macos/say-writes-audio", True, detail)
            return
        plain = folder / "heliox-say-plain.aiff"
        plain.unlink(missing_ok=True)
        subprocess.run(["say", "-o", str(plain), TEXT], capture_output=True, check=False, timeout=60)
        plain_seconds = aiff_seconds(plain)
        listed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False).stdout.splitlines()
        voices = f"say -v '?' lists {len(listed)} voice(s){': ' + listed[0].split('#')[0].strip() if listed else ''}"
        if plain_seconds >= 0.5:
            # The system can speak to a file and Heliox's call did not: a fault a person's Mac would have too.
            check(
                "speech-macos/say-writes-audio",
                False,
                f"{detail}; a plain say -o wrote {plain_seconds:.2f} s; {voices}",
            )
        else:
            skip(
                "speech-macos/say-writes-audio",
                f"this runner writes no speech to a file even for a plain say -o ({plain_seconds:.2f} s, "
                f"{plain.stat().st_size if plain.exists() else 0} bytes); {voices}; Heliox's call: {detail}",
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
    elif sys.platform == "darwin":
        import mac_front

        mac_front.diag("at the start of speech_check")
        speak()
        mac_front.diag("at the end of speech_check")
    else:
        speak()
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
