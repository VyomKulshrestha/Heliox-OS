"""The system volume set, stepped, muted and unmuted for real through pilot.system.volume, read back each time, for
the ``platform-readers`` job.

Linux: PulseAudio is running from the job's audio steps; a null sink is loaded and made the default, so there is a
sink to set. macOS and Windows: the runner's own default output, if it has one; a runner with no output device is
reported as such (one NOT-POSSIBLE line), since that is what Heliox would say there too. The level and mute state
found at the start are put back at the end.

It changes the runner's volume, so it runs only on a runner (ci_only). Each check prints one ``CHECK`` line; the exit
code is the number of failed checks.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_only import require_runner  # noqa: E402

require_runner("sets, steps, mutes and unmutes this computer's output volume")

from pilot.system import volume  # noqa: E402
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


async def main() -> int:
    system = str(CURRENT_PLATFORM)
    if CURRENT_PLATFORM == Platform.LINUX:
        subprocess.run(["pactl", "load-module", "module-null-sink", "sink_name=heliox_volume"], check=False)
        subprocess.run(["pactl", "set-default-sink", "heliox_volume"], check=False)
    start = await volume.volume_state()
    if start.level is None:
        print(f"NOT-POSSIBLE volume-{system}/read: the runner has no output device to read ({start})", flush=True)
        try:
            await volume.volume_set(30)
            check(f"volume-{system}/no-device-is-said", False, "a set without a device claimed success")
        except RuntimeError as error:
            check(f"volume-{system}/no-device-is-said", True, str(error))
        return failures
    check(f"volume-{system}/call-available", volume.volume_call_available())
    said = await volume.volume_set(30)
    state = await volume.volume_state()
    check(
        f"volume-{system}/set-30-reads-back",
        state.level == 30 and said.startswith("Volume is at 30"),
        f"{said!r} {state}",
    )
    said = await volume.volume_step(-10)
    state = await volume.volume_state()
    check(f"volume-{system}/step-down-reads-back", state.level == 20 and "20" in said, f"{said!r} {state}")
    said = await volume.volume_step(10)
    check(f"volume-{system}/step-up-reads-back", "30" in said, said)
    said = await volume.volume_mute(True)
    state = await volume.volume_state()
    check(f"volume-{system}/mute", state.muted is True and said in {"Muted.", "Already muted."}, f"{said!r} {state}")
    said = await volume.volume_mute(True)
    check(f"volume-{system}/mute-again-says-already", said == "Already muted.", said)
    said = await volume.volume_mute(False)
    state = await volume.volume_state()
    check(f"volume-{system}/unmute", state.muted is False and said == "Unmuted.", f"{said!r} {state}")
    said = await volume.volume_mute(False)
    check(f"volume-{system}/unmute-when-not-muted-changes-nothing", said == "Already unmuted.", said)
    # put back what was there
    await volume.volume_set(start.level)
    if start.muted is not None:
        await volume.volume_mute(start.muted)
    return failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
