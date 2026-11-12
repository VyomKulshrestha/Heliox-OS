"""Two real daemons started in the same instant (``python -m pilot.server`` twice, the same home folder). One must
serve and the other must print the one sentence and exit with code 3; the token files on disk must be the
survivor's (its window token authenticates). Run with the environment's Python and HOME set to the CI's empty home.
Prints CHECK lines; exit code 1 if one fails. The survivor is stopped at the end.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 240.0


def main() -> int:
    sys.path.insert(0, str(HERE))
    import first_run_rpc_check as rpc

    copies = [
        subprocess.Popen(
            [sys.executable, "-m", "pilot.server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        started = time.monotonic()
        exited = None
        while time.monotonic() - started < LIMIT and exited is None:
            exited = next((copy for copy in copies if copy.poll() is not None), None)
            time.sleep(0.2)
        if exited is None:
            print(f"CHECK two-copies/one-stops: FAIL - both copies still run after {LIMIT:.0f} s")
            return 1
        stderr = exited.stderr.read() if exited.stderr else ""
        sentence = next((line for line in stderr.splitlines() if line.startswith("Heliox: Port")), "")
        ok_exit = exited.returncode == 3 and "already in use" in sentence
        print(
            f"CHECK two-copies/one-stops: {'PASS' if ok_exit else 'FAIL'} - the second copy exited with code "
            f"{exited.returncode} after {time.monotonic() - started:.1f} s: {sentence or stderr.strip()[-300:]!r}"
        )
        survivor = next(copy for copy in copies if copy is not exited)
        if survivor.poll() is not None:
            print(f"CHECK two-copies/one-serves: FAIL - the other copy exited too, with code {survivor.returncode}")
            return 1
        answered = asyncio.run(rpc.main(LIMIT)) == 0  # authenticates with the token files on disk
        print(
            f"CHECK two-copies/one-serves: {'PASS' if answered else 'FAIL'} - the surviving copy "
            f"{'answers with the token files on disk (they are its own)' if answered else 'does not answer'}"
        )
        return 0 if ok_exit and answered else 1
    finally:
        for copy in copies:
            if copy.poll() is None:
                copy.terminate()
                try:
                    copy.wait(30)
                except subprocess.TimeoutExpired:
                    copy.kill()


if __name__ == "__main__":
    sys.exit(main())
