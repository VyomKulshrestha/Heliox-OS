"""Store growth at ten times the requests: does each request still cost the same bytes when the stores are bigger?

Runs ``benchmarks/store_growth.py`` (the real request path into a temporary data folder; no model, no network) at
200 requests and at 2,000, and compares bytes per request for each store. Growth that is linear in requests is
expected and is what RELIABILITY.md states per day and per year; a store whose cost per request rises with its size
(an index rewritten, a file copied whole on every write) fails. Also prints the time per request at each size.

    python daemon/tests/platform_real/store_growth_check.py

What this cannot show: a year of a real person's mix, and stores the benchmark does not drive (the activity timeline,
screen reading). Prints CHECK lines; exit code 1 if one fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

DAEMON = Path(__file__).resolve().parents[2]
SIZES = (200, 2000)
SLACK = (1.25, 64.0)  # the larger run may cost at most 25 % more, plus 64 bytes, per request
LINE = re.compile(r"^(?P<store>[^:{]+): (?P<bytes>\d+) bytes per request")


def measure(requests: int) -> tuple[dict[str, int], float]:
    began = time.monotonic()
    done = subprocess.run(
        [sys.executable, str(DAEMON / "benchmarks" / "store_growth.py"), str(requests)],
        cwd=DAEMON,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    took = time.monotonic() - began
    found = {
        match["store"].strip(): int(match["bytes"]) for match in map(LINE.match, done.stdout.splitlines()) if match
    }
    if done.returncode != 0 or not found:
        raise RuntimeError(
            f"store_growth.py {requests} failed ({done.returncode}): {(done.stdout + done.stderr)[-600:]}"
        )
    print(f"INFO {requests} requests in {took:.0f} s ({took / requests * 1000:.0f} ms each, start-up included)")
    for line in done.stdout.splitlines():
        print(f"INFO   {line}")
    return found, took


def main() -> int:
    small, _small_took = measure(SIZES[0])
    large, _large_took = measure(SIZES[1])
    failed = 0
    for store, per_request in small.items():
        bigger = large.get(store)
        bound = per_request * SLACK[0] + SLACK[1]
        ok = bigger is not None and bigger <= bound
        failed += not ok
        print(
            f"CHECK store-growth/{store.lower().replace(' ', '-')}: {'PASS' if ok else 'FAIL'} - {per_request} bytes "
            f"per request at {SIZES[0]} requests, {bigger} at {SIZES[1]} (bound {bound:.0f})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
