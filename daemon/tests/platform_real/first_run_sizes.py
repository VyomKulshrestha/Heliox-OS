"""Download and disk sizes for the first-run screen, measured by the ``first-run`` CI job.

With ``HELIOX_MEASURE_PIP_CACHE`` set, the setup scripts print after each install:

- ``@@size <step> <KiB>``: pip's download cache so far (it starts empty), so a step's download is the difference;
- ``@@disk <step> <KiB>``: Heliox's whole folder so far. For the core that is everything the person has after setup
  (environment, and Python and uv when Heliox brought its own); for a group it is the difference.

This turns a setup log into megabytes per step and writes them into the UI's data file, so the screen's numbers are
measured rather than typed in.

    python first_run_sizes.py show   <setup.log>                                  # print {"core": {...}, ...}
    python first_run_sizes.py update <firstRunSizes.json> <system> <run> <setup.log>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def step_sizes(log: str) -> dict[str, dict[str, float]]:
    """{step: {"mb": download, "disk_mb": disk}} from the cumulative ``@@size`` and ``@@disk`` lines of a log."""
    sizes: dict[str, dict[str, float]] = {}
    previous = {"@@size": 0, "@@disk": 0}
    for line in log.splitlines():
        parts = line.strip().split()
        if len(parts) != 3 or parts[0] not in previous or not parts[2].isdigit():
            continue
        total = int(parts[2])
        first_disk = parts[0] == "@@disk" and previous["@@disk"] == 0
        delta = total if first_disk else max(0, total - previous[parts[0]])
        key = "mb" if parts[0] == "@@size" else "disk_mb"
        sizes.setdefault(parts[1], {})[key] = round(delta / 1024, 1)
        previous[parts[0]] = total
    return sizes


def update(data: dict, system: str, run: str, sizes: dict[str, dict[str, float]]) -> dict:
    """Write measured sizes for ``system`` ("windows", "macos" or "linux") into the data file's structure."""
    for step, measured in sizes.items():
        entry = data["sizes"].setdefault(step, {})
        current = dict(entry.get(system) or entry.get("all") or {})
        for key, value in measured.items():
            current[key] = max(1, round(value))
        current["measured"] = run
        entry[system] = current
    return data


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "show":
        print(json.dumps(step_sizes(Path(argv[1]).read_text(encoding="utf-8", errors="replace"))))
        return 0
    if len(argv) >= 5 and argv[0] == "update":
        target = Path(argv[1])
        data = json.loads(target.read_text(encoding="utf-8"))
        sizes = step_sizes(Path(argv[4]).read_text(encoding="utf-8", errors="replace"))
        target.write_text(json.dumps(update(data, argv[2], argv[3], sizes), indent=2) + "\n", encoding="utf-8")
        print(f"{argv[2]}: {sizes}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
