"""What a first run and a removal leave in the home folder, for the real-removal CI job.

    python home_snapshot.py save BEFORE        # list the home folder (four levels down, links not followed)
    python home_snapshot.py left BEFORE [--ignore PATH ...]   # what is new since, and whose it is

Anything left that carries Heliox's names (heliox, com.helioxos.app, a .pilot folder that was not there) is a finding
and fails the check. Everything else new is listed as left by other programs (pip's and uv's caches, for example,
which the removal screen names as shared caches it leaves alone). ``--ignore`` leaves out the checks' own files
by their exact path (on a Linux runner the job's temporary folder is inside the home folder, and its state files
carry Heliox's name: CI on 0a8edad reported them as left behind).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DEPTH = 4
# A new .pilot folder that is still there was made by Heliox's daemon (nothing else on a fresh runner uses it).
HELIOX = re.compile(r"heliox|[\\/]\.pilot([\\/]|$)", re.IGNORECASE)


def listing(home: Path, depth: int = DEPTH) -> set[str]:
    found: set[str] = set()

    def walk(folder: Path, level: int) -> None:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return
        for entry in entries:
            found.add(os.path.relpath(entry.path, home))
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            if is_dir and level < depth:
                walk(Path(entry.path), level + 1)

    walk(home, 1)
    return found


def new_tops(before: set[str], after: set[str]) -> list[str]:
    """Paths new since ``before``, without the children of a new folder. Pure."""
    new = sorted(after - before)
    tops: list[str] = []
    for path in new:
        if not any(path.startswith(top + os.sep) for top in tops):
            tops.append(path)
    return tops


def ignored_paths(argv: list[str], home: Path) -> set[str]:
    """The exact paths after each ``--ignore``, relative to ``home`` as the listing writes them. Pure."""
    paths = [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "--ignore"]
    return {os.path.relpath(os.path.abspath(path), home) for path in paths}


def main(argv: list[str]) -> int:
    mode, record = argv[0], Path(argv[1])
    home = Path.home()
    ignore = ignored_paths(argv[2:], home)
    if mode == "save":
        record.write_text("\n".join(sorted(listing(home))), encoding="utf-8")
        print(f"home {home}: {len(listing(home))} paths recorded")
        return 0
    before = set(record.read_text(encoding="utf-8").splitlines())
    tops = [path for path in new_tops(before, listing(home)) if path not in ignore]
    heliox = [path for path in tops if HELIOX.search(os.sep + path)]
    others = [path for path in tops if path not in heliox]
    print(f"left by other programs: {others}")
    print(f"CHECK removal/nothing-of-heliox-left-in-home: {'FAIL' if heliox else 'PASS'} - {heliox}", flush=True)
    return 1 if heliox else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
