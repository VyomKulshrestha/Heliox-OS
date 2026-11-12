"""A protected folder named in another case, on the runner's real file system, through the action validator.

On macOS (APFS, case-insensitive by default) ``~/HelioxProtected/secret.txt`` and ``~/helioxprotected/secret.txt``
are the same file; a rule that compared case-sensitively could be stepped around by changing one letter. The check
first shows the file system treats the two spellings as one file, then asks ``ActionValidator`` to delete the file
by the other spelling with ``~/HelioxProtected`` protected: it must refuse. On Linux (ext4, case-sensitive) the
other spelling is another, non-existent path, and the validator must let it through, as the file system would.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ci_only import require_runner

require_runner("creates and deletes a HelioxProtected folder in the home folder")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def main() -> int:
    from pilot.actions import Action, ActionType, FileParams
    from pilot.config import PilotConfig
    from pilot.security.validator import ActionValidator, ValidationError

    folder = Path.home() / "HelioxProtected"
    other = Path.home() / "helioxprotected" / "secret.txt"
    folder.mkdir(exist_ok=True)
    (folder / "secret.txt").write_text("protected\n", encoding="utf-8")
    try:
        same = other.exists() and os.path.samefile(folder / "secret.txt", other)
        config = PilotConfig()
        config.restrictions.protected_folders = [str(folder)]
        action = Action(action_type=ActionType.FILE_DELETE, target=str(other), parameters=FileParams(path=str(other)))
        try:
            ActionValidator(config).validate_action(action)
            outcome = "allowed"
        except ValidationError as exc:
            outcome = f"refused: {exc}"
        system = "macos" if sys.platform == "darwin" else "linux"
        if system == "macos":
            check("protected-macos/apfs-treats-both-spellings-as-one-file", same, f"{other} exists: {other.exists()}")
            check("protected-macos/other-case-is-refused", outcome.startswith("refused"), outcome)
        else:
            check("protected-linux/other-spelling-is-another-path", not other.exists())
            check("protected-linux/other-spelling-is-not-refused", outcome == "allowed", outcome)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return failures


if __name__ == "__main__":
    sys.exit(main())
