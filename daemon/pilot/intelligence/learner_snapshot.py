"""A saved state of the online learner, so a start replays only the events after it instead of the whole ledger.

Why a keyed, restricted pickle and not JSON: the learner's drift detectors are river's ADWIN, a Cython object whose
state is reachable only through its pickle protocol (``river.drift.adwin_c.AdaptiveWindowing`` and ``Bucket``). A JSON
format would have to copy ADWIN's private layout for each river version; anything less would make a loaded learner
decide differently from a full replay as soon as drift is next checked. So the snapshot is a pickle, and it is only
ever loaded when all of these hold:

- its HMAC-SHA256 matches, keyed by a secret kept in Heliox's key vault (the system keyring), not on disk beside it;
- the header's format, model version, river and numpy versions and reset point match this installation;
- every class it names is on a short allow-list (river's model, optimiser, drift detector and vector types, numpy's
  array rebuild, a few standard containers and the learner's own two classes); anything else is refused before it
  is built.

On any refusal the caller replays the whole ledger in the background instead. With no vault (no system keyring),
nothing is saved or loaded.

File layout: one line of JSON (the header, with ``hmac``), a newline, then the pickle bytes. The HMAC covers the header
without its ``hmac`` field (sorted keys) and the pickle bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import pickle
from pathlib import Path
from typing import Any

FORMAT = "heliox-learner-snapshot/1"

ALLOWED_GLOBALS = frozenset(
    {
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("collections", "Counter"),
        ("collections", "deque"),
        ("datetime", "datetime"),
        ("datetime", "timezone"),
        ("datetime", "timedelta"),
        ("numpy", "dtype"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("river.drift.adwin", "ADWIN"),
        ("river.drift.adwin_c", "AdaptiveWindowing"),
        ("river.drift.adwin_c", "Bucket"),
        ("river.drift.adwin_c", "__pyx_unpickle_AdaptiveWindowing"),
        ("river.drift.adwin_c", "__pyx_unpickle_Bucket"),
        ("river.linear_model.log_reg", "LogisticRegression"),
        ("river.optim.initializers", "Zeros"),
        ("river.optim.losses", "Log"),
        ("river.optim.schedulers", "Constant"),
        ("river.optim.sgd", "SGD"),
        ("river.utils.vectordict", "VectorDict"),
        ("river.utils.vectordict", "__pyx_unpickle_VectorDict"),
        ("pilot.intelligence.online_learning", "_VerifiedClassifier"),
        ("pilot.intelligence.online_learning", "_ReplaySample"),
        ("pilot.intelligence.online_learning", "AdaptationScore"),
    }
)


class SnapshotRefused(Exception):
    """Why a snapshot was not used, in plain words; the caller replays the whole ledger instead."""


class _Restricted(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise SnapshotRefused(f"it names {module}.{name}, which a learner snapshot never contains")
        return super().find_class(module, name)


def _versions() -> dict[str, str]:
    import numpy
    import river

    return {"river": river.__version__, "numpy": numpy.__version__}


def _sign(key: bytes, header: dict[str, Any], payload: bytes) -> str:
    unsigned = json.dumps({k: v for k, v in header.items() if k != "hmac"}, sort_keys=True).encode("utf-8")
    return hmac.new(key, unsigned + b"\n" + payload, hashlib.sha256).hexdigest()


def write(
    path: Path, key: bytes, state: dict[str, Any] | bytes, *, model_version: str, reset_before: int, sequence: int
) -> None:
    """Write the snapshot beside the old one and rename it over, so a crash leaves the old one or the new one.
    ``state`` may already be pickled (the learner serialises it under its lock)."""
    payload = state if isinstance(state, bytes) else pickle.dumps(state, protocol=5)
    header: dict[str, Any] = {
        "format": FORMAT,
        "model_version": model_version,
        "reset_before_sequence": reset_before,
        "sequence": sequence,
        **_versions(),
    }
    header["hmac"] = _sign(key, header, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".new")
    with open(staged, "wb") as handle:
        handle.write(json.dumps(header, sort_keys=True).encode("utf-8") + b"\n" + payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)


def read(path: Path, key: bytes, *, model_version: str, reset_before: int) -> tuple[int, dict[str, Any]]:
    """The snapshot's sequence and state. Raises SnapshotRefused, with the reason, for anything it will not load."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise SnapshotRefused("there is no saved learner state yet") from error
    except OSError as error:
        raise SnapshotRefused(f"the saved learner state could not be read ({error})") from error
    head, separator, payload = raw.partition(b"\n")
    try:
        header = json.loads(head)
    except ValueError as error:
        raise SnapshotRefused("the saved learner state is not in a known format") from error
    if not separator or not isinstance(header, dict) or header.get("format") != FORMAT:
        raise SnapshotRefused("the saved learner state is not in a known format")
    if not hmac.compare_digest(str(header.get("hmac", "")), _sign(key, header, payload)):
        raise SnapshotRefused("the saved learner state does not match its key, so it was not trusted")
    expected = {"model_version": model_version, "reset_before_sequence": reset_before, **_versions()}
    for name, value in expected.items():
        if header.get(name) != value:
            raise SnapshotRefused(f"the saved learner state was made with a different {name.replace('_', ' ')}")
    try:
        state = _Restricted(io.BytesIO(payload)).load()
    except SnapshotRefused:
        raise
    except Exception as error:
        raise SnapshotRefused(f"the saved learner state could not be rebuilt ({type(error).__name__})") from error
    if not isinstance(state, dict):
        raise SnapshotRefused("the saved learner state is not in a known format")
    return int(header["sequence"]), state
