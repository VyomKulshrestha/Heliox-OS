"""Hash checks for committed text evidence whose line endings depend on the checkout.

Some frozen reports pin the SHA-256 of a JSON manifest as it sat on the author's
disk (LF or CRLF). Git normalizes text to LF, and Windows checkouts convert it
back to CRLF, so the same committed file hashes differently per platform.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def text_sha256_variants(path: Path) -> set[str]:
    """SHA-256 of the file as checked out, with LF endings, and with CRLF endings."""
    data = path.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    return {hashlib.sha256(variant).hexdigest() for variant in (data, lf, lf.replace(b"\n", b"\r\n"))}


def assert_pinned_text_hash(pinned: str, path: Path) -> None:
    assert pinned in text_sha256_variants(path), f"{path.name}: pinned {pinned[:12]} matches no line-ending variant"
