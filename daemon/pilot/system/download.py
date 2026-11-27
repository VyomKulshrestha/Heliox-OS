"""Download files from URLs, through the checked outbound path."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pilot.system import outbound

logger = logging.getLogger("pilot.system.download")


def _facts(resp: Any, data: bytes) -> str:
    """What the server said about the body and what was received, for the post-condition check.

    The final address is kept without its query, which can hold a signed token.
    """
    headers = resp.headers
    final = urlsplit(str(resp.url))
    return json.dumps(
        {
            "content_length": headers.get("content-length"),
            "content_encoding": headers.get("content-encoding") or "identity",
            "content_type": headers.get("content-type", ""),
            "final_url": urlunsplit((final.scheme, final.netloc, final.path, "", "")),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )


async def download_file(
    url: str, output_path: str, overwrite: bool = False, *, allow_private_network: bool = False
) -> str:
    """Download a file from a URL to a local path, through the checked outbound path (pilot.system.outbound).

    An address on this computer or the local network is refused unless the step carries ``allow_private_network``,
    which makes it need the person's approval. A refused or failed download writes nothing and is never handed to
    another download program.
    """
    out_path = Path(output_path)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}. Set overwrite=true.")

    resp = await outbound.fetch("GET", url, allow_private_network=allow_private_network, timeout=120)
    if resp.status_code >= 400:
        from pilot.system.api_client import HttpStatusError

        raise HttpStatusError("GET", str(resp.url), resp.status_code, getattr(resp, "reason_phrase", ""))
    data = resp.content
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    size = len(data)
    return f"Downloaded {url} -> {output_path} ({size:,} bytes)\n[DOWNLOAD] {_facts(resp, data)}"
