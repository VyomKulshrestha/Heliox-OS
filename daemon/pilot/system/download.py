"""Download files from URLs, through the checked outbound path, streamed to disk.

The body is written as it arrives to a temporary file beside the target and moved into place only when it is
complete. A file larger than the limit is refused before the first byte when the server states its length, and
cut off as soon as the stream passes the limit when it does not; a download that is refused, cut or fails leaves
nothing behind.

The limit is ``DEFAULT_MAX_BYTES`` (2 GiB) unless the step carries a larger ``max_bytes``, which makes the step need
the person's approval (pilot.actions.Action.permission_tier). 2 GiB holds the documents, archives and application
installers people ask for by voice, while a disk image or a dataset (usually several gigabytes) is something a
person asks for knowingly ("download this 6 GB image"), and a page that streams without end cannot fill the disk.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pilot.actions import DOWNLOAD_DEFAULT_MAX_BYTES
from pilot.system import outbound

logger = logging.getLogger("pilot.system.download")

DEFAULT_MAX_BYTES = DOWNLOAD_DEFAULT_MAX_BYTES
# Room kept free on the disk beyond the file itself, so a download never fills it to the last byte.
FREE_SPACE_RESERVE = 256 * 1024**2


class DownloadRefused(OSError):
    """The download was not made, or was stopped, for a reason asking again unchanged cannot alter."""


def _size(count: int) -> str:
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if count >= scale:
            return f"{count / scale:.1f} {unit}"
    return f"{count} bytes"


def _facts(resp: Any, size: int, sha256: str) -> str:
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
            "bytes": size,
            "sha256": sha256,
        }
    )


def _too_large(limit: int, stated: int | None) -> DownloadRefused:
    said = f" (the server says {_size(stated)})" if stated else ""
    return DownloadRefused(
        f"The file is larger than the {_size(limit)} download limit{said}, so I did not keep it. Ask for it again "
        "with its size if you want it."
    )


def _no_room(folder: Path, needed: int, free: int) -> DownloadRefused:
    return DownloadRefused(
        errno.ENOSPC,
        f"There is not enough free disk space in {folder} for this download: it needs {_size(needed)} and "
        f"{_size(free)} is free.",
    )


async def download_file(
    url: str,
    output_path: str,
    overwrite: bool = False,
    *,
    allow_private_network: bool = False,
    max_bytes: int | None = None,
) -> str:
    """Download a file from a URL to a local path, through the checked outbound path (pilot.system.outbound).

    An address on this computer or the local network is refused unless the step carries ``allow_private_network``,
    which makes it need the person's approval. A refused or failed download writes nothing and is never handed to
    another download program.
    """
    out_path = Path(output_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}. Set overwrite=true.")
    limit = int(max_bytes) if max_bytes and int(max_bytes) > 0 else DEFAULT_MAX_BYTES

    async def consume(resp: Any) -> tuple[int, str]:
        if resp.status_code >= 400:
            from pilot.system.api_client import HttpStatusError

            raise HttpStatusError("GET", str(resp.url), resp.status_code, getattr(resp, "reason_phrase", ""))
        length = str(resp.headers.get("content-length") or "")
        identity = (resp.headers.get("content-encoding") or "identity").lower() == "identity"
        stated = int(length) if length.isdigit() else None
        if stated is not None and identity and stated > limit:
            raise _too_large(limit, stated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(out_path.parent).free
        if stated is not None and stated + FREE_SPACE_RESERVE > free:
            raise _no_room(out_path.parent, stated, free)
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed and moved or removed below
            dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".part", delete=False
        )
        part = Path(handle.name)
        digest = hashlib.sha256()
        size = 0
        try:
            with handle:
                async with outbound.body_chunks(resp) as chunks:
                    async for chunk in chunks:
                        size += len(chunk)
                        if size > limit:
                            raise _too_large(limit, stated)
                        try:
                            handle.write(chunk)
                        except OSError as exc:
                            if exc.errno == errno.ENOSPC:
                                raise _no_room(out_path.parent, size, shutil.disk_usage(out_path.parent).free) from None
                            raise
                        digest.update(chunk)
            if stated is not None and identity and size != stated:
                raise DownloadRefused(
                    f"The download stopped early: the server said {stated:,} bytes and sent {size:,}, so I did not "
                    "keep it."
                )
            os.replace(part, out_path)
        except BaseException:
            with contextlib.suppress(OSError):
                part.unlink()
            raise
        return size, digest.hexdigest()

    holder: dict[str, Any] = {}

    async def keep(resp: Any) -> tuple[int, str]:
        holder["resp"] = resp
        return await consume(resp)

    size, sha256 = await outbound.fetch_streamed(
        "GET", url, keep, allow_private_network=allow_private_network, timeout=120
    )
    return f"Downloaded {url} -> {output_path} ({size:,} bytes)\n[DOWNLOAD] {_facts(holder['resp'], size, sha256)}"
