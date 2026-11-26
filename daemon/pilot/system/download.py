"""Download files from URLs.

Cross-platform using httpx (preferred) or OS-native tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pilot.config import PilotConfig
from pilot.system.http_client import create_httpx_client
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

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


async def download_file(url: str, output_path: str, overwrite: bool = False) -> str:
    """Download a file from a URL to a local path."""
    out_path = Path(output_path)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}. Set overwrite=true.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try httpx first (in our deps)
    try:
        async with create_httpx_client(PilotConfig.load(), follow_redirects=True, timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
            out_path.write_bytes(data)
            size = len(data)
            return f"Downloaded {url} -> {output_path} ({size:,} bytes)\n[DOWNLOAD] {_facts(resp, data)}"
    except Exception as e:
        logger.warning("httpx download failed, falling back to OS tools: %s", e)

    # Fallback to OS tools
    if CURRENT_PLATFORM == Platform.WINDOWS:
        # The address and the file are data ($HelioxArgs), never part of the command text.
        code, out, err = await run_powershell(
            "Invoke-WebRequest -Uri $HelioxArgs['url'] -OutFile $HelioxArgs['out'] -UseBasicParsing",
            values={"url": url, "out": output_path},
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        code, out, err = await run_command(["curl", "-fSL", "-o", output_path, url])
    else:
        code, out, err = await run_command(["wget", "-O", output_path, url])
        if code != 0:
            code, out, err = await run_command(["curl", "-fSL", "-o", output_path, url])

    if code != 0:
        raise RuntimeError(f"Download failed: {err.strip()}")

    size = out_path.stat().st_size if out_path.exists() else 0
    return f"Downloaded {url} -> {output_path} ({size:,} bytes)"
