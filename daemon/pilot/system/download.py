"""Download files from URLs.

Cross-platform using httpx (preferred) or OS-native tools.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command, run_powershell

logger = logging.getLogger("pilot.system.download")


async def download_file(url: str, output_path: str, overwrite: bool = False) -> str:
    """Download a file from a URL to a local path."""
    out_path = Path(output_path)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}. Set overwrite=true.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try httpx first (in our deps)
    try:
        import httpx

        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            size = len(resp.content)
            return f"Downloaded {url} -> {output_path} ({size:,} bytes)"
    except ImportError:
        pass
    except Exception as e:
        logger.warning("httpx download failed, falling back to OS tools: %s", e)

    # Fallback to OS tools
    if CURRENT_PLATFORM == Platform.WINDOWS:
        code, out, err = await run_powershell(
            f"Invoke-WebRequest -Uri '{url}' -OutFile '{output_path}' -UseBasicParsing"
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
