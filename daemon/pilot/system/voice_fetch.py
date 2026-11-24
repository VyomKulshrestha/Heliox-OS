"""Fetching a local voice's files ahead of use, with the person's consent (the face picker's "Use this face and
download its voice").

Kokoro and Pocket TTS fetch their own files from Hugging Face the first time they speak. When a person picks a face
whose voice is not on the computer and agrees to the download, Heliox fetches those same files first, into the same
Hugging Face cache, while the current voice keeps speaking; the switch to the new voice comes afterwards, between two
replies (``LocalTTSWorkerManager.between_replies``). Which files each engine needs, and their sizes, are data in
``speech_models.json`` ("voices": "files", "voice_file", "voice_bytes"), the one place they are written down.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pilot.system.voice_default import VOICES

HUB_HOST = "huggingface.co"
REACH_SECONDS = 3.0


@dataclass(frozen=True)
class HubFile:
    repo: str
    path: str
    revision: str | None = None


def files_for(engine: str, voice: str, *, model_on_disk: bool, voices: Mapping[str, Any] = VOICES) -> list[HubFile]:
    """The files using ``voice`` of ``engine`` fetches: the voice's own file, and the model's when it is not on disk.
    Pocket's model is fetched by Pocket itself (its file names follow its own configuration); only its voices are
    listed here, which is all a face needs while Pocket already speaks."""
    entry = voices.get(engine) or {}
    files = []
    if not model_on_disk:
        files += [HubFile(str(f["repo"]), str(f["path"]), f.get("revision")) for f in entry.get("files", [])]
    spec = entry.get("voice_file")
    if spec and voice:
        files.append(HubFile(str(spec["repo"]), str(spec["path"]).format(voice=voice), spec.get("revision")))
    return files


def download_bytes(engine: str, *, model_on_disk: bool, voices: Mapping[str, Any] = VOICES) -> int:
    """About how much ``files_for`` fetches (the manifest's "about" sizes)."""
    entry = voices.get(engine) or {}
    size = int(entry.get("voice_bytes", 0))
    return size if model_on_disk else size + int(entry.get("download_bytes", 0))


def network_ok(
    environ: Mapping[str, str] | None = None,
    reach: Callable[[], None] | None = None,
) -> tuple[bool, str]:
    """Whether a download can start now, and the reason in a few words when it cannot. Asks nothing of the network
    but whether huggingface.co answers on its port (``reach``; nothing is sent)."""
    environ = os.environ if environ is None else environ
    if environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False, "Heliox is set to work offline."
    try:
        if reach is None:
            socket.create_connection((HUB_HOST, 443), timeout=REACH_SECONDS).close()
        else:
            reach()
    except OSError:
        return False, "No connection to huggingface.co right now."
    return True, ""


def fetch(files: list[HubFile], download: Callable[..., Any] | None = None) -> None:
    """Fetch ``files`` into the Hugging Face cache, one after another (a file already there is not fetched again).
    Raises on the first that fails."""
    if download is None:
        from huggingface_hub import hf_hub_download as download
    for item in files:
        download(repo_id=item.repo, filename=item.path, revision=item.revision)
