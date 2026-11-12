"""The speech models Heliox can use, and how the person downloads them: never by themselves.

Each model is listed with what it is for, its size, licence and source. A download starts only when the person
presses it (the first-run screen or Settings, Voice), resumes after an interruption (HTTP Range into a ``.part``
file), is checked against the SHA-256 written here before it counts, and lands under Heliox's own data folder:
``<data>/models/<folder>``, where DATA_DIR already follows each system's convention (``%LOCALAPPDATA%`` on Windows,
``~/Library/Application Support`` on macOS, ``$XDG_DATA_HOME`` or ``~/.local/share`` on Linux).

The catalogue lives in ``speech_models.json`` beside this file, which the first-run screen also bundles. The
checksums were taken from the files as their own libraries fetched them (Hugging Face verifies large files
against the same SHA-256 when it serves them) and, for Parakeet, from the release archive.

Pure parts (the catalogue, paths, status, plan of what is left to fetch) are tested everywhere; the network part
takes an injectable opener so tests never touch the network.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.speech_models")

CHUNK = 1 << 20


@dataclass(frozen=True)
class ModelFile:
    name: str  # path inside the model folder
    size: int
    sha256: str
    url: str = ""  # direct download of this file ("" when it comes from the archive)


@dataclass(frozen=True)
class SpeechModel:
    id: str
    title: str
    purpose: str
    folder: str
    licence: str
    source: str
    languages: tuple[str, ...]
    files: tuple[ModelFile, ...]
    archive_url: str = ""  # a .tar.bz2 holding the files under ``archive_root``
    archive_size: int = 0
    archive_sha256: str = ""
    archive_root: str = ""
    required_for: str = ""  # the setting or feature that needs it
    part_of_voice: bool = False  # comes with the Voice choice itself (Whisper base), not as an extra

    @property
    def download_bytes(self) -> int:
        return self.archive_size or sum(f.size for f in self.files)

    @property
    def disk_bytes(self) -> int:
        return sum(f.size for f in self.files)


MANIFEST = Path(__file__).with_name("speech_models.json")


def load_catalogue(path: Path = MANIFEST) -> tuple[SpeechModel, ...]:
    """The catalogue from speech_models.json, the one file the daemon and the first-run screen both read."""
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    models = []
    for entry in data["models"]:
        archive = entry.get("archive") or {}
        language = entry.get("language", "multilingual")
        models.append(
            SpeechModel(
                id=entry["id"],
                title=entry["title"],
                purpose=entry["purpose"],
                folder=entry["folder"],
                licence=entry["licence"],
                source=entry["source"],
                languages=(language,),
                files=tuple(
                    ModelFile(f["name"], int(f["bytes"]), f["sha256"], f.get("url", "")) for f in entry["files"]
                ),
                archive_url=archive.get("url", ""),
                archive_size=int(archive.get("bytes", 0)),
                archive_sha256=archive.get("sha256", ""),
                archive_root=archive.get("root", ""),
                required_for=entry.get("required_for", ""),
                part_of_voice=bool(entry.get("part_of_voice", False)),
            )
        )
    return tuple(models)


CATALOGUE: tuple[SpeechModel, ...] = load_catalogue()
BY_ID = {model.id: model for model in CATALOGUE}


def speaks_english(language: str, system_locale: str | None = None) -> bool:
    """Whether the recommended set is the English one: the voice language setting, or for "auto" the system's
    locale ("English_United States" on Windows, "en_GB" on macOS and Linux). Unknown counts as English."""
    language = (language or "auto").strip().casefold()
    if language != "auto":
        return language.startswith("en")
    if system_locale is None:
        import locale

        try:
            system_locale = locale.getlocale()[0] or os.environ.get("LANG", "")
        except (ValueError, TypeError):
            system_locale = ""
    folded = (system_locale or "").casefold()
    return not folded or folded.startswith(("en", "english", "c.", "posix")) or folded == "c"


def _host_name(url: str) -> str:
    """Where a download comes from, as a person would say it: "GitHub", "Hugging Face", else the bare host name
    (a mirror or a company server: "models.example.com")."""
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").casefold()
    if not host:
        return ""
    if host == "github.com" or host.endswith((".github.com", ".githubusercontent.com")):
        return "GitHub"
    if host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co"):
        return "Hugging Face"
    return host.removeprefix("www.")


def _joined(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def default_whisper_model(language: str = "auto", system_locale: str | None = None) -> str:
    """The Whisper size a first-run person's settings start with: the one the recommended download brings for
    their language ("base" with Parakeet for English, "small" otherwise; see the manifest's "recommended")."""
    return "base" if speaks_english(language, system_locale) else "small"


def recommended(language: str, data_dir: Path | None = None, *, path: Path = MANIFEST) -> dict[str, Any]:
    """The set of speech models recommended for this language setting, its download size and what is still
    missing: {"key", "title", "why", "ids", "download_bytes", "missing", "missing_bytes"}."""
    import json

    sets = json.loads(path.read_text(encoding="utf-8")).get("recommended", {})
    key = "en" if speaks_english(language) else "other"
    chosen = sets.get(key) or {}
    models = [BY_ID[i] for i in chosen.get("ids", []) if i in BY_ID]

    def present(model: SpeechModel) -> bool:
        if installed(model, data_dir):
            return True
        if model.id.startswith("whisper-"):  # also found in the Hugging Face cache, where Whisper puts it itself
            from pilot.system.model_presence import whisper_present

            return whisper_present(model.id.split("-", 1)[1])
        return False

    missing = [m for m in models if not present(m)]
    partial = 0
    for m in missing:
        folder = model_path(m, data_dir)
        partial += sum(f.stat().st_size for f in folder.glob("*.part")) if folder.is_dir() else 0
    hosts: list[str] = []
    for m in models:
        for url in [m.archive_url, *(f.url for f in m.files)]:
            host = _host_name(url)
            if host and host not in hosts:
                hosts.append(host)
    return {
        "key": key,
        "title": chosen.get("title", ""),
        "why": chosen.get("why", ""),
        "ids": [m.id for m in models],
        "download_bytes": sum(m.download_bytes for m in models),
        "missing": [m.id for m in missing],
        "missing_bytes": sum(m.download_bytes for m in missing),
        "partial_bytes": partial,  # already fetched by a stopped download: the next one continues from here
        "from": _joined(hosts),
        "where": (f"Downloaded once from {_joined(hosts)}; t" if hosts else "Downloaded once; t")
        + "he models stay on this computer and recognise your speech here.",
    }


def models_dir(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        from pilot.config import DATA_DIR

        data_dir = Path(DATA_DIR)
    return Path(data_dir) / "models"


def model_path(model: SpeechModel, data_dir: Path | None = None) -> Path:
    return models_dir(data_dir) / model.folder


def installed(model: SpeechModel, data_dir: Path | None = None) -> bool:
    """Every file present at its size (the full checksum is checked once, when the download finishes)."""
    folder = model_path(model, data_dir)
    return all((folder / f.name).is_file() and (folder / f.name).stat().st_size == f.size for f in model.files)


def status(data_dir: Path | None = None) -> list[dict[str, Any]]:
    out = []
    for model in CATALOGUE:
        folder = model_path(model, data_dir)
        part = sum(p.stat().st_size for p in folder.glob("*.part")) if folder.is_dir() else 0
        out.append(
            {
                "id": model.id,
                "title": model.title,
                "purpose": model.purpose,
                "languages": list(model.languages),
                "licence": model.licence,
                "source": model.source,
                "download_bytes": model.download_bytes,
                "disk_bytes": model.disk_bytes,
                "required_for": model.required_for,
                "part_of_voice": model.part_of_voice,
                "installed": installed(model, data_dir),
                "partial_bytes": part,
                "path": str(folder),
            }
        )
    return out


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class DownloadError(Exception):
    """A sentence the person can read: what went wrong and what to do."""


Opener = Callable[[str, int], Any]  # (url, start byte) -> response with .status, .headers, .read(n)


def _default_opener(url: str, start: int) -> Any:
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "Heliox"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(request, timeout=30)  # noqa: S310 - fixed https URLs from the catalogue


@dataclass
class Progress:
    model_id: str
    state: str = "queued"  # queued | downloading | verifying | ready | failed | cancelled
    received: int = 0
    total: int = 0
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "state": self.state,
            "received": self.received,
            "total": self.total,
            "message": self.message,
        }


@dataclass
class Downloader:
    """Fetches one model at a time on a worker thread; report() is called with Progress as it goes."""

    data_dir: Path | None = None
    opener: Opener = _default_opener
    report: Callable[[Progress], None] = lambda _progress: None
    _cancel: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._cancel.set()

    def run(self, model_id: str) -> Progress:
        model = BY_ID.get(model_id)
        progress = Progress(model_id, total=0)
        if model is None:
            progress.state, progress.message = "failed", f"There is no speech model called {model_id}."
            self.report(progress)
            return progress
        progress.total = model.download_bytes
        self._cancel.clear()
        folder = model_path(model, self.data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if installed(model, self.data_dir):
                progress.state, progress.received = "ready", progress.total
                progress.message = f"{model.title} is ready."
            elif model.archive_url:
                self._fetch_archive(model, folder, progress)
            else:
                self._fetch_files(model, folder, progress)
        except DownloadError as error:
            progress.state, progress.message = "failed", str(error)
        except _Cancelled:
            progress.state, progress.message = "cancelled", "Download stopped. It continues from here next time."
        self.report(progress)
        return progress

    # ── the parts ──

    def _fetch_files(self, model: SpeechModel, folder: Path, progress: Progress) -> None:
        done = 0
        for item in model.files:
            target = folder / item.name
            if target.is_file() and target.stat().st_size == item.size:
                done += item.size
                continue
            part = folder / f"{item.name}.part"
            self._fetch(item.url, part, item.size, progress, base=done)
            progress.state, progress.message = "verifying", f"Checking {item.name}…"
            self.report(progress)
            if sha256_of(part) != item.sha256:
                part.unlink(missing_ok=True)
                raise DownloadError(
                    f"{item.name} did not match its checksum, so it was deleted. Try again; if it fails again, the "
                    "download source may have changed."
                )
            part.replace(target)
            done += item.size
        progress.state, progress.received = "ready", progress.total
        progress.message = f"{model.title} is ready."

    def _fetch_archive(self, model: SpeechModel, folder: Path, progress: Progress) -> None:
        part = folder / "download.tar.bz2.part"
        self._fetch(model.archive_url, part, model.archive_size, progress, base=0)
        progress.state, progress.message = "verifying", "Checking the download…"
        self.report(progress)
        if sha256_of(part) != model.archive_sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(
                "The download did not match its checksum, so it was deleted. Try again; if it fails again, the "
                "download source may have changed."
            )
        wanted = {f"{model.archive_root}/{f.name}": f for f in model.files}
        with tarfile.open(part, "r:bz2") as archive:
            for member in archive.getmembers():
                item = wanted.get(member.name)
                if item is None or not member.isfile():
                    continue  # only the listed files, by exact name: nothing can be written outside the folder
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, open(folder / f"{item.name}.part", "wb") as out:
                    shutil.copyfileobj(source, out, CHUNK)
        for item in model.files:
            extracted = folder / f"{item.name}.part"
            if not extracted.is_file() or sha256_of(extracted) != item.sha256:
                raise DownloadError(f"{item.name} was missing or damaged in the download. Try again.")
            extracted.replace(folder / item.name)
        part.unlink(missing_ok=True)
        progress.state, progress.received = "ready", progress.total
        progress.message = f"{model.title} is ready."

    def _fetch(self, url: str, part: Path, size: int, progress: Progress, *, base: int) -> None:
        start = part.stat().st_size if part.is_file() else 0
        if start >= size:
            return
        progress.state = "downloading"
        try:
            response = self.opener(url, start)
        except OSError as error:
            raise DownloadError(
                f"Could not reach the download ({type(error).__name__}). Check the internet connection and try "
                "again; what was already downloaded is kept."
            ) from error
        status_code = int(getattr(response, "status", 200) or 200)
        if start and status_code != 206:
            start = 0  # the server ignored the range: start this file again
        mode = "ab" if start else "wb"
        received = start
        try:
            with response, open(part, mode) as out:
                while True:
                    if self._cancel.is_set():
                        raise _Cancelled
                    block = response.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    received += len(block)
                    progress.received = base + received
                    progress.message = (
                        f"Downloading… {progress.received // (1 << 20)} of {progress.total // (1 << 20)} MB"
                    )
                    self.report(progress)
        except OSError as error:
            raise DownloadError(
                f"The download was interrupted ({type(error).__name__}). Press download again to continue from "
                f"{received // (1 << 20)} MB."
            ) from error
        if received < size:
            raise DownloadError(
                f"The download stopped early ({received // (1 << 20)} of {size // (1 << 20)} MB). Press download again "
                "to continue."
            )


class _Cancelled(Exception):
    pass


def local_path(model_id: str, data_dir: Path | None = None) -> Path | None:
    """The folder of an installed model, for the loaders; None when it is not downloaded."""
    model = BY_ID[model_id]
    return model_path(model, data_dir) if installed(model, data_dir) else None
