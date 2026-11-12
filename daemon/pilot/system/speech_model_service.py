"""The daemon side of speech-model downloads: four RPCs and one notification, used by the first-run screen and
Settings, Voice. A download starts only when one of them asks; one model downloads at a time, on a worker thread.

RPCs (registered in server.py):
  speech_models_list                  -> {"status": "ok", "models": [...], "active": id | "", "waiting": [ids],
                                          "recommended": {...}}  (pilot.system.speech_models.recommended)
  speech_model_download {"id"}        -> {"status": "started" | "queued" | "ready" | "error", ...}
                                         one at a time; asked during another download, it waits its turn
  speech_model_cancel {"id"}          -> {"status": "cancelling" | "idle"}
  speech_model_remove {"id"}          -> {"status": "removed" | "error", ...}
Notification:
  speech_model_progress {"id", "state", "received", "total", "message"}
    state: queued | downloading | verifying | ready | failed | cancelled
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pilot.system import speech_models

logger = logging.getLogger("pilot.system.speech_model_service")

Broadcast = Callable[[str, dict[str, Any]], Awaitable[None]]


class SpeechModelService:
    def __init__(
        self,
        broadcast: Broadcast,
        *,
        data_dir: Path | None = None,
        opener: speech_models.Opener | None = None,
        on_ready: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._broadcast = broadcast
        self._data_dir = data_dir
        self._opener = opener
        self._on_ready = on_ready
        self._active: str = ""
        self._downloader: speech_models.Downloader | None = None
        self._lock = threading.Lock()
        self._task: asyncio.Task[None] | None = None
        self._last_sent = float("-inf")
        self._waiting: list[str] = []  # asked for while another downloaded: started in order, one at a time

    def list(self, language: str = "auto") -> dict[str, Any]:
        return {
            "status": "ok",
            "models": speech_models.status(self._data_dir),
            "active": self._active,
            "waiting": list(self._waiting),
            "recommended": speech_models.recommended(language, self._data_dir),
        }

    async def download(self, model_id: str) -> dict[str, Any]:
        model = speech_models.BY_ID.get(str(model_id or ""))
        if model is None:
            return {"status": "error", "message": f"There is no speech model called {model_id!r}."}
        if speech_models.installed(model, self._data_dir):
            await self._finished(model.id)
            return {"status": "ready", "message": f"{model.title} is already downloaded."}
        with self._lock:
            if self._active:
                if model.id != self._active and model.id not in self._waiting:
                    self._waiting.append(model.id)
                waiting = True
            else:
                self._active = model.id
                waiting = False
        if waiting and model.id == self._active:
            return {"status": "started", "id": model.id, "total": model.download_bytes}  # already under way
        if waiting:
            await self._broadcast(
                "speech_model_progress",
                speech_models.Progress(
                    model.id, "queued", total=model.download_bytes, message="Waiting for the other download…"
                ).as_dict(),
            )
            return {"status": "queued", "id": model.id, "message": "It starts when the download before it ends."}
        loop = asyncio.get_running_loop()

        def report(progress: speech_models.Progress) -> None:
            import time

            now = time.monotonic()
            if progress.state == "downloading" and now - self._last_sent < 0.5:
                return  # at most two progress notifications a second
            self._last_sent = now
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._broadcast("speech_model_progress", progress.as_dict()))
            )

        kwargs: dict[str, Any] = {"data_dir": self._data_dir, "report": report}
        if self._opener is not None:
            kwargs["opener"] = self._opener
        downloader = speech_models.Downloader(**kwargs)
        self._downloader = downloader

        async def run() -> None:
            try:
                result = await asyncio.to_thread(downloader.run, model.id)
            except Exception as error:  # a bug must not leave the slot taken
                logger.exception("Speech model download failed")
                result = speech_models.Progress(model.id, "failed", message=f"The download failed ({error}).")
                await self._broadcast("speech_model_progress", result.as_dict())
            finally:
                with self._lock:
                    self._active = ""
                    self._downloader = None
                    following = self._waiting.pop(0) if self._waiting else ""
            if result.state == "ready":
                await self._finished(model.id)
            if following:
                await self.download(following)

        await self._broadcast(
            "speech_model_progress",
            speech_models.Progress(model.id, "queued", total=model.download_bytes, message="Starting…").as_dict(),
        )
        self._task = asyncio.ensure_future(run())
        return {"status": "started", "id": model.id, "total": model.download_bytes}

    def cancel(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            if self._downloader is not None and self._active == model_id:
                self._downloader.cancel()
                return {"status": "cancelling"}
            if model_id in self._waiting:
                self._waiting.remove(model_id)
                return {"status": "cancelled"}
        return {"status": "idle"}

    def remove(self, model_id: str) -> dict[str, Any]:
        model = speech_models.BY_ID.get(str(model_id or ""))
        if model is None:
            return {"status": "error", "message": f"There is no speech model called {model_id!r}."}
        if self._active == model.id:
            return {"status": "error", "message": "Stop the download first."}
        folder = speech_models.model_path(model, self._data_dir)
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        return {"status": "removed", "id": model.id}

    async def _finished(self, model_id: str) -> None:
        if self._on_ready is not None:
            try:
                await self._on_ready(model_id)
            except Exception:
                logger.warning("Could not switch to the downloaded speech model", exc_info=True)
