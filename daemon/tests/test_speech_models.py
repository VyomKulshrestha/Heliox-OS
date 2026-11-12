"""Speech models are downloaded only when the person asks, resume after an interruption, are checked against their
SHA-256 before they count, and land under Heliox's data folder. No network: a fake opener serves the bytes."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from pilot.system import speech_models
from pilot.system.speech_models import CATALOGUE, Downloader, ModelFile, SpeechModel, installed, status


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200) -> None:
        super().__init__(data)
        self.status = status


class Server:
    """Serves files by URL, honouring Range; can cut a transfer short once to simulate a dropped connection."""

    def __init__(self, files: dict[str, bytes], cut_after: int | None = None, ranges: bool = True) -> None:
        self.files, self.cut_after, self.ranges = files, cut_after, ranges
        self.requests: list[tuple[str, int]] = []

    def __call__(self, url: str, start: int):
        self.requests.append((url, start))
        data = self.files[url]
        if start and self.ranges:
            body, status = data[start:], 206
        else:
            body, status = data, 200
        if self.cut_after is not None:
            body, self.cut_after = body[: self.cut_after], None
        return Response(body, status)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_model(contents: dict[str, bytes]) -> SpeechModel:
    return SpeechModel(
        id="test-files",
        title="Test model",
        purpose="",
        folder="test-files",
        licence="MIT",
        source="test",
        languages=("en",),
        files=tuple(ModelFile(n, len(d), sha(d), f"https://example.test/{n}") for n, d in contents.items()),
    )


@pytest.fixture
def catalogue(monkeypatch):
    def use(model: SpeechModel) -> SpeechModel:
        monkeypatch.setitem(speech_models.BY_ID, model.id, model)
        return model

    return use


def test_the_catalogue_says_what_each_model_is_its_size_licence_and_source():
    ids = [model.id for model in CATALOGUE]
    assert ids == [
        "whisper-base",
        "parakeet-tdt-0.6b-v2-int8",
        "whisper-small",
        "streaming-zipformer-en",
        "smart-turn-v3",
    ]
    for model in CATALOGUE:
        assert model.title and model.purpose and model.licence and model.source and model.languages
        assert all(len(f.sha256) == 64 for f in model.files)
        assert model.download_bytes > 0 and (
            model.archive_url or all(f.url.startswith("https://") for f in model.files)
        )
    parakeet = speech_models.BY_ID["parakeet-tdt-0.6b-v2-int8"]
    assert parakeet.languages == ("en",) and parakeet.archive_size == 482440461  # English only, said in the app


def test_files_land_under_the_data_folder_and_count_only_when_their_checksum_matches(tmp_path, catalogue):
    contents = {"model.bin": b"weights" * 1000, "tokens.txt": b"a b c"}
    model = catalogue(file_model(contents))
    server = Server({f"https://example.test/{n}": d for n, d in contents.items()})
    reports = []
    result = Downloader(data_dir=tmp_path, opener=server, report=lambda p: reports.append(p.as_dict())).run(model.id)
    assert result.state == "ready" and installed(model, tmp_path)
    assert (tmp_path / "models" / "test-files" / "model.bin").read_bytes() == contents["model.bin"]
    assert reports[-1]["state"] == "ready" and any(r["state"] == "verifying" for r in reports)
    again = Downloader(data_dir=tmp_path, opener=server).run(model.id)
    assert again.state == "ready" and len(server.requests) == 2  # nothing fetched twice


def test_an_interrupted_download_continues_where_it_stopped(tmp_path, catalogue):
    data = bytes(range(256)) * 5000
    model = catalogue(file_model({"model.bin": data}))
    server = Server({"https://example.test/model.bin": data}, cut_after=300_000)
    first = Downloader(data_dir=tmp_path, opener=server).run(model.id)
    assert first.state == "failed" and "continue" in first.message
    second = Downloader(data_dir=tmp_path, opener=server).run(model.id)
    assert second.state == "ready"
    assert server.requests[1] == ("https://example.test/model.bin", 300_000)  # a Range request from the cut


def test_a_server_that_ignores_ranges_starts_the_file_again(tmp_path, catalogue):
    data = b"x" * 400_000
    model = catalogue(file_model({"model.bin": data}))
    server = Server({"https://example.test/model.bin": data}, cut_after=100_000, ranges=False)
    Downloader(data_dir=tmp_path, opener=server).run(model.id)
    assert Downloader(data_dir=tmp_path, opener=server).run(model.id).state == "ready"
    assert (tmp_path / "models" / "test-files" / "model.bin").read_bytes() == data


def test_a_damaged_file_is_deleted_and_not_installed(tmp_path, catalogue):
    good = b"good" * 100
    model = catalogue(file_model({"model.bin": good}))
    server = Server({"https://example.test/model.bin": b"evil" * 100})
    result = Downloader(data_dir=tmp_path, opener=server).run(model.id)
    assert result.state == "failed" and "checksum" in result.message
    assert not installed(model, tmp_path)
    assert not list((tmp_path / "models" / "test-files").glob("*"))


def test_offline_says_so_and_keeps_what_was_downloaded(tmp_path, catalogue):
    model = catalogue(file_model({"model.bin": b"z" * 10}))

    def offline(url, start):
        raise OSError("getaddrinfo failed")

    result = Downloader(data_dir=tmp_path, opener=offline).run(model.id)
    assert result.state == "failed" and "internet connection" in result.message


def test_an_archive_is_checked_then_only_its_listed_files_are_taken(tmp_path, catalogue):
    files = {"encoder.onnx": b"enc" * 500, "tokens.txt": b"t"}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        for name, data in {**files, "../../evil.txt": b"no"}.items():
            info = tarfile.TarInfo(f"pkg/{name}" if not name.startswith("..") else name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    blob = buffer.getvalue()
    model = catalogue(
        SpeechModel(
            id="test-archive",
            title="Archive model",
            purpose="",
            folder="test-archive",
            licence="CC-BY-4.0",
            source="test",
            languages=("en",),
            files=tuple(ModelFile(n, len(d), sha(d)) for n, d in files.items()),
            archive_url="https://example.test/pkg.tar.bz2",
            archive_size=len(blob),
            archive_sha256=sha(blob),
            archive_root="pkg",
        )
    )
    result = Downloader(data_dir=tmp_path, opener=Server({"https://example.test/pkg.tar.bz2": blob})).run(model.id)
    assert result.state == "ready" and installed(model, tmp_path)
    folder = tmp_path / "models" / "test-archive"
    assert sorted(p.name for p in folder.iterdir()) == ["encoder.onnx", "tokens.txt"]
    assert not (tmp_path / "evil.txt").exists() and not Path(tmp_path.parent / "evil.txt").exists()


def test_status_lists_every_model_with_what_is_on_disk(tmp_path):
    rows = status(tmp_path)
    assert [row["id"] for row in rows] == [model.id for model in CATALOGUE]
    assert all(row["installed"] is False and row["partial_bytes"] == 0 for row in rows)
    assert rows[1]["path"].endswith("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")


def test_cancelling_keeps_the_part_for_next_time(tmp_path, catalogue):
    data = b"q" * (3 << 20)
    model = catalogue(file_model({"model.bin": data}))
    downloader = Downloader(data_dir=tmp_path, opener=Server({"https://example.test/model.bin": data}))
    downloader.report = lambda progress: downloader.cancel() if progress.received >= (1 << 20) else None
    result = downloader.run(model.id)
    assert result.state == "cancelled"
    assert (tmp_path / "models" / "test-files" / "model.bin.part").stat().st_size >= 1 << 20


# ── the service behind the RPCs ──


@pytest.mark.asyncio
async def test_the_service_downloads_on_request_reports_progress_and_switches_when_ready(tmp_path, catalogue):
    import asyncio

    from pilot.system.speech_model_service import SpeechModelService

    contents = {"model.bin": b"w" * 2_500_000}
    model = catalogue(file_model(contents))
    sent, ready = [], []

    async def broadcast(method, params):
        sent.append((method, dict(params)))

    async def on_ready(model_id):
        ready.append(model_id)

    service = SpeechModelService(
        broadcast, data_dir=tmp_path, opener=Server({"https://example.test/model.bin": contents["model.bin"]}),
        on_ready=on_ready,
    )  # fmt: skip
    assert service.list()["active"] == ""
    started = await service.download(model.id)
    assert started["status"] == "started"
    assert (await service.download(model.id))["status"] == "started"  # already under way: not started twice
    assert service.list()["waiting"] == []
    for _ in range(200):
        if ready:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)
    states = [params["state"] for method, params in sent if method == "speech_model_progress"]
    assert states[0] == "queued" and states[-1] == "ready" and "verifying" in states
    assert ready == [model.id] and service.list()["active"] == ""
    assert (await service.download(model.id))["status"] == "ready"
    assert service.remove(model.id)["status"] == "removed" and not installed(model, tmp_path)
    assert (await service.download("nope"))["status"] == "error"


@pytest.mark.asyncio
async def test_parakeet_becomes_the_final_pass_only_when_its_folder_was_not_chosen_by_hand():
    from unittest.mock import AsyncMock

    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server.config.save = lambda *a, **k: None
    server._broadcast_notification = AsyncMock()
    await server._speech_model_ready("parakeet-tdt-0.6b-v2-int8")
    assert server.config.voice.final_pass == "parakeet"
    other = PilotServer(PilotConfig())
    other.config.save = lambda *a, **k: None
    other._broadcast_notification = AsyncMock()
    other.config.voice.parakeet_model_dir = "D:/my-models/parakeet"
    await other._speech_model_ready("parakeet-tdt-0.6b-v2-int8")
    assert other.config.voice.final_pass == "whisper"  # the person pointed it elsewhere: left alone
    await other._speech_model_ready("smart-turn-v3")
    assert other.config.voice.final_pass == "whisper"


def test_the_manifest_is_the_single_catalogue_and_has_the_fields_the_first_run_screen_reads():
    import json

    data = json.loads(speech_models.MANIFEST.read_text(encoding="utf-8"))
    for entry in data["models"]:
        assert {"id", "title", "purpose", "language", "size_bytes", "licence", "sources", "files"} <= set(entry)
        assert all({"name", "sha256", "bytes"} <= set(f) for f in entry["files"])
    assert [entry["id"] for entry in data["models"]] == [model.id for model in CATALOGUE]


def test_the_loaders_use_a_downloaded_model_from_the_data_folder(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    import pilot.system.voice as voice

    folder = tmp_path / "models" / "faster-whisper-base"
    folder.mkdir(parents=True)
    for f in speech_models.BY_ID["whisper-base"].files:
        with open(folder / f.name, "wb") as out:
            out.truncate(f.size)  # the right sizes; installed() checks sizes, the checksum was checked on download
    monkeypatch.setattr(speech_models, "models_dir", lambda data_dir=None: tmp_path / "models")
    made = []
    with (
        patch.dict(
            "sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=lambda name, **k: made.append(name))}
        ),
        patch.object(voice, "_faster_whisper_runtime", return_value=("cpu", "int8")),
    ):
        voice._faster_whisper_model_cache.clear()
        voice._get_faster_whisper_model("base")
        voice._faster_whisper_model_cache.clear()
    assert made == [str(folder)]


@pytest.mark.asyncio
async def test_a_second_model_asked_for_during_a_download_waits_its_turn(tmp_path, catalogue):
    import asyncio

    from pilot.system.speech_model_service import SpeechModelService

    first_bytes, second_bytes = {"model.bin": b"a" * 1_500_000}, {"model.bin": b"b" * 1_000}
    first = catalogue(file_model(first_bytes))
    import dataclasses

    second = catalogue(
        dataclasses.replace(
            file_model(second_bytes),
            id="test-second",
            folder="test-second",
            files=(ModelFile("model.bin", 1_000, sha(b"b" * 1_000), "https://example.test/second.bin"),),
        )
    )
    ready: list[str] = []

    async def broadcast(method, params):
        return None

    async def on_ready(model_id):
        ready.append(model_id)

    files = {
        "https://example.test/model.bin": first_bytes["model.bin"],
        "https://example.test/second.bin": b"b" * 1_000,
    }
    service = SpeechModelService(broadcast, data_dir=tmp_path, opener=Server(files), on_ready=on_ready)
    assert (await service.download(first.id))["status"] == "started"
    assert (await service.download(second.id))["status"] == "queued"
    assert service.list()["waiting"] == [second.id]
    for _ in range(300):
        if len(ready) == 2:
            break
        await asyncio.sleep(0.02)
    assert ready == [first.id, second.id]  # in the order asked, one at a time


def test_the_recommended_set_follows_the_language_setting():
    english = speech_models.recommended("en")
    assert english["ids"] == ["parakeet-tdt-0.6b-v2-int8", "whisper-base"]
    assert english["download_bytes"] == sum(speech_models.BY_ID[i].download_bytes for i in english["ids"])
    assert speech_models.recommended("de")["ids"] == ["whisper-small"]
    assert speech_models.speaks_english("auto", "English_United States")  # Windows
    assert speech_models.speaks_english("auto", "en_GB")  # macOS, Linux
    assert speech_models.speaks_english("auto", "C")  # a bare Linux container: nothing says otherwise
    assert not speech_models.speaks_english("auto", "de_DE")
    assert not speech_models.speaks_english("auto", "German_Germany")


def test_the_recommended_set_says_where_it_comes_from_and_counts_a_stopped_download(tmp_path):
    offer = speech_models.recommended("en", tmp_path)
    assert offer["from"] == "GitHub and Hugging Face" and "stay on this computer" in offer["where"]
    assert offer["partial_bytes"] == 0
    parakeet = speech_models.BY_ID["parakeet-tdt-0.6b-v2-int8"]
    folder = speech_models.model_path(parakeet, tmp_path)
    folder.mkdir(parents=True)
    (folder / "download.tar.bz2.part").write_bytes(b"x" * 1234)  # what Stop leaves behind
    assert speech_models.recommended("en", tmp_path)["partial_bytes"] == 1234


def test_a_mirror_or_company_server_is_named_by_its_host_and_the_sentence_has_no_hole(tmp_path, monkeypatch):
    import json

    mirror = SpeechModel(
        id="mirror-model", title="Mirror (Whisper base)", purpose="", folder="mirror", licence="MIT", source="mirror",
        languages=("en",),
        files=(ModelFile("model.bin", 10, "0" * 64, "https://models.example.com/whisper/model.bin"),),
    )  # fmt: skip
    bare = SpeechModel(
        id="bare-model", title="Bare", purpose="", folder="bare", licence="MIT", source="?", languages=("en",),
        files=(ModelFile("model.bin", 10, "0" * 64, ""),),
    )  # fmt: skip
    monkeypatch.setitem(speech_models.BY_ID, mirror.id, mirror)
    monkeypatch.setitem(speech_models.BY_ID, bare.id, bare)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"recommended": {"en": {"ids": ["mirror-model"]}, "other": {"ids": ["bare-model"]}}})
    )
    offer = speech_models.recommended("en", tmp_path, path=manifest)
    assert offer["from"] == "models.example.com"
    assert offer["where"].startswith("Downloaded once from models.example.com; the models stay on this computer")
    nowhere = speech_models.recommended("de", tmp_path, path=manifest)  # no URL at all
    assert nowhere["from"] == "" and nowhere["where"].startswith("Downloaded once; the models stay")
    assert "from ;" not in nowhere["where"] and "  " not in nowhere["where"]
