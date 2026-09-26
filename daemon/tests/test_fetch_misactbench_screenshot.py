"""Bounded public ZIP access checks, without network calls."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from scripts import fetch_misactbench_screenshot as fetch
from scripts.probe_misactbench_visual import PROMPT_V1

PNG = b"\x89PNG\r\n\x1a\n" + b"fixture"


def _archive(name: str) -> tuple[bytes, bytes]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, PNG)
    data = stream.getvalue()
    marker = data.rfind(b"PK\x05\x06")
    _, _, _, _, entries, size, offset, _ = struct.unpack_from("<4s4H2LH", data, marker)
    assert entries == 1
    return data, data[offset:offset + size]


def test_central_and_member_crc_with_bounded_ranges(monkeypatch) -> None:
    name = "trajectories/t/step_0.png"
    archive, central = _archive(name)
    index = fetch.parse_central(central, 1)
    monkeypatch.setattr(fetch, "ARCHIVE_BYTES", len(archive))
    monkeypatch.setattr(fetch, "get_range",
                        lambda start, end, *, max_bytes: archive[start:end + 1])
    assert fetch.member_bytes(name, index[name]) == PNG
    changed = {**index[name], "crc32": 0}
    with pytest.raises(ValueError, match="CRC"):
        fetch.member_bytes(name, changed)


def test_central_rejects_path_traversal_and_incomplete_data() -> None:
    _, central = _archive("../outside.png")
    with pytest.raises(ValueError, match="Unsafe"):
        fetch.parse_central(central, 1)
    with pytest.raises(ValueError, match="ended early"):
        fetch.parse_central(central[:20], 1)


def test_visual_probe_pins_selected_cases_and_prompt() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    selection = root / "misactbench_visual_selection_v1.json"
    report = json.loads((root / "misactbench_visual_gpu_recheck_v1.json").read_text())
    case_ids = json.loads(selection.read_text())["case_ids"]
    assert len(case_ids) == len(set(case_ids)) == 16
    assert report["selection_sha256"] == hashlib.sha256(selection.read_bytes()).hexdigest()
    assert report["prompt_v1_sha256"] == hashlib.sha256(PROMPT_V1.encode()).hexdigest()
    assert report["archive_central_sha256"] == fetch.CENTRAL_SHA256
