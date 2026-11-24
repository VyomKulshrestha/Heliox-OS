"""Heliox's own voice on the card stays ready while idle unless another component that could use the card asks for it.

A live run on 24 November released the idle voice on the card at 288 s with 3.3 to 3.6 GB free, and the next reply
came in the system voice: Parakeet had found the card held by the voice when it loaded, and that counted as wanting
the card, although Parakeet without the CUDA build of sherpa-onnx never runs there. A component that could not use the
card no longer counts. Made-up machines only: no card is read, nothing loads.
"""

from __future__ import annotations

from pilot.system import gpu_arbiter, gpu_paths, parakeet_asr
from pilot.system.gpu_arbiter import GIB, GpuArbiter, GpuInfo, MachineReading, Needs

MADE_UP_ENV: dict[str, str] = {}  # a made-up machine with every card visible
VOICE_HOLDS = {"component": "voice", "pid": 7}


def _machine() -> GpuArbiter:
    card = GpuInfo(0, "RTX", 60.0, 7 * GIB, 8 * GIB)
    reading = MachineReading("nvidia", (card,), 8 * GIB, 16 * GIB, "")
    return GpuArbiter(environ=MADE_UP_ENV, read=lambda: reading, holder=lambda: VOICE_HOLDS)


def test_a_component_that_could_not_use_the_card_does_not_count_as_wanting_it():
    arbiter = _machine()
    assert arbiter.check("parakeet", Needs(gpu_bytes=GIB), would_use=False).code == "busy"
    assert arbiter.wants() == 0
    assert arbiter.check("ocr", Needs(gpu_bytes=GIB)).code == "busy"  # it could: it counts, as before
    assert arbiter.wants() == 1 and arbiter.last_wanted_by == "ocr"


def test_parakeet_without_its_cuda_build_leaves_the_voice_its_card(monkeypatch):
    arbiter = _machine()
    monkeypatch.setattr(gpu_arbiter, "_shared", arbiter)
    monkeypatch.setattr(gpu_paths, "sherpa_onnx_cuda", lambda: False)
    device, _why = parakeet_asr.choose_provider()
    assert device == "cpu" and arbiter.wants() == 0
    monkeypatch.setattr(gpu_paths, "sherpa_onnx_cuda", lambda: True)  # with it, a held card is a real want
    device, why = parakeet_asr.choose_provider()
    assert device == "cpu" and "in use" in why and arbiter.wants() == 1


def test_screen_reading_with_a_cpu_build_of_pytorch_leaves_the_voice_its_card(monkeypatch):
    from pilot.system import vision

    arbiter = _machine()
    monkeypatch.setattr(gpu_arbiter, "_shared", arbiter)
    monkeypatch.setattr(gpu_paths, "torch_build", lambda platform=None: "cpu")
    read: list[bool] = []

    class Reader:
        def __init__(self, gpu: bool) -> None:
            self.gpu = gpu

        def readtext(self, image):
            read.append(self.gpu)
            return []

    monkeypatch.setattr(vision, "_easyocr_reader", lambda module, language, gpu: Reader(gpu))
    assert vision._easyocr_read(object(), object()) == []
    assert read == [False] and arbiter.wants() == 0
