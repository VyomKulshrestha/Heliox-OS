"""Nothing in the daemon process opens a CUDA context by itself: the swarm hardware check reads nvidia-smi through the
GPU arbiter, and Chroma's embedding model is pinned to the CPU."""

import sys
from types import ModuleType, SimpleNamespace

from pilot.memory import store
from pilot.swarm.swarm_manager import SwarmManager
from pilot.system import gpu_arbiter
from pilot.system.gpu_arbiter import GIB, GpuArbiter, GpuInfo, MachineReading


class _ExplodingTorch(ModuleType):
    """A torch whose CUDA must not be touched."""

    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = SimpleNamespace(
            is_available=self._fail, get_device_name=self._fail, get_device_properties=self._fail
        )

    @staticmethod
    def _fail(*_args):
        raise AssertionError("the daemon touched torch.cuda")


async def _hardware(monkeypatch, reading: MachineReading) -> dict:
    monkeypatch.setitem(sys.modules, "torch", _ExplodingTorch())
    monkeypatch.setattr(gpu_arbiter, "_shared", GpuArbiter(read=lambda: reading, holder=lambda: None))
    return await SwarmManager(SimpleNamespace(server=SimpleNamespace(port=8785)))._detect_local_hardware()


async def test_swarm_hardware_reads_the_card_without_cuda(monkeypatch):
    card = GpuInfo(0, "NVIDIA GeForce RTX 4060 Laptop GPU", 70.0, 6 * GIB, 8 * GIB)
    hardware = await _hardware(monkeypatch, MachineReading("nvidia", (card,), 4 * GIB, 16 * GIB))
    assert hardware["has_gpu"] is True
    assert hardware["gpu_name"] == "NVIDIA GeForce RTX 4060 Laptop GPU" and hardware["gpu_vram_gb"] == 8.0


async def test_swarm_hardware_on_apple_silicon_and_cpu_only(monkeypatch):
    for kind in ("apple", "cpu"):
        hardware = await _hardware(monkeypatch, MachineReading(kind, host_free_bytes=GIB, host_total_bytes=8 * GIB))
        assert hardware["has_gpu"] is False and hardware["gpu_name"] is None and hardware["gpu_vram_gb"] == 0


def test_chroma_embeddings_are_pinned_to_the_cpu(monkeypatch):
    built = []

    class ONNXMiniLM_L6_V2:  # noqa: N801 - mirrors chromadb's class name
        def __init__(self, preferred_providers=None):
            built.append(preferred_providers)

    functions = ModuleType("chromadb.utils.embedding_functions")
    functions.ONNXMiniLM_L6_V2 = ONNXMiniLM_L6_V2
    monkeypatch.setitem(sys.modules, "chromadb", ModuleType("chromadb"))
    monkeypatch.setitem(sys.modules, "chromadb.utils", ModuleType("chromadb.utils"))
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", functions)

    assert isinstance(store._cpu_embedding_function(), ONNXMiniLM_L6_V2)
    assert built == [["CPUExecutionProvider"]]


def test_chroma_embeddings_fall_back_to_chromas_default_when_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", None)
    assert store._cpu_embedding_function() is None
