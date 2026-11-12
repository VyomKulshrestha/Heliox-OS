"""Nothing in the daemon process opens a CUDA context by itself: Chroma's embedding model is pinned to the CPU."""

import sys
from types import ModuleType

from pilot.memory import store


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
