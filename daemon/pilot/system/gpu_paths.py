"""Whether the GPU path of each local speech model is installed, and telling "not installed" from "the card failed".

On the owner's laptop (12 November) the card and its driver were fine, but Whisper's runtime (ctranslate2, built for
CUDA 12) found no ``cublas64_12.dll``: nothing Heliox installs provides it. The load failed, Whisper went to the
processor, and the failure was recorded with the GPU arbiter as a fault of the card, so a missing library counted
toward switching a healthy card off for everything. Here the two are told apart:

* a library, runtime build or provider that is not installed is "GPU path not installed": said once in the model
  health status with what is missing, never counted against the card;
* a real device error (out of memory, a CUDA or cuDNN error from a loaded library, a crash) stays a fault.

It also says where each model can run before anything is loaded, from what is installed (package metadata and the
library files), so the health status can say "on the processor, because ..." without importing PyTorch into the
daemon, and it makes the NVIDIA libraries installed by the ``voice-gpu`` extra findable before the first load.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
import threading
from pathlib import Path

logger = logging.getLogger("pilot.system.gpu_paths")

# What ctranslate2 (Faster-Whisper) loads for CUDA 12, by system. cudnn is loaded too, by its main library's name.
WHISPER_GPU_LIBRARIES = {
    "win32": ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"),
    "linux": ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"),
}
INSTALL_HINT = "Settings > Optional parts > Voice on the graphics card"

_LIBRARY = re.compile(
    r"([\w.+-]+\.(?:dll|so(?:\.\d+)*|dylib))\b[^\n]*?(?:not found|cannot be loaded|could not be loaded|"
    r"cannot open shared object|no such file|is not found)",
    re.IGNORECASE,
)
_MISSING = (
    (re.compile(r"torch not compiled with cuda", re.IGNORECASE), "a CUDA build of PyTorch"),
    (
        re.compile(r"cuda(?:executionprovider)?[^\n]*not (?:in )?available", re.IGNORECASE),
        "the GPU build of the ONNX runtime",
    ),  # fmt: skip
    (re.compile(r"cuda driver version is insufficient", re.IGNORECASE), "a newer NVIDIA driver"),
    (
        re.compile(r"dll load failed|cannot load library|could not load (?:dynamic )?library", re.IGNORECASE),
        "a GPU library",
    ),  # fmt: skip
)

_lock = threading.Lock()
_not_installed: dict[str, str] = {}
_prepared = False


def missing_library(text: str) -> str:
    """What is not installed, when ``text`` is an error about a library or runtime build that is missing; else "".
    "Library cublas64_12.dll is not found or cannot be loaded" -> "cublas64_12.dll"."""
    found = _LIBRARY.search(text or "")
    if found:
        return found.group(1)
    for pattern, what in _MISSING:
        if pattern.search(text or ""):
            return what
    return ""


def note_not_installed(component: str, what: str) -> bool:
    """``component``'s GPU path is not installed (``what`` is missing). Logged once per component; True the first
    time. Never a fault of the card."""
    with _lock:
        first = component not in _not_installed
        _not_installed[component] = what
    if first:
        logger.info("GPU path for %s is not installed (%s missing); it runs on the processor", component, what)
    return first


def not_installed(component: str) -> str:
    """What is missing for ``component``'s GPU path, or "" (installed, or not known to be missing)."""
    with _lock:
        return _not_installed.get(component, "")


def reset() -> None:
    """Forget what was noted (tests, and after the optional part is installed and Heliox restarted)."""
    global _prepared
    with _lock:
        _not_installed.clear()
    _prepared = False


def on_processor_because_missing(what: str) -> str:
    return f"On the processor, because its GPU libraries are not installed ({what} missing): {INSTALL_HINT}."


# ── The NVIDIA libraries from pip (the voice-gpu extra) ──


# The NVIDIA wheels Whisper's runtime (ctranslate2) loads from, and what they need themselves. On Linux PyPI's
# PyTorch also installs nccl, nvshmem, cusparselt and others, which have nothing to do with Whisper: their folders are
# never added to the search path.
WHISPER_NVIDIA_PARTS = ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "nvjitlink")


def nvidia_library_dirs() -> list[Path]:
    """Folders of the NVIDIA wheels Whisper needs (WHISPER_NVIDIA_PARTS) in this environment that hold their
    libraries: ``nvidia/<part>/bin`` on Windows, ``nvidia/<part>/lib`` elsewhere. [] when none is installed."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    leaf = "bin" if sys.platform == "win32" else "lib"
    found: list[Path] = []
    for root in spec.submodule_search_locations:
        for part in WHISPER_NVIDIA_PARTS:
            folder = Path(root) / part / leaf
            if folder.is_dir():
                found.append(folder)
    return found


def prepare_gpu_libraries() -> list[Path]:
    """Make the NVIDIA wheels' libraries findable before the first GPU load, once. Windows: each folder is added to
    the DLL search path and to PATH (ctranslate2 loads cuBLAS by name). Linux: the libraries are loaded by full path,
    so a later load by name finds them already in the process. Returns the folders used."""
    global _prepared
    folders = nvidia_library_dirs()
    if _prepared or not folders:
        return folders
    _prepared = True
    if sys.platform == "win32":
        for folder in folders:
            try:
                os.add_dll_directory(str(folder))
            except (OSError, AttributeError):
                continue
        os.environ["PATH"] = os.pathsep.join([*(str(f) for f in folders), os.environ.get("PATH", "")])
    elif sys.platform.startswith("linux"):
        for folder in folders:
            for name in ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9"):
                path = folder / name
                if path.exists():
                    try:
                        ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                    except OSError:
                        logger.debug("Could not load %s", path, exc_info=True)
    return folders


def _loadable(name: str) -> bool:
    try:
        if sys.platform == "win32":
            ctypes.WinDLL(name)
        else:
            ctypes.CDLL(name)
        return True
    except OSError:
        return False


def whisper_gpu_missing(platform: str | None = None, loadable=_loadable) -> list[str]:
    """The CUDA 12 libraries Whisper's runtime needs that cannot be loaded here ([] when all can). Only asked when an
    NVIDIA card is present: it loads them into the process. macOS has no CUDA: [] (it never gets here)."""
    platform = platform or sys.platform
    key = "win32" if platform == "win32" else ("linux" if platform.startswith("linux") else "")
    if not key:
        return []
    prepare_gpu_libraries()
    return [name for name in WHISPER_GPU_LIBRARIES[key] if not loadable(name)]


# ── What is installed, read without importing it ──


def torch_build(platform: str | None = None) -> str:
    """The installed PyTorch's build: "cuda", "cpu", "mps" (macOS) or "" (not installed). From its metadata only:
    importing PyTorch into the daemon is what the worker exists to avoid. PyPI's Windows wheel is a CPU build, its
    Linux wheel a CUDA one (it requires NVIDIA's packages); the CUDA index's wheels say "+cu" in the version."""
    platform = platform or sys.platform
    try:
        dist = importlib.metadata.distribution("torch")
    except importlib.metadata.PackageNotFoundError:
        return ""
    version = str(dist.version)
    if "+cpu" in version:
        return "cpu"
    if "+cu" in version or "+rocm" in version:
        return "cuda"
    if platform == "darwin":
        return "mps"
    requires = " ".join(dist.requires or [])
    return "cuda" if "nvidia-" in requires else "cpu"


def sherpa_onnx_cuda() -> bool:
    """The installed sherpa-onnx carries the CUDA provider (its CUDA build, which PyPI does not have)."""
    try:
        spec = importlib.util.find_spec("sherpa_onnx")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin:
        return False
    folder = Path(spec.origin).parent
    names = ("onnxruntime_providers_cuda.dll", "libonnxruntime_providers_cuda.so")
    return any(any(folder.rglob(name)) for name in names)
