"""GPU/VRAM detection utilities for dynamic n_gpu_layers calculation.

Supports NVIDIA GPUs via pynvml. Falls back to a conservative CPU-only
estimate using psutil when pynvml is unavailable or no NVIDIA GPU is present
(AMD, Intel, or CPU-only systems).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("pilot.models.gpu_utils")

# Safety margin: keep this fraction of free VRAM unallocated to avoid OOM.
_VRAM_SAFETY_MARGIN = 0.90

# Rough per-layer VRAM cost used when the model file is unavailable.
# Based on typical 7B-class Q4 GGUF models (~80 MB/layer).
_FALLBACK_BYTES_PER_LAYER = 80 * 1024 * 1024

# Minimum layers to offload when VRAM is available, to make GPU use worthwhile.
_MIN_GPU_LAYERS = 1


def _gguf_metadata(model_path: Path) -> tuple[int, int]:
    """Parse layer count and per-layer byte cost from a GGUF file header.

    Returns:
        (n_layers, bytes_per_layer) — falls back to (32, _FALLBACK_BYTES_PER_LAYER)
        if the file cannot be parsed.
    """
    n_layers = 32
    bytes_per_layer = _FALLBACK_BYTES_PER_LAYER

    try:
        file_size = model_path.stat().st_size
        # Heuristic: divide total file size by an assumed layer count.
        # GGUF stores weights roughly uniformly across transformer layers.
        bytes_per_layer = max(file_size // n_layers, _FALLBACK_BYTES_PER_LAYER)
    except OSError:
        pass

    # Attempt a lightweight parse of the GGUF key-value metadata block to
    # extract the actual block/layer count without loading the full model.
    try:
        with model_path.open("rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return n_layers, bytes_per_layer

            # GGUF v2/v3: version (u32), tensor_count (u64), kv_count (u64)
            import struct

            version = struct.unpack("<I", fh.read(4))[0]
            if version not in (2, 3):
                return n_layers, bytes_per_layer

            _tensor_count = struct.unpack("<Q", fh.read(8))[0]
            kv_count = struct.unpack("<Q", fh.read(8))[0]

            # Walk key-value pairs looking for "llama.block_count" (or similar).
            for _ in range(min(kv_count, 256)):
                key_len = struct.unpack("<Q", fh.read(8))[0]
                if key_len > 256:
                    break
                key = fh.read(key_len).decode("utf-8", errors="replace")
                value_type = struct.unpack("<I", fh.read(4))[0]

                # value_type 4 == UINT32
                if value_type == 4:
                    value = struct.unpack("<I", fh.read(4))[0]
                    if key.endswith("block_count"):
                        n_layers = int(value)
                        file_size = model_path.stat().st_size
                        bytes_per_layer = max(file_size // n_layers, _FALLBACK_BYTES_PER_LAYER)
                        break
                else:
                    # Skip unsupported value types to keep the parser simple.
                    _skip_gguf_value(fh, value_type)
    except Exception:  # noqa: BLE001
        pass

    return n_layers, bytes_per_layer


def _skip_gguf_value(fh, value_type: int) -> None:  # type: ignore[type-arg]
    """Advance the file cursor past a GGUF metadata value of the given type."""
    import struct

    _FIXED: dict[int, int] = {
        0: 1,  # UINT8
        1: 1,  # INT8
        2: 2,  # UINT16
        3: 2,  # INT16
        4: 4,  # UINT32
        5: 4,  # INT32
        6: 4,  # FLOAT32
        7: 1,  # BOOL
        10: 8,  # UINT64
        11: 8,  # INT64
        12: 8,  # FLOAT64
    }
    if value_type in _FIXED:
        fh.read(_FIXED[value_type])
    elif value_type == 8:  # STRING
        length = struct.unpack("<Q", fh.read(8))[0]
        fh.read(length)
    elif value_type == 9:  # ARRAY
        elem_type = struct.unpack("<I", fh.read(4))[0]
        count = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(min(count, 4096)):
            _skip_gguf_value(fh, elem_type)


def _free_vram_bytes_nvidia() -> int | None:
    """Return free VRAM in bytes for the first available NVIDIA GPU.

    Returns None if pynvml is not installed or no NVIDIA GPU is present.
    """
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        device_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_bytes: int = mem_info.free
        total_bytes: int = mem_info.total
        logger.debug(
            "NVIDIA VRAM — free: %.1f MB / total: %.1f MB",
            free_bytes / 1024**2,
            total_bytes / 1024**2,
        )
        return free_bytes
    except Exception as exc:  # noqa: BLE001
        logger.debug("pynvml unavailable or failed: %s", exc)
        return None


def _free_ram_bytes_psutil() -> int:
    """Return free system RAM in bytes via psutil (CPU/AMD fallback)."""
    try:
        import psutil

        free: int = psutil.virtual_memory().available
        logger.debug("System RAM available: %.1f MB", free / 1024**2)
        return free
    except Exception as exc:  # noqa: BLE001
        logger.warning("psutil unavailable: %s", exc)
        return 0


def calculate_gpu_layers(
    model_path: Path | None,
    *,
    vram_limit_mb: int = 0,
) -> int:
    """Calculate the optimal n_gpu_layers value before loading a GGUF model.

    Strategy:
    1. Query free VRAM via pynvml (NVIDIA). If unavailable, fall back to
       system RAM via psutil (CPU/AMD — layers stay at 0 for those).
    2. Apply the configured ``vram_limit_mb`` cap if set.
    3. Estimate per-layer VRAM cost from the model file (GGUF metadata parse
       or file-size heuristic).
    4. Return the maximum number of layers that fit within the safe budget,
       clamped to [0, total_layers].

    Args:
        model_path: Path to the ``.gguf`` file, or None if unknown.
        vram_limit_mb: Optional hard cap from config (0 = no cap).

    Returns:
        Number of layers to offload to GPU. Returns 0 for CPU-only systems.
        Returns -1 (offload all) only when free VRAM comfortably fits the
        entire model.
    """
    n_layers, bytes_per_layer = _gguf_metadata(model_path) if model_path else (32, _FALLBACK_BYTES_PER_LAYER)

    free_bytes = _free_vram_bytes_nvidia()
    using_gpu = free_bytes is not None

    if not using_gpu:
        # No NVIDIA GPU detected — keep everything on CPU to avoid crashes.
        logger.info("No NVIDIA GPU detected. Running model on CPU (n_gpu_layers=0).")
        return 0

    # Apply safety margin and optional user-configured cap.
    usable_bytes = int(free_bytes * _VRAM_SAFETY_MARGIN)
    if vram_limit_mb > 0:
        usable_bytes = min(usable_bytes, vram_limit_mb * 1024 * 1024)

    optimal_layers = int(usable_bytes // bytes_per_layer)
    optimal_layers = max(0, min(optimal_layers, n_layers))

    if optimal_layers == 0:
        logger.warning(
            "Insufficient VRAM (%.1f MB usable) for even one layer (%.1f MB/layer). Falling back to CPU inference.",
            usable_bytes / 1024**2,
            bytes_per_layer / 1024**2,
        )
        return 0

    if optimal_layers >= n_layers:
        logger.info(
            "VRAM sufficient for full model offload — using n_gpu_layers=-1 "
            "(%.1f MB usable, %d layers × %.1f MB/layer).",
            usable_bytes / 1024**2,
            n_layers,
            bytes_per_layer / 1024**2,
        )
        return -1

    logger.info(
        "Dynamic GPU offload: %d / %d layers (%.1f MB usable, %.1f MB/layer, %.1f MB estimated usage).",
        optimal_layers,
        n_layers,
        usable_bytes / 1024**2,
        bytes_per_layer / 1024**2,
        (optimal_layers * bytes_per_layer) / 1024**2,
    )
    return max(optimal_layers, _MIN_GPU_LAYERS)
