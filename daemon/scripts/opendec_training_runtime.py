"""Preflight the runtime profile declared by the Open-Jev training extra."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Mapping

from packaging.version import InvalidVersion, Version

TORCH_MINIMUM = Version("2.8.0")
PINNED_VERSIONS = {
    "transformers": Version("5.10.2"),
    "peft": Version("0.19.1"),
    "accelerate": Version("1.13.0"),
}


def validate_training_versions(actual: Mapping[str, str]) -> None:
    """Reject unsupported packages before loading Qwen3.5 or allocating VRAM."""
    problems = []
    torch_value = actual.get("torch")
    try:
        if torch_value is None or Version(torch_value) < TORCH_MINIMUM:
            problems.append(f"torch {torch_value or 'missing'} (requires >= {TORCH_MINIMUM})")
    except InvalidVersion:
        problems.append(f"torch {torch_value} (version is not parseable)")

    for package, expected in PINNED_VERSIONS.items():
        value = actual.get(package)
        try:
            compatible = value is not None and Version(value) == expected
        except InvalidVersion:
            compatible = False
        if not compatible:
            problems.append(f"{package} {value or 'missing'} (requires {expected})")

    if problems:
        raise RuntimeError(
            "Unsupported Open-Jev/Qwen3.5 training stack: " + "; ".join(problems)
            + ". Use the versions in Open-Jev's [train] extra."
        )


def require_training_runtime() -> dict[str, str]:
    """Read installed versions and catch an incompatible optional torchvision."""
    names = ("torch", *PINNED_VERSIONS)
    actual: dict[str, str] = {}
    missing = []
    for name in names:
        try:
            actual[name] = version(name)
        except PackageNotFoundError:
            missing.append(name)
    if missing:
        actual.update({name: "missing" for name in missing})
    validate_training_versions(actual)

    try:
        version("torchvision")
    except PackageNotFoundError:
        pass
    else:
        try:
            importlib.import_module("torchvision")
        except Exception as exc:
            raise RuntimeError(
                "torchvision is installed but incompatible with the selected Torch build; "
                "install a matching torchvision or remove the optional package"
            ) from exc
    return actual
