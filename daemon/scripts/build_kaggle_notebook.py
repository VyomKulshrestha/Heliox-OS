"""Build a self-contained Kaggle notebook that trains one Gnomon candidate from this repository's tracked code.

The notebook carries a zip of the training script's own import closure (every repo module it can import, found by
reading the code), limited to tracked files (``git ls-files``), so nothing kept local-only through
``.git/info/exclude`` (reference-model outputs and anything derived from them) can travel with it; the build fails
if any such path would be included. Kaggle caps a notebook's source at 1 MB, which the closure fits. On Kaggle it installs the pinned libraries, downloads the pinned public CC0
dataset and backbone, runs the training command and zips the output for download.

    python scripts/build_kaggle_notebook.py --name gnomon_2b_v7 --output <scratch>/gnomon_2b_v7.ipynb -- <train args>

Kaggle: Code -> New notebook -> File -> Import notebook; Session options: GPU T4 x2, Internet on; Save Version ->
Save & Run All (Commit). The output zip appears under the version's Output tab.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DAEMON = REPO / "daemon"
TRAIN_SCRIPT = DAEMON / "scripts" / "train_general_decision.py"
PACKAGES = ("pilot", "benchmarks", "scripts")
# Code only, plus the frozen protocol: benchmark result files (some hold reference-model comparison answers) never
# travel into a training environment, even though training does not read them.
EXTRA_FILES = ("daemon/benchmarks/gnomon_mixture_protocol_v7_20261109.json",)
MAX_NOTEBOOK_BYTES = 1_000_000
LIBRARIES = "transformers==5.10.2 peft==0.19.1 accelerate==1.13.0 bitsandbytes pyarrow huggingface_hub"


def _excluded() -> set[str]:
    path = REPO / ".git" / "info" / "exclude"
    if not path.exists():
        return set()
    return {
        line.strip().lstrip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _module_file(name: str) -> Path | None:
    parts = name.split(".")
    if parts[0] not in PACKAGES:
        return None
    base = DAEMON.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return base / "__init__.py" if (base / "__init__.py").is_file() else None


def _imported_names(path: Path) -> set[str]:
    rel = path.relative_to(DAEMON).with_suffix("").parts
    package = list(rel[:-1]) if rel[-1] != "__init__" else list(rel[:-1])
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
            module = ".".join([*base, node.module] if node.module else base) if node.level else (node.module or "")
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def import_closure(start: Path = TRAIN_SCRIPT) -> set[str]:
    """Every pilot/benchmarks/scripts module the script can import, at any depth, as repo-relative paths."""
    seen: set[Path] = set()
    todo = [start]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imported_names(path):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):  # each parent package's __init__ runs too
                found = _module_file(".".join(parts[:i]))
                if found is not None and found not in seen:
                    todo.append(found)
    return {path.relative_to(REPO).as_posix() for path in seen}


def bundle() -> tuple[bytes, list[str]]:
    tracked = set(
        subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    )
    excluded = _excluded()
    wanted = import_closure() | set(EXTRA_FILES)
    untracked = sorted(wanted - tracked)
    if untracked:
        raise SystemExit(f"Commit these first; only tracked files are bundled: {untracked}")
    chosen = sorted(wanted)
    leaked = sorted(set(chosen) & excluded)
    if leaked:
        raise SystemExit(f"Refusing to bundle local-only files: {leaked}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in chosen:
            archive.write(REPO / name, name)
    return buffer.getvalue(), chosen


def _code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def notebook(name: str, train_args: list[str], payload: bytes) -> dict:
    sys.path.insert(0, str(REPO / "daemon"))
    from benchmarks.general_decision import DATASET, DATASET_REVISION
    from pilot.intelligence.typed_decision import PINNED_BACKBONES

    encoded = base64.b64encode(payload).decode("ascii")
    command = " ".join(shlex.quote(arg) for arg in ["python", "scripts/train_general_decision.py", *train_args])
    configs = ["release-v2-redistributable", "email-selection-control-v1", "mailroom-control-v1"]
    cells = [
        _code(
            f"!pip install -q {LIBRARIES}\n"
            "!pip uninstall -y -q torchao || true\n"
            "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
        ),
        _code(
            "import base64, io, zipfile, pathlib\n"
            f"PAYLOAD = '{encoded}'\n"
            "root = pathlib.Path('/kaggle/working/heliox')\n"
            "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)\n"
            "print(sum(1 for _ in root.rglob('*.py')), 'python files')"
        ),
        _code(
            "from huggingface_hub import snapshot_download\n"
            f"snapshot_download({DATASET!r}, repo_type='dataset', revision={DATASET_REVISION!r}, "
            f"allow_patterns=[f'data/{{c}}/*' for c in {configs!r}])\n"
            f"snapshot_download('Qwen/Qwen3.5-2B', revision={PINNED_BACKBONES['Qwen/Qwen3.5-2B']!r})\n"
            "print('downloads ready')"
        ),
        _code(
            "import subprocess, sys, time\n"
            "started = time.time()\n"
            f"command = {command!r}\n"
            "with open('/kaggle/working/train.log', 'w') as log:\n"
            "    result = subprocess.run(command, shell=True, cwd='/kaggle/working/heliox/daemon', stdout=log, "
            "stderr=subprocess.STDOUT, env={**__import__('os').environ, 'PYTHONUNBUFFERED': '1'})\n"
            "print('exit', result.returncode, 'after', round(time.time() - started), 's')\n"
            "print(open('/kaggle/working/train.log').read()[-4000:])"
        ),
        _code(
            "import shutil, hashlib\n"
            f"out = shutil.make_archive('/kaggle/working/{name}', 'zip', '/kaggle/working', '{name}_out')\n"
            "print(out, hashlib.sha256(open(out, 'rb').read()).hexdigest())"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },  # fmt: skip
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    payload, files = bundle()
    book = notebook(args.name, train_args, payload)
    text = json.dumps(book)
    if len(text.encode("utf-8")) >= MAX_NOTEBOOK_BYTES:
        raise SystemExit(f"The notebook is {len(text) / 1e6:.2f} MB; Kaggle accepts less than 1 MB")
    args.output.write_text(text, encoding="utf-8")
    print(f"{args.output}: {len(files)} files, bundle {len(payload) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
