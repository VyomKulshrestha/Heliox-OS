"""Fetch what the timing run needs on a CI runner, through the product's own download path; report size and source.

  python -m benchmarks.conversation_bench.ci_prefetch [--summary FILE]

The speech models come from Heliox's catalogue (``pilot.system.speech_models``): the recommended English set
(Parakeet for the words, Whisper base for the wake phrase) and the two models the listener needs for turn-taking
(streaming Zipformer, Smart Turn), each checked against its SHA-256 and stored under Heliox's data folder
(under ``HELIOX_BENCH_ROOT``), which the workflow caches between runs. The local voice (Kokoro) is not in the catalogue: its
weights and the voices the synthetic scenarios use load through Kokoro's own Hugging Face fetch, into the Hugging
Face cache, also cached. Writes a Markdown table (model, size, source, fetched or already cached) for the job
summary. Only for CI or a fresh machine: the bench itself never downloads.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

CATALOGUE_IDS = ("parakeet-tdt-0.6b-v2-int8", "whisper-base", "streaming-zipformer-en", "smart-turn-v3")
VOICES = ("af_heart", "am_michael", "bm_lewis", "af_bella")


def _mb(n: int) -> str:
    return f"{n / 2**20:.0f} MB"


def fetch_catalogue() -> list[tuple[str, str, str, str]]:
    from pilot.system import speech_models

    rows = []
    for model_id in CATALOGUE_IDS:
        model = speech_models.BY_ID[model_id]
        before = speech_models.installed(model)
        started = time.perf_counter()
        progress = speech_models.Downloader().run(model_id)
        if progress.state != "ready":
            raise SystemExit(f"{model_id}: {progress.state}: {progress.message}")
        how = "already cached" if before else f"fetched in {time.perf_counter() - started:.0f} s"
        rows.append((model.title, _mb(model.download_bytes), model.source, how))
    return rows


def fetch_voice() -> tuple[str, str, str, str]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from huggingface_hub import scan_cache_dir

    from pilot.system import kokoro_tts

    kokoro_tts.configure(device="cpu")
    started = time.perf_counter()
    for voice in VOICES:
        kokoro_tts._get_pipeline()
        kokoro_tts._generate("Ready.", voice)  # loads the weights and this voice, as the voice worker does
    size = 0
    for repo in scan_cache_dir().repos:
        if repo.repo_id == kokoro_tts.REPO_ID:
            size = repo.size_on_disk
    return (f"Kokoro 82M and voices {', '.join(VOICES)}", _mb(size), f"Hugging Face, {kokoro_tts.REPO_ID}",
            f"ready in {time.perf_counter() - started:.0f} s")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_bench.ci_prefetch")
    parser.add_argument("--summary", default=None, help="append a Markdown table here (the job summary)")
    args = parser.parse_args(argv)
    from pilot.real_folders import use_scratch_folders

    # The same scratch root ci_timing runs on, so the models land where the timing run looks for them.
    use_scratch_folders(Path(os.environ.get("HELIOX_BENCH_ROOT") or ".heliox-bench"))
    rows = [*fetch_catalogue(), fetch_voice()]
    from pilot.config import DATA_DIR

    lines = [
        "### Speech models for the timing run",
        "",
        f"Stored under `{Path(DATA_DIR) / 'models'}` and the Hugging Face cache; both cached between runs.",
        "",
        "| model | size | source | this run |",
        "|---|---|---|---|",
        *(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
