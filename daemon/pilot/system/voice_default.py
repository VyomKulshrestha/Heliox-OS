"""The voice a new install speaks with, chosen by the computer and the language.

On a computer without a usable graphics card, Pocket TTS streams on the processor and the answer's first sound came
about 0.6 s sooner than with Kokoro there (one development laptop, measured live: 1.48 to 1.59 s against 2.10 s).
On a graphics card the two were about even. So the voice a new install starts with follows a small table in
``speech_models.json`` ("voice_rules"), which the first-run screen reads too, so it shows the same pick before
anything is downloaded. The first row that matches the computer and the voice language wins:

- ``nvidia_card``: an NVIDIA card the voice can use and PyTorch built with CUDA. Kokoro, on the card.
- ``apple_silicon``: Kokoro, as before. Neither voice has been measured on a Mac yet.
- ``processor`` (every other computer): Pocket TTS for English; Kokoro for the languages Pocket has not passed the
  read-back check in; Kokoro, as before, for every other language.

The table is applied only to an install with no voice of its own yet: the person never chose one
(``voice.tts_voice_chosen`` is false), the engine is a local one, no local voice model is on the computer and no local
voice has ever loaded here. Anyone whose voice has already spoken keeps it, chosen or not: a changed voice reads as a
different companion. They may see one suggestion in Settings, Voice (``suggestion``), and nothing changes unless they
press it.

A face whose engine has no approved voice for it keeps Kokoro with the face's own Kokoro voice
(``pilot.system.face_voice``), so a face never speaks with a voice of the wrong kind. The small-machine rule (the
system voice on a computer with less than 10 GB) and every memory and heat rule still apply afterwards, unchanged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST = Path(__file__).with_name("speech_models.json")
MACHINES = ("nvidia_card", "apple_silicon", "processor")
LOCAL_ENGINES = ("kokoro_tts", "pocket_tts")
DEFAULT_ENGINE = "kokoro_tts"  # VoiceConfig's default, and the engine when no row matches
# The Hugging Face cache folders the two engines fetch their models into, and a file that is there only once a model
# has been fetched. Read as files: importing huggingface_hub here is not needed for a yes or no.
MODEL_FILES = {
    "kokoro_tts": (("models--hexgrad--Kokoro-82M",), "kokoro-v1_0.pth"),
    "pocket_tts": (
        ("models--kyutai--pocket-tts-without-voice-cloning", "models--kyutai--pocket-tts"),
        "model.safetensors",
    ),
}


@dataclass(frozen=True)
class Rule:
    machine: str
    languages: tuple[str, ...]  # two-letter codes, or "*" for any
    engine: str
    why: str = ""

    def matches(self, machine: str, language: str) -> bool:
        return self.machine == machine and ("*" in self.languages or language in self.languages)


def load_rules(path: Path = MANIFEST) -> tuple[Rule, ...]:
    """The rows of speech_models.json's "voice_rules", in order (the first match wins)."""
    data = json.loads(path.read_text(encoding="utf-8")).get("voice_rules", {})
    rows = []
    for row in data.get("rows", []):
        machine, engine = str(row["machine"]), str(row["engine"])
        if machine not in MACHINES or engine not in LOCAL_ENGINES:
            raise ValueError(f"voice_rules: unknown machine {machine!r} or engine {engine!r}")
        rows.append(Rule(machine, tuple(str(code) for code in row["languages"]), engine, str(row.get("why", ""))))
    return tuple(rows)


def load_voices(path: Path = MANIFEST) -> dict[str, dict[str, Any]]:
    """speech_models.json's "voices" by engine: what each local voice is, its size, licence and source."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry["engine"]): dict(entry) for entry in data.get("voices", [])}


RULES: tuple[Rule, ...] = load_rules()
VOICES: dict[str, dict[str, Any]] = load_voices()


def pocket_voice_terms(voice: str, voices: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """A built-in Pocket voice's source and terms ("voice_terms" in speech_models.json); {} for a name not listed."""
    entry = (VOICES if voices is None else voices).get("pocket_tts", {})
    return dict(entry.get("voice_terms", {}).get(voice, {}))


def pocket_voice_allowed(voice: str) -> bool:
    """Its terms allow any use (CC0 or CC BY): the only Pocket voices Heliox may set, pick or bring with a face."""
    return pocket_voice_terms(voice).get("use") == "allowed"


def pocket_voices_offered() -> list[str]:
    """Allowed Pocket voices that were also listened to and read back: the ones Settings offers."""
    terms = VOICES.get("pocket_tts", {}).get("voice_terms", {})
    return [name for name, term in terms.items() if term.get("use") == "allowed" and term.get("listened")]


def default_voice(engine: str) -> str:
    return str(VOICES.get(engine, {}).get("default_voice", "af_heart" if engine == "kokoro_tts" else "alba"))


def pick(machine: str, language: str, rules: tuple[Rule, ...] = RULES) -> Rule:
    """The first row for this kind of computer and language; Kokoro when none matches."""
    language = (language or "en").split("-")[0].split("_")[0].casefold()
    for rule in rules:
        if rule.matches(machine, language):
            return rule
    return Rule(machine, ("*",), DEFAULT_ENGINE, "")


def machine_kind(
    reading: Any,
    *,
    torch_build: str,
    hidden: bool = False,
    tts_device: str = "auto",
    needs_bytes: int = 0,
    visible: tuple[int, ...] | None = None,
) -> str:
    """Which row of the table this computer reads, from the GPU arbiter's reading (``pilot.system.gpu_arbiter``):
    "apple_silicon"; "nvidia_card" when an NVIDIA card (the first visible one, as the voice uses) has ``needs_bytes``
    of graphics memory in all and PyTorch is the CUDA build; "processor" otherwise, also when the cards are hidden
    (CUDA_VISIBLE_DEVICES), the voice is set to the processor, or the card did not report. A hot or busy card counts
    as usable: heat and load pass, and the choice of a voice is made once."""
    kind = getattr(reading, "kind", "cpu")
    if kind == "apple":
        return "apple_silicon"
    if kind != "nvidia" or getattr(reading, "error", "") or hidden or tts_device == "cpu" or torch_build != "cuda":
        return "processor"
    cards = [gpu for gpu in getattr(reading, "gpus", ()) if visible is None or gpu.index in visible]
    if visible is not None:
        cards.sort(key=lambda gpu: visible.index(gpu.index))
    first = cards[:1]
    if first and (first[0].total_bytes is None or first[0].total_bytes >= needs_bytes):
        return "nvidia_card"
    return "processor"


def machine_now(tts_device: str = "auto", environ: Mapping[str, str] | None = None) -> str:
    """machine_kind for this computer now: the arbiter's reading (nvidia-smi, cached for 2 s) and the installed
    PyTorch's build, read from its metadata without importing it."""
    from pilot.system.gpu_arbiter import VOICE_NEEDS, arbiter, visible_from_env
    from pilot.system.gpu_paths import torch_build
    from pilot.system.local_tts_worker import cuda_hidden

    environ = os.environ if environ is None else environ
    shared = arbiter()
    return machine_kind(
        shared.reading(),
        torch_build=torch_build(),
        hidden=cuda_hidden(environ),
        tts_device=tts_device,
        needs_bytes=VOICE_NEEDS.gpu_bytes + shared.headroom_bytes,
        visible=visible_from_env(environ.get("CUDA_VISIBLE_DEVICES")),
    )


def hf_cache_dir(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """Where Hugging Face keeps downloaded models, as huggingface_hub works it out: HF_HUB_CACHE, else HF_HOME/hub,
    else XDG_CACHE_HOME/huggingface/hub, else ~/.cache/huggingface/hub (on every system)."""
    environ = os.environ if environ is None else environ
    if environ.get("HF_HUB_CACHE"):
        return Path(environ["HF_HUB_CACHE"]).expanduser()
    if environ.get("HF_HOME"):
        return Path(environ["HF_HOME"]).expanduser() / "hub"
    if environ.get("XDG_CACHE_HOME"):
        return Path(environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return (home or Path.home()) / ".cache" / "huggingface" / "hub"


def voice_models_on_disk(cache: Path | None = None) -> frozenset[str]:
    """The local engines whose model is already in the Hugging Face cache (any revision, any language)."""
    cache = hf_cache_dir() if cache is None else cache
    found = set()
    for engine, (folders, name) in MODEL_FILES.items():
        for folder in folders:
            snapshots = cache / folder / "snapshots"
            try:
                if snapshots.is_dir() and any(snapshots.rglob(name)):
                    found.add(engine)
                    break
            except OSError:
                continue
    return frozenset(found)


def pocket_voice_on_disk(voice: str, cache: Path | None = None) -> bool:
    """A Pocket TTS voice is in the Hugging Face cache, so speaking with it fetches nothing: its embedding beside the
    model without voice cloning (languages/<language>/embeddings/<voice>.safetensors)."""
    cache = hf_cache_dir() if cache is None else cache
    snapshots = cache / "models--kyutai--pocket-tts-without-voice-cloning" / "snapshots"
    try:
        return bool(voice) and snapshots.is_dir() and any(snapshots.rglob(f"{voice}.safetensors"))
    except OSError:
        return False


def voice_ever_loaded(store: Path | None = None) -> bool:
    """A local voice has loaded on this computer before: the voice store (``local_tts_worker.SpeechStore``, under
    Heliox's cache folder) holds load times or made speech. The store outlives a cleared Hugging Face cache."""
    if store is None:
        from pilot.config import CACHE_DIR

        store = Path(CACHE_DIR) / "voice_heads"
    for name in ("loads.json", "index.json"):
        try:
            if json.loads((store / name).read_text(encoding="utf-8")):
                return True
        except (OSError, ValueError):
            continue
    return False


@dataclass(frozen=True)
class FirstVoice:
    engine: str
    voice: str
    why: str
    machine: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def first_voice(
    *,
    engine: str,
    voice: str,
    chosen: bool,
    machine: str,
    language: str,
    models_on_disk: frozenset[str] | set[str],
    ever_loaded: bool,
    face_voices: Mapping[str, str] | None = None,
    rules: tuple[Rule, ...] = RULES,
) -> FirstVoice | None:
    """The engine and voice to write for an install with no voice of its own yet, or None to leave the settings as
    they are (a chosen voice, the system voice, a voice model already here, a voice that has spoken here before, or
    nothing to change). ``face_voices``: the chosen face's approved voice per engine ("" for none yet), None without a
    face; an engine with no approved voice for the face gives way to Kokoro with the face's Kokoro voice."""
    if chosen or engine not in LOCAL_ENGINES or models_on_disk or ever_loaded:
        return None
    rule = pick(machine, language, rules)
    target = rule.engine
    if face_voices is not None and not face_voices.get(target):
        target = "kokoro_tts"
    if target == engine:
        return None
    new_voice = (face_voices or {}).get(target) or default_voice(target)
    if target == "pocket_tts" and not pocket_voice_allowed(new_voice):
        target, new_voice = "kokoro_tts", (face_voices or {}).get("kokoro_tts") or default_voice("kokoro_tts")
        if target == engine:
            return None
    return FirstVoice(target, new_voice, rule.why, machine)


def suggestion(
    *,
    engine: str,
    machine: str,
    language: str,
    small_machine: bool,
    pocket_installed: bool,
    pocket_on_disk: bool,
    rules: tuple[Rule, ...] = RULES,
) -> dict[str, Any] | None:
    """The one quiet suggestion Settings, Voice may show to a person whose voice already speaks: on a computer where
    the table would start a new install on Pocket TTS, someone hearing Kokoro is told Pocket would answer sooner and
    what it downloads. Nothing is switched or downloaded unless they press it. None when it does not apply (another
    engine, a card, a Mac, a language Pocket has not passed, a small computer that speaks with the system voice
    anyway, or Pocket not installed)."""
    if engine != "kokoro_tts" or small_machine or not pocket_installed:
        return None
    if pick(machine, language, rules).engine != "pocket_tts":
        return None
    pocket = VOICES.get("pocket_tts", {})
    size = int(pocket.get("download_bytes", 0))
    fetch = (
        "Its model is already on this computer."
        if pocket_on_disk
        else f"It downloads about {round(size / 1_000_000)} MB the first time it speaks."
    )
    return {
        "engine": "pocket_tts",
        "voice": default_voice("pocket_tts"),
        "download_bytes": 0 if pocket_on_disk else size,
        "say": (
            "On this computer Pocket TTS would answer about 0.6 s sooner than Kokoro (measured on one development "
            f"laptop's processor). {fetch} Your voice stays as it is unless you switch."
        ),
    }
