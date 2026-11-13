"""Scripts and benches outside pytest that can reach Heliox's folders either move every folder first
(use_scratch_folders, directly or through the bench driver) or are the owner's own tools, which stop without
--real-folders. Heliox's stores have no single opener to guard, so this is checked over the files instead: a file
that names a folder, a store, a server, a listener or the budget tracker is "reaching".
"""

from __future__ import annotations

import re
from pathlib import Path

DAEMON = Path(__file__).resolve().parents[1]
FOLDERS = (DAEMON / "benchmarks", DAEMON / "scripts")
REACHING = re.compile(
    r"\b(?:DATA_DIR|CONFIG_DIR|STATE_DIR|RUNTIME_DIR|CACHE_DIR|SETUP_DIR|CONFIG_FILE|RESTRICTIONS_FILE|\w*DB_FILE)\b"
    r"|\b(?:PilotServer|Harness|ContinuousVoiceListener|WakeWordCalibrator|VoiceCalibrationStore|BudgetTracker|"
    r"ExperienceLedger|DurableTaskStore|WorkflowCheckpointStore|AuditLogger|RecordedWorkflowStore|PlanHistory\w*|"
    r"MemoryStore|TemporalMemory\w*|ensure_dirs)\("
)
# Open only a file the person names on the command line (no default in Heliox's folders).
GIVEN_PATH_ONLY = {"scripts/train_world_visual_transition.py"}
# The conversation agent moves these onto use_scratch_folders and real_budget_tracker in its own patch
# (conv_isolation); each comes off this list in that patch. Nothing may be added to it.
PENDING: set[str] = set()
# The owner's own tools: they read or write the person's real Heliox data on purpose and stop without the flag.
OWNER_TOOLS = {
    "scripts/review_local_scope.py",
    "scripts/evaluate_local_scope_reviews.py",
    "scripts/train_recorded_workflow_outcome.py",
    "benchmarks/conversation_bench/replay.py",  # its build step: the person's settings and models
    "benchmarks/conversation_bench/turn_split.py",
}


def _files() -> list[Path]:
    return sorted(path for folder in FOLDERS for path in folder.rglob("*.py") if "pilot" in path.read_text("utf-8"))


def _imported(path: Path, text: str) -> list[Path]:
    """The benchmark modules ``text`` imports: ``from .x import``, ``from . import x, y``, ``from benchmarks...``."""
    bench = path.parent
    names = [
        a or b for a, b in re.findall(r"from \.(\w+) import|from benchmarks\.conversation_bench\.(\w+) import", text)
    ]
    for group in re.findall(r"from \. import \(?([\w, \n]+?)\)?$", text, flags=re.MULTILINE):
        names += [part.split(" as ")[0].strip() for part in group.split(",") if part.strip()]
    found = [bench / f"{name}.py" for name in names]
    found += [DAEMON / "benchmarks" / f"{name}.py" for name in re.findall(r"from benchmarks\.(\w+) import", text)]
    return [module for module in found if module.exists() and module != path]


def _moves(path: Path, text: str, seen: set[Path] | None = None) -> bool:
    """It calls use_scratch_folders itself, or imports a benchmark module that does (directly or through another)."""
    if "use_scratch_folders(" in text:
        return True
    seen = seen if seen is not None else {path}
    for module in _imported(path, text):
        if module in seen:
            continue
        seen.add(module)
        if _moves(module, module.read_text("utf-8"), seen):
            return True
    return False


def test_every_reaching_tool_moves_its_folders_or_is_an_owner_tool_that_asks_for_the_flag():
    problems = []
    for path in _files():
        text = path.read_text("utf-8")
        name = path.relative_to(DAEMON).as_posix()
        if name in OWNER_TOOLS:
            if "require_real_folders_flag(" not in text:
                problems.append(f"{name}: an owner tool that does not ask for --real-folders")
            continue
        if name in GIVEN_PATH_ONLY or name in PENDING:
            continue
        if REACHING.search(text) and not _moves(path, text):
            problems.append(f"{name}: names {REACHING.search(text).group(0)} without use_scratch_folders")
    assert problems == []


def test_the_named_lists_name_files_that_exist():
    assert all((DAEMON / name).exists() for name in OWNER_TOOLS | GIVEN_PATH_ONLY | PENDING)


def test_a_pending_file_that_already_moves_its_folders_comes_off_the_list():
    moved = [name for name in PENDING if _moves(DAEMON / name, (DAEMON / name).read_text("utf-8"))]
    assert moved == []
