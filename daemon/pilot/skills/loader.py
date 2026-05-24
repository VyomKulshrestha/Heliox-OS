"""Dynamic discovery and loading of ``Skill`` implementations from ``.py`` files."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pilot.config import CONFIG_DIR
from pilot.skills.base import Skill, SkillContext

logger = logging.getLogger("pilot.skills.loader")

# Shipped examples (read-only package path)
_BUNDLED_DIR = Path(__file__).resolve().parent / "bundled"
# User / third-party drop-ins (no edits under pilot/ required)
_DEFAULT_USER_DIR = CONFIG_DIR / "skills"


@dataclass
class SkillLoadRecord:
    """Outcome of loading one file (may register zero or more skills)."""

    path: str
    success: bool
    skill_ids: list[str] = field(default_factory=list)
    error: str | None = None


class SkillRegistry:
    """Discover ``*.py`` files, import modules, instantiate concrete ``Skill`` classes."""

    def __init__(self, search_dirs: Sequence[Path] | None = None) -> None:
        dirs = list(search_dirs) if search_dirs is not None else [_BUNDLED_DIR, _DEFAULT_USER_DIR]
        self._search_dirs = [Path(p).expanduser().resolve() for p in dirs]
        self._skills: dict[str, Skill] = {}
        self._skill_sources: dict[str, str] = {}  # skill_id -> file path
        self.last_load_records: list[SkillLoadRecord] = []

    @property
    def search_dirs(self) -> list[Path]:
        return list(self._search_dirs)

    def add_search_dir(self, path: Path | str, *, first: bool = False) -> None:
        """Register an extra directory (e.g. enterprise bundle path)."""
        p = Path(path).expanduser().resolve()
        if first:
            self._search_dirs.insert(0, p)
        elif p not in self._search_dirs:
            self._search_dirs.append(p)

    def discover_files(self) -> list[Path]:
        """List candidate ``.py`` files (non-private) under search dirs, shallow only."""
        found: list[Path] = []
        for d in self._search_dirs:
            if not d.is_dir():
                continue
            for py in sorted(d.glob("*.py")):
                if py.name.startswith("_"):
                    continue
                found.append(py)
        return found

    def load_all(self) -> list[SkillLoadRecord]:
        """Clear registry and load every discovered file. Returns per-file records."""
        self._skills.clear()
        self._skill_sources.clear()
        records: list[SkillLoadRecord] = []

        for d in self._search_dirs:
            d.mkdir(parents=True, exist_ok=True)

        for path in self.discover_files():
            records.append(self._load_file(path))

        self.last_load_records = records
        ok = sum(1 for r in records if r.success)
        logger.info(
            "Skill discovery: %d files processed, %d ok, %d skills registered",
            len(records),
            ok,
            len(self._skills),
        )
        return records

    def reload(self) -> list[SkillLoadRecord]:
        """Reload all skills (re-reads files from disk)."""
        return self.load_all()

    def _load_file(self, path: Path) -> SkillLoadRecord:
        path = path.resolve()
        spath = str(path)
        mod_name = f"pilot_skill_{path.stem}_{abs(hash(spath)) % 10_000_000}"

        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return SkillLoadRecord(path=spath, success=False, error="invalid import spec")

            module = importlib.util.module_from_spec(spec)
            # Register before exec so relative imports inside the file behave
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning("Skill file failed to import: %s — %s", spath, e, exc_info=False)
            sys.modules.pop(mod_name, None)
            return SkillLoadRecord(path=spath, success=False, error=str(e))

        registered: list[str] = []
        try:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj is Skill or not issubclass(obj, Skill):
                    continue
                if inspect.isabstract(obj):
                    continue
                if obj.__module__ != mod_name:
                    # Subclass defined elsewhere and only imported here
                    continue
                try:
                    inst = obj()
                except Exception as e:
                    logger.warning("Skill class %s in %s failed to instantiate: %s", obj.__name__, spath, e)
                    continue

                sid = inst.skill_id.strip()
                if not sid:
                    logger.warning("Skill %s in %s has empty skill_id; skipped", obj.__name__, spath)
                    continue
                if sid in self._skills:
                    logger.warning("Duplicate skill_id %r (%s wins over earlier)", sid, spath)
                self._skills[sid] = inst
                self._skill_sources[sid] = spath
                registered.append(sid)
        except Exception as e:
            logger.error("Skill scan failed for %s: %s", spath, e, exc_info=True)
            sys.modules.pop(mod_name, None)
            return SkillLoadRecord(path=spath, success=False, error=str(e))

        if not registered:
            msg = "no concrete Skill subclasses found"
            logger.info("Skill file %s: %s", spath, msg)
            return SkillLoadRecord(path=spath, success=False, error=msg)

        logger.info("Loaded skills from %s: %s", spath, ", ".join(registered))
        return SkillLoadRecord(path=spath, success=True, skill_ids=registered)

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def list_skills(self) -> list[dict[str, Any]]:
        """Metadata for UI / JSON-RPC."""
        out: list[dict[str, Any]] = []
        for sid, sk in sorted(self._skills.items(), key=lambda x: x[0]):
            out.append(
                {
                    "skill_id": sid,
                    "name": sk.name,
                    "description": sk.description,
                    "source": self._skill_sources.get(sid, ""),
                }
            )
        return out

    def planner_prompt_block(self) -> str:
        """Append to planner system prompt so the LLM can emit ``skill_run`` actions."""
        if not self._skills:
            return ""
        lines = [
            'Registered custom skills — use action_type "skill_run" with parameters:',
            '  {"skill_id": "<id>", "arguments": { ... }}',
            "Skills:",
        ]
        for sid, sk in sorted(self._skills.items(), key=lambda x: x[0]):
            lines.append(f"  - {sid}: {sk.description}")
        return "\n".join(lines)

    async def run(self, skill_id: str, arguments: dict[str, Any], ctx: SkillContext) -> str:
        skill = self._skills.get(skill_id)
        if skill is None:
            known = ", ".join(sorted(self._skills)) or "(none)"
            return f"Unknown skill_id {skill_id!r}. Loaded: {known}"
        try:
            return await skill.run(arguments or {}, ctx)
        except Exception as e:
            logger.exception("Skill %s raised", skill_id)
            return f"Skill {skill_id!r} failed: {e}"
