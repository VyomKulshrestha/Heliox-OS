"""Git conflict parser and LLM resolution bridge.

Complies with Ruff linter and strict typing constraints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pilot.actions import Action
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.system.git_conflict")


def extract_conflict_blocks(content: str) -> list[dict[str, str]]:
    """Scan file lines to extract git conflict blocks bounded by <<<<<<<, =======, and >>>>>>>.

    Returns:
        A list of dicts, each with keys 'original_hunk', 'conflict_hunk', and 'full_block'.
    """
    lines = content.splitlines(keepends=True)
    blocks: list[dict[str, str]] = []

    in_conflict = False
    in_original = False

    current_block_lines: list[str] = []
    original_lines: list[str] = []
    conflict_lines: list[str] = []

    for line in lines:
        if line.startswith("<<<<<<<"):
            in_conflict = True
            in_original = True
            current_block_lines = [line]
            original_lines = []
            conflict_lines = []
        elif in_conflict:
            current_block_lines.append(line)
            if line.startswith("======="):
                in_original = False
            elif line.startswith(">>>>>>>"):
                in_conflict = False
                original_hunk = "".join(original_lines)
                conflict_hunk = "".join(conflict_lines)
                full_block = "".join(current_block_lines)
                blocks.append(
                    {
                        "original_hunk": original_hunk,
                        "conflict_hunk": conflict_hunk,
                        "full_block": full_block,
                    }
                )
            else:
                if in_original:
                    original_lines.append(line)
                else:
                    conflict_lines.append(line)

    return blocks


def in_git_work_tree(path: Path) -> bool:
    """Inside a git working tree: a ``.git`` folder (or, in a linked worktree, file) in a folder above it."""
    return any((parent / ".git").exists() for parent in path.parents)


def resolution_refusal(
    path: object, full_block: object, resolved_code: object, proposals: Iterable[str] = ()
) -> str | None:
    """Why one conflict resolution may not be written, or None when it may.

    It may be written only to a file that exists, named by its full path, inside a git working tree, which holds
    ``full_block`` exactly once as a whole conflict block (``<<<<<<<`` to ``>>>>>>>``), and only as one of that
    block's two sides or a resolution Heliox proposed for it (``proposals``). Where the file may be is the
    validator's to decide, as for any file Heliox writes (path rules, Heliox's own folders, protected folders).
    """
    if not isinstance(path, str) or not isinstance(full_block, str) or not isinstance(resolved_code, str):
        return "path, full_block and resolved_code must be text"
    if not path.strip() or not full_block:
        return "Missing required params: path, full_block, resolved_code"
    target = Path(path)
    if not target.is_absolute():
        return "The file must be named by its full path"
    if not target.is_file():
        return f"{path} is not a file that exists"
    try:
        resolved = target.resolve()
        content = resolved.read_text("utf-8")
    except (OSError, UnicodeDecodeError, RuntimeError) as error:
        return f"{path} could not be read as text: {error}"
    if not in_git_work_tree(resolved):
        return f"{path} is not inside a git repository"
    blocks = [block for block in extract_conflict_blocks(content) if block["full_block"] == full_block]
    if not blocks:
        return "That conflict is not in the file any more (it may have been resolved or changed already)"
    if content.count(full_block) != 1:
        return "That conflict appears more than once in the file, so Heliox cannot tell which one to resolve"
    allowed = {blocks[0]["original_hunk"], blocks[0]["conflict_hunk"], *proposals}
    if resolved_code not in allowed:
        return "Only one side of the conflict, or the resolution Heliox proposed for it, can be written"
    return None


def git_resolution_action(path: str, full_block: str, resolved_code: str) -> Action:
    """The action that writes one checked resolution, run through the executor like any file change: the
    validator's path checks, the undo journal (git_resolve is undoable as a file write) and the task receipts."""
    from pilot.actions import Action, ActionType, GitResolveParams

    return Action(
        action_type=ActionType.GIT_RESOLVE,
        target=path,
        parameters=GitResolveParams(path=path, full_block=full_block, resolved_code=resolved_code),
    )


async def resolve_conflicts_in_file(filepath: str, model_router: ModelRouter) -> list[dict[str, str]]:
    """Read a target file, locate git conflict blocks, and query the LLM to get a structured resolution.

    Strictly enforces the JSON schema format.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    content = await asyncio.to_thread(p.read_text, "utf-8")
    blocks = extract_conflict_blocks(content)

    if not blocks:
        return []

    resolved_blocks: list[dict[str, str]] = []

    # Get JSON schema to show in system instructions
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "git_conflict_resolution.json"
    schema_text = "{}"
    if schema_path.exists():
        schema_text = schema_path.read_text("utf-8")

    for block in blocks:
        prompt = (
            f"Please resolve the following git merge conflict.\n\n"
            f"Original code (Local / HEAD):\n```\n{block['original_hunk']}```\n\n"
            f"Conflict code (Incoming / Other):\n```\n{block['conflict_hunk']}```\n\n"
            f"Output a JSON object exactly matching the schema. Your output MUST contain the resolved, clean, merged code block "
            f"in the `proposed_resolution_code` field.\n"
            f"Do not include Markdown backticks around the raw JSON output."
        )

        system_instruction = (
            f"You are a master software engineer specializing in git conflict resolution.\n"
            f"Analyze the two competing versions of the code, preserve crucial functionality from both sides, "
            f"and produce a clean, bug-free, unified resolution.\n"
            f"You must strictly output a valid JSON object matching this schema:\n{schema_text}"
        )

        try:
            # Query the LLM using ModelRouter with json_mode=True
            response_text = await model_router.generate(
                prompt,
                system=system_instruction,
                json_mode=True,
                temperature=0.1,
            )

            # Clean response text just in case LLM added markdown fences
            clean_response = response_text.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            parsed = json.loads(clean_response)

            # The two sides are the file's own, never the model's: the window writes the side the person picks, and
            # until 13 November 2026 a model that rewrote "original_hunk" made "keep mine" write the model's text.
            resolved_blocks.append(
                {
                    "path": str(p.resolve()),
                    "original_hunk": block["original_hunk"],
                    "conflict_hunk": block["conflict_hunk"],
                    "proposed_resolution_code": parsed.get("proposed_resolution_code", ""),
                    "full_block": block["full_block"],
                }
            )
        except Exception as e:
            logger.exception("Failed to resolve git conflict chunk with LLM")
            # Fallback block
            resolved_blocks.append(
                {
                    "path": str(p.resolve()),
                    "original_hunk": block["original_hunk"],
                    "conflict_hunk": block["conflict_hunk"],
                    "proposed_resolution_code": block["original_hunk"],  # Fallback to local
                    "full_block": block["full_block"],
                }
            )

    return resolved_blocks
