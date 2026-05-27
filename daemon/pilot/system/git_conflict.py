"""Git conflict parser and LLM resolution bridge.

Complies with Ruff linter and strict typing constraints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

            resolved_blocks.append(
                {
                    "path": str(p.resolve()),
                    "original_hunk": parsed.get("original_hunk", block["original_hunk"]),
                    "conflict_hunk": parsed.get("conflict_hunk", block["conflict_hunk"]),
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
