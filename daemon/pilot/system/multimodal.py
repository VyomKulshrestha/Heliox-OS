"""Multi-Modal Chat History — rich formatting for pilot outputs.

Handles images, tables, code blocks, file trees, and system stats
so the frontend can render rich UI components instead of plain text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("pilot.system.multimodal")


@dataclass
class ChatNode:
    """A node in the multimodal chat payload."""

    type: str  # text, image, table, code, file_tree, error, link
    content: Any


class MultiModalFormatter:
    """Formats executor outputs into rich multimodal nodes."""

    def format_output(self, action_type: str, output: str, params: dict) -> list[dict]:
        """Convert a raw string output into a list of multimodal nodes."""
        nodes: list[ChatNode] = []

        # 1. Images (Screenshots / OCR)
        if "screenshot" in action_type or "ocr" in action_type or "vision" in action_type:
            # Check if output contains a known image path
            if output and ("\n" not in output) and (output.endswith(".png") or output.endswith(".jpg")):
                nodes.append(ChatNode(type="image", content={"path": output}))
            else:
                nodes.append(ChatNode(type="text", content=output))

        # 2. JSON Structures (Data tables, process lists, elements)
        elif output.strip().startswith("{") and output.strip().endswith("}"):
            try:
                data = json.loads(output)
                if isinstance(data, dict):
                    if "matches" in data:
                        # Screen finding matches
                        nodes.append(
                            ChatNode(
                                type="table",
                                content={
                                    "columns": ["Text", "Center (X,Y)", "Confidence"],
                                    "rows": [
                                        [m["text"], str(m["center"]), f"{m['confidence']:.2f}"] for m in data["matches"]
                                    ],
                                },
                            )
                        )
                    elif "elements" in data:
                        # UI element map
                        nodes.append(
                            ChatNode(
                                type="table",
                                content={
                                    "columns": ["ID", "Type", "Text", "Center"],
                                    "rows": [
                                        [str(e["id"]), e["type"], e["text"], str(e["center"])] for e in data["elements"]
                                    ],
                                },
                            )
                        )
                    elif "total_commands" in data:
                        # Context stats
                        nodes.append(
                            ChatNode(
                                type="table",
                                content={
                                    "columns": ["Metric", "Value"],
                                    "rows": [[k, str(v)] for k, v in data.items() if not isinstance(v, list)],
                                },
                            )
                        )
                    else:
                        nodes.append(ChatNode(type="code", content={"language": "json", "code": output}))
                else:
                    nodes.append(ChatNode(type="code", content={"language": "json", "code": output}))
            except json.JSONDecodeError:
                nodes.append(ChatNode(type="text", content=output))

        # JSON Lists
        elif output.strip().startswith("[") and output.strip().endswith("]"):
            try:
                data = json.loads(output)
                if data and isinstance(data, list) and isinstance(data[0], dict):
                    # Guess columns from first object
                    cols = list(data[0].keys())
                    rows = [[str(item.get(c, "")) for c in cols] for item in data[:50]]
                    nodes.append(ChatNode(type="table", content={"columns": cols, "rows": rows}))
                else:
                    nodes.append(ChatNode(type="code", content={"language": "json", "code": output}))
            except json.JSONDecodeError:
                nodes.append(ChatNode(type="text", content=output))

        # 3. Code Execution
        elif action_type in ("code_execute", "shell_script"):
            lang = params.get("language", "shell")
            nodes.append(ChatNode(type="code", content={"language": lang, "code": params.get("code", "")}))
            nodes.append(ChatNode(type="text", content=f"**Output:**\n```\n{output}\n```"))

        # 4. File Trees (ls, find)
        elif action_type in ("file_list", "file_search"):
            # Simple heuristic: if it looks like paths, make it a tree
            if len(output.splitlines()) > 2 and ("\\" in output or "/" in output):
                nodes.append(ChatNode(type="file_tree", content={"paths": output.splitlines()[:100]}))
            else:
                nodes.append(ChatNode(type="text", content=output))

        # 5. Generic text fallback
        else:
            if "error" in output.lower() or "failed" in output.lower() or "exception" in output.lower():
                nodes.append(ChatNode(type="error", content=output))
            elif "http" in output and len(output.splitlines()) == 1:
                nodes.append(ChatNode(type="link", content={"url": output.strip()}))
            else:
                nodes.append(ChatNode(type="text", content=output))

        return [asdict(n) for n in nodes]


def format_action_result(action_type: str, output: str, params: dict) -> str:
    """Format an action result into a JSON multimodal payload."""
    formatter = MultiModalFormatter()
    nodes = formatter.format_output(action_type, output, params)
    return json.dumps({"type": "multimodal_result", "action": action_type, "nodes": nodes})
