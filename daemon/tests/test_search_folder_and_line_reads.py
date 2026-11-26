"""Two reads the live run got wrong, on temporary files only.

- file_search_content was given the folder as "path" with "directory" empty, searched the daemon's own working folder
  and answered "No matches ... in  (*)". Both names now mean the folder, two different folders is an error, no folder
  is an error, and "no matches" says where it looked.
- "Show me the last lines of this log" became code that needs approval. file_read takes "tail_lines" or "head_lines",
  and its check compares the same lines of the file read again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pilot.actions import Action, ActionResult, ActionType, FileParams
from pilot.agents.planner import Planner
from pilot.agents.postconditions.reads import file_read_matches_file
from pilot.system import file_intel
from pilot.system.filesystem import file_read, some_lines


def test_the_folder_may_be_given_as_path_or_directory_but_not_as_two_folders() -> None:
    normalize = Planner._normalize_params
    params = normalize(ActionType.FILE_SEARCH_CONTENT, {"path": "C:\\w\\workspace", "search_text": "budget"}, "")
    assert params["directory"] == "C:\\w\\workspace" and "path" not in params
    same = normalize(ActionType.FILE_SEARCH_CONTENT, {"path": "C:/w/workspace/", "directory": "C:\\w\\workspace"}, "")
    assert same["directory"] == "C:\\w\\workspace"
    with pytest.raises(ValueError, match="'directory' and 'path'"):
        normalize(ActionType.FILE_SEARCH_CONTENT, {"path": "C:\\a", "directory": "C:\\b"}, "")
    assert normalize(ActionType.FILE_SEARCH_CONTENT, {"query": "budget", "folder": "C:\\a"}, "") == {
        "search_text": "budget",
        "directory": "C:\\a",
    }
    assert normalize(ActionType.FILE_LIST, {"directory": "C:\\a"}, "")["path"] == "C:\\a"


def test_the_handler_takes_either_name_and_refuses_none_or_two() -> None:
    assert file_intel.search_folder("", "C:\\w") == "C:\\w"
    assert file_intel.search_folder("C:\\w", "") == "C:\\w"
    with pytest.raises(ValueError, match="two folders"):
        file_intel.search_folder("C:\\a", "C:\\b")
    with pytest.raises(ValueError, match="not given a folder"):
        file_intel.search_folder("", "", "")


@pytest.mark.asyncio
async def test_no_matches_says_which_folder_was_searched(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("nothing here", encoding="utf-8")
    answer = await file_intel.search_file_contents(str(tmp_path), "quarterly budget", "*")
    assert answer == f"No matches for 'quarterly budget' in any file in {tmp_path.resolve()} and its subfolders."
    with pytest.raises(ValueError):
        await file_intel.search_file_contents("", "quarterly budget", "*")


@pytest.mark.asyncio
async def test_a_read_returns_only_the_lines_asked_for_and_its_check_agrees(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_bytes(b"one\r\ntwo\nthree\nfour\nfive\nsix\n")
    assert await file_read(str(log), tail_lines=2) == "five\nsix\n"
    assert await file_read(str(log), head_lines=2) == "one\ntwo\n"
    assert await file_read(str(log)) == "one\ntwo\nthree\nfour\nfive\nsix\n"
    assert some_lines("a\nb", tail_lines=5) == "a\nb"

    action = Action(
        action_type=ActionType.FILE_READ, target=str(log), parameters=FileParams(path=str(log), tail_lines=2)
    )
    good = await file_read_matches_file(ActionResult(action=action, success=True, output="five\nsix\n"))
    assert good.passed and good.observed and "last 2 line(s)" in good.detail
    bad = await file_read_matches_file(ActionResult(action=action, success=True, output="six\n"))
    assert not bad.passed and bad.observed


def test_head_and_tail_together_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        FileParams(path="a.log", head_lines=1, tail_lines=1)
