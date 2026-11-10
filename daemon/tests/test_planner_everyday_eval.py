"""The everyday planner eval's acceptance checker, and the planner rules it measures.

The eval itself calls a model and is opt-in (scripts/run_planner_everyday_eval.py); these tests
need no model and execute nothing.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from benchmarks.planner_everyday import RULES, check_plan, load_cases, norm_path, path_flavour_for
from pilot.actions import Action, ActionType, FileParams
from pilot.agents import planner as planner_module
from pilot.agents.multi_agent import AgentRole, MultiAgentRouter
from pilot.agents.planner import SYSTEM_PROMPT, Planner, _expand_text_edits, _strictly_inside

WIN_HOME = "C:\\Users\\me"
POSIX_HOME = "/home/me"
CASES = {case["id"]: case for case in load_cases()}


def _step(kind: str, target: str = "", use_previous_output: bool = False, **params) -> dict:
    return {
        "action_type": kind,
        "target": target or str(params.get("path") or params.get("name") or ""),
        "parameters": params,
        "use_previous_output": use_previous_output,
    }


def _check(case_id: str, steps: list[dict], home: str = WIN_HOME, **kwargs):
    return check_plan(CASES[case_id], steps, home=home, **kwargs)


# --- the case file ----------------------------------------------------------------------------------


def test_case_file_is_well_formed_and_covers_the_review_set_and_the_launch_job():
    valid = {kind.value for kind in ActionType}
    ids = [case["id"] for case in load_cases()]
    assert len(ids) == len(set(ids)) >= 48
    for case in CASES.values():
        specs = [case.get("steps") or []] + list(case.get("alternatives") or [])
        kinds = set(case.get("allowed") or []) | set(case.get("code_ok") or []) | set(case.get("not_undoable_ok") or [])
        for spec in [s for alternative in specs for s in alternative] + list(case.get("forbid") or []):
            kind = spec.get("type")
            kinds.update([kind] if isinstance(kind, str) else kind or [])
        assert kinds <= valid, (case["id"], kinds - valid)

    review = json.loads(Path("benchmarks/gnomon_everyday_review_v1.json").read_text(encoding="utf-8"))
    correct = [case["request"] for case in review["cases"] if case["label"] == "correct"]
    ours = [case["request"] for case in CASES.values() if case["source"] == "gnomon_everyday_review_v1"]
    assert ours == correct and len(ours) == 28

    launch_sentence = "Get my launch folder on the Desktop ready to ship."
    assert launch_sentence in Path("tests/test_multi_part_file_jobs.py").read_text(encoding="utf-8")
    assert any(case["request"] == launch_sentence for case in CASES.values())


# --- the checker ------------------------------------------------------------------------------------


def test_a_shell_delete_is_rejected_where_a_typed_delete_exists():
    drafts = WIN_HOME + "\\Desktop\\launch\\drafts"
    shell = _check(
        "a09_delete_drafts_folder", [_step("shell_command", drafts, command="rmdir", args=["/s", "/q", drafts])]
    )
    assert not shell.passed
    assert shell.rules["no_code"] is False and shell.rules["undo"] is False and shell.rules["types"] is False

    typed = _check("a09_delete_drafts_folder", [_step("file_delete", path=drafts, recursive=True)])
    assert typed.passed, typed.failures

    not_recursive = _check("a09_delete_drafts_folder", [_step("file_delete", path=drafts)])
    assert not_recursive.rules["params"] is False


def test_a_forbidden_shell_step_fails_even_where_a_shell_step_is_allowed():
    launch = WIN_HOME + "\\Desktop\\launch"
    zip_and_delete = [
        _step("shell_script", launch, script=f"Remove-Item -Recurse '{launch}\\drafts'", interpreter="powershell"),
    ]
    result = _check("a21_launch_ready_to_ship", zip_and_delete)
    assert result.rules["no_code"] is False

    only_zip = [
        _step("file_delete", path=launch + "\\drafts", recursive=True),
        _step("shell_script", launch, script=f"Compress-Archive -Path '{launch}' -DestinationPath '{launch}.zip'"),
    ]
    assert _check("a21_launch_ready_to_ship", only_zip).passed


def test_two_equivalent_correct_plans_both_pass():
    assert _check("g17_mute", [_step("volume_mute", "volume", mute=True)]).passed
    assert _check("g17_mute", [_step("volume_set", "volume", level=0)]).passed

    source = WIN_HOME + "\\Desktop\\launch\\screenshots"
    into_folder = _step("file_copy", path=source, destination=WIN_HOME + "\\Documents", recursive=True)
    named_copy = _step("file_copy", path=source, destination=WIN_HOME + "\\Documents\\screenshots", recursive=True)
    assert _check("a07_copy_screenshots_folder", [into_folder]).passed
    assert _check("a07_copy_screenshots_folder", [named_copy]).passed


@pytest.mark.parametrize(
    ("home", "sep"),
    [(WIN_HOME, "\\"), (POSIX_HOME, "/")],
    ids=["windows", "posix"],
)
def test_a_wrong_path_is_caught_on_either_system(home, sep):
    launch = sep.join([home, "Desktop", "launch"])
    right = _step("file_move", path=launch + sep + "post-final.md", destination=launch + sep + "post-v17.md")
    wrong_folder = _step(
        "file_move", path=launch + sep + "post-final.md", destination=sep.join([home, "Desktop", "post-v17.md"])
    )
    wrong_file = _step("file_move", path=launch + sep + "README.txt", destination=launch + sep + "post-v17.md")
    assert _check("a05_rename_post", [right], home=home).passed
    assert _check("a05_rename_post", [wrong_folder], home=home).rules["params"] is False
    assert _check("a05_rename_post", [wrong_file], home=home).rules["params"] is False


def test_an_append_must_keep_the_old_text_and_read_before_writing():
    path = WIN_HOME + "\\Desktop\\launch\\checklist.txt"
    read = _step("file_read", path=path)
    keep = _step("file_write", path=path, content="{PREV_OUTPUT}\nship it", use_previous_output=True)
    assert _check("a01_append_ship_it", [read, keep]).passed

    overwrite = _step("file_write", path=path, content="ship it")
    assert _check("a01_append_ship_it", [overwrite]).rules["params"] is False

    backwards = _check("a01_append_ship_it", [keep, read])
    assert backwards.rules["order"] is False

    code = _step("code_execute", "append", use_previous_output=True, code="print(PREV_OUTPUT + 'ship it')")
    with_code = _check("a01_append_ship_it", [read, code, keep])
    assert with_code.rules["no_code"] is False and with_code.rules["max_steps"] is False


def test_a_three_part_job_is_checked_in_order():
    source = WIN_HOME + "\\Desktop\\launch\\checklist.txt"
    copy = WIN_HOME + "\\Documents\\checklist.txt"
    steps = [
        _step("file_copy", path=source, destination=copy),
        _step("file_read", path=copy),
        _step("file_write", path=copy, content="{PREV_OUTPUT}\nprint badges", use_previous_output=True),
    ]
    assert _check("a18_copy_then_append", steps).passed
    reordered = _check("a18_copy_then_append", [steps[1], steps[2], steps[0]])
    assert reordered.rules["order"] is False


def test_asking_counts_only_where_the_request_does_not_say_which_file():
    asked = check_plan(CASES["g02_copy_logo"], [], explanation="Which logo, and where is the website folder?")
    assert asked.passed and asked.asked

    refused = check_plan(CASES["a05_rename_post"], [], explanation="Which file do you mean?")
    assert not refused.passed

    failed = check_plan(CASES["g04_volume_30"], None, error="Planning failed: timeout")
    assert not failed.passed and set(failed.rules) == set(RULES)


def test_read_only_extra_steps_pass_only_where_looking_first_is_allowed():
    move = _step("file_move", path="C:\\work\\report.txt", destination="C:\\work\\report-final.txt")
    look = _step("file_search", path=WIN_HOME, pattern="report.txt")
    assert _check("g01_rename_report", [look, move]).passed
    assert _check("a05_rename_post", [look, move]).rules["types"] is False


def test_paths_compare_by_the_rules_of_the_system_they_were_made_for():
    assert path_flavour_for(WIN_HOME) is PureWindowsPath
    assert path_flavour_for(POSIX_HOME) is PurePosixPath
    # Windows: separators and case do not matter.
    windows = PureWindowsPath
    assert norm_path("C:\\Users\\Me\\Desktop\\", windows) == norm_path("c:/users/me/desktop", windows)
    # POSIX: case matters and a backslash is part of a name.
    posix = PurePosixPath
    assert norm_path("/home/me/Desktop/", posix) == "/home/me/Desktop"
    assert norm_path("/home/me/Desktop", posix) != norm_path("/home/me/desktop", posix)
    assert norm_path("/home/me/a\\b", posix) == "/home/me/a\\b"


def test_a_posix_plan_is_checked_with_case_sensitive_paths():
    launch = POSIX_HOME + "/Desktop/launch"
    right = _step("file_move", path=launch + "/post-final.md", destination=launch + "/post-v17.md")
    wrong_case = _step("file_move", path=launch + "/POST-FINAL.md", destination=launch + "/post-v17.md")
    assert _check("a05_rename_post", [right], home=POSIX_HOME).passed
    assert _check("a05_rename_post", [wrong_case], home=POSIX_HOME).rules["params"] is False


# --- planner rules this eval measures -------------------------------------------------------------


def _prompt() -> str:
    return SYSTEM_PROMPT.format(os="Windows", path_style="Windows", home=WIN_HOME)


def test_prompt_teaches_typed_text_edits_and_the_smallest_plan():
    prompt = _prompt()
    assert 'file_write {"path": P,\n  "content": "{PREV_OUTPUT}\\nnew line"}' in prompt
    assert '"content": "new line\\n{PREV_OUTPUT}"' in prompt
    assert "the file_read must come directly before its file_write" in prompt
    assert "SMALLEST CORRECT PLAN" in prompt
    assert "no file_read after a write to check it" in prompt
    assert "CODE AND SHELL ARE THE LAST RESORT" in prompt
    assert "STEP ORDER" in prompt
    assert 'volume_set {"level": 30}' in prompt and 'clipboard_write {"content"' in prompt
    # An empty folder has no typed action; the shell is allowed for that one job.
    assert '{"command": "mkdir", "args": ["C:\\path\\new folder"]}' in prompt
    new_rules = prompt[prompt.index("19b.") : prompt.index("20. ")]
    assert "\u2014" not in new_rules


def test_a_one_step_append_is_expanded_into_a_read_and_a_write_that_keep_the_file():
    path = "C:\\Users\\me\\todo.txt"
    raw = [
        {
            "action_type": "file_write",
            "target": path,
            "parameters": {"path": path, "content": "buy milk", "mode": "append"},
        }
    ]
    expanded = _expand_text_edits(raw)
    assert [a["action_type"] for a in expanded] == ["file_read", "file_write"]
    assert expanded[1]["parameters"]["content"] == "{PREV_OUTPUT}\nbuy milk"
    assert expanded[1]["use_previous_output"] is True

    top = _expand_text_edits([{"action_type": "file_prepend", "parameters": {"path": path, "content": "first"}}])
    assert top[1]["parameters"]["content"] == "first\n{PREV_OUTPUT}"

    already_read = [{"action_type": "file_read", "parameters": {"path": path}}] + raw
    assert [a["action_type"] for a in _expand_text_edits(already_read)] == ["file_read", "file_write"]

    plain = [{"action_type": "file_write", "parameters": {"path": path, "content": "whole new text"}}]
    assert _expand_text_edits(plain) == plain


def test_parsed_plan_with_append_mode_never_replaces_the_file_with_the_new_line():
    planner = Planner.__new__(Planner)
    path = "C:\\Users\\me\\todo.txt"
    raw = json.dumps(
        {
            "explanation": "Add buy milk to the todo file.",
            "actions": [
                {
                    "action_type": "file_write",
                    "target": path,
                    "parameters": {"path": path, "content": "buy milk", "mode": "append"},
                }
            ],
        }
    )
    plan = planner._parse_response(raw, "Add 'buy milk' to my todo file")
    assert plan.error is None
    assert [a.action_type for a in plan.actions] == [ActionType.FILE_READ, ActionType.FILE_WRITE]
    assert plan.actions[1].parameters.content == "{PREV_OUTPUT}\nbuy milk"


def _file(kind: ActionType, path: str, content: str | None = None, use_previous_output: bool = False) -> Action:
    return Action(
        action_type=kind,
        target=path,
        parameters=FileParams(path=path, content=content),
        use_previous_output=use_previous_output,
    )


# The post-processor compares paths by the rules of the machine the daemon runs on. Each test runs
# under both rules, with paths written for that system, so Windows and POSIX are covered on any OS.
FLAVOURS = [
    pytest.param(PureWindowsPath, "C:\\Users\\me\\Desktop", "\\", id="windows"),
    pytest.param(PurePosixPath, "/home/me/Desktop", "/", id="posix"),
]


@pytest.mark.parametrize(("flavour", "desktop", "sep"), FLAVOURS)
def test_the_read_of_a_file_that_is_then_written_back_is_kept(monkeypatch, flavour, desktop, sep):
    # Seen on 10 November 2026: the planner returned read, code, write, read; the post-processor dropped
    # the first read because a later write had the same path, so the append would have lost the file.
    monkeypatch.setattr(planner_module, "_path_flavour", lambda: flavour)
    path = sep.join([desktop, "launch", "checklist.txt"])
    actions = [
        _file(ActionType.FILE_READ, path),
        _file(ActionType.FILE_WRITE, path, "{PREV_OUTPUT}\nship it", use_previous_output=True),
    ]
    kept = Planner.__new__(Planner)._postprocess_actions(actions)
    assert [a.action_type for a in kept] == [ActionType.FILE_READ, ActionType.FILE_WRITE]


@pytest.mark.parametrize(("flavour", "desktop", "sep"), FLAVOURS)
def test_a_leading_read_of_a_folder_a_later_step_writes_into_is_still_dropped(monkeypatch, flavour, desktop, sep):
    monkeypatch.setattr(planner_module, "_path_flavour", lambda: flavour)
    folder = sep.join([desktop, "report"])
    actions = [
        _file(ActionType.FILE_READ, folder),
        _file(ActionType.FILE_WRITE, folder + sep + "summary.txt", "done"),
    ]
    kept = Planner.__new__(Planner)._postprocess_actions(actions)
    assert [a.action_type for a in kept] == [ActionType.FILE_WRITE]


def test_the_post_processor_uses_the_running_systems_path_rules():
    import os

    expected = PureWindowsPath if os.name == "nt" else PurePosixPath
    assert planner_module._path_flavour() is expected


@pytest.mark.parametrize(
    ("flavour", "path", "folder", "inside"),
    [
        (PureWindowsPath, "C:\\Users\\me\\report\\a.txt", "C:\\Users\\me\\report", True),
        (PureWindowsPath, "C:/Users/me/Report/a.txt", "c:\\users\\me\\report", True),
        (PureWindowsPath, "C:\\Users\\me\\report.txt", "C:\\Users\\me\\report.txt", False),
        (PureWindowsPath, "C:\\Users\\me\\reports\\a.txt", "C:\\Users\\me\\report", False),
        (PurePosixPath, "/home/me/report/a.txt", "/home/me/report", True),
        (PurePosixPath, "/home/me/report.txt", "/home/me/report.txt", False),
        (PurePosixPath, "/home/me/Report/a.txt", "/home/me/report", False),
        (PurePosixPath, "/home/me/a.txt", "", False),
    ],
)
def test_strictly_inside_follows_each_systems_path_rules(flavour, path, folder, inside):
    assert _strictly_inside(path, folder, flavour) is inside


def test_specialist_routing_matches_whole_words():
    router = MultiAgentRouter(None)
    assert AgentRole.CODE_AGENT not in router.classify("Get the latest Node LTS installer")
    assert AgentRole.FORENSICS_AGENT not in router.classify("Copy the logo into the website folder")
    assert AgentRole.MONITOR_AGENT not in router.classify("Open the program")
    assert router.classify("run the tests") == [AgentRole.CODE_AGENT]
    assert AgentRole.SYSTEM_AGENT in router.classify("Put this on my clipboard")
    assert AgentRole.FORENSICS_AGENT in router.classify("Analyze the auth logs")


def test_a_plan_after_a_sentence_of_prose_is_still_read():
    planner = Planner.__new__(Planner)
    plan_json = json.dumps(
        {
            "explanation": "Set the volume to 30.",
            "actions": [{"action_type": "volume_set", "target": "volume", "parameters": {"level": 30}}],
        }
    )
    for raw in (f"I'll set it now. {plan_json}", f"Setting the volume.\n```json\n{plan_json}\n```"):
        plan = planner._parse_response(raw, "Set the volume to 30 percent")
        assert plan.error is None
        assert [a.action_type for a in plan.actions] == [ActionType.VOLUME_SET]

    still_bad = planner._parse_response("I cannot plan this {not json}", "x")
    assert still_bad.error and still_bad.error.startswith("JSON PARSING FAILED")
