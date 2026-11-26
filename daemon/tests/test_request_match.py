"""The plan is held against the request in code (pilot.agents.request_match).

Two lists: plans a person would call right (they must pass, including requests that name nothing, that point back
with "it" or "again", that have several steps, or that are not in English) and plans that do another request's work
(they must be refused). The last test counts both lists, the way the live run's records were counted."""

from __future__ import annotations

import pytest

from pilot.actions import Action, ActionType, ApiRequestParams, CodeExecParams, FileParams, ShellCommandParams
from pilot.agents import request_match


def copy(src: str, dst: str) -> Action:
    return Action(action_type=ActionType.FILE_COPY, target=src, parameters=FileParams(path=src, destination=dst))


def move(src: str, dst: str) -> Action:
    return Action(action_type=ActionType.FILE_MOVE, target=src, parameters=FileParams(path=src, destination=dst))


def delete(path: str) -> Action:
    return Action(action_type=ActionType.FILE_DELETE, target=path, parameters=FileParams(path=path))


def write(path: str, content: str = "x", previous: bool = False) -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE,
        target=path,
        parameters=FileParams(path=path, content=content),
        use_previous_output=previous,
    )


def read(path: str, **lines: int) -> Action:
    return Action(action_type=ActionType.FILE_READ, target=path, parameters=FileParams(path=path, **lines))


def slack(message: str) -> Action:
    return Action(action_type=ActionType.API_SLACK, target="slack", parameters=ApiRequestParams(message=message))


def mkdir(path: str) -> Action:
    return Action(
        action_type=ActionType.SHELL_COMMAND, target=path, parameters=ShellCommandParams(command="mkdir", args=[path])
    )


W = "C:\\w"
D = "C:\\Users\\p"

# (request, plan, earlier texts "it" may point to)
RIGHT = [
    ("Tidy my downloads", [move(f"{D}\\Downloads\\a.pdf", f"{D}\\Downloads\\PDFs\\a.pdf")], []),
    ("Delete it", [delete(f"{W}\\b.txt")], [f"Copy {W}\\a.txt to {W}\\b.txt"]),
    (
        "Copy that one too",
        [copy(f"{W}\\c.txt", f"{W}\\backup\\c.txt")],
        [f"Copy {W}\\a.txt to {W}\\backup", f"Copied a.txt to backup. c.txt is in {W} as well."],
    ),
    (
        f"Make a folder {W}\\archive and move {W}\\notes\\note_backup.txt into it",
        [mkdir(f"{W}\\archive"), move(f"{W}\\notes\\note_backup.txt", f"{W}\\archive\\note_backup.txt")],
        [],
    ),
    (f"Lösche {W}\\alt.txt", [delete(f"{W}\\alt.txt")], []),
    (f"Copia {W}\\a.txt en {W}\\b.txt", [copy(f"{W}\\a.txt", f"{W}\\b.txt")], []),
    ("Borra el archivo informe.pdf del escritorio", [delete(f"{D}\\Desktop\\informe.pdf")], []),
    (
        f"Rename {W}\\notes\\note_copy.txt to note_backup.txt in the same folder",
        [move(f"{W}\\notes\\note_copy.txt", f"{W}\\notes\\note_backup.txt")],
        [],
    ),
    (
        f"Add the line 'Also renew the parking permit.' at the end of the file {W}\\notes\\note.txt",
        [read(f"{W}\\notes\\note.txt"), write(f"{W}\\notes\\note.txt", "{PREV_OUTPUT}\nAlso renew", True)],
        [],
    ),
    (f"Back up {D}\\Documents to E:\\Backup", [copy(f"{D}\\Documents", "E:\\Backup\\Documents")], []),
    (
        f"Write a note to {W}\\note.txt saying: delete the old files tomorrow",
        [write(f"{W}\\note.txt", "delete the old files tomorrow")],
        [],
    ),
    (
        "Move all PDFs from Downloads to Documents\\PDFs",
        [move(f"{D}\\Downloads\\a.pdf", f"{D}\\Documents\\PDFs\\a.pdf")],
        [],
    ),
    ("Post 'Build is green' to Slack", [slack("Build is green")], []),
    ("Send it to Slack", [read(f"{W}\\sum.txt"), slack("{PREV_OUTPUT}")], [f"Write a summary to {W}\\sum.txt"]),
    (f"Delete the duplicates in {W}\\photos", [delete(f"{W}\\photos\\a (1).jpg")], []),
    (
        "Save the page title of https://example.com to title.txt on my desktop",
        [write(f"{D}\\Desktop\\title.txt", "Example Domain")],
        [],
    ),
    ("Copy /home/p/a.txt to /home/p/backup/", [copy("/home/p/a.txt", "/home/p/backup/a.txt")], []),
    (f"Copy C:\\My Files\\a.txt to D:\\Backup", [copy("C:\\My Files\\a.txt", "D:\\Backup\\a.txt")], []),
    (f"Clean up {W}\\tmp", [delete(f"{W}\\tmp\\old.log")], []),
    ("Do that again", [copy(f"{W}\\a.txt", f"{W}\\b.txt")], [f"Copy {W}\\a.txt to {W}\\b.txt"]),
]

WRONG = [
    # The live run: "Delete note_backup.txt" planned as the earlier turn's copy.
    (
        f"Delete {W}\\archive\\note_backup.txt",
        [copy(f"{W}\\notes\\does_not_exist.txt", f"{W}\\notes\\x.txt")],
        [f"Copy {W}\\notes\\does_not_exist.txt to {W}\\notes\\x.txt"],
    ),
    (f"Delete {W}\\a.txt", [delete(f"{W}\\b.txt")], []),
    (
        f"Copy {W}\\a.txt to {W}\\b.txt",
        [copy(f"{W}\\a.txt", f"{W}\\old_dest.txt")],
        [f"Copy {W}\\a.txt to {W}\\old_dest.txt"],
    ),
    ("Delete it", [delete(f"{W}\\other.txt")], [f"Copy {W}\\a.txt to {W}\\b.txt"]),
    ("Delete it", [copy(f"{W}\\a.txt", f"{W}\\b.txt")], [f"Copy {W}\\a.txt to {W}\\b.txt"]),
    ("Post 'Build is green' to Slack", [slack("Deploy failed")], []),
    (f"Write hello to {W}\\a.txt", [slack("hello")], []),
    (
        f"Write a file {W}\\readonly\\blocked.txt containing the word hello",
        [write(f"{W}\\elsewhere\\blocked.txt", "hello")],
        [],
    ),
]

# Known limit: a request in another language has no verb the lists know, so a copy for "delete" in German passes
# when it uses the named file. (The prompt fix is what keeps the model on the right request there.)
KNOWN_ACCEPTED = [(f"Lösche {W}\\alt.txt", [copy(f"{W}\\alt.txt", f"{W}\\neu.txt")], [])]


@pytest.mark.parametrize(("request_text", "plan", "earlier"), RIGHT)
def test_a_plan_that_does_what_was_asked_passes(request_text, plan, earlier) -> None:
    assert request_match.check_plan(request_text, plan, earlier=earlier) is None


@pytest.mark.parametrize(("request_text", "plan", "earlier"), WRONG)
def test_a_plan_for_another_request_is_refused(request_text, plan, earlier) -> None:
    assert request_match.check_plan(request_text, plan, earlier=earlier) is not None


def test_the_known_limit_for_other_languages_is_what_it_says() -> None:
    [(request_text, plan, earlier)] = KNOWN_ACCEPTED
    assert request_match.check_plan(request_text, plan, earlier=earlier) is None


def test_the_live_runs_delete_is_refused_for_its_kind_and_named_in_a_sentence() -> None:
    request_text, plan, earlier = WRONG[0]
    problem = request_match.check_plan(request_text, plan, earlier=earlier)
    assert problem is not None and problem.rule == "wrong_kind"
    sentence = request_match.first_refusal(problem, request_text)
    assert sentence.startswith(request_match.REFUSED_FIRST)
    assert "copy does_not_exist.txt to x.txt" in sentence and "Nothing was changed" in sentence
    assert request_match.is_refusal(sentence)


def test_a_repair_may_not_copy_to_an_earlier_turns_destination() -> None:
    request_text = f"Copy {W}\\big\\big300.bin to {W}\\big\\big300_tampered.bin"
    failed = [copy(f"{W}\\big\\big300.bin", f"{W}\\big\\big300_tampered.bin")]
    repair = [copy(f"{W}\\big\\big300.bin", f"{W}\\big\\big300_copy.bin")]
    earlier = [f"Copy {W}\\big\\big300.bin to {W}\\big\\big300_copy.bin"]
    problem = request_match.check_repair(request_text, repair, failed=failed, earlier_requests=earlier)
    assert problem is not None and problem.rule == "repair_earlier_turn"
    assert request_match.repair_refusal(problem) == (
        "I did not try again: the retry would copy big300.bin to big300_copy.bin, which this request did not name."
    )
    # Retrying the failed step itself is a repair.
    assert request_match.check_repair(request_text, failed, failed=failed, earlier_requests=earlier) is None


def test_a_repair_may_not_write_a_path_nobody_named() -> None:
    request_text = f"Copy {W}\\a.txt to {W}\\b.txt"
    failed = [copy(f"{W}\\a.txt", f"{W}\\b.txt")]
    problem = request_match.check_repair(request_text, [copy(f"{W}\\a.txt", f"{W}\\b2.txt")], failed=failed)
    assert problem is not None and problem.rule == "repair_new_path"


def test_a_repair_that_redoes_one_copy_of_three_passes() -> None:
    request_text = f"Copy {W}\\big.bin to {W}\\d.bin, to {W}\\e.bin and to {W}\\t.bin"
    failed = [copy(f"{W}\\big.bin", f"{W}\\{name}.bin") for name in ("d", "e", "t")]
    assert request_match.check_repair(request_text, failed[2:], failed=failed) is None


def test_read_only_plans_and_code_are_not_checked_here() -> None:
    assert request_match.check_plan(f"Delete {W}\\a.txt", [read(f"{W}\\b.txt")]) is None


def test_a_file_name_in_a_path_is_not_a_verb() -> None:
    named = request_match.named_in(f"Delete {W}\\archive\\note_backup.txt")
    assert named.kinds == frozenset({"delete"}) and not named.open_verb


def test_paths_with_spaces_and_two_paths_in_one_line() -> None:
    named = request_match.named_in("Copy C:\\My Files\\a.txt to D:\\Backup")
    assert named.absolute == ("c:\\my files\\a.txt", "d:\\backup")


def test_reading_lines_with_code_gets_a_hint_and_a_typed_read_does_not() -> None:
    code = Action(
        action_type=ActionType.CODE_EXECUTE,
        target="",
        parameters=CodeExecParams(code="lines = PREV_OUTPUT.splitlines()\nprint('\\n'.join(lines[-5:]))"),
        use_previous_output=True,
    )
    assert "tail_lines" in request_match.read_done_by_code([read(f"{W}\\app.log"), code])
    assert request_match.read_done_by_code([read(f"{W}\\app.log", tail_lines=5)]) == ""
    tail = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="tail",
        parameters=ShellCommandParams(command="tail", args=["-n", "5", "/var/log/app.log"]),
    )
    assert "tail_lines" in request_match.read_done_by_code([tail])


def test_counts_over_both_lists() -> None:
    false_refusals = [r for r, p, e in RIGHT if request_match.check_plan(r, p, earlier=e) is not None]
    false_accepts = [r for r, p, e in WRONG if request_match.check_plan(r, p, earlier=e) is None]
    assert (len(RIGHT), len(false_refusals)) == (20, 0)
    assert (len(WRONG), len(false_accepts)) == (8, 0)


# ── other languages and scripts ──────────────────────────────────────

HINDI = "रिपोर्ट.txt"
JAPANESE = "報告書.txt"
ARABIC = "تقرير.txt"


def nfd(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFD", text)


@pytest.mark.parametrize(
    ("request_text", "plan"),
    [
        # Hindi: the file name has vowel signs (combining marks), the plan spells it decomposed.
        (f"{W}\\{HINDI} को हटाओ", [delete(nfd(f"{W}\\{HINDI}"))]),
        (f"{HINDI} हटाओ", [delete(f"{D}\\Desktop\\{HINDI}")]),
        # Japanese: the path runs straight into the next word, with and without an extension.
        (f"{W}\\{JAPANESE}を削除して", [delete(f"{W}\\{JAPANESE}")]),
        (f"{W}\\写真を整理して", [move(f"{W}\\写真\\a.jpg", f"{W}\\写真\\2026\\a.jpg")]),
        # Arabic, right to left around a left-to-right path.
        (f"احذف الملف {W}\\{ARABIC}", [delete(f"{W}\\{ARABIC}")]),
        # A kind the English lists would call wrong is not refused in a language they do not read.
        (f"{W}\\{HINDI} की कॉपी {W}\\backup में बनाओ", [copy(f"{W}\\{HINDI}", f"{W}\\backup\\{HINDI}")]),
        (f"{W}\\{JAPANESE}を{W}\\backupにコピーして", [copy(f"{W}\\{JAPANESE}", f"{W}\\backup\\{JAPANESE}")]),
        (f"انسخ {W}\\{ARABIC} إلى {W}\\backup", [copy(f"{W}\\{ARABIC}", f"{W}\\backup\\{ARABIC}")]),
        # Combining characters in a Latin name: typed composed, planned decomposed.
        (f"Delete {W}\\café.txt", [delete(nfd(f"{W}\\café.txt"))]),
        (f"Delete {nfd('résumé.pdf')}", [delete(f"{D}\\Desktop\\résumé.pdf")]),
    ],
)
def test_requests_in_other_scripts_are_never_refused_for_their_words(request_text, plan) -> None:
    assert request_match.check_plan(request_text, plan) is None


@pytest.mark.parametrize(
    "request_text",
    [f"{W}\\{HINDI} को हटाओ", f"{W}\\{JAPANESE}を削除して", f"احذف الملف {W}\\{ARABIC}"],
)
def test_no_verb_is_read_from_hindi_japanese_or_arabic(request_text) -> None:
    named = request_match.named_in(request_text)
    assert named.kinds == frozenset() and named.targets


@pytest.mark.parametrize(
    "request_text",
    [
        f"{W}\\archive\\note_backup.txt को हटाओ",
        f"{W}\\archive\\note_backup.txtを削除して",
        f"احذف {W}\\archive\\note_backup.txt",
    ],
)
def test_a_plan_for_another_file_is_still_refused_in_any_script(request_text) -> None:
    plan = [copy(f"{W}\\notes\\does_not_exist.txt", f"{W}\\notes\\x.txt")]
    problem = request_match.check_plan(request_text, plan)
    assert problem is not None and problem.rule == "off_target"
