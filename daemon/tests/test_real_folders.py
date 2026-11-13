"""Tools outside the daemon (benches, evals, replay tools, measuring scripts) never use the person's real folders."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pilot import config, real_folders
from pilot.real_folders import RealFoldersRefused, refuse_real_folders


@pytest.fixture
def as_on_the_real_folders(monkeypatch):
    """This process as a tool started by the person on their own account: no test root, the home is their own."""
    monkeypatch.setattr(config, "_TEST_ROOT", None)
    monkeypatch.setattr(real_folders, "account_home", lambda: Path.home())
    monkeypatch.setattr(real_folders, "_daemon_entry", False)
    monkeypatch.setattr(sys, "argv", ["tool"])


def test_the_suite_runs_under_its_test_root():
    assert refuse_real_folders("a test") == "test root"


def test_a_tool_on_the_real_folders_is_refused(as_on_the_real_folders):
    with pytest.raises(RealFoldersRefused, match="real Heliox folders"):
        refuse_real_folders("The bench")


def test_the_explicit_flag_or_the_daemon_allows_the_real_folders(as_on_the_real_folders, monkeypatch):
    assert refuse_real_folders("The bench", allow_real=True) == "real"
    monkeypatch.setattr(sys, "argv", ["tool", "--real-folders"])
    assert refuse_real_folders("The bench") == "real"
    monkeypatch.setattr(sys, "argv", ["tool"])
    real_folders.enter_as_daemon()
    assert refuse_real_folders("The daemon") == "real"


def test_a_fake_home_is_allowed_only_when_every_folder_is_inside_it(as_on_the_real_folders, monkeypatch, tmp_path):
    fake = tmp_path / "home"
    monkeypatch.setattr(real_folders, "account_home", lambda: tmp_path / "the-real-one")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    inside = {name: fake / name / "heliox-os" for name in ("config", "data", "state", "runtime", "cache", "setup")}
    monkeypatch.setattr(real_folders, "heliox_folders", lambda: inside)
    assert refuse_real_folders("The bench") == "fake home"

    # On Linux the runtime folder is /run/user/<uid> unless XDG_RUNTIME_DIR moves too: still the real one.
    monkeypatch.setattr(real_folders, "heliox_folders", lambda: {**inside, "runtime": tmp_path / "run" / "heliox-os"})
    with pytest.raises(RealFoldersRefused, match="runtime"):
        refuse_real_folders("The bench")


def test_starting_a_server_outside_the_daemon_is_refused(as_on_the_real_folders):
    from pilot.config import PilotConfig
    from pilot.server import PilotServer

    with pytest.raises(RealFoldersRefused, match="Heliox server"):
        PilotServer(PilotConfig())


def test_a_listener_outside_the_daemon_does_not_save_the_wake_calibration(as_on_the_real_folders):
    from pilot.system.voice_calibration import VoiceCalibrationStore, WakeWordVariant

    target = config.CACHE_DIR / "voice_calibration" / "wake_variants.json"
    target.unlink(missing_ok=True)

    VoiceCalibrationStore().save({"hey helios": WakeWordVariant(text="hey helios", confirmed_count=5)})

    assert not target.exists()


def test_use_scratch_folders_keeps_the_test_root_it_finds():
    assert real_folders.use_scratch_folders(Path("elsewhere")) == config._TEST_ROOT


def test_a_process_a_test_starts_finds_the_same_test_root():
    shown = subprocess.run(
        [sys.executable, "-c", "import pilot.config as c; print(c._TEST_ROOT); print(c.CACHE_DIR)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.splitlines()

    assert shown[0] == str(config._TEST_ROOT)
    assert Path(shown[1]) == config.CACHE_DIR


def test_a_scratch_root_moves_every_folder_in_a_new_process(tmp_path):
    code = (
        "import os; os.environ.pop('HELIOX_TEST_ROOT', None)\n"
        "from pathlib import Path; import sys\n"
        "from pilot.real_folders import use_scratch_folders, refuse_real_folders\n"
        "root = use_scratch_folders(Path(sys.argv[1]))\n"
        "from pilot import config\n"
        "print(refuse_real_folders('a tool'))\n"
        "print(all(str(p).startswith(str(root)) for p in (config.DATA_DIR, config.CACHE_DIR, config.SETUP_DIR,"
        " config.RUNTIME_DIR, config.CONFIG_DIR, config.STATE_DIR)))\n"
    )
    shown = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "scratch")], capture_output=True, text=True, timeout=60, check=True
    ).stdout.split()

    assert shown == ["test", "root", "True"]


def test_the_suite_and_its_processes_never_reach_the_system_keyring():
    import keyring
    import keyring.backends.fail

    assert isinstance(keyring.get_keyring(), keyring.backends.fail.Keyring)
    shown = subprocess.run(
        [sys.executable, "-c", "import keyring; print(type(keyring.get_keyring()).__module__)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()
    assert shown == "keyring.backends.fail"


def test_saving_the_settings_outside_the_daemon_on_the_real_folders_is_refused(as_on_the_real_folders):
    from pilot.config import PilotConfig

    before = config.CONFIG_FILE.read_text(encoding="utf-8") if config.CONFIG_FILE.exists() else None
    with pytest.raises(RealFoldersRefused, match="settings"):
        PilotConfig().save()
    after = config.CONFIG_FILE.read_text(encoding="utf-8") if config.CONFIG_FILE.exists() else None
    assert after == before


def test_a_save_keeps_the_file_it_replaces_and_a_bounded_number_of_copies(monkeypatch, tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("[model]\nprovider = 'cloud'\n", encoding="utf-8")
    monkeypatch.setattr(config, "SAVED_COPIES", 3)

    config._save_with_backup(target, "[model]\n")
    copies = config._saved_copies(target)
    assert len(copies) == 1 and copies[0].read_text(encoding="utf-8") == "[model]\nprovider = 'cloud'\n"
    assert target.read_text(encoding="utf-8") == "[model]\n"

    config._save_with_backup(target, "[model]\n")  # the same content again: nothing to keep
    assert len(config._saved_copies(target)) == 1

    for number in range(5):
        config._save_with_backup(target, f"# {number}\n")
    # The newest three, and today's earliest (the owner's file from the first save) as the day's copy.
    kept = config._saved_copies(target)
    assert len(kept) == 4 and kept[0].read_text(encoding="utf-8") == "[model]\nprovider = 'cloud'\n"


def test_copies_beyond_the_newest_keep_the_earliest_of_each_day_for_thirty_days(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SAVED_COPIES", 2)
    names = [
        "20261001-080000",  # 42 days before: too old
        "20261020-080000",  # earliest of 20 October: kept
        "20261020-090000",
        "20261111-070000",  # earliest of 11 November: kept
        "20261111-230000",
        "20261112-100000",  # earliest of 12 November, also among the newest two
        "20261112-110000",  # newest
    ]
    copies = [tmp_path / f"config.toml.saved-{name}" for name in names]

    kept = config._copies_to_keep(copies, "20261112")

    assert sorted(path.name[-15:] for path in kept) == [
        "20261020-080000",
        "20261111-070000",
        "20261112-100000",
        "20261112-110000",
    ]


def test_a_copy_that_is_not_ours_is_never_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SAVED_COPIES", 1)
    target = tmp_path / "config.toml"
    target.write_text("a = 1\n", encoding="utf-8")
    mine = tmp_path / "config.toml.saved-by-hand"
    mine.write_text("kept\n", encoding="utf-8")
    for number in range(3):
        config._save_with_backup(target, f"a = {number + 2}\n")
    assert mine.exists()


def test_a_save_on_a_new_file_keeps_nothing(tmp_path):
    target = tmp_path / "restrictions.toml"
    config._save_with_backup(target, "x = 1\n")
    assert config._saved_copies(target) == [] and target.read_text(encoding="utf-8") == "x = 1\n"


def test_the_settings_saved_from_the_daemon_keep_the_previous_file(monkeypatch, tmp_path):
    from pilot.config import PilotConfig

    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(config, "RESTRICTIONS_FILE", tmp_path / "restrictions.toml")
    first = PilotConfig()
    first.save()
    first.model.provider = "cloud" if first.model.provider != "cloud" else "ollama"
    first.save()

    assert config._saved_copies(config.CONFIG_FILE)


def test_a_paid_run_copies_the_persons_model_settings_onto_its_own_config(monkeypatch, tmp_path):
    from pilot.config import PilotConfig

    own = tmp_path / "account"
    (own / ".config" / "heliox-os").mkdir(parents=True)
    (own / ".config" / "heliox-os" / "config.toml").write_text(
        "[model]\nprovider = 'cloud'\ncloud_provider = 'claude'\nnot_a_setting = 1\n", encoding="utf-8"
    )
    monkeypatch.setattr(real_folders, "account_home", lambda: own)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    run_config = PilotConfig()

    assert sorted(real_folders.apply_owner_model_settings(run_config)) == ["cloud_provider", "provider"]
    assert run_config.model.provider == "cloud" and run_config.model.cloud_provider == "claude"
    assert real_folders.real_budget_db() == own / ".local" / "share" / "heliox-os" / "pilot.db"
    owners_file = own / ".config" / "heliox-os" / "config.toml"
    assert owners_file != config.CONFIG_FILE  # the run's config saves to the run's folder


def test_the_guard_protects_the_accounts_own_folders_when_the_home_is_faked(monkeypatch, tmp_path):
    import folder_guard

    own = tmp_path / "own"
    folders = folder_guard.real_folders(home=own)
    assert own / ".local" / "share" / "heliox-os" in folders and own / ".cache" / "heliox" in folders
    if sys.platform == "win32":
        assert own / "AppData" / "Local" / "heliox-os" / "runtime" in folders
    assert folder_guard.true_home() == real_folders.account_home()


def _owner_with(monkeypatch, tmp_path, text: str) -> Path:
    own = tmp_path / "account"
    (own / ".config" / "heliox-os").mkdir(parents=True)
    (own / ".config" / "heliox-os" / "config.toml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(real_folders, "account_home", lambda: own)
    for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    return own


VOICE = ("voice.final_pass", "voice.turn_hold_ms", "voice.input_device", "voice.parakeet_model_dir")


def _voice_owner(monkeypatch, tmp_path) -> Path:
    own = _owner_with(monkeypatch, tmp_path, "")
    models = (own / ".local" / "share" / "heliox-os" / "models" / "parakeet").as_posix()
    (own / ".config" / "heliox-os" / "config.toml").write_text(
        f"""[model]
provider = 'cloud'
[voice]
final_pass = 'parakeet'
turn_hold_ms = 1800.0
input_device = 'Headset (USB)'
tts_device = 'cuda'
parakeet_model_dir = '{models}'
barge_in_enabled = false
[conversation]
claude_model = 'claude-haiku-5-5'
share_screen_text = true
[live]
enabled = true
""",
        encoding="utf-8",
    )
    return own


def test_a_run_copies_only_the_named_fields_and_never_devices_or_paths_into_their_folders(monkeypatch, tmp_path):
    from pilot.config import PilotConfig

    _voice_owner(monkeypatch, tmp_path)
    run_config = PilotConfig()
    device_before, barge_before = run_config.voice.input_device, run_config.voice.barge_in_enabled
    share_before = run_config.conversation.share_screen_text

    copied = real_folders.apply_owner_settings(
        run_config, ("model", "voice", "conversation"), only=(*VOICE, "voice.tts_device", "conversation.claude_model")
    )

    assert sorted(copied) == ["conversation.claude_model", "model.provider", "voice.final_pass", "voice.turn_hold_ms"]
    assert run_config.voice.final_pass == "parakeet" and run_config.voice.parakeet_model_dir == ""
    assert run_config.voice.input_device == device_before  # a named device field is still this computer's own
    assert run_config.voice.barge_in_enabled == barge_before  # set by the person, not named: not copied
    assert run_config.conversation.share_screen_text == share_before
    assert run_config.live.enabled is False  # [live] is not a section a run copies


def test_a_voice_or_conversation_section_needs_its_fields_named(monkeypatch, tmp_path):
    from pilot.config import PilotConfig

    _voice_owner(monkeypatch, tmp_path)
    for section in ("voice", "conversation"):
        with pytest.raises(ValueError, match="only="):
            real_folders.apply_owner_settings(PilotConfig(), ("model", section))


def test_skip_leaves_a_named_field_or_section_alone(monkeypatch, tmp_path):
    from pilot.config import PilotConfig

    _owner_with(monkeypatch, tmp_path, "[model]\nprovider = 'cloud'\n[voice]\nfinal_pass = 'parakeet'\n")
    run_config = PilotConfig()

    copied = real_folders.apply_owner_settings(
        run_config, ("model", "voice"), only=("voice.final_pass",), skip=("voice.final_pass", "model")
    )

    assert copied == [] and run_config.voice.final_pass != "parakeet"


def test_only_the_named_sections_can_be_read(monkeypatch, tmp_path):
    _owner_with(monkeypatch, tmp_path, "[live]\nenabled = true\n")
    with pytest.raises(ValueError, match="not a section"):
        real_folders.owner_settings("live")


def _stamps(folder: Path) -> dict[str, tuple[int, int]]:
    found = {}
    for path in folder.rglob("*"):
        if path.is_file():
            info = path.stat()
            found[str(path)] = (info.st_size, info.st_mtime_ns)
    return found


@pytest.mark.asyncio
async def test_a_paid_runs_budget_write_is_named_as_its_own_and_any_other_change_is_not(monkeypatch, tmp_path):
    own = tmp_path / "own"
    data = own / ".local" / "share" / "heliox-os"
    data.mkdir(parents=True)
    notes = data / "notes.json"
    notes.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(real_folders, "account_home", lambda: own)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(real_folders, "_paid_stamps", {})
    before = _stamps(data)

    tracker = real_folders.real_budget_tracker(config.PilotConfig().model, "the test run")
    await tracker.initialize()
    await tracker.record_usage("claude", "claude-haiku", 100, 20)
    await tracker.close()
    notes.write_text('[{"text": "not the run"}]', encoding="utf-8")  # something else changed a file meanwhile

    said = dict(real_folders.describe_changes(before, _stamps(data)))
    assert said[str(data / "pilot.db")] == real_folders.PAID_RUN
    assert said[str(notes)] == real_folders.UNKNOWN
    lines = (config.STATE_DIR / real_folders.PAID_RUN_NOTES).read_text(encoding="utf-8").splitlines()
    assert any(line.endswith(f"the test run: recording a call's spend wrote {data / 'pilot.db'}") for line in lines)


def test_a_store_changed_after_the_runs_own_write_is_not_called_the_runs():
    path = "/somewhere/pilot.db"
    real_folders._paid_stamps[real_folders._key(Path(path))] = (10, 1)
    try:
        assert real_folders.who_changed(path, (10, 1)) == real_folders.PAID_RUN
        assert real_folders.who_changed(path, (12, 2)) == real_folders.UNKNOWN  # written again by someone else
        assert real_folders.who_changed(path, None) == real_folders.UNKNOWN  # removed
    finally:
        real_folders._paid_stamps.pop(real_folders._key(Path(path)), None)
