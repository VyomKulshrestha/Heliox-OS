import pytest

from pilot.config import PilotConfig, _config_to_dict, _merge_config, _validate_config_types


def test_default_persisted_config_is_fully_covered_by_validator():
    raw = _config_to_dict(PilotConfig())
    raw.pop("restrictions")

    _validate_config_types(raw)


def test_calendar_preview_cognitive_and_typed_decision_config_round_trip_through_merge():
    raw = {
        "calendar": {
            "enabled": True,
            "caldav_url": "https://calendar.example.test/dav",
            "caldav_username": "person@example.test",
            "caldav_password_provider": "caldav",
            "ics_files": ["local.ics"],
        },
        "email": {
            "enabled": True,
            "imap_host": "imap.example.test",
            "smtp_host": "smtp.example.test",
            "smtp_port": 587,
            "username": "person@example.test",
            "password_provider": "email",
        },
        "preview": {"enabled": True, "confirm_timeout_seconds": 45},
        "cognitive": {"enabled": False},
        "typed_decision": {
            "enabled": True,
            "artifact_dir": "C:/models/decision-candidate",
            "report_path": "C:/models/decision-candidate/report.json",
        },
    }

    _validate_config_types(raw)
    config = _merge_config(PilotConfig(), raw)

    assert config.calendar.enabled is True
    assert config.calendar.caldav_url == "https://calendar.example.test/dav"
    assert config.calendar.ics_files == ["local.ics"]
    assert config.email.enabled is True
    assert config.email.smtp_port == 587
    assert config.preview.enabled is True
    assert config.preview.confirm_timeout_seconds == 45
    assert config.cognitive.enabled is False
    assert config.typed_decision.enabled is True
    assert config.typed_decision.artifact_dir == "C:/models/decision-candidate"
    assert config.typed_decision.report_path.endswith("report.json")


@pytest.mark.parametrize("section", ["calender", "cognitiv", "prevew"])
def test_unknown_top_level_config_section_is_rejected(section):
    with pytest.raises(ValueError, match="Invalid config section"):
        _validate_config_types({section: {"enabled": True}})


def test_preview_typo_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="preview.enabld"):
        _validate_config_types({"preview": {"enabld": True}})


def test_numeric_config_does_not_accept_boolean_subclass():
    with pytest.raises(ValueError, match="calendar"):
        _validate_config_types({"calendar": {"ics_files": True}})
