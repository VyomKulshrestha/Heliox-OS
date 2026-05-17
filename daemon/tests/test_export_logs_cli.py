import json
import zipfile

from pilot import server


def test_export_logs_archive_includes_logs_config_and_audit(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    export_dir = tmp_path / "exports"

    (state_dir / "nested").mkdir(parents=True)
    data_dir.mkdir()
    config_dir.mkdir()
    (state_dir / "pilot.log").write_text("daemon log\n", encoding="utf-8")
    (state_dir / "nested" / "trace.log").write_text("trace\n", encoding="utf-8")
    config_file = config_dir / "config.toml"
    audit_file = data_dir / "audit.jsonl"
    permission_audit = data_dir / "permission_audit.db"
    config_file.write_text("server = {}\n", encoding="utf-8")
    audit_file.write_text('{"event":"start"}\n', encoding="utf-8")
    permission_audit.write_bytes(b"sqlite")

    monkeypatch.setattr(server, "STATE_DIR", state_dir)
    monkeypatch.setattr(server, "CONFIG_FILE", config_file)
    monkeypatch.setattr(server, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(server, "PERMISSION_AUDIT_DB_FILE", permission_audit)

    archive_path = server.export_logs_archive(export_dir)

    assert archive_path.parent == export_dir
    assert archive_path.name.startswith("heliox-diagnostics-")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert {
            "logs/pilot.log",
            "logs/nested/trace.log",
            "config/config.toml",
            "audit/audit.jsonl",
            "audit/permission_audit.db",
            "manifest.json",
        }.issubset(names)
        manifest = json.loads(archive.read("manifest.json"))

    assert "logs/pilot.log" in manifest["included_files"]
    assert manifest["source_paths"]["state_dir"] == str(state_dir)


def test_export_logs_archive_succeeds_when_optional_files_are_missing(monkeypatch, tmp_path):
    state_dir = tmp_path / "missing-state"
    export_dir = tmp_path / "exports"
    monkeypatch.setattr(server, "STATE_DIR", state_dir)
    monkeypatch.setattr(server, "CONFIG_FILE", tmp_path / "missing.toml")
    monkeypatch.setattr(server, "AUDIT_FILE", tmp_path / "missing-audit.jsonl")
    monkeypatch.setattr(server, "PERMISSION_AUDIT_DB_FILE", tmp_path / "missing-audit.db")

    archive_path = server.export_logs_archive(export_dir)

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["manifest.json"]
        manifest = json.loads(archive.read("manifest.json"))

    assert manifest["included_files"] == []
