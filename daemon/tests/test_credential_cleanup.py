"""Removing Heliox's secrets from the system credential store, with a fake keyring (no real store is touched)."""

from __future__ import annotations

import json
from pathlib import Path

from pilot.system import credential_cleanup as cc

ROOT = Path(__file__).resolve().parents[2]


class FakeKeyring:
    def __init__(self, entries: dict[tuple[str, str], str], locked: set[tuple[str, str]] = frozenset()) -> None:
        self.entries = dict(entries)
        self.locked = set(locked)

    def get_password(self, service, account):
        return self.entries.get((service, account))

    def delete_password(self, service, account):
        if (service, account) in self.locked:
            raise PermissionError("the store refused")
        del self.entries[(service, account)]


def test_the_services_come_from_the_modules_that_own_them_and_match_the_apps_list():
    from pilot.security.vault import VAULT_SERVICE
    from pilot.workflows.recorded_workflows import KEYRING_SERVICE

    names = cc.services()
    assert VAULT_SERVICE in names and KEYRING_SERVICE in names and len(names) == 4
    shipped = json.loads((ROOT / "tauri-app" / "src-tauri" / "setup" / "credential_services.json").read_text("utf-8"))
    assert shipped["services"] == names  # the desktop app's fallback list when the environment cannot run


def test_everything_the_store_shows_and_every_known_name_is_removed():
    store = FakeKeyring(
        {
            ("pilot-ai-command-center", "anthropic"): "sk-ant-x",
            ("pilot-ai-command-center", "ssh:prod:private-key"): "pem",
            ("pilot-ai-command-center", "air_handoff:phone"): "s",
            ("Heliox OS Recorded Workflows", "local-recordings-v1"): "k",
            ("someone-else", "anthropic"): "not Heliox's",
        }
    )

    def entries(service):  # what a search of the store by service returns
        return [account for (s, account) in store.entries if s == service and account != "anthropic"]

    result = cc.remove_all(
        "linux",
        keyring_module=store,
        entries=entries,
        names=["pilot-ai-command-center", "Heliox OS Recorded Workflows"],
        known={
            "pilot-ai-command-center": ["anthropic", "openai"],
            "Heliox OS Recorded Workflows": ["local-recordings-v1"],
        },
    )
    assert store.entries == {("someone-else", "anthropic"): "not Heliox's"}
    assert sorted(result.removed) == [
        "Heliox OS Recorded Workflows / local-recordings-v1",
        "pilot-ai-command-center / air_handoff:phone",
        "pilot-ai-command-center / anthropic",
        "pilot-ai-command-center / ssh:prod:private-key",
    ]
    assert result.remaining == []


def test_what_cannot_be_removed_is_listed_with_its_exact_name_and_how():
    store = FakeKeyring(
        {("pilot-ai-command-center", "anthropic"): "k"}, locked={("pilot-ai-command-center", "anthropic")}
    )
    result = cc.remove_all(
        "win32",
        keyring_module=store,
        entries=lambda s: [],
        names=["pilot-ai-command-center"],
        known={"pilot-ai-command-center": ["anthropic"]},
    )
    report = result.as_dict("win32")
    assert report["remaining"] == ["pilot-ai-command-center / anthropic"] and "Credential Manager" in report["how"]
    assert "Keychain Access" in cc.how_to_remove("darwin") and "Seahorse" in cc.how_to_remove("linux")


def test_a_store_that_cannot_be_searched_still_gets_the_known_names_removed():
    store = FakeKeyring({("pilot-ai-command-center", "openai"): "k"})

    def broken(service):
        raise RuntimeError("no Secret Service on this session")

    result = cc.remove_all(
        "linux",
        keyring_module=store,
        entries=broken,
        names=["pilot-ai-command-center"],
        known={"pilot-ai-command-center": ["openai"]},
    )
    assert store.entries == {} and result.removed == ["pilot-ai-command-center / openai"]
    assert result.errors and "cannot search" in result.errors[0]


def test_each_systems_search_reads_names_only():
    assert (
        cc.parse_security_output('    "acct"<blob>="anthropic"\n    "svce"<blob>="pilot-ai-command-center"')
        == "anthropic"
    )
    assert (
        cc.parse_security_output("security: SecKeychainSearchCopyNext: The specified item could not be found.") is None
    )
    creds = [
        {"TargetName": "pilot-ai-command-center", "UserName": "openai"},
        {"TargetName": "anthropic@pilot-ai-command-center", "UserName": "anthropic"},
        {"TargetName": "git:https://github.com", "UserName": "sam"},
    ]
    assert cc.windows_entries("pilot-ai-command-center", lambda: creds) == ["openai", "anthropic"]
    assert cc.linux_entries("svc", lambda attrs: [{"service": "svc", "username": "a"}, {"service": "svc"}]) == ["a"]
    assert cc.macos_entries("svc", lambda args: (0, '"acct"<blob>="b"')) == ["b"]
    assert cc.macos_entries("svc", lambda args: (44, "not found")) == []


def _platform_real(name: str):
    import importlib.util

    path = ROOT / "daemon" / "tests" / "platform_real" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"platform_real_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ci_checks_read_cmdkey_and_tell_heliox_leftovers_from_other_programs():
    store = _platform_real("credential_store_check")
    listing = (
        "Currently stored credentials:\n\n"
        "    Target: LegacyGeneric:target=pilot-ai-command-center\n    Type: Generic\n    User: heliox-ci-12\n\n"
        "    Target: WindowsLive:target=virtualapp/didlogical\n"
        "    Target: heliox-ci-12@heliox-co-adaptation\n"
    )
    assert store.windows_targets(listing) == {
        "pilot-ai-command-center",
        "virtualapp/didlogical",
        "heliox-ci-12@heliox-co-adaptation",
    }

    home = _platform_real("home_snapshot")
    sep = "/" if "/" in str(Path("a/b")) else "\\"
    before = {".cache", f".cache{sep}pip"}
    after = before | {
        f".cache{sep}uv",
        f".cache{sep}uv{sep}wheels",
        ".pilot",
        f".pilot{sep}other.conf",
        f".config{sep}heliox-os",
    }
    tops = home.new_tops(before, after)
    assert tops == [f".cache{sep}uv", f".config{sep}heliox-os", ".pilot"]
    assert [t for t in tops if home.HELIOX.search(sep + t)] == [f".config{sep}heliox-os", ".pilot"]


def test_an_empty_windows_store_is_an_empty_list_and_the_learner_key_is_known():
    class Win32Error(Exception):
        def __init__(self, code, function, message):
            super().__init__(code, function, message)
            self.winerror = code

    assert cc.empty_store_error(Win32Error(1168, "CredEnumerate", "Element not found"))
    assert cc.empty_store_error(Exception(1168, "CredEnumerate", "Element not found"))
    assert not cc.empty_store_error(Win32Error(5, "CredEnumerate", "Access is denied"))
    from pilot.security.vault import VAULT_SERVICE
    from pilot.server import LEARNER_SNAPSHOT_KEY

    assert LEARNER_SNAPSHOT_KEY in cc.known_accounts()[VAULT_SERVICE]
