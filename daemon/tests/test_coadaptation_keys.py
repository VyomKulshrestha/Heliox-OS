"""Key providers of the co-adaptation store (spec 10.6, tests of 22.2). The OS keyring is always monkeypatched."""

from __future__ import annotations

import base64

import keyring
import keyring.backends.fail
import pytest

from pilot.coadaptation.keys import (
    CoAdaptationKeyMissing,
    CoAdaptationKeyUnavailable,
    InMemoryKeyProvider,
    KeyProvider,
    OsKeyringKeyProvider,
)

PK = "p-3fa29c1e"


class _SecureBackend:
    priority = 1


class _ZeroPriorityBackend:
    priority = 0


@pytest.fixture
def fake_keyring(monkeypatch):
    stored: dict[tuple[str, str], str] = {}
    calls: list[str] = []

    def set_password(service, account, value):
        calls.append("set")
        stored[(service, account)] = value

    def delete_password(service, account):
        calls.append("delete")
        stored.pop((service, account))

    monkeypatch.setattr(keyring, "get_keyring", lambda: _SecureBackend())
    monkeypatch.setattr(keyring, "get_password", lambda service, account: stored.get((service, account)))
    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "delete_password", delete_password)
    return stored, calls


def test_fails_closed_on_the_fail_backend(monkeypatch):
    monkeypatch.setattr(keyring, "get_keyring", lambda: keyring.backends.fail.Keyring())
    provider = OsKeyringKeyProvider()
    assert provider.available() is False
    with pytest.raises(CoAdaptationKeyUnavailable):
        provider.get(PK)
    with pytest.raises(CoAdaptationKeyUnavailable):
        provider.get_or_create(PK)
    with pytest.raises(CoAdaptationKeyUnavailable):
        provider.delete(PK)


def test_fails_closed_on_a_zero_priority_backend(monkeypatch):
    monkeypatch.setattr(keyring, "get_keyring", lambda: _ZeroPriorityBackend())
    assert OsKeyringKeyProvider().available() is False
    with pytest.raises(CoAdaptationKeyUnavailable):
        OsKeyringKeyProvider().get_or_create(PK)


def test_fails_closed_when_the_backend_raises(monkeypatch):
    def broken():
        raise RuntimeError("no session bus")

    monkeypatch.setattr(keyring, "get_keyring", broken)
    assert OsKeyringKeyProvider().available() is False


def test_get_never_creates_a_key(fake_keyring):
    stored, calls = fake_keyring
    provider = OsKeyringKeyProvider()
    assert provider.available() is True
    with pytest.raises(CoAdaptationKeyMissing):
        provider.get(PK)
    assert stored == {}
    assert calls == []


def test_get_or_create_creates_once_under_the_participant_key(fake_keyring):
    stored, calls = fake_keyring
    provider = OsKeyringKeyProvider()
    first = provider.get_or_create(PK)
    second = provider.get_or_create(PK)
    assert first == second
    assert len(first) == 32
    assert calls == ["set"]
    assert list(stored) == [("heliox-co-adaptation", PK)]
    assert base64.b64decode(stored[("heliox-co-adaptation", PK)]) == first
    assert provider.get(PK) == first


def test_delete_reports_whether_a_key_existed(fake_keyring):
    stored, _calls = fake_keyring
    provider = OsKeyringKeyProvider()
    provider.get_or_create(PK)
    assert provider.delete(PK) is True
    assert stored == {}
    assert provider.delete(PK) is False
    with pytest.raises(CoAdaptationKeyMissing):
        provider.get(PK)


def test_invalid_stored_key_fails_closed(fake_keyring):
    stored, _calls = fake_keyring
    stored[("heliox-co-adaptation", PK)] = base64.b64encode(b"short").decode()
    with pytest.raises(CoAdaptationKeyUnavailable):
        OsKeyringKeyProvider().get(PK)
    stored[("heliox-co-adaptation", PK)] = "not base64 !"
    with pytest.raises(CoAdaptationKeyUnavailable):
        OsKeyringKeyProvider().get(PK)


@pytest.mark.parametrize("account", ["Lab A", "alice", "p-3FA29C1E", "p-3fa29c1e:stream"])
def test_account_names_are_participant_keys_only(fake_keyring, account):
    with pytest.raises(ValueError):
        OsKeyringKeyProvider().get_or_create(account)


def test_in_memory_provider_behaves_like_the_keyring():
    provider = InMemoryKeyProvider()
    assert isinstance(provider, KeyProvider)
    with pytest.raises(CoAdaptationKeyMissing):
        provider.get(PK)
    assert PK not in provider
    key = provider.get_or_create(PK)
    assert provider.get_or_create(PK) == key
    assert provider.get(PK) == key
    assert len(provider) == 1
    assert provider.delete(PK) is True
    assert provider.delete(PK) is False
    assert PK not in provider


def test_in_memory_provider_can_be_unavailable():
    provider = InMemoryKeyProvider(available=False)
    assert provider.available() is False
    with pytest.raises(CoAdaptationKeyUnavailable):
        provider.get_or_create(PK)
    with pytest.raises(CoAdaptationKeyUnavailable):
        provider.get(PK)
    provider.set_available(True)
    assert len(provider.get_or_create(PK)) == 32


def test_in_memory_provider_accepts_preloaded_keys():
    key = bytes(range(32))
    provider = InMemoryKeyProvider(keys={PK: key})
    assert provider.get(PK) == key
    with pytest.raises(ValueError):
        InMemoryKeyProvider(keys={PK: b"short"})
