"""Remove every secret Heliox put in the system credential store, for "Remove everything".

Heliox keeps secrets in the operating system's credential store through ``keyring`` (Windows Credential Manager,
the macOS Keychain, the Secret Service on Linux) under four services, each named by the module that owns it:

- ``pilot.security.vault.VAULT_SERVICE``: provider API keys, SSH keys and passphrases, paired devices, the receipt
  signing key, the mesh secret;
- ``pilot.coadaptation.keys`` (``OsKeyringKeyProvider.SERVICE``): per-participant study keys;
- ``pilot.neural.recording.NeuralRecordingKeyStore.SERVICE``: per-subject recording keys;
- ``pilot.workflows.recorded_workflows.KEYRING_SERVICE``: the recorded-workflows key.

Accounts under the vault are partly dynamic (a provider name, ``ssh:<host>:private-key``, ``air_handoff:<device>``),
so the store is searched by service where the system allows it (Windows: the credential list; macOS: the
``security`` tool; Linux: Secret Service attributes) and the accounts Heliox knows of are tried by name as well.
What could not be removed is reported with the exact service and account and how to remove it by hand.

The desktop app runs ``python -m pilot.system.credential_cleanup remove`` with the environment's Python before
it deletes the environment, with a time limit; it prints one JSON object. ``list`` prints the same without removing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


def services() -> list[str]:
    """The keyring services Heliox uses, from the modules that own them."""
    from pilot.coadaptation.keys import OsKeyringKeyProvider
    from pilot.neural.recording import NeuralRecordingKeyStore
    from pilot.security.vault import VAULT_SERVICE
    from pilot.workflows.recorded_workflows import KEYRING_SERVICE

    return [VAULT_SERVICE, OsKeyringKeyProvider.SERVICE, NeuralRecordingKeyStore.SERVICE, KEYRING_SERVICE]


def known_accounts() -> dict[str, list[str]]:
    """Accounts Heliox knows by name, per service (searching by service finds the rest where it can)."""
    from pilot.security.vault import KNOWN_PROVIDERS, VAULT_SERVICE
    from pilot.workflows.recorded_workflows import KEYRING_ACCOUNT, KEYRING_SERVICE
    from pilot.workflows.task_receipts import SIGNING_KEY_NAME

    vault = [*KNOWN_PROVIDERS, SIGNING_KEY_NAME]
    try:
        from pilot.config import PilotConfig

        config = PilotConfig.load()
        for host in getattr(getattr(config, "ssh", None), "allowed_hosts", []) or []:
            vault += [p for p in (host.private_key_provider, host.passphrase_provider) if p]
    except Exception:
        pass
    return {VAULT_SERVICE: sorted(set(vault)), KEYRING_SERVICE: [KEYRING_ACCOUNT]}


# ── finding entries by service, per system ──


def parse_security_output(text: str) -> str | None:
    """The account name in ``security find-generic-password`` output ('"acct"<blob>="openai"'), or None. Pure."""
    match = re.search(r'"acct"<blob>=(?:"(.*)"|0x[0-9A-Fa-f]+\s+"(.*)")', text)
    if not match:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def windows_entries(service: str, enumerate_credentials: Callable[[], list[dict]] | None = None) -> list[str]:
    """Accounts under ``service`` in Windows Credential Manager (keyring names a target "service" or
    "account@service")."""
    if enumerate_credentials is None:
        from win32ctypes.pywin32 import win32cred  # keyring's own dependency on Windows

        def enumerate_credentials() -> list[dict]:
            return list(win32cred.CredEnumerate(None, 0) or [])

    accounts = []
    for credential in enumerate_credentials():
        target = str(credential.get("TargetName") or "")
        if target == service or target.endswith("@" + service):
            accounts.append(str(credential.get("UserName") or target.split("@", 1)[0]))
    return accounts


def macos_entries(
    service: str, run: Callable[[list[str]], tuple[int, str]] | None = None, limit: int = 200
) -> list[str]:
    """Accounts under ``service`` in the login keychain, read one by one with ``security`` (attributes only)."""

    def default_run(args: list[str]) -> tuple[int, str]:
        done = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        return done.returncode, done.stdout + done.stderr

    run = run or default_run
    code, out = run(["/usr/bin/security", "find-generic-password", "-s", service])
    account = parse_security_output(out) if code == 0 else None
    return [account] if account is not None else []


def linux_entries(service: str, search: Callable[[dict], list[dict]] | None = None) -> list[str]:
    """Accounts under ``service`` in the Secret Service (keyring stores attributes "service" and "username")."""
    if search is None:
        import secretstorage

        def search(attributes: dict) -> list[dict]:
            connection = secretstorage.dbus_init()
            collection = secretstorage.get_default_collection(connection)
            return [item.get_attributes() for item in collection.search_items(attributes)]

    return [str(attrs.get("username", "")) for attrs in search({"service": service}) if attrs.get("username")]


@dataclass
class Cleanup:
    removed: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self, platform: str) -> dict:
        return {
            "removed": self.removed,
            "remaining": self.remaining,
            "errors": self.errors,
            "how": how_to_remove(platform),
        }


def how_to_remove(platform: str) -> str:
    if platform == "win32":
        return "Open Credential Manager > Windows Credentials and remove the entries named above (service@account)."
    if platform == "darwin":
        return "Open Keychain Access, search for each service named above in the login keychain and delete it."
    return "Open Passwords and Keys (Seahorse) or your desktop's wallet, search for each service named above and delete it."


def remove_all(
    platform: str = sys.platform,
    *,
    keyring_module=None,
    entries: Callable[[str], list[str]] | None = None,
    names: Iterable[str] | None = None,
    known: dict[str, list[str]] | None = None,
    rounds: int = 50,
) -> Cleanup:
    """Delete every Heliox entry the store shows; list what is left with its exact service and account."""
    if keyring_module is None:
        import keyring as keyring_module
    if entries is None:
        finder = {"win32": windows_entries, "darwin": macos_entries}.get(platform, linux_entries)

        def entries(service: str) -> list[str]:
            return finder(service)

    result = Cleanup()
    known = known if known is not None else known_accounts()
    for service in names if names is not None else services():
        for _ in range(rounds):
            try:
                found = entries(service)
            except Exception as error:  # the store cannot be searched here: fall back to names
                result.errors.append(f"{service}: cannot search ({type(error).__name__})")
                found = []
            candidates = [a for a in dict.fromkeys([*found, *known.get(service, [])]) if a]
            deleted = False
            for account in candidates:
                try:
                    if keyring_module.get_password(service, account) is None:
                        continue
                    keyring_module.delete_password(service, account)
                    result.removed.append(f"{service} / {account}")
                    deleted = True
                except Exception as error:
                    label = f"{service} / {account}"
                    if label not in result.remaining:
                        result.remaining.append(label)
                        result.errors.append(f"{label}: {type(error).__name__}")
            if not deleted:
                break
        try:
            for account in entries(service):
                label = f"{service} / {account}"
                if label not in result.remaining:
                    result.remaining.append(label)
        except Exception:
            pass
    return result


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "list"
    if mode == "remove":
        print(json.dumps(remove_all().as_dict(sys.platform)))
        return 0
    if mode == "services":
        print(json.dumps(services()))
        return 0
    found: dict[str, list[str]] = {}
    finder = {"win32": windows_entries, "darwin": macos_entries}.get(sys.platform, linux_entries)
    for service in services():
        try:
            found[service] = finder(service)
        except Exception as error:
            found[service] = [f"(cannot search: {type(error).__name__})"]
    print(json.dumps(found))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
