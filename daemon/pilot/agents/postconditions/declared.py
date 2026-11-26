"""Declared post-conditions: the ``expect`` probe of actions whose effect is arbitrary.

A shell command, a script on another machine, a page script, an MCP or plugin tool can do anything, so no fixed
verifier can say whether it did what was meant. A plan may instead declare, on the action, a read-only probe and what
it must find (``Action.expect``, see ``pilot.actions.Expectation``). After the action succeeded this module runs the
probe once and compares.

Read-only is decided by construction, never trusted: ``probe_refusal`` accepts a probe only when its type is allowed
for the action it checks, its permission tier is not above the action's, it is not elevated, it passes the same
``ActionValidator`` rules as any action, and a shell or ssh probe is one allow-listed command with no shell syntax at
all. A refused probe never runs and never blocks the action: the step stays "reported", with the reason.

No ``expect`` keeps today's status: the step is reported, not verified.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from pilot.actions import (
    READ_ONLY_HTTP_METHODS,
    Action,
    ActionResult,
    ActionType,
    Expectation,
)
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified
from pilot.agents.unconfirmed import Kind
from pilot.timeouts import bounded

logger = logging.getLogger("pilot.agents.postconditions.declared")

METHOD = "declared_expect_postcondition"
NO_EXPECT_DETAIL = "No independent postcondition for this action type"
CHECK_TIMEOUT_SECONDS = 30.0
LOCAL_PROBE_SECONDS = 15
REMOTE_PROBE_SECONDS = 25
OUTPUT_LIMIT = 64 * 1024
FOUND_EXCERPT = 120

# The fifteen actions whose effect is arbitrary and that a declared probe can check. api_webhook is the sixteenth:
# a webhook has no read-back, so it stays a receipt and is not registered here.
DECLARED_ACTIONS = (
    ActionType.SSH_COMMAND,
    ActionType.SSH_SCRIPT,
    ActionType.BROWSER_EXECUTE_JS,
    ActionType.MCP_CALL_TOOL,
    ActionType.PLUGIN_CALL,
    ActionType.API_REQUEST,
    ActionType.API_GITHUB,
    ActionType.SHELL_COMMAND,
    ActionType.SHELL_SCRIPT,
    ActionType.CODE_EXECUTE,
    ActionType.CODE_GENERATE_AND_RUN,
    ActionType.PTY_EXEC,
    ActionType.DBUS_CALL,
    ActionType.SKILL_RUN,
    ActionType.WASM_CALL,
)
NO_READ_BACK = frozenset({ActionType.API_WEBHOOK})
ALL_ARBITRARY_EFFECT_ACTIONS = frozenset(DECLARED_ACTIONS) | NO_READ_BACK

_LOCAL_EFFECT = frozenset(
    {
        ActionType.SHELL_COMMAND,
        ActionType.SHELL_SCRIPT,
        ActionType.PTY_EXEC,
        ActionType.CODE_EXECUTE,
        ActionType.CODE_GENERATE_AND_RUN,
        ActionType.SKILL_RUN,
        ActionType.WASM_CALL,
        ActionType.PLUGIN_CALL,
    }
)
_LOCAL_READS = frozenset({ActionType.FILE_READ, ActionType.FILE_HASH, ActionType.FILE_LIST, ActionType.REGISTRY_READ})
_BROWSER_READS = frozenset({ActionType.BROWSER_PAGE_INFO, ActionType.BROWSER_EXTRACT})
_BROWSER_ATTRIBUTES = frozenset({"innerText", "value", "href", "src", "title"})
_PROBE_HEADERS = frozenset(
    {"authorization", "accept", "accept-language", "user-agent", "x-api-key", "x-github-api-version"}
)
_DBUS_PROPERTIES = "org.freedesktop.DBus.Properties"

# ---------------------------------------------------------------------------------------------------------------
# The command allow-list for shell and ssh probes.
#
# A probe command is split on spaces. Every word may hold only these characters, so no word can be shell syntax:
# no ; & | < > ` $ ( ) { } [ ] * ? ' " \ ! # or a newline. Locally the words go to run_command (no shell); on a
# remote host the remote shell reads them, which is why the set leaves out everything a shell would act on.
# ``~`` only expands to a home folder and ``%`` means nothing to a shell; both are kept for paths and git formats.
# ---------------------------------------------------------------------------------------------------------------
_WORD = re.compile(r"^[A-Za-z0-9_./:@%+=,~-]{1,1024}$")
_NUMBER = re.compile(r"^[0-9]{1,5}$")
MAX_COMMAND_WORDS = 8

PATH, NUM, NAME = "<path>", "<n>", "<name>"
_GIT_FORMATS = frozenset({"--format=%H", "--format=%s", "--format=%an", "--format=%ad"})
_SYSTEMCTL_QUERIES = frozenset({"is-active", "is-enabled", "is-failed"})

# Each form is the exact sequence of words: a literal, a set of literals, or a placeholder.
_FORMS: tuple[tuple[Any, ...], ...] = (
    ("test", frozenset({"-e", "-f", "-d", "-s", "-r", "-x", "-L"}), PATH),
    ("stat", PATH),
    ("cat", PATH),
    ("head", "-n", NUM, PATH),
    ("tail", "-n", NUM, PATH),
    ("wc", frozenset({"-l", "-c"}), PATH),
    ("sha256sum", PATH),
    ("md5sum", PATH),
    ("shasum", "-a", "256", PATH),
    ("ls", PATH),
    ("ls", frozenset({"-l", "-a", "-la", "-al", "-ld", "-1"}), PATH),
    ("readlink", PATH),
    ("git", "rev-parse", NAME),
    ("git", "-C", PATH, "rev-parse", NAME),
    ("git", "log", "-1", _GIT_FORMATS),
    ("git", "-C", PATH, "log", "-1", _GIT_FORMATS),
    ("systemctl", _SYSTEMCTL_QUERIES, NAME),
    ("systemctl", "--user", _SYSTEMCTL_QUERIES, NAME),
    ("pgrep", "-x", NAME),
    ("pidof", NAME),
    ("id", "-u"),
    ("whoami",),
    ("hostname",),
    ("uname",),
    ("uname", frozenset({"-a", "-r", "-s", "-m"})),
    ("getent", frozenset({"passwd", "group"}), NAME),
    ("grep", "-F", frozenset({"-q", "-c"}), NAME, PATH),
)
ALLOWED_COMMANDS = frozenset(form[0] for form in _FORMS)


def command_words(command: str, args: list[str] | None = None) -> list[str]:
    """The probe's words: ``command`` split on spaces when no separate ``args`` are given."""
    words = [command, *args] if args else command.split(" ")
    return [word for word in words if word != ""]


def command_refusal(words: list[str]) -> str | None:
    """Why ``words`` is not an allow-listed read-only command, or None when it is."""
    if not words:
        return "the probe has no command"
    if len(words) > MAX_COMMAND_WORDS:
        return f"the probe command has more than {MAX_COMMAND_WORDS} words"
    for word in words:
        if not _WORD.match(word):
            bad = sorted({char for char in word if not re.match(r"[A-Za-z0-9_./:@%+=,~-]", char)})
            shown = " ".join(repr(char) for char in bad[:4]) or "an empty word"
            return f"the probe command holds characters a check may not use ({shown})"
    if words[0] not in ALLOWED_COMMANDS:
        return f"'{words[0]}' is not one of the read-only commands a check may run"
    for form in _FORMS:
        if form[0] == words[0] and len(form) == len(words) and all(map(_word_fits, form, words)):
            return None
    flags = [word for word in words[1:] if word.startswith("-")]
    if flags:
        return f"'{words[0]}' may not be run with {' '.join(flags)} in a check"
    return f"'{' '.join(words)}' is not in a form a check may run"


def _word_fits(slot: Any, word: str) -> bool:
    if isinstance(slot, frozenset):
        return word in slot
    if slot == NUM:
        return bool(_NUMBER.match(word))
    if slot in (PATH, NAME):
        return not word.startswith("-")
    return word == slot


def command_paths(words: list[str]) -> list[str]:
    """The words of an allow-listed command that name a path (checked against the file rules for local probes)."""
    for form in _FORMS:
        if form[0] == words[0] and len(form) == len(words) and all(map(_word_fits, form, words)):
            return [word for slot, word in zip(form, words, strict=True) if slot == PATH]
    return []


# ---------------------------------------------------------------------------------------------------------------
# Context: the model router gives the person's configuration (for ActionValidator) and the vault (for ssh).
# ---------------------------------------------------------------------------------------------------------------
_router: Any = None


def bind_router(router: Any) -> None:
    """Called by ``Verifier.__init__``: the configuration and vault the actions themselves use."""
    global _router
    _router = router


def _config() -> Any | None:
    if _router is None:
        return None
    try:
        config = _router.get_config()
    except Exception:  # noqa: BLE001
        return None
    from pilot.config import PilotConfig

    return config if isinstance(config, PilotConfig) else None


# ---------------------------------------------------------------------------------------------------------------
# Which probe may check which action.
# ---------------------------------------------------------------------------------------------------------------
def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def probe_refusal(action: Action) -> str | None:
    """Why ``action.expect``'s probe may not run, or None when it may. Deterministic; reads no state.

    The validator rules that need the person's configuration are checked separately by ``validator_refusal``.
    """
    expect = action.expect
    if expect is None:
        return "no check was declared"
    probe = expect.probe
    kind = action.action_type
    if kind in NO_READ_BACK:
        return "a webhook has no read-back, so only the server's receipt is kept"
    if kind not in DECLARED_ACTIONS:
        return f"{kind.value} does not take a declared check"
    if kind in {ActionType.API_REQUEST, ActionType.API_GITHUB}:
        method = str(getattr(action.parameters, "method", "GET")).upper()
        if method in READ_ONLY_HTTP_METHODS:
            return "a read changes nothing, so there is nothing to check afterwards"
    if probe.requires_root or probe.destructive or bool(getattr(probe.parameters, "elevated", False)):
        return "a check may not run elevated or be marked destructive"
    if probe.use_previous_output:
        return "a check may not take another step's output"

    refusal = _type_refusal(action, probe)
    if refusal:
        return refusal
    if expect.kind == "exit_code" and probe.action_type not in {ActionType.SHELL_COMMAND, ActionType.SSH_COMMAND}:
        return "an exit code can only be expected from a shell or ssh check"
    try:
        probe_tier, action_tier = probe.permission_tier, action.permission_tier
    except Exception as exc:  # noqa: BLE001
        return f"the check's permission tier could not be decided: {exc}"
    if probe_tier > action_tier:
        return (
            f"the check ({probe.action_type.value}, tier {probe_tier.name}) needs more permission "
            f"than the step it checks (tier {action_tier.name})"
        )
    return None


def _type_refusal(action: Action, probe: Action) -> str | None:
    kind, ptype = action.action_type, probe.action_type
    p = probe.parameters

    if kind in {ActionType.SSH_COMMAND, ActionType.SSH_SCRIPT}:
        if ptype != ActionType.SSH_COMMAND:
            return "a step on another machine can only be checked with an ssh_command on the same host"
        if str(getattr(p, "host", "")) != str(getattr(action.parameters, "host", "")):
            return "a check may only connect to the host the step ran on"
        return command_refusal(command_words(str(getattr(p, "command", ""))))

    if kind == ActionType.BROWSER_EXECUTE_JS:
        if ptype not in _BROWSER_READS:
            return "a page script can only be checked by reading the page (browser_page_info or browser_extract)"
        if ptype == ActionType.BROWSER_EXTRACT:
            if not str(getattr(p, "selector", "")).strip() or len(str(getattr(p, "selector", ""))) > 300:
                return "browser_extract needs a selector of at most 300 characters"
            if bool(getattr(p, "multiple", False)):
                return "a page check reads one element"
            if str(getattr(p, "attribute", "innerText")) not in _BROWSER_ATTRIBUTES:
                return "a page check reads innerText, value, href, src or title"
        return None

    if kind in {ActionType.API_REQUEST, ActionType.API_GITHUB}:
        if ptype != kind:
            return f"an {kind.value} can only be checked with another {kind.value} that reads"
        if str(getattr(p, "method", "GET")).upper() not in {"GET", "HEAD"}:
            return "an HTTP check may only use GET or HEAD"
        if getattr(p, "body", None):
            return "an HTTP check may not send a body"
        headers = {str(name).lower() for name in (getattr(p, "headers", None) or {})}
        if headers - _PROBE_HEADERS:
            return f"an HTTP check may not send the header {sorted(headers - _PROBE_HEADERS)[0]}"
        if bool(getattr(p, "allow_private_network", False)) and not bool(
            getattr(action.parameters, "allow_private_network", False)
        ):
            return "an HTTP check may not reach a private network the step did not"
        if kind == ActionType.API_REQUEST:
            probe_host = _host(str(getattr(p, "url", "")))
            if not probe_host or probe_host != _host(str(getattr(action.parameters, "url", ""))):
                return "an HTTP check may only read from the host the step sent to"
        elif not str(getattr(p, "endpoint", "")).startswith("/"):
            return "a GitHub check reads a path on api.github.com, such as /repos/owner/name"
        return None

    if kind == ActionType.MCP_CALL_TOOL:
        if ptype != ActionType.MCP_CALL_TOOL:
            return "an MCP tool can only be checked with a read-only tool of the same server"
        if str(getattr(p, "mcp_server", "")).strip() != str(getattr(action.parameters, "mcp_server", "")).strip():
            return "an MCP check may only use the server the step used"
        from pilot.actions import PermissionTier
        from pilot.mcp_client import tool_permission_tier

        if tool_permission_tier(str(p.mcp_server), str(p.mcp_tool)) != PermissionTier.READ_ONLY:
            return f"the MCP tool {p.mcp_tool} does not declare itself read-only"
        return None

    if kind == ActionType.DBUS_CALL:
        if ptype != ActionType.DBUS_CALL:
            return "a DBus call can only be checked by reading a property of the same service"
        a = action.parameters
        if str(getattr(p, "bus", "")) != str(getattr(a, "bus", "")) or str(getattr(p, "service", "")) != str(
            getattr(a, "service", "")
        ):
            return "a DBus check may only read from the bus and service the step called"
        if str(getattr(p, "interface", "")) != _DBUS_PROPERTIES or str(getattr(p, "method", "")) not in {
            "Get",
            "GetAll",
        }:
            return "a DBus check may only call org.freedesktop.DBus.Properties Get or GetAll"
        return None

    if kind in _LOCAL_EFFECT:
        if ptype in _LOCAL_READS:
            return None
        if ptype == ActionType.SHELL_COMMAND:
            if getattr(p, "working_directory", None):
                return "a shell check runs without a working folder; give full paths"
            return command_refusal(command_words(str(getattr(p, "command", "")), list(getattr(p, "args", []) or [])))
        return (
            "a step on this computer can only be checked with file_read, file_hash, file_list, registry_read "
            "or an allow-listed shell_command"
        )
    return f"{kind.value} does not take a declared check"


def validator_refusal(action: Action) -> str | None:
    """The same validator rules every action meets (protected and private folders, blocked commands, root)."""
    config = _config()
    if config is None:
        return None
    from pilot.actions import FileParams
    from pilot.security.validator import ActionValidator

    expect = action.expect
    if expect is None:
        return None
    probe = expect.probe.model_copy(deep=True)
    if not probe.target.strip():
        # The validator wants a target for file reads, and checks it as a path; the planner fills it the same way.
        probe.target = str(getattr(probe.parameters, "path", "") or "check")
    candidates = [probe]
    if probe.action_type == ActionType.SHELL_COMMAND:
        # Validated as the words that will run: the command name, then its arguments, then each path as a file_read.
        words = command_words(str(probe.parameters.command), list(getattr(probe.parameters, "args", []) or []))
        probe.parameters = probe.parameters.model_copy(update={"command": words[0] if words else "", "args": words[1:]})
        candidates += [
            Action(action_type=ActionType.FILE_READ, target=path, parameters=FileParams(path=path))
            for path in command_paths(words)
        ]
    validator = ActionValidator(config)
    for candidate in candidates:
        try:
            validator.validate_action(candidate, 0)
        except Exception as exc:  # noqa: BLE001
            return f"the same rules as any action refuse it: {exc}"
    return None


# ---------------------------------------------------------------------------------------------------------------
# Plain words for the plan card and the receipt.
# ---------------------------------------------------------------------------------------------------------------
def describe_probe(probe: Action) -> str:
    p = probe.parameters
    t = probe.action_type
    if t == ActionType.SSH_COMMAND:
        return f"runs `{getattr(p, 'command', '')}` on {getattr(p, 'host', '')}"
    if t == ActionType.SHELL_COMMAND:
        words = command_words(str(getattr(p, "command", "")), list(getattr(p, "args", []) or []))
        return f"runs `{' '.join(words)}` on this computer"
    if t == ActionType.FILE_READ:
        return f"reads {getattr(p, 'path', '')}"
    if t == ActionType.FILE_HASH:
        return f"hashes {getattr(p, 'path', '')} ({getattr(p, 'hash_algorithm', 'sha256')})"
    if t == ActionType.FILE_LIST:
        return f"lists {getattr(p, 'path', '')}"
    if t == ActionType.REGISTRY_READ:
        return f"reads the registry value {getattr(p, 'key_path', '')} {getattr(p, 'value_name', '')}".rstrip()
    if t == ActionType.BROWSER_PAGE_INFO:
        return "reads the open page's address and title"
    if t == ActionType.BROWSER_EXTRACT:
        return f"reads {getattr(p, 'attribute', 'innerText')} of `{getattr(p, 'selector', '')}` on the open page"
    if t in {ActionType.API_REQUEST, ActionType.API_GITHUB}:
        where = _host(str(getattr(p, "url", ""))) if t == ActionType.API_REQUEST else "api.github.com"
        path = "" if t == ActionType.API_REQUEST else str(getattr(p, "endpoint", ""))
        return f"sends {str(getattr(p, 'method', 'GET')).upper()} to {where}{path}"
    if t == ActionType.MCP_CALL_TOOL:
        return f"calls the read-only tool {getattr(p, 'mcp_server', '')}.{getattr(p, 'mcp_tool', '')}"
    if t == ActionType.DBUS_CALL:
        return f"reads a property of {getattr(p, 'service', '')} over DBus"
    return f"runs {t.value}"


def describe_expectation(expect: Expectation) -> str:
    kind, value = expect.kind, expect.value
    if kind == "exit_code":
        return "expects it to succeed" if value == 0 else f"expects exit code {value}"
    if kind == "equals":
        return f"expects exactly `{value}`"
    if kind == "contains":
        return f"expects it to contain `{value}`"
    if kind == "pattern":
        return f"expects it to match `{value}`"
    if kind == "json_equals":
        return f"expects {expect.field} to be `{value}`"
    if kind == "exists":
        return "expects it to be there"
    return "expects it to be gone"


def describe_check(action: Action) -> str | None:
    """The plan card's line for a step of the sixteen types: the check that will be made, or why none will be.

    None for any other action type, and for a step that is neither irreversible nor carries an ``expect``
    (it says nothing new: such steps are reported, as today).
    """
    if action.action_type not in ALL_ARBITRARY_EFFECT_ACTIONS:
        return None
    if action.expect is None:
        try:
            irreversible = action.is_irreversible
        except Exception:  # noqa: BLE001
            irreversible = False
        if not irreversible:
            return None
        if action.action_type in NO_READ_BACK:
            return "None: a webhook has no read-back, so Heliox will only keep the server's receipt."
        return "None declared: Heliox will report what the step said, without checking it."
    refusal = probe_refusal(action) or validator_refusal(action)
    if refusal:
        return f"None: the check was refused because {refusal}."
    probe = describe_probe(action.expect.probe)
    return f"{probe[:1].upper()}{probe[1:]} and {describe_expectation(action.expect)}."


def plan_expect_notes(actions: list[Action]) -> list[tuple[int, str]]:
    """``(step index, line)`` for every step of the sixteen types that has something to say in the plan card."""
    notes = []
    for index, action in enumerate(actions):
        line = describe_check(action)
        if line is not None:
            notes.append((index, line))
    return notes


# ---------------------------------------------------------------------------------------------------------------
# Running a probe.
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reading:
    """What a probe saw. ``present`` is None when the probe cannot tell; ``unavailable`` says why it did not run."""

    output: str = ""
    exit_code: int | None = None
    present: bool | None = None
    unavailable: str = ""


def _absent_exit(code: int) -> bool | None:
    if code == 0:
        return True
    if 1 <= code <= 3:
        return False
    return None


async def _run_local_shell(words: list[str]) -> Reading:
    from pilot.system.platform_detect import run_command

    code, out, err = await run_command(words, timeout=LOCAL_PROBE_SECONDS)
    if code == -1 and (err.startswith("Command not found") or err.startswith("Command error")):
        return Reading(unavailable=f"'{words[0]}' is not available on this computer")
    if code == -1 and err == "Command timed out":
        return Reading(unavailable=f"the check did not finish within {LOCAL_PROBE_SECONDS} s")
    return Reading(output=out[:OUTPUT_LIMIT], exit_code=code, present=_absent_exit(code))


async def _run_ssh(host: str, words: list[str]) -> Reading:
    from pilot.agents.ssh_agent import SshAgent

    if _router is None:
        return Reading(unavailable="SSH settings are not available to the verifier")
    try:
        code, out, _err = await bounded(
            SshAgent(_router).run_probe(host, " ".join(words), timeout_seconds=REMOTE_PROBE_SECONDS - 5),
            REMOTE_PROBE_SECONDS,
        )
    except TimeoutError:
        return Reading(unavailable=f"the check on {host} did not finish within {REMOTE_PROBE_SECONDS} s")
    except Exception as exc:  # noqa: BLE001
        return Reading(unavailable=f"could not reach {host} for the check: {_short(str(exc))}")
    return Reading(output=out[:OUTPUT_LIMIT], exit_code=code, present=_absent_exit(code))


def _read_file(probe: Action) -> Reading:
    import hashlib
    from pathlib import Path

    p = probe.parameters
    path = Path(os.path.expanduser(str(getattr(p, "path", ""))))
    try:
        if probe.action_type == ActionType.FILE_LIST:
            names = sorted(entry.name for entry in path.iterdir())
            return Reading(output="\n".join(names[:500]), present=True)
        if probe.action_type == ActionType.FILE_HASH:
            digest = hashlib.new(str(getattr(p, "hash_algorithm", "sha256")))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return Reading(output=digest.hexdigest(), present=True)
        with path.open("rb") as handle:
            data = handle.read(OUTPUT_LIMIT)
        return Reading(output=data.decode("utf-8", errors="replace"), present=True)
    except FileNotFoundError:
        return Reading(present=False)
    except NotADirectoryError:
        return Reading(present=False)
    except OSError as exc:
        return Reading(unavailable=f"could not read {path}: {exc.strerror or exc}")


async def _read_registry(probe: Action) -> Reading:
    from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell

    if CURRENT_PLATFORM != Platform.WINDOWS:
        return Reading(unavailable="registry checks are only possible on Windows")
    p = probe.parameters
    script = "Get-ItemPropertyValue -LiteralPath ('Registry::' + $HelioxArgs['key']) -Name $HelioxArgs['name']"
    code, out, err = await run_powershell(
        script,
        timeout=LOCAL_PROBE_SECONDS,
        values={"key": str(getattr(p, "key_path", "")), "name": str(getattr(p, "value_name", ""))},
    )
    if code == 0:
        return Reading(output=out.strip(), present=True)
    if "does not exist" in err or "Cannot find path" in err:
        return Reading(present=False)
    return Reading(unavailable=f"the registry could not be read: {_short(err.strip())}")


async def _read_browser(probe: Action) -> Reading:
    from pilot.system import browser

    if not browser.has_active_session():
        return Reading(unavailable="no browser page is open to read")
    if probe.action_type == ActionType.BROWSER_PAGE_INFO:
        return Reading(output=await browser.browser_get_page_info(), present=True)
    p = probe.parameters
    text = await browser.browser_extract(str(p.selector), str(p.attribute), False)
    if text.startswith("Element not found:"):
        return Reading(present=False)
    return Reading(output=text[:OUTPUT_LIMIT], present=True)


async def _read_http(probe: Action) -> Reading:
    from pilot.config import PilotConfig
    from pilot.system.api_client import _request_with_validated_redirects
    from pilot.system.http_client import create_httpx_client

    p = probe.parameters
    method = str(getattr(p, "method", "GET")).upper()
    headers = dict(getattr(p, "headers", None) or {})
    if probe.action_type == ActionType.API_GITHUB:
        url = f"https://api.github.com{p.endpoint}"
        headers.setdefault("Accept", "application/vnd.github+json")
        from pilot.system import connections

        try:
            token = connections.choose(
                getattr(p, "token", None),
                (await connections.service("github")).get(connections.GITHUB_TOKEN),
                service_name="github",
                what="GitHub token",
            )
        except ValueError:
            token = ""
        if token and not any(name.lower() == "authorization" for name in headers):
            headers["Authorization"] = f"Bearer {token}"
        private = False
    else:
        url = str(p.url)
        private = bool(getattr(p, "allow_private_network", False))
    config = _config() or PilotConfig.load()
    async with create_httpx_client(config, timeout=REMOTE_PROBE_SECONDS - 5, follow_redirects=False, verify=True) as c:
        response = await _request_with_validated_redirects(
            c,
            method,
            url,
            allow_private_network=private,
            request_kwargs={"params": dict(getattr(p, "params", None) or {}), "headers": headers},
        )
    status = int(response.status_code)
    body = "" if method == "HEAD" else response.text[:OUTPUT_LIMIT]
    if 200 <= status < 300:
        return Reading(output=body, present=True)
    if status in {404, 410}:
        return Reading(output=body, present=False)
    return Reading(unavailable=f"the check's request answered HTTP {status}")


async def _read_mcp(probe: Action) -> Reading:
    from pilot.mcp_client import default_manager

    p = probe.parameters
    text = await default_manager().call(str(p.mcp_server).strip(), str(p.mcp_tool).strip(), dict(p.mcp_arguments))
    return Reading(output=str(text)[:OUTPUT_LIMIT], present=True)


async def _read_dbus(probe: Action) -> Reading:
    from pilot.system.dbus_client import call_dbus_method

    p = probe.parameters
    text = await call_dbus_method(p.bus, p.service, p.object_path, p.interface, p.method, list(p.args) or None)
    return Reading(output=str(text)[:OUTPUT_LIMIT], present=True)


async def run_probe(probe: Action) -> Reading:
    """Run an accepted probe. Only called after ``probe_refusal`` and ``validator_refusal`` returned None."""
    t = probe.action_type
    p = probe.parameters
    if t == ActionType.SSH_COMMAND:
        return await _run_ssh(str(p.host), command_words(str(p.command)))
    if t == ActionType.SHELL_COMMAND:
        return await _run_local_shell(command_words(str(p.command), list(p.args or [])))
    if t in {ActionType.FILE_READ, ActionType.FILE_HASH, ActionType.FILE_LIST}:
        return await bounded(asyncio.to_thread(_read_file, probe), LOCAL_PROBE_SECONDS)
    if t == ActionType.REGISTRY_READ:
        return await _read_registry(probe)
    if t in _BROWSER_READS:
        return await bounded(_read_browser(probe), REMOTE_PROBE_SECONDS)
    if t in {ActionType.API_REQUEST, ActionType.API_GITHUB}:
        return await bounded(_read_http(probe), REMOTE_PROBE_SECONDS)
    if t == ActionType.MCP_CALL_TOOL:
        return await bounded(_read_mcp(probe), REMOTE_PROBE_SECONDS)
    if t == ActionType.DBUS_CALL:
        return await bounded(_read_dbus(probe), LOCAL_PROBE_SECONDS)
    return Reading(unavailable=f"{t.value} cannot be used as a check")


# ---------------------------------------------------------------------------------------------------------------
# Comparing.
# ---------------------------------------------------------------------------------------------------------------
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]+-----.*?(?:-----END [A-Z ]+-----|$)", re.DOTALL),
    re.compile(r"\b(?:sk|gh[opsu]|github_pat)_[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)\b\s*[:=]\s*[\"']?[^\s\"',;]{6,}"
    ),
    # A long token-like run, but not a plain hex digest (a file hash or a commit id is what checks compare).
    re.compile(r"(?=[A-Za-z0-9+/=_-]*[G-Zg-z+/=_])[A-Za-z0-9+/=_-]{40,}"),
)


def _short(text: str) -> str:
    """At most FOUND_EXCERPT characters of ``text``, on one line, with secrets masked."""
    masked = text
    for pattern in _SECRET_PATTERNS:
        masked = pattern.sub("[REDACTED]", masked)
    masked = " ".join(masked.split())
    return masked if len(masked) <= FOUND_EXCERPT else masked[: FOUND_EXCERPT - 3].rstrip() + "..."


def _json_field(text: str, field: str) -> tuple[bool, Any]:
    try:
        data: Any = json.loads(text)
    except (TypeError, ValueError):
        return False, None
    for part in field.split("."):
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            return False, None
    return True, data


def _same_value(found: Any, expected: Any) -> bool:
    if isinstance(found, (dict, list)) or isinstance(expected, (dict, list)):
        return found == expected
    if isinstance(found, bool) or isinstance(expected, bool):
        return str(found).lower() == str(expected).lower()
    return found == expected or str(found) == str(expected)


def _met(expect: Expectation) -> str:
    """What was found when the declaration holds, for the VERIFIED line."""
    kind, value = expect.kind, _short(str(expect.value))
    return {
        "exit_code": "it succeeded" if expect.value == 0 else f"it exited with {value}",
        "equals": f"the output is `{value}`",
        "contains": f"the output contains `{value}`",
        "pattern": f"the output matches `{value}`",
        "json_equals": f"{expect.field} is `{value}`",
        "exists": "it is there",
        "absent": "it is gone",
    }[kind]


def _expected(expect: Expectation) -> str:
    """What was declared, for the MISMATCH line."""
    kind, value = expect.kind, _short(str(expect.value))
    return {
        "exit_code": f"exit code {value}",
        "equals": f"the output `{value}`",
        "contains": f"output containing `{value}`",
        "pattern": f"output matching `{value}`",
        "json_equals": f"{expect.field} = `{value}`",
        "exists": "it to be there",
        "absent": "it to be gone",
    }[kind]


def compare(expect: Expectation, reading: Reading, probe_text: str) -> Outcome:
    """The outcome of one reading against what was declared. Found text is short and masked."""
    if reading.unavailable:
        return unobserved(f"not verified: {reading.unavailable}", kind=Kind.COULD_NOT_LOOK)
    kind, value = expect.kind, expect.value
    expected = f"{probe_text}: expected {_expected(expect)}"

    if kind == "exit_code":
        if reading.exit_code is None:
            return unobserved("not verified: the check gave no exit code", kind=Kind.COULD_NOT_LOOK)
        if reading.exit_code == value:
            return verified(f"{probe_text}; {_met(expect)}, as declared")
        return mismatch(f"{expected}, found exit code {reading.exit_code}")

    if kind in {"exists", "absent"}:
        if reading.present is None:
            code = f" (exit code {reading.exit_code})" if reading.exit_code is not None else ""
            return unobserved(
                f"not verified: the check could not tell whether it is there{code}", kind=Kind.COULD_NOT_LOOK
            )
        if reading.present == (kind == "exists"):
            return verified(f"{probe_text}; {_met(expect)}, as declared")
        return mismatch(f"{expected}, found {'it is still there' if kind == 'absent' else 'it is not there'}")

    if reading.present is False:
        return mismatch(f"{expected}, found nothing there")
    if reading.exit_code not in (None, 0):
        return mismatch(f"{expected}, but the check failed with exit code {reading.exit_code}")

    if kind == "json_equals":
        found_field, found = _json_field(reading.output, expect.field)
        if not found_field:
            return mismatch(f"{expected}, found no field {expect.field}")
        if _same_value(found, value):
            return verified(f"{probe_text}; {_met(expect)}, as declared")
        return mismatch(f"{expected}, found `{_short(json.dumps(found, ensure_ascii=False))}`")

    text = reading.output.strip()
    assert isinstance(value, str)
    if kind == "equals":
        matched = text == value.strip()
    elif kind == "contains":
        matched = value in reading.output
    else:
        matched = fnmatch.fnmatchcase(text, value)
    if matched:
        return verified(f"{probe_text}; {_met(expect)}, as declared")
    return mismatch(f"{expected}, found `{_short(text)}`" if text else f"{expected}, found no output")


# ---------------------------------------------------------------------------------------------------------------
# The registered verifier.
# ---------------------------------------------------------------------------------------------------------------
@register(*DECLARED_ACTIONS, method=METHOD, precheck=False, timeout=CHECK_TIMEOUT_SECONDS)
async def declared_expectation(result: ActionResult) -> Outcome:
    """Run the action's declared probe once and compare; without ``expect`` the step stays reported."""
    action = result.action
    if action.expect is None:
        return unobserved(NO_EXPECT_DETAIL, kind=Kind.NOT_OBSERVABLE)
    cached = result._expect_outcome
    if isinstance(cached, Outcome):
        return cached
    outcome = await _check(result)
    result._expect_outcome = outcome
    return outcome


async def _check(result: ActionResult) -> Outcome:
    action = result.action
    if not result.success or not result.executed:
        # The registry is only reached for a success; a dry run or an already-satisfied skip did not run.
        return unobserved(
            "not verified: the step did not run, so its check did not run either", kind=Kind.COULD_NOT_LOOK
        )
    refusal = probe_refusal(action) or validator_refusal(action)
    if refusal:
        return unobserved(f"not verified: the check was refused because {refusal}", kind=Kind.COULD_NOT_LOOK)
    expect = action.expect
    assert expect is not None
    probe_text = describe_probe(expect.probe)
    try:
        reading = await run_probe(expect.probe)
    except TimeoutError:
        return unobserved("not verified: the check did not finish in time", kind=Kind.COULD_NOT_LOOK)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Declared check for %s could not run: %s", action.action_type.value, type(exc).__name__)
        return unobserved(
            f"not verified: the check could not run: {_short(str(exc)) or type(exc).__name__}", kind=Kind.COULD_NOT_LOOK
        )
    return compare(expect, reading, probe_text)
