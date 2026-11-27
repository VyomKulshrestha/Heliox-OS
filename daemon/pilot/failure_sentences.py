"""What Heliox says when a step fails because of the system: a plain sentence that names the thing and says what a
person can do, and whether asking again could change anything.

A handler's error is text (``ActionResult.error``): Python's ``[Errno N]`` (errno on macOS and Linux, and the C
runtime's errno on Windows), ``[WinError N]`` (Windows error numbers), or the words of a library or a tool. Each
platform's numbers are listed apart because they differ: ENAMETOOLONG is 36 on Linux and 63 on macOS, connection
refused is 111 on Linux, 61 on macOS and 10061 (WSAECONNREFUSED) on Windows.

The raw text stays in the receipt and the log; the person sees and hears the sentence. A failure of a ``FINAL_KINDS``
kind cannot be changed by asking again: the repair round does not retry it (pilot.server's terminal check), and a
step that already failed this way is never put to the person for approval a second time in the same task.
"""

from __future__ import annotations

import re
import sys
from enum import StrEnum

from pilot.reply_rules import folder_of, for_screen, host_of, name_of

# A failure's shown form: its sentence on the first line, then what the system said, as it said it, on a line that
# starts with this. The first line alone is the sentence (pilot.conversation.action_talk.result_sentence says it).
SYSTEM_SAID = "The system said: "


class FailureKind(StrEnum):
    NO_PERMISSION = "no_permission"
    READ_ONLY = "read_only"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    IN_USE = "in_use"
    DISK_FULL = "disk_full"
    PATH_TOO_LONG = "path_too_long"
    NO_NETWORK = "no_network"
    NAME_NOT_RESOLVED = "name_not_resolved"
    TIMED_OUT = "timed_out"
    REFUSED = "refused"
    SIGN_IN_REFUSED = "sign_in_refused"
    NOT_CONNECTED = "not_connected"
    # The web: a site was reached and answered, or could not be trusted.
    PAGE_NOT_THERE = "page_not_there"  # HTTP 404, 410
    SITE_REFUSED = "site_refused"  # HTTP 401, 403: a sign-in or a permission
    SITE_TROUBLE = "site_trouble"  # HTTP 429 and 5xx: overloaded or broken, try later
    SITE_DECLINED = "site_declined"  # any other HTTP 4xx
    REDIRECT_LOOP = "redirect_loop"
    CERTIFICATE = "certificate"
    GATEWAY_DENIED = "gateway_denied"  # Heliox's own rules refused the step before it was sent
    # The network layer refused an address on this computer, the local network or a metadata service that the person
    # did not name themselves (pilot.system.outbound).
    LOCAL_ADDRESS_REFUSED = "local_address_refused"
    NO_ELEMENT = "no_element"  # a page loaded, and nothing on it matches the selector asked for
    # Git's common endings.
    PUSH_BEHIND = "push_behind"  # the remote has commits the local branch lacks
    PUSH_PROTECTED = "push_protected"
    NOTHING_TO_COMMIT = "nothing_to_commit"
    NOT_A_REPOSITORY = "not_a_repository"
    NO_REMOTE = "no_remote"
    MERGE_CONFLICT = "merge_conflict"
    GIT_SIGN_IN = "git_sign_in"


# Asking again cannot change these. NOT_FOUND is final only when the request itself named the missing thing (a path
# the planner guessed may be found by another plan): see ``is_final``.
FINAL_KINDS = frozenset(
    {
        FailureKind.NO_PERMISSION,
        FailureKind.READ_ONLY,
        FailureKind.SIGN_IN_REFUSED,
        FailureKind.NOT_FOUND,
        FailureKind.NOT_CONNECTED,
        FailureKind.PAGE_NOT_THERE,
        FailureKind.SITE_REFUSED,
        FailureKind.REDIRECT_LOOP,
        FailureKind.CERTIFICATE,
        FailureKind.NO_ELEMENT,
        FailureKind.GATEWAY_DENIED,
        FailureKind.LOCAL_ADDRESS_REFUSED,
        FailureKind.PUSH_BEHIND,
        FailureKind.PUSH_PROTECTED,
        FailureKind.NOTHING_TO_COMMIT,
        FailureKind.NOT_A_REPOSITORY,
        FailureKind.NO_REMOTE,
        FailureKind.MERGE_CONFLICT,
        FailureKind.GIT_SIGN_IN,
    }
)

# An HTTP answer as pilot.system.api_client.HttpStatusError writes it: "GET to https://host failed with HTTP 404 Not
# Found (hint): <body excerpt>". The status and the server's short reason are kept; the body never is.
_HTTP = re.compile(r"failed with HTTP (?P<status>\d{3})(?P<reason>(?: [A-Za-z][A-Za-z' -]*?)?)(?=\s*\(|:|$)")
# Checked before the system's numbers and words, in order: an HTTP status, then a certificate or redirect problem,
# then git's endings. Each pattern is one that only that source writes.
EARLY_WORDS: tuple[tuple[FailureKind, re.Pattern[str]], ...] = (
    (
        FailureKind.CERTIFICATE,
        re.compile(
            r"certificate verify failed|CERTIFICATE_VERIFY_FAILED|SSLCertVerificationError|self[- ]signed certificate|"
            r"certificate has expired|hostname mismatch",
            re.I,
        ),
    ),
    (FailureKind.REDIRECT_LOOP, re.compile(r"exceeded \d+ redirects|too many redirects|redirect loop", re.I)),
    (FailureKind.GATEWAY_DENIED, re.compile(r"^\s*gateway denied\b", re.I)),
    (
        FailureKind.LOCAL_ADDRESS_REFUSED,
        re.compile(
            r"^I don't fetch from addresses on this computer or your own network|did not approve, so I did not follow it"
        ),
    ),
    (FailureKind.NO_ELEMENT, re.compile(r"\bno element matches the selector\b", re.I)),
    (FailureKind.NOT_CONNECTED, re.compile(r"\bis not connected\. connect it in\b", re.I)),
    (FailureKind.PUSH_PROTECTED, re.compile(r"protected branch|GH006|pre-receive hook declined", re.I)),
    (
        FailureKind.PUSH_BEHIND,
        re.compile(
            r"\[rejected\][^\n]*\((?:fetch first|non-fast-forward)\)|updates were rejected because the (?:remote contains|"
            r"tip of your current branch is behind)",
            re.I,
        ),
    ),
    (
        FailureKind.MERGE_CONFLICT,
        re.compile(r"^CONFLICT \(|merge conflict|unmerged files|fix conflicts and then", re.I | re.M),
    ),
    (FailureKind.NOTHING_TO_COMMIT, re.compile(r"nothing to commit|nothing staged to commit|nothing is staged", re.I)),
    (
        FailureKind.NO_REMOTE,
        re.compile(
            r"'[^']+' does not appear to be a git repository|no configured push destination|no such remote", re.I
        ),
    ),
    (FailureKind.NOT_A_REPOSITORY, re.compile(r"not a git repository", re.I)),
    (
        FailureKind.GIT_SIGN_IN,
        re.compile(r"authentication failed for|could not read username|permission denied \(publickey\)", re.I),
    ),
)


def http_status(error: str) -> tuple[int, str] | None:
    """(status, the server's short reason) of an HTTP answer in an error, or None."""
    found = _HTTP.search(str(error or ""))
    return (int(found.group("status")), found.group("reason").strip()) if found else None


# errno by platform. Python on Windows reports the C runtime's errno for file calls ([Errno 13] for a read-only
# folder) and the Windows error number for others ([WinError 5]).
ERRNO = {
    "linux": {
        1: FailureKind.NO_PERMISSION,  # EPERM
        2: FailureKind.NOT_FOUND,  # ENOENT
        13: FailureKind.NO_PERMISSION,  # EACCES
        16: FailureKind.IN_USE,  # EBUSY
        17: FailureKind.ALREADY_EXISTS,  # EEXIST
        26: FailureKind.IN_USE,  # ETXTBSY
        28: FailureKind.DISK_FULL,  # ENOSPC
        30: FailureKind.READ_ONLY,  # EROFS
        36: FailureKind.PATH_TOO_LONG,  # ENAMETOOLONG
        100: FailureKind.NO_NETWORK,  # ENETDOWN
        101: FailureKind.NO_NETWORK,  # ENETUNREACH
        110: FailureKind.TIMED_OUT,  # ETIMEDOUT
        111: FailureKind.REFUSED,  # ECONNREFUSED
        113: FailureKind.NO_NETWORK,  # EHOSTUNREACH
        122: FailureKind.DISK_FULL,  # EDQUOT
        -2: FailureKind.NAME_NOT_RESOLVED,  # EAI_NONAME
        -3: FailureKind.NAME_NOT_RESOLVED,  # EAI_AGAIN
        -5: FailureKind.NAME_NOT_RESOLVED,  # EAI_NODATA
    },
    "darwin": {
        1: FailureKind.NO_PERMISSION,
        2: FailureKind.NOT_FOUND,
        13: FailureKind.NO_PERMISSION,
        16: FailureKind.IN_USE,
        17: FailureKind.ALREADY_EXISTS,
        26: FailureKind.IN_USE,
        28: FailureKind.DISK_FULL,
        30: FailureKind.READ_ONLY,
        50: FailureKind.NO_NETWORK,  # ENETDOWN
        51: FailureKind.NO_NETWORK,  # ENETUNREACH
        60: FailureKind.TIMED_OUT,  # ETIMEDOUT
        61: FailureKind.REFUSED,  # ECONNREFUSED
        63: FailureKind.PATH_TOO_LONG,  # ENAMETOOLONG
        65: FailureKind.NO_NETWORK,  # EHOSTUNREACH
        69: FailureKind.DISK_FULL,  # EDQUOT
        8: FailureKind.NAME_NOT_RESOLVED,  # EAI_NONAME
    },
    "win32": {
        2: FailureKind.NOT_FOUND,
        13: FailureKind.NO_PERMISSION,
        17: FailureKind.ALREADY_EXISTS,
        28: FailureKind.DISK_FULL,
        38: FailureKind.PATH_TOO_LONG,  # the C runtime's ENAMETOOLONG
        11001: FailureKind.NAME_NOT_RESOLVED,  # getaddrinfo: WSAHOST_NOT_FOUND
        11002: FailureKind.NAME_NOT_RESOLVED,  # WSATRY_AGAIN
        11004: FailureKind.NAME_NOT_RESOLVED,  # WSANO_DATA
    },
}
WINERROR = {
    2: FailureKind.NOT_FOUND,  # ERROR_FILE_NOT_FOUND
    3: FailureKind.NOT_FOUND,  # ERROR_PATH_NOT_FOUND
    5: FailureKind.NO_PERMISSION,  # ERROR_ACCESS_DENIED
    19: FailureKind.READ_ONLY,  # ERROR_WRITE_PROTECT
    32: FailureKind.IN_USE,  # ERROR_SHARING_VIOLATION
    33: FailureKind.IN_USE,  # ERROR_LOCK_VIOLATION
    39: FailureKind.DISK_FULL,  # ERROR_HANDLE_DISK_FULL
    53: FailureKind.NO_NETWORK,  # ERROR_BAD_NETPATH
    80: FailureKind.ALREADY_EXISTS,  # ERROR_FILE_EXISTS
    112: FailureKind.DISK_FULL,  # ERROR_DISK_FULL
    121: FailureKind.TIMED_OUT,  # ERROR_SEM_TIMEOUT
    183: FailureKind.ALREADY_EXISTS,  # ERROR_ALREADY_EXISTS
    206: FailureKind.PATH_TOO_LONG,  # ERROR_FILENAME_EXCED_RANGE
    1231: FailureKind.NO_NETWORK,  # ERROR_NETWORK_UNREACHABLE
    10050: FailureKind.NO_NETWORK,  # WSAENETDOWN
    10051: FailureKind.NO_NETWORK,  # WSAENETUNREACH
    10060: FailureKind.TIMED_OUT,  # WSAETIMEDOUT
    10061: FailureKind.REFUSED,  # WSAECONNREFUSED
    10065: FailureKind.NO_NETWORK,  # WSAEHOSTUNREACH
    11001: FailureKind.NAME_NOT_RESOLVED,  # WSAHOST_NOT_FOUND
    11002: FailureKind.NAME_NOT_RESOLVED,
}
# The words of the system, of libraries (httpx, requests, aiohttp, smtplib, git) and of Heliox's own handlers, for
# errors that carry no number. Checked in order: the first match decides.
WORDS: tuple[tuple[FailureKind, re.Pattern[str]], ...] = (
    (
        FailureKind.SIGN_IN_REFUSED,
        re.compile(
            r"refused the sign-in|authentication (?:failed|required)|invalid (?:credentials|token|api key)|"
            r"bad credentials|\b401\b.*unauthori[sz]ed|unauthori[sz]ed.*\b401\b|username and password not accepted",
            re.I,
        ),
    ),
    (FailureKind.READ_ONLY, re.compile(r"read-only file system|write[- ]protected|is read-only", re.I)),
    (
        FailureKind.NO_PERMISSION,
        re.compile(
            r"permission denied|access is denied|access denied|operation not permitted|not permitted to|"
            r"requires elevation|insufficient privileges",
            re.I,
        ),
    ),
    (
        FailureKind.DISK_FULL,
        re.compile(r"no space left on device|disk (?:is )?full|not enough space|quota exceeded", re.I),
    ),
    (
        FailureKind.PATH_TOO_LONG,
        re.compile(r"file ?name too long|path too long|filename or extension is too long", re.I),
    ),
    (
        FailureKind.IN_USE,
        re.compile(
            r"being used by another process|used by another program|resource busy|device or resource busy|"
            r"text file busy|file is locked|is locked by",
            re.I,
        ),
    ),
    (FailureKind.ALREADY_EXISTS, re.compile(r"already exists|file exists", re.I)),
    (
        FailureKind.NAME_NOT_RESOLVED,
        re.compile(
            r"name or service not known|nodename nor servname|getaddrinfo failed|temporary failure in name resolution|"
            r"could not resolve host|no address associated with hostname|name resolution",
            re.I,
        ),
    ),
    (
        FailureKind.NO_NETWORK,
        re.compile(
            r"network is unreachable|network is down|no route to host|host is unreachable|no internet|"
            r"network path was not found",
            re.I,
        ),
    ),
    (FailureKind.REFUSED, re.compile(r"connection refused|actively refused", re.I)),
    (FailureKind.TIMED_OUT, re.compile(r"timed out|timeout|did not answer within|took too long", re.I)),
    (
        FailureKind.NOT_FOUND,
        re.compile(
            r"no such file or directory|cannot find the (?:file|path)|file not found|path not found|source not found|"
            r"does not exist|was not found|not found:",
            re.I,
        ),
    ),
)

_ERRNO = re.compile(r"\[Errno (-?\d+)\]")
_WINERROR = re.compile(r"\[WinError (\d+)\]")
# The path a system error names: Python's "[Errno 13] Permission denied: 'C:\\x\\y.txt'" or "Source not found: C:\\x".
_QUOTED_PATH = re.compile(r"""['"](?P<path>(?:[A-Za-z]:[\\/]|\\\\|/|~[\\/])[^'"]+)['"]""")
_BARE_PATH = re.compile(r"(?P<path>(?:[A-Za-z]:[\\/]|\\\\|(?<![\w.])/)[^\s'\"]+)")


def platform_key(platform: str | None = None) -> str:
    name = platform or sys.platform
    return "win32" if name.startswith("win") else "darwin" if name == "darwin" else "linux"


def classify(error: str, *, platform: str | None = None) -> FailureKind | None:
    """The kind of a system failure from its text, or None when it is none of the common ones."""
    text = str(error or "")
    answered = http_status(text)
    if answered is not None:
        status = answered[0]
        if status in {404, 410}:
            return FailureKind.PAGE_NOT_THERE
        if status in {401, 403}:
            return FailureKind.SITE_REFUSED
        if status == 429 or status >= 500:
            return FailureKind.SITE_TROUBLE
        return FailureKind.SITE_DECLINED
    for kind, pattern in EARLY_WORDS:
        if pattern.search(text):
            return kind
    found = _WINERROR.search(text)
    if found and int(found.group(1)) in WINERROR:
        return WINERROR[int(found.group(1))]
    found = _ERRNO.search(text)
    if found:
        number = int(found.group(1))
        kind = ERRNO[platform_key(platform)].get(number)
        if kind is not None:
            # EACCES is also what Linux and macOS give for a read-only mount when the words say so.
            if kind is FailureKind.NO_PERMISSION and re.search(r"read-only", text, re.I):
                return FailureKind.READ_ONLY
            return kind
    for kind, pattern in WORDS:
        if pattern.search(text):
            return kind
    return None


def named_path(error: str) -> str:
    """The path a system error names, with a repr's doubled backslashes put back; "" when it names none."""
    from pilot.reply_rules import undoubled

    text = undoubled(str(error or ""))
    found = _QUOTED_PATH.search(text) or _BARE_PATH.search(text)
    return found.group("path").rstrip(".,;:)") if found else ""


# What each action does, in a few words, for "I couldn't <verb> <thing>".
_VERBS = {
    "file_read": "read",
    "file_write": "write",
    "file_delete": "delete",
    "file_move": "move",
    "file_copy": "copy",
    "file_list": "list",
    "file_search": "search",
    "directory_summary": "look through",
    "directory_size": "measure",
    "file_hash": "hash",
    "file_compare": "compare",
    "file_permissions": "change the permissions of",
    "file_parse": "read",
    "file_search_content": "search",
    "download_file": "download",
    "screenshot": "save",
    "browser_screenshot": "save",
    "git_status": "read the status of",
    "git_diff": "read the changes in",
    "git_log": "read the history of",
    "git_branch": "switch the branch of",
    "git_stage": "stage",
    "git_commit": "commit to",
    "git_push": "push",
    "open_application": "open",
    "open_url": "open",
    "api_request": "reach",
    "api_scrape": "read",
    "api_webhook": "reach",
    "api_github": "reach",
    "api_send_email": "send the email to",
    "api_slack": "post to",
    "api_discord": "post to",
    "process_kill": "stop",
    "trigger_delete": "delete the trigger",
    "workspace_index": "index",
    "log_analyze": "read",
    "registry_write": "change",
    "service_start": "start",
    "service_stop": "stop",
    "service_restart": "restart",
    "package_install": "install",
    "package_remove": "remove",
}
_NETWORK_ACTIONS = frozenset(
    {"api_request", "api_scrape", "api_webhook", "api_github", "api_slack", "api_discord", "download_file", "open_url"}
)


def _thing(action: object, error: str) -> tuple[str, str]:
    """(what the step was about, by its name; the folder it is in, or "")."""
    action_type = str(getattr(getattr(action, "action_type", None), "value", getattr(action, "action_type", "")))
    params = getattr(action, "parameters", None)
    if action_type in _NETWORK_ACTIONS or action_type.startswith("api_"):
        url = str(getattr(params, "url", "") or getattr(params, "webhook_url", "") or getattr(action, "target", ""))
        if action_type == "api_slack":
            return "Slack", ""
        if action_type == "api_discord":
            return "Discord", ""
        if action_type == "api_github":
            return "GitHub", ""
        if action_type == "api_send_email":
            return str(getattr(params, "to", "") or "the recipient"), ""
        host = host_of(url) if url else ""
        if host:
            return host, ""
    path = named_path(error)
    if not path:
        for field in ("path", "repo_path", "output_path", "directory", "log_path", "folder_path"):
            value = getattr(params, field, None)
            if isinstance(value, str) and value.strip():
                path = value
                break
    if not path:
        name = str(getattr(params, "name", "") or getattr(action, "target", "") or "")
        return name_of(name) if name else "it", ""
    return name_of(path), folder_of(path)


def failure_sentence(action: object, error: str, *, platform: str | None = None) -> tuple[str, str]:
    """(shown, spoken) for a step that failed with ``error``. A common system failure gets a plain sentence that names
    the thing and says what a person can do; any other failure says what could not be done, the shown form adds the
    error as the system wrote it (backslashes as written)."""
    from pilot.reply_rules import for_ear

    action_type = str(getattr(getattr(action, "action_type", None), "value", getattr(action, "action_type", "")))
    verb = _VERBS.get(action_type, "do")
    kind = classify(error, platform=platform)
    # A system failure names the path it hit; any other error is about the step's own target.
    thing, folder = _thing(action, error if kind is not None else "")
    where = f" in {folder}" if folder and folder != thing else ""
    if verb == "do" and kind is None:
        # No verb for this action: say which step failed, in its own words ("The system info step failed").
        raw = for_screen(without_markup(" ".join(str(error or "").split())), limit=400)
        lead = f"The {action_type.replace('_', ' ') or 'last'} step failed"
        return (f"{lead}.\n{SYSTEM_SAID}{raw}" if raw else f"{lead}."), for_ear(
            f"{lead}. The details are on screen." if raw else f"{lead}."
        )
    did = f"{verb} {thing}{where}" if verb != "do" else f"do that with {thing}" if thing != "it" else "do that"
    sentences = {
        FailureKind.NO_PERMISSION: f"I couldn't {did}: permission was denied. You can change its permissions, or "
        "choose another place.",
        FailureKind.READ_ONLY: f"I couldn't {did}: that place is read-only. Choose a folder you can write to.",
        FailureKind.NOT_FOUND: f"I couldn't find {thing}{where}. Check the name, or tell me where it is.",
        FailureKind.ALREADY_EXISTS: f"{thing[:1].upper() + thing[1:]} already exists{where}. Tell me to replace it, "
        "or give a new name.",
        FailureKind.IN_USE: f"I couldn't {did}: another program is using it. Close that program and ask me again.",
        FailureKind.DISK_FULL: f"I couldn't {did}: the disk is full. Free some space and ask me again.",
        FailureKind.PATH_TOO_LONG: f"I couldn't {did}: its path is too long. Use a shorter folder path or name.",
        FailureKind.NO_NETWORK: f"I couldn't {did}: there's no network connection. Check that you're online and ask "
        "me again.",
        FailureKind.NAME_NOT_RESOLVED: f"I couldn't {did}: that address could not be found. Check its spelling and "
        "your connection.",
        FailureKind.TIMED_OUT: f"I couldn't {did}: it took too long to answer. You can ask me again in a moment.",
        FailureKind.REFUSED: f"I couldn't {did}: the connection was refused. Check that the service is running.",
        FailureKind.SIGN_IN_REFUSED: f"I couldn't {did}: the sign-in was refused. Check the account in Settings.",
    }
    sentences.update(_web_and_git_sentences(action, thing, error))
    if kind is not None:
        sentence = sentences[kind]
        answered = http_status(error)
        if answered is not None:
            # What the server said, in its own short words; its page stays in the receipt.
            status, reason = answered
            return f"{sentence}\nThe server answered: HTTP {status}{f' {reason}' if reason else ''}", for_ear(sentence)
        if kind is FailureKind.NO_ELEMENT:
            held = re.split(r"(?<=\.)\s+", " ".join(str(error or "").split()), maxsplit=1)
            return (f"{sentence}\n{held[1]}" if len(held) > 1 else sentence), for_ear(sentence)
        return sentence, for_ear(sentence)
    raw = for_screen(without_markup(" ".join(str(error or "").split())), limit=400)
    lead = f"I couldn't {did}."
    shown = f"{lead}\n{SYSTEM_SAID}{raw}" if raw else lead
    return shown, for_ear(f"{lead} The details are on screen." if raw else lead)


_MARKUP = re.compile(r"<(?:!doctype|html|head|body|\?xml)\b.*$", re.IGNORECASE | re.DOTALL)


def without_markup(text: str) -> str:
    """An error with a web page's HTML in it, cut where the page starts: the page stays in the receipt."""
    cut = _MARKUP.sub("", str(text or "")).rstrip(" :")
    return cut + (" (the page it sent is in the receipt)" if cut != str(text or "").rstrip(" :") else "")


def _web_and_git_sentences(action: object, thing: str, error: str) -> dict[FailureKind, str]:
    """The sentences for a site that answered or could not be trusted, a service not connected, and git's endings."""
    params = getattr(action, "parameters", None)
    repo = name_of(str(getattr(params, "repo_path", "") or "")) or "the repository"
    branch = str(getattr(params, "branch", "") or "").strip()
    remote = str(getattr(params, "remote", "") or "").strip() or "origin"
    pushed = branch or "the branch"
    selector = str(getattr(params, "selector", "") or "").strip()
    kind_of = str(getattr(getattr(action, "action_type", None), "value", ""))
    sends = kind_of.startswith("api_") or kind_of in {"download_file", "open_url"}
    verb_for_gateway = "call" if sends else "run that on"
    service = thing if thing and thing != "it" else "That service"
    where = re.search(r"connect it in ([^.]+)", str(error or ""), re.I)
    return {
        FailureKind.NOT_CONNECTED: f"{service} isn't connected yet. Connect it in "
        f"{where.group(1).strip().replace(' > ', ', ') if where else 'Settings'}, then ask me again.",
        FailureKind.PAGE_NOT_THERE: f"{thing} has no page at that address. Check the address and ask me again.",
        FailureKind.SITE_REFUSED: f"{thing} refused the request: it needs a sign-in or a permission I don't have.",
        FailureKind.SITE_TROUBLE: f"{thing} is overloaded or having trouble right now. Try again in a few minutes.",
        FailureKind.SITE_DECLINED: f"{thing} turned the request down. The details are on screen.",
        FailureKind.REDIRECT_LOOP: f"{thing} kept sending me to another address without reaching a page, so I stopped.",
        FailureKind.CERTIFICATE: f"I didn't open {thing}: its security certificate could not be checked, so the "
        "connection could not be trusted.",
        FailureKind.LOCAL_ADDRESS_REFUSED: _local_refusal(thing, error),
        FailureKind.GATEWAY_DENIED: f"I didn't {verb_for_gateway} {thing}: my safety rules don't allow that "
        f"from here, so {'nothing was sent' if sends else 'nothing ran'}. Asking again won't change it.",
        FailureKind.NO_ELEMENT: f"Nothing on {thing} matches {selector or 'what was asked for'}. The page loaded; "
        "what it does hold is on screen.",
        FailureKind.PUSH_BEHIND: f"I couldn't push {pushed}: {remote} has commits you don't have yet. Pull first, "
        "then push again.",
        FailureKind.PUSH_PROTECTED: f"I couldn't push {pushed}: {remote} protects that branch, so it takes changes "
        "only through a pull request.",
        FailureKind.NOTHING_TO_COMMIT: f"There was nothing to commit in {repo}: no changes are staged.",
        FailureKind.NOT_A_REPOSITORY: f"{repo[:1].upper() + repo[1:]} is not a git repository.",
        FailureKind.NO_REMOTE: f"I couldn't push {pushed}: {repo} has no remote called {remote}.",
        FailureKind.MERGE_CONFLICT: f"I couldn't finish that in {repo}: there is a merge conflict to resolve first.",
        FailureKind.GIT_SIGN_IN: f"I couldn't reach {remote}: it refused the sign-in. Check your Git credentials.",
    }


def _local_refusal(thing: str, error: str) -> str:
    """The sentence for a request the network layer refused: the address, what it is, and how the person can ask
    for it. A redirect that left an approved local address keeps the network layer's own sentence."""
    text = str(error or "").strip()
    if "so I did not follow it" in text:
        return text if text.endswith(".") else text + "."
    found = re.search(r"\. (\S+?)(?: \(([^)]*)\))? is (.+?)\.$", text)
    host = found.group(1) if found else (thing if thing and thing != "it" else "that address")
    what = found.group(3) if found else "on this computer or your own network"
    what = f"an address {what}" if what.startswith("on ") else what
    return (
        f"I didn't fetch {host}: it's {what}, and I fetch from those only when you ask for that address yourself. "
        "Type the address in your request if you meant it."
    )


def is_final(action: object, error: str, request: str = "", *, platform: str | None = None) -> bool:
    """Whether asking again cannot change this failure: no permission, read-only, a refused sign-in, or a thing not
    found that the request itself named."""
    kind = classify(error, platform=platform)
    if kind not in FINAL_KINDS:
        return False
    if kind is not FailureKind.NOT_FOUND:
        return True
    thing, _folder = _thing(action, error)
    return bool(thing) and thing != "it" and thing.casefold() in str(request or "").casefold()
