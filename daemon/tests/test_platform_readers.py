"""The focused-text readers on Windows, macOS and Linux share one privacy decision, and each platform layer is
tested here with a fake of its accessibility interface, on every system CI runs."""

from __future__ import annotations

import asyncio

import pytest

from pilot.live import focused_text, focused_text_linux, focused_text_macos
from pilot.live.focused_text_policy import (
    FIELD,
    MAX_CHARS,
    TERMINAL,
    ElementFacts,
    FocusedText,
    Permission,
    PermissionGate,
    admit,
    never_read,
    same_field,
)

# ── the shared decision ──


def test_admit_passes_text_only_from_a_writable_non_secure_field():
    field = admit(ElementFacts(role="entry", pid=3, kind=FIELD, field="f1", text="hello"))
    assert field.editable and field.text == "hello" and field.field == "f1" and field.kind == FIELD
    term = admit(ElementFacts(role="terminal", kind=TERMINAL, field="t", text="$ ls"))
    assert term.editable and term.kind == TERMINAL
    secure = admit(ElementFacts(role="password text", secure=True, kind=FIELD, field="p", text="hunter2"))
    assert secure.password and not secure.editable and secure.text == "" and secure.field == ""
    page = admit(ElementFacts(role="document web", kind="", field="d", text="article text"))
    assert not page.editable and page.text == "" and page.field == ""
    odd = admit(ElementFacts(role="x", kind="maybe", text="x"))
    assert not odd.editable and odd.text == ""
    long = admit(ElementFacts(role="entry", kind=FIELD, field="f", text="a" * (MAX_CHARS + 50) + "END"))
    assert len(long.text) == MAX_CHARS and long.text.endswith("END")


def test_never_read_covers_heliox_password_managers_system_stores_and_exclusions():
    assert never_read("Heliox-OS", "Heliox-OS", is_heliox=True)
    assert never_read("1Password", "Vault")
    assert never_read("chrome", "Bitwarden - Google Chrome")
    assert never_read("Keychain Access", "login")
    assert never_read("Passwords", "")  # macOS 15's Passwords app
    assert never_read("seahorse", "Passwords and Keys")
    assert never_read("Slack", "general", ["slack"])
    assert never_read("slack.exe", "general", ["  Slack "])
    # A page about passwords in an ordinary app is read; the system store names match the whole app name only.
    assert not never_read("chrome", "Change your passwords - Google Chrome")
    assert not never_read("Code", "server.py", ["", "  "])


def test_same_field_follows_identity_not_title():
    a = FocusedText("hi", "Edit", 1, False, True, FIELD, "42.7")
    b = FocusedText("hi there", "Edit", 1, False, True, FIELD, "42.7")
    c = FocusedText("hi", "Edit", 1, False, True, FIELD, "42.8")
    assert same_field(a, b) and not same_field(a, c) and not same_field(None, a)
    assert not same_field(FocusedText("", "Edit", 1, False), FocusedText("", "Edit", 1, False))


def test_windows_replies_go_through_the_same_decision():
    reply = focused_text.parse_reply(
        {"ok": True, "type": "ControlType.Edit", "password": True, "editable": True, "kind": "field", "text": "x"}
    )
    assert reply == admit(ElementFacts(role="Edit", secure=True, kind=FIELD, text="x"))


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_permission_gate_prompts_once_and_notices_a_later_grant():
    clock = Clock()
    granted = [False]
    checks: list[float] = []
    prompts: list[int] = []

    def check() -> Permission:
        checks.append(clock.t)
        return Permission.GRANTED if granted[0] else Permission.NOT_ASKED

    gate = PermissionGate(check, lambda: prompts.append(1), recheck_seconds=10.0, clock=clock)
    assert not gate.allowed() and gate.state == Permission.NOT_ASKED
    for _ in range(50):  # a polling loop never prompts and re-checks only every 10 s
        assert not gate.allowed()
    assert prompts == [] and len(checks) == 1
    assert gate.request() and not gate.request() and prompts == [1]
    granted[0] = True
    clock.t = 5.0
    assert not gate.allowed()  # not re-checked yet
    clock.t = 10.5
    assert gate.allowed() and gate.state == Permission.GRANTED
    assert not gate.request() and prompts == [1]


def test_permission_gate_treats_a_failing_check_as_cannot_tell():
    def boom() -> Permission:
        raise OSError("no framework")

    gate = PermissionGate(boom, None)
    assert not gate.allowed() and gate.state == Permission.UNKNOWN and not gate.request()


def test_create_reader_picks_the_platform(monkeypatch):
    from pilot.system.platform_detect import Platform

    monkeypatch.setattr(focused_text, "CURRENT_PLATFORM", Platform.LINUX)
    assert isinstance(focused_text.create_reader(), focused_text_linux.LinuxFocusedTextReader)
    monkeypatch.setattr(focused_text, "CURRENT_PLATFORM", Platform.MACOS)
    assert isinstance(focused_text.create_reader(), focused_text_macos.MacFocusedTextReader)
    monkeypatch.setattr(focused_text, "CURRENT_PLATFORM", Platform.WINDOWS)
    assert isinstance(focused_text.create_reader(), focused_text.WindowsFocusedTextReader)


# ── Linux: AT-SPI with a fake bus ──


def bits(*state: int) -> list[int]:
    words = [0, 0]
    for s in state:
        words[s // 32] |= 1 << (s % 32)
    return words


L = focused_text_linux
APP = ":1.42"


class FakeBus(L.Transport):
    """An accessibility bus with one application: a window holding an entry, a password field and a terminal."""

    unique_name = ":1.99"

    def __init__(self) -> None:
        self.title = "Untitled - Editor"
        self.nodes = {
            "/win": {"role": "frame", "state": [L.STATE_ACTIVE, L.STATE_SHOWING], "children": ["/entry", "/pw"]},
            "/entry": {
                "role": "entry",
                "state": [L.STATE_EDITABLE, L.STATE_SHOWING],
                "text": "draft",
                "parent": "/win",
            },
            "/pw": {
                "role": "password text",
                "state": [L.STATE_EDITABLE, L.STATE_SHOWING],
                "text": "s3cret",
                "parent": "/win",
            },
            "/label": {"role": "label", "state": [L.STATE_SHOWING], "text": "Name", "parent": "/win"},
            # GTK 4: a GtkPasswordEntry ("text box", numeric role password text) holding the text widget that has
            # focus ("text box", numeric role text). Only the entry's numeric role says it is a password.
            "/gtk4pw": {
                "role": "text box",
                "number": L.ROLE_PASSWORD_TEXT,
                "state": [L.STATE_SHOWING],
                "parent": "/win",
            },
            "/gtk4pw/text": {
                "role": "text box",
                "number": 61,
                "state": [L.STATE_EDITABLE, L.STATE_SHOWING],
                "text": "s3cret",
                "parent": "/gtk4pw",
            },
            "/term": {"role": "terminal", "state": [L.STATE_SHOWING], "text": "$ make", "parent": "/win"},
        }
        self.focus = "/entry"
        self.pending: list[L.Ref] = []
        self.calls: list[tuple[str, str]] = []

    def call(self, bus, path, interface, method, signature="", body=()):
        self.calls.append((path, method))
        if path in getattr(self, "gone", ()):
            raise RuntimeError("[org.freedesktop.DBus.Error.UnknownObject] No such object")
        if method == "GetConnectionUnixProcessID":
            return (4242,)
        if path == L.ROOT_PATH and method == "GetChildren":
            return ([(APP, "/app")],)
        if path == "/app" and method == "GetChildren":
            return ([(APP, "/win")],)
        node = self.nodes[path]
        if method == "GetState":
            state = list(node["state"]) + ([L.STATE_FOCUSED] if path == self.focus else [])
            return (bits(*state),)
        if method == "GetRoleName":
            return (node["role"],)
        if method == "GetRole":
            if node.get("number", 0) is None:
                raise RuntimeError("[org.freedesktop.DBus.Error.UnknownMethod] GetRole")
            default = {"frame": 23, "entry": 79, "password text": 40, "label": 29, "terminal": 59}
            return (node.get("number", default.get(node["role"], 61)),)
        if method == "GetChildren":
            return ([(APP, child) for child in node.get("children", [])],)
        if method == "GetInterfaces":
            return ([L.ACCESSIBLE] + ([L.TEXT] if "text" in node else []),)
        if method == "Get":
            prop = body[1]
            if prop == "CharacterCount":
                return (("i", len(node["text"])),)
            if prop == "Name":
                return (("s", self.title if path == "/win" else ""),)
            if prop == "Parent":
                return (("(so)", (APP, node.get("parent", "/org/a11y/atspi/null"))),)
        if method == "GetText":
            start, end = body
            return (node["text"][start:end],)
        raise KeyError((path, method))

    def events(self):
        found, self.pending = self.pending, []
        return found


def test_atspi_state_bits_and_classification():
    assert L.states([1 << 7 | 1 << 12, 1 << (43 - 32)]) == {7, 12, 43}
    assert L.classify("password text", {L.STATE_EDITABLE}, [L.TEXT]) == (True, "")
    assert L.classify("entry", {L.STATE_EDITABLE}, [L.TEXT]) == (False, FIELD)
    assert L.classify("entry", {L.STATE_EDITABLE, L.STATE_READ_ONLY}, [L.TEXT]) == (False, "")
    assert L.classify("entry", {L.STATE_EDITABLE}, []) == (False, "")
    assert L.classify("terminal", set(), [L.TEXT]) == (False, TERMINAL)
    assert L.classify("label", {L.STATE_EDITABLE}, [L.TEXT]) == (False, "")
    assert L.classify("document web", set(), [L.TEXT]) == (False, "")
    assert L.focus_from_event(("focused", 1, 0, ("i", 0)), ":1.5", "/a/b") == (":1.5", "/a/b")
    assert L.focus_from_event(("focused", 0, 0, ("i", 0)), ":1.5", "/a/b") is None
    assert L.focus_from_event(("checked", 1, 0, ("i", 0)), ":1.5", "/a/b") is None


def test_atspi_reads_the_focused_entry_found_by_search_then_follows_events(tmp_path, monkeypatch):
    clock = Clock()
    bus = FakeBus()
    client = L.AtspiClient(bus, clock=clock)
    ref = client.focused()
    assert ref == (APP, "/entry")
    read = client.describe(ref)
    assert read.editable and read.text == "draft" and read.pid == 4242 and read.field == f"atspi:{APP}/entry"
    # Focus moves to the terminal; the event says so and no search is needed.
    bus.focus = "/term"
    bus.pending.append((APP, "/term"))
    bus.calls.clear()
    term = client.describe(client.focused())
    assert term.kind == TERMINAL and term.text == "$ make"
    assert not any(method == "GetChildren" for _path, method in bus.calls)


def test_atspi_refuses_a_password_field_before_asking_for_its_text():
    bus = FakeBus()
    bus.focus = "/pw"
    client = L.AtspiClient(bus, clock=Clock())
    read = client.describe(client.focused())
    assert read.password and not read.editable and read.text == ""
    assert not any(path == "/pw" and method in ("GetText", "Get") for path, method in bus.calls)


def test_a_gtk4_password_entry_is_refused_by_its_numeric_role_and_its_parents():
    # Role names, numeric roles and the parent chain as GTK 4 gives them over AT-SPI (gtkatspicontext: GetRoleName
    # returns the ARIA-style "text box", GetRole returns password text only for the GtkPasswordEntry). The next
    # CI run records the real dump (DIAG lines of toolkit_check.py gtk4).
    assert L.password_by_role(["text box", "text box"], [61, L.ROLE_PASSWORD_TEXT])
    assert L.password_by_role(["password text"], [61])
    assert L.password_by_role(["text box"], [None]), "a field whose numeric role cannot be read is not read"
    assert not L.password_by_role(["text box", "frame"], [61, 23])
    bus = FakeBus()
    bus.focus = "/gtk4pw/text"
    client = L.AtspiClient(bus, clock=Clock())
    read = client.describe((APP, "/gtk4pw/text"))
    assert read.password and not read.editable and read.text == ""
    assert not any(path == "/gtk4pw/text" and method == "GetText" for path, method in bus.calls)
    bus.nodes["/entry"]["number"] = None  # a toolkit that cannot say its numeric role: refused, not read
    assert client.describe((APP, "/entry")).text == ""


def test_atspi_field_identity_holds_while_typing_and_when_the_title_changes():
    bus = FakeBus()
    client = L.AtspiClient(bus, clock=Clock())
    first = client.describe(client.focused())
    assert client.window_title() == "Untitled - Editor"
    bus.nodes["/entry"]["text"] = "draft and more"
    bus.title = "*draft and more - Editor"
    second = client.describe(client.focused())
    assert client.window_title() == "*draft and more - Editor"
    assert same_field(first, second) and second.text == "draft and more"
    bus.focus = "/pw"
    bus.pending.append((APP, "/pw"))
    assert not same_field(second, client.describe(client.focused()))


def test_atspi_search_is_rate_limited_and_own_events_are_ignored():
    clock = Clock()
    bus = FakeBus()
    bus.focus = "/nowhere"
    bus.nodes["/nowhere"] = {"role": "panel", "state": [], "parent": "/win"}
    client = L.AtspiClient(bus, clock=clock)
    assert client.focused() is None
    bus.calls.clear()
    assert client.focused() is None and bus.calls == []  # within SEARCH_EVERY_SECONDS: no second search
    bus.pending.append((bus.unique_name, "/entry"))  # an event from Heliox's own connection
    bus.focus = "/entry"
    assert client.focused() is None
    clock.t = L.SEARCH_EVERY_SECONDS + 1
    assert client.focused() == (APP, "/entry")


def test_atspi_long_text_reads_only_the_tail():
    bus = FakeBus()
    bus.nodes["/entry"]["text"] = "x" * 10_000 + "tail"
    client = L.AtspiClient(bus, clock=Clock())
    read = client.describe(client.focused())
    assert len(read.text) == MAX_CHARS and read.text.endswith("tail")


def test_atspi_foreground_uses_the_process_name(tmp_path, monkeypatch):
    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "comm").write_text("gedit\n")
    original = L.comm_name
    monkeypatch.setattr(L, "comm_name", lambda pid: original(pid, tmp_path))
    client = L.AtspiClient(FakeBus(), clock=Clock())
    assert client.foreground() == ("gedit", "Untitled - Editor")
    assert original(1, tmp_path) == ""


def test_linux_reader_explains_a_missing_binding_and_a_missing_bus(monkeypatch):
    shared = L._Shared()
    monkeypatch.setattr(L, "SHARED", shared)

    def no_jeepney():
        raise ImportError("jeepney")

    shared.make = no_jeepney
    reader = L.LinuxFocusedTextReader()
    assert asyncio.run(reader.read()) is None
    assert reader.status().state == "unavailable" and "jeepney" in reader.status().message

    shared2 = L._Shared()
    monkeypatch.setattr(L, "SHARED", shared2)

    def no_bus():
        raise ConnectionRefusedError("no bus")

    shared2.make = no_bus
    assert asyncio.run(reader.read()) is None
    assert reader.status().state == "unavailable" and "at-spi2-core" in reader.status().message
    assert shared2.retry_at > 0  # backs off instead of reconnecting on every poll


def test_linux_reader_reads_through_the_shared_client(monkeypatch):
    shared = L._Shared()
    shared.make = FakeBus
    monkeypatch.setattr(L, "SHARED", shared)
    monkeypatch.setattr(L, "toolkit_accessibility_enabled", lambda: False)
    reader = L.LinuxFocusedTextReader()
    read = asyncio.run(reader.read())
    assert read is not None and read.text == "draft"
    status = reader.status()
    assert status.state == "ready" and "toolkit-accessibility" in status.message
    asyncio.run(reader.close())
    assert shared.client is None


# ── macOS: the Accessibility API with a fake backend ──


M = focused_text_macos


class FakeAX:
    def __init__(self) -> None:
        self.trusted_now = True
        self.prompts = 0
        self.element = {"AXRole": "AXTextArea", "AXSubrole": "", "AXValue": "Dear Sam", "hash": 77, "settable": True}
        self.bundle = "com.apple.TextEdit"
        self.title = "Untitled"
        self.text_reads = 0

    def trusted(self):
        return Permission.GRANTED if self.trusted_now else Permission.NOT_ASKED

    def prompt(self):
        self.prompts += 1

    def focused(self):
        return self.element

    def focused_app(self):
        return {"app": True}

    def attr(self, element, name):
        if "app" in element:
            return {"AXTitle": self.title} if name == "AXFocusedWindow" else None
        if name == "AXTitle":
            return element.get("AXTitle")
        return element.get(name)

    def settable(self, element, name):
        return element["settable"]

    def pid(self, element):
        return 501

    def element_hash(self, element):
        return element["hash"]

    def bundle_id(self, pid):
        return self.bundle

    def app_name(self, pid):
        return "TextEdit"

    def text_tail(self, element):
        self.text_reads += 1
        return element["AXValue"]


def test_macos_classification():
    assert M.classify("AXTextField", "AXSecureTextField", True, "com.x", False) == (True, "")
    assert M.classify("AXTextArea", "", True, "com.x", True) == (True, "")  # secure keyboard entry is on
    assert M.classify("AXTextArea", "", True, "com.x", False) == (False, FIELD)
    assert M.classify("AXTextField", "", False, "com.x", False) == (False, "")  # not settable: read-only
    assert M.classify("AXWebArea", "", False, "com.apple.Safari", False) == (False, "")
    assert M.classify("AXTextArea", "", False, "com.apple.Terminal", False) == (False, TERMINAL)
    assert M.field_id(501, -1).startswith("ax:501:") and not M.field_id(1, 2).startswith("42.")


def test_macos_reads_a_text_area_and_keeps_its_identity_when_the_title_changes():
    ax = FakeAX()
    reader = M.MacReader(ax, secure_input=lambda: False)
    first = reader.read()
    assert first is not None and first.editable and first.text == "Dear Sam" and first.pid == 501
    assert reader.foreground() == ("TextEdit", "Untitled")
    ax.element = {**ax.element, "AXValue": "Dear Sam, thanks"}  # a new reference to the same element
    ax.title = "Untitled — Edited"
    second = reader.read()
    assert reader.foreground() == ("TextEdit", "Untitled — Edited")
    assert same_field(first, second) and second.text == "Dear Sam, thanks"


def test_macos_never_reads_a_secure_field_or_during_secure_input():
    ax = FakeAX()
    ax.element = {
        "AXRole": "AXTextField",
        "AXSubrole": "AXSecureTextField",
        "AXValue": "pw",
        "hash": 1,
        "settable": True,
    }
    read = M.MacReader(ax, secure_input=lambda: False).read()
    assert read is not None and read.password and read.text == "" and ax.text_reads == 0
    ax2 = FakeAX()
    read2 = M.MacReader(ax2, secure_input=lambda: True).read()
    assert read2 is not None and read2.password and ax2.text_reads == 0


def test_macos_without_permission_reads_nothing_and_prompts_only_when_asked_once():
    ax = FakeAX()
    ax.trusted_now = False
    reader = M.MacReader(ax, secure_input=lambda: False)
    for _ in range(20):
        assert reader.read() is None
    assert ax.prompts == 0 and ax.text_reads == 0
    assert reader.gate.request() and not reader.gate.request() and ax.prompts == 1


def test_macos_reader_status_messages(monkeypatch):
    monkeypatch.setattr(M, "_reader", None)
    monkeypatch.setattr(M, "_error", M.NO_BINDING)
    reader = M.MacFocusedTextReader()
    assert asyncio.run(reader.read()) is None
    assert reader.status().state == "unavailable" and "pyobjc" in reader.status().message
    assert not reader.request_permission()
    ax = FakeAX()
    ax.trusted_now = False
    monkeypatch.setattr(M, "_reader", M.MacReader(ax, secure_input=lambda: False))
    monkeypatch.setattr(M, "_error", "")
    status = reader.status()
    assert status.state == "needs_permission" and "Accessibility" in status.message
    assert status.settings_url.startswith("x-apple.systempreferences:")
    assert reader.request_permission() and not reader.request_permission() and ax.prompts == 1


def test_macos_front_window_owner_skips_menu_bar_and_overlays():
    windows = [
        {"kCGWindowLayer": 25, "kCGWindowOwnerName": "Control Centre"},
        {"kCGWindowLayer": 0, "kCGWindowOwnerName": "Safari"},
        {"kCGWindowLayer": 0, "kCGWindowOwnerName": "Finder"},
    ]
    assert M.front_window_owner(windows) == "Safari"
    assert M.front_window_owner([]) == ""


# ── application names ──


def test_bundle_display_name_from_an_executable_path():
    import plistlib
    from pathlib import Path

    from pilot.system.app_names import bundle_display_name

    plists = {
        Path("/Applications/Visual Studio Code.app/Contents/Info.plist"): {
            "CFBundleName": "Code",
            "CFBundleDisplayName": "Visual Studio Code",
        },
        Path("/Applications/Slack.app/Contents/Info.plist"): {"CFBundleName": "Slack"},
    }

    def read(path: Path) -> bytes:
        return plistlib.dumps(plists[path])

    assert (
        bundle_display_name("/Applications/Visual Studio Code.app/Contents/MacOS/Electron", read)
        == "Visual Studio Code"
    )
    assert bundle_display_name("/Applications/Slack.app/Contents/MacOS/Slack", read) == "Slack"
    assert bundle_display_name("/usr/bin/python3", read) == ""
    assert bundle_display_name("/Applications/Gone.app/Contents/MacOS/Gone", read) == ""


def test_desktop_entry_name_matches_file_exec_and_wm_class():
    from pilot.system.app_names import desktop_entry_name, parse_desktop_entry

    entries = [
        (
            "org.gnome.Nautilus",
            "[Desktop Entry]\nType=Application\nName=Files\nName[de]=Dateien\nExec=nautilus --new-window %U\n",
        ),
        (
            "code",
            "[Desktop Entry]\nName=Visual Studio Code\nExec=/usr/share/code/code --unity-launch %F\nStartupWMClass=Code\n",
        ),
        (
            "org.gnome.TextEditor",
            "[Desktop Entry]\nName=Text Editor\nExec=gnome-text-editor %U\n[Desktop Action new]\nName=New Window\n",
        ),
        ("hidden-tool", "[Desktop Entry]\nName=Hidden Helper\nExec=env FOO=1 tool\nNoDisplay=true\n"),
        ("link", "[Desktop Entry]\nType=Link\nName=Website\nURL=https://example.com\n"),
    ]
    assert desktop_entry_name("nautilus", entries) == "Files"
    assert desktop_entry_name("code", entries) == "Visual Studio Code"
    assert desktop_entry_name("gnome-text-editor", entries) == "Text Editor"
    assert desktop_entry_name("tool", entries) == "Hidden Helper"
    assert desktop_entry_name("unknown", entries) == ""
    assert parse_desktop_entry(entries[2][1])["Name"] == "Text Editor"  # a later group does not overwrite it


# ── macOS permissions ──


def test_macos_permission_states_and_messages():
    from pilot.system import macos_permissions as mp

    assert mp.av_status(0) == mp.PermissionState.NOT_ASKED
    assert mp.av_status(2) == mp.PermissionState.DENIED and mp.av_status(3) == mp.PermissionState.GRANTED
    assert mp.av_status(9) == mp.PermissionState.CANNOT_TELL
    assert mp.ae_status(0) == mp.PermissionState.GRANTED and mp.ae_status(-1743) == mp.PermissionState.DENIED
    assert mp.ae_status(-1744) == mp.PermissionState.NOT_ASKED and mp.ae_status(-600) == mp.PermissionState.CANNOT_TELL
    assert mp.trusted_status(None) == mp.PermissionState.CANNOT_TELL
    assert mp.trusted_status(True) == mp.PermissionState.GRANTED
    denied = mp.report("microphone", mp.PermissionState.DENIED)
    assert "Microphone was turned off" in denied.message and "Heliox-OS" in denied.message
    assert denied.settings_url.endswith("Privacy_Microphone")
    assert mp.report("accessibility", mp.PermissionState.GRANTED).message == ""
    assert set(denied.as_dict()) == {"name", "state", "label", "message", "settings_url"}


@pytest.mark.skipif(__import__("sys").platform == "darwin", reason="the real checks run on macOS")
def test_macos_permissions_are_inert_elsewhere():
    from pilot.system import macos_permissions as mp

    assert mp.snapshot() == []
    assert mp.microphone_state() == mp.PermissionState.CANNOT_TELL
    assert mp.accessibility_state() == mp.PermissionState.CANNOT_TELL


def test_macos_launch_services_front_app_parsing():
    assert M.parse_lsappinfo_front("ASN:0x0-0x2c02c:\n") == "ASN:0x0-0x2c02c:"
    assert M.parse_lsappinfo_front("") == "" and M.parse_lsappinfo_front("error") == ""
    assert M.parse_lsappinfo_name('"LSDisplayName"="Safari"\n') == "Safari"
    assert M.parse_lsappinfo_name('"LSDisplayName"=[ NULL ]') == ""
    assert M.parse_lsappinfo_name("") == ""
    outputs = {"front": "ASN:0x0-0x1:\n", "info": '"LSDisplayName"="Visual Studio Code"\n'}
    assert M.launch_services_front_app(lambda args: outputs[args[1]]) == "Visual Studio Code"
    assert M.launch_services_front_app(lambda args: "") == ""

    def broken(args):
        raise OSError("no lsappinfo")

    assert M.launch_services_front_app(broken) == ""


def test_macos_window_list_skips_transparent_windows():
    windows = [
        {"kCGWindowLayer": 0, "kCGWindowOwnerName": "Overlay", "kCGWindowAlpha": 0.0},
        {"kCGWindowLayer": 0, "kCGWindowOwnerName": "Mail", "kCGWindowAlpha": 1.0},
    ]
    assert M.front_window_owner(windows) == "Mail"


def test_vision_ocr_lines_come_out_in_reading_order():
    from pilot.system.macos_vision_ocr import reading_order

    # Bottom-left origin: a larger y is higher on the page.
    boxes = [
        ("world", 0.5, 0.800, 0.05),
        ("second line", 0.1, 0.600, 0.05),
        ("Hello", 0.1, 0.805, 0.05),
        ("", 0.1, 0.2, 0.05),
    ]
    assert reading_order(boxes) == ["Hello world", "second line"]
    assert reading_order([]) == []


def _swift_route(monkeypatch, tmp_path, compile_result):
    """vision with pyobjc missing, a fresh helper state, the helper folder in tmp_path and swiftc faked."""
    import sys
    import threading

    from pilot.system import vision

    monkeypatch.setitem(sys.modules, "pilot.system.macos_vision_ocr", None)  # import fails like a missing extra
    monkeypatch.setattr(vision, "_swift_state", vision._SwiftHelperState())
    monkeypatch.setattr(vision, "_swift_helper_dir", lambda: tmp_path / "vision-ocr")
    monkeypatch.setattr(vision.shutil, "which", lambda name: "/usr/bin/swiftc" if name == "swiftc" else None)
    compiles = []
    release = threading.Event()

    def compile_helper(source, output):
        compiles.append(source.read_text(encoding="utf-8") == vision.SWIFT_OCR_SOURCE)
        release.wait(5)
        code, error = compile_result
        if code == 0:
            output.write_bytes(b"helper")
        return code, error

    monkeypatch.setattr(vision, "_compile_swift_helper", compile_helper)
    return vision, compiles, release


def test_macos_ocr_without_pyobjc_compiles_the_swift_helper_once_in_the_background(monkeypatch, tmp_path):
    import asyncio

    import pytest

    vision, compiles, release = _swift_route(monkeypatch, tmp_path, (0, ""))
    runs = []

    async def run(args, **kw):
        runs.append(kw.get("timeout"))
        return (0, "from swift\n", "")

    monkeypatch.setattr(vision, "run_command", run)

    async def two_reads_at_once():
        return await asyncio.gather(
            vision._ocr_macos_native(b"png"), vision._ocr_macos_native(b"png"), return_exceptions=True
        )

    # The read that finds no helper does not wait for the compile: it says a one-time preparation is under way,
    # and a second read at the same moment neither waits nor starts another compile.
    first, second = asyncio.run(two_reads_at_once())
    assert isinstance(first, RuntimeError) and "one-time step" in str(first)
    assert isinstance(second, RuntimeError) and "one-time step" in str(second)
    release.set()
    vision._swift_state.thread.join(5)
    assert compiles == [True]
    assert asyncio.run(vision._ocr_macos_native(b"png")) == "from swift"
    assert asyncio.run(vision._ocr_macos_native(b"png")) == "from swift"
    assert compiles == [True] and runs == [vision.SWIFT_OCR_RUN_TIMEOUT] * 2
    folder = tmp_path / "vision-ocr"
    assert [p.name for p in folder.iterdir() if p.suffix in (".swift", ".part")] == []
    if __import__("os").name != "nt":
        assert folder.stat().st_mode & 0o777 == 0o700


def test_a_failed_swift_compile_is_remembered_and_never_retried_in_the_same_run(monkeypatch, tmp_path):
    import asyncio

    import pytest

    vision, compiles, release = _swift_route(monkeypatch, tmp_path, (-1, "timed out"))
    release.set()
    with pytest.raises(RuntimeError, match="one-time step"):
        asyncio.run(vision._ocr_macos_native(b"png"))
    vision._swift_state.thread.join(5)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="restart Heliox to try again"):
            asyncio.run(vision._ocr_macos_native(b"png"))
    assert compiles == [True]

    # A quick failure (no licence accepted, no SDK) is remembered the same way, with the compiler's last line.
    vision, compiles, release = _swift_route(monkeypatch, tmp_path, (1, "error: unable to find sdk 'macosx'"))
    release.set()
    with pytest.raises(RuntimeError, match="one-time step"):
        asyncio.run(vision._ocr_macos_native(b"png"))
    vision._swift_state.thread.join(5)
    with pytest.raises(RuntimeError, match=r"unable to find sdk.*xcode-select --install"):
        asyncio.run(vision._ocr_macos_native(b"png"))
    assert compiles == [True]


def test_macos_ocr_without_pyobjc_or_developer_tools_says_what_to_install(monkeypatch, tmp_path):
    import asyncio
    import sys

    import pytest

    from pilot.system import vision

    monkeypatch.setitem(sys.modules, "pilot.system.macos_vision_ocr", None)
    monkeypatch.setattr(vision, "_swift_state", vision._SwiftHelperState())
    monkeypatch.setattr(vision, "_swift_helper_dir", lambda: tmp_path)
    monkeypatch.setattr(vision.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"pilot-daemon\[macos\].*xcode-select --install"):
        asyncio.run(vision._ocr_macos_native(b"png"))


def test_macos_screen_ocr_says_screen_recording_is_missing(monkeypatch):
    import asyncio

    from pilot.system import macos_permissions as mp
    from pilot.system import vision
    from pilot.system.platform_detect import Platform

    async def capture(_region):
        return b"wallpaper"

    async def must_not_ocr(_img):
        raise AssertionError("the wallpaper must not be read")

    monkeypatch.setattr(vision, "CURRENT_PLATFORM", Platform.MACOS)
    monkeypatch.setattr(vision, "_capture_screenshot_bytes", capture)
    monkeypatch.setattr(vision, "_ocr_macos_native", must_not_ocr)
    monkeypatch.setattr(mp, "screen_recording_state", lambda: mp.PermissionState.NOT_ASKED)
    answer = asyncio.run(vision.screen_ocr())
    assert answer.startswith("[OCR unavailable:") and "Screen & System Audio Recording" in answer
    with pytest.raises(vision.OCRUnavailableError, match="Screen"):
        asyncio.run(vision.screen_ocr(allow_slow_fallback=False))
    monkeypatch.setattr(mp, "screen_recording_state", lambda: mp.PermissionState.CANNOT_TELL)

    async def ocr(_img):
        return "text"

    monkeypatch.setattr(vision, "_ocr_macos_native", ocr)
    assert asyncio.run(vision.screen_ocr()) == "text"


def test_macos_frontmost_app_is_cached_briefly(monkeypatch):
    calls = []

    def front(run=None):
        calls.append(1)
        return "Mail"

    clock = [100.0]
    monkeypatch.setattr(M, "launch_services_front_app", front)
    monkeypatch.setattr(M, "_front_cache", [-1e9, ""])
    monkeypatch.setattr(M.time, "monotonic", lambda: clock[0])
    assert M.frontmost_app_name() == "Mail" and M.frontmost_app_name() == "Mail" and len(calls) == 1
    clock[0] += M.FRONT_APP_CACHE_SECONDS + 0.01
    assert M.frontmost_app_name() == "Mail" and len(calls) == 2


# ── whether a field's window still exists (Live drops a card when it has gone) ──


def test_window_handles_for_macos_and_linux_fields_resolve_through_their_readers(monkeypatch):
    from pilot.live import focused_text_linux, focused_text_macos

    assert focused_text.window_handle("42.30540886") == 30540886  # Windows unchanged
    mac = focused_text.window_handle("ax:501:1f")
    linux = focused_text.window_handle("atspi::1.42/org/a11y/atspi/accessible/4")
    assert isinstance(mac, int) and isinstance(linux, int) and mac != linux
    assert focused_text.window_handle("ax:501:1f") == mac  # stable for the same field
    answers = {"mac": None, "linux": True}
    monkeypatch.setattr(focused_text_macos, "field_exists", lambda field: answers["mac"])
    monkeypatch.setattr(focused_text_linux, "field_exists", lambda field: answers["linux"])
    assert focused_text.window_exists(mac) is True  # cannot tell: the card stays
    assert focused_text.window_exists(linux) is True
    answers.update(mac=False, linux=False)
    assert focused_text.window_exists(mac) is False and focused_text.window_exists(linux) is False


def test_macos_field_exists_from_the_element_and_its_process(monkeypatch):
    assert M.element_answer(0) is True and M.element_answer(M.AX_ERROR_INVALID_ELEMENT) is False
    assert M.element_answer(M.AX_ERROR_CANNOT_COMPLETE) is None

    class Backend:
        valid = True
        window: bool | None = True

        def element_valid(self, element):
            return self.valid

        def window_open(self, element):
            return self.window

    backend = Backend()
    monkeypatch.setattr(M, "_reader", type("R", (), {"backend": backend})())
    monkeypatch.setattr(M, "_ELEMENTS", {})
    monkeypatch.setattr(M, "pid_alive", lambda pid: pid == 501)
    M.remember_element("ax:501:1f", object())
    assert M.field_exists("ax:501:1f") is True
    backend.window = False  # the window closed but the app keeps its text field alive (seen on a real Mac in CI)
    assert M.field_exists("ax:501:1f") is False
    backend.window = None  # the app's window list cannot be read: cannot tell, the card stays
    assert M.field_exists("ax:501:1f") is None
    backend.valid, backend.window = False, True  # the element itself is gone
    assert M.field_exists("ax:501:1f") is False
    assert M.field_exists("ax:777:2a") is False  # the app quit
    assert M.field_exists("ax:501:99") is None  # never read here: cannot tell


def test_macos_window_open_answer_needs_the_window_in_its_apps_list():
    assert M.window_open_answer(0, True, True) is True
    assert M.window_open_answer(0, True, False) is False
    assert M.window_open_answer(0, True, None) is None
    assert M.window_open_answer(M.AX_ERROR_INVALID_ELEMENT, False, None) is False
    assert M.window_open_answer(M.AX_ERROR_CANNOT_COMPLETE, False, None) is None
    assert M.window_open_answer(0, False, None) is None  # a field with no window attribute: cannot tell


def test_linux_field_exists_from_the_accessibility_bus(monkeypatch):
    assert L.ref_from_field("atspi::1.42/org/a11y/atspi/accessible/4") == (":1.42", "/org/a11y/atspi/accessible/4")
    assert L.ref_from_field("ax:1:2") is None
    assert L.gone_from_error("[org.freedesktop.DBus.Error.ServiceUnknown] The name :1.42 was not provided")
    assert not L.gone_from_error("TimeoutError")
    shared = L._Shared()
    bus = FakeBus()
    shared.make = lambda: bus
    monkeypatch.setattr(L, "SHARED", shared)
    field = f"atspi:{APP}/entry"
    assert L.field_exists(field) is True
    bus.nodes["/entry"]["state"] = [L.STATE_DEFUNCT]
    assert L.field_exists(field) is False
    bus.nodes["/entry"]["state"] = [L.STATE_EDITABLE]
    bus.gone = {"/entry"}  # the window was destroyed
    assert L.field_exists(field) is False
