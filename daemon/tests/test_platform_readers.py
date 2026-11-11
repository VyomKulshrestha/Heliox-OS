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
            "/term": {"role": "terminal", "state": [L.STATE_SHOWING], "text": "$ make", "parent": "/win"},
        }
        self.focus = "/entry"
        self.pending: list[L.Ref] = []
        self.calls: list[tuple[str, str]] = []

    def call(self, bus, path, interface, method, signature="", body=()):
        self.calls.append((path, method))
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
