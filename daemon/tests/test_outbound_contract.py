"""Only pilot.system.outbound opens a connection for a plan step.

Every module under daemon/pilot is read (not imported). A call that builds an HTTP client or opens a connection, or
that starts a download program (curl, wget, certutil, bitsadmin, PowerShell's web cmdlets), is allowed only in the
places listed here, each with the reason it is not a plan step's request. A new place fails this test until it goes
through pilot.system.outbound.fetch or is added here with its reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1] / "pilot"

# Calls that build an HTTP client or open a connection, by the last parts of their dotted name.
CONNECTORS = {
    "httpx.AsyncClient",
    "httpx.Client",
    "httpx.get",
    "httpx.post",
    "httpx.put",
    "httpx.patch",
    "httpx.delete",
    "httpx.head",
    "httpx.request",
    "httpx.stream",
    "AsyncClient",
    "create_httpx_client",
    "urlopen",
    "urlretrieve",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.patch",
    "requests.delete",
    "requests.head",
    "requests.request",
    "requests.Session",
    "aiohttp.ClientSession",
    "ClientSession",
    "websockets.connect",
    "socket.create_connection",
    "asyncio.open_connection",
    "HTTPConnection",
    "HTTPSConnection",
    "urllib3.PoolManager",
    "PoolManager",
    "caldav.DAVClient",
    "DAVClient",
}
DOWNLOAD_PROGRAMS = {"curl", "wget", "certutil", "bitsadmin", "aria2c"}
DOWNLOAD_CMDLETS = ("invoke-webrequest", "invoke-restmethod", "start-bitstransfer", "net.webclient", "downloadfile(")

# (module, what it calls) -> why it is not a plan step's request.
ALLOWED = {
    ("system/outbound.py", "create_httpx_client"): "the checked path itself",
    ("system/http_client.py", "httpx.AsyncClient"): "the client factory the checked path and the model providers use",
    ("models/cloud.py", "create_httpx_client"): "the model provider the person configured",
    ("models/ollama.py", "create_httpx_client"): "the local model server the person configured",
    ("conversation/fast_model.py", "create_httpx_client"): "the fast model provider the person configured",
    ("live/grounded.py", "httpx.AsyncClient"): "the local Ollama model at the configured base address",
    ("system/vision.py", "create_httpx_client"): "the vision model provider or local Ollama the person configured",
    ("system/speech_models.py", "urllib.request.urlopen"): "fixed https addresses from the speech model catalogue",
    ("system/gesture.py", "urllib.request.urlretrieve"): "one fixed https address of the hand model",
    ("system/voice_fetch.py", "socket.create_connection"): "a reachability probe of the fixed model hub host",
    ("mcp_server.py", "websockets.connect"): "the MCP bridge to this daemon's own local socket",
    ("network/peer_connection.py", "websockets.connect"): "a device the person paired",
    ("plugins/marketplace.py", "urlopen"): "the marketplace registry from settings, read when the person browses it",
    ("agents/calendar_agent.py", "caldav.DAVClient"): "the CalDAV account the person configured",
    ("system/calendar_readback.py", "caldav.DAVClient"): "the CalDAV account the person configured",
    # Plugin code runs in the plugin runner (plugins/native_runner.py), which lets it reach only the domains its
    # manifest declares, and every plugin_call needs the person's approval.
    ("plugins/developer/developer_plugin.py", "urlopen"): "plugin code, limited to its declared domains",
    ("plugins/homeassistant/homeassistant_plugin.py", "urlopen"): "plugin code, limited to its declared domains",
    ("plugins/media/media_plugin.py", "urlopen"): "plugin code, limited to its declared domains",
    ("plugins/marketplace_catalog/home-assistant/plugin.py", "urllib.request.urlopen"): "plugin code, as above",
    ("plugins/marketplace_catalog/spotify-control/plugin.py", "urllib.request.urlopen"): "plugin code, as above",
    ("plugins/marketplace_catalog/weather/plugin.py", "urllib.request.urlopen"): "plugin code, as above",
}
# Modules whose text names download programs only to refuse or recognise them, never to run them.
NAMES_ONLY = {
    "security/sanitizer.py": "the shell sanitizer's list of refused programs",
    "conversation/quick.py": "the quick-answer path's list of programs it never runs",
    "live/detectors.py": "a pattern that recognises a piped download on screen",
}


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _connector(name: str) -> bool:
    """A module call by its full name (httpx.AsyncClient), or a bare name under any prefix (urllib.request.urlopen)."""
    return any(name == known or ("." not in known and name.endswith("." + known)) for known in CONNECTORS)


def _modules() -> list[Path]:
    return sorted(path for path in PILOT.rglob("*.py") if "__pycache__" not in path.parts)


def _findings() -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    for path in _modules():
        rel = path.relative_to(PILOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                if name and _connector(name):
                    found.append((rel, name, node.lineno))
                first = node.args[0] if node.args else None
                if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
                    first = first.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    program = first.value.strip().split(" ")[0].lower().removesuffix(".exe")
                    if program in DOWNLOAD_PROGRAMS:
                        found.append((rel, f"runs {program}", node.lineno))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.lower()
                if any(cmdlet in text for cmdlet in DOWNLOAD_CMDLETS) and rel not in NAMES_ONLY:
                    found.append((rel, "PowerShell download cmdlet", node.lineno))
    return found


def test_only_the_checked_module_opens_a_connection_for_a_plan_step() -> None:
    unexpected = [f"{rel}:{line} {name}" for rel, name, line in _findings() if (rel, name) not in ALLOWED]
    assert unexpected == [], (
        "A connection is opened outside pilot.system.outbound. Send it through outbound.fetch, or, if no plan "
        "step can reach it, add it to ALLOWED with the reason: " + ", ".join(unexpected)
    )


def test_every_allowed_place_still_exists() -> None:
    present = {(rel, name) for rel, name, _line in _findings()}
    stale = sorted(f"{rel} {name}" for rel, name in ALLOWED if (rel, name) not in present)
    assert stale == [], f"Remove these from ALLOWED, they no longer open a connection: {stale}"


def test_the_download_action_and_the_senders_use_the_checked_module() -> None:
    for rel in ("system/download.py", "system/api_client.py", "agents/rss_agent.py"):
        text = (PILOT / rel).read_text(encoding="utf-8")
        assert "outbound.fetch(" in text, rel
    for rel in ("agents/postconditions/declared.py", "agents/postconditions/messaging.py"):
        assert "outbound.fetch(" in (PILOT / rel).read_text(encoding="utf-8"), rel


def test_the_contract_catches_a_new_client_and_a_download_program(tmp_path) -> None:
    module = tmp_path / "pilot" / "system" / "new_fetcher.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import httpx\n"
        "from pilot.system.platform_detect import run_command\n"
        "async def go(url):\n"
        "    async with httpx.AsyncClient() as client:\n"
        "        await client.get(url)\n"
        "    await run_command(['curl', '-o', 'x', url])\n"
        "    await run_powershell('Invoke-WebRequest -Uri $u')\n",
        encoding="utf-8",
    )
    globals()["PILOT"] = tmp_path / "pilot"
    try:
        names = {name for _rel, name, _line in _findings()}
    finally:
        globals()["PILOT"] = Path(__file__).resolve().parents[1] / "pilot"
    assert {"httpx.AsyncClient", "runs curl", "PowerShell download cmdlet"} <= names
