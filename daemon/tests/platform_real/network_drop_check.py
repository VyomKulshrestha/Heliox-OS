"""The network really goes away in the middle of a request, in front of Heliox's real HTTP clients (CI: ubuntu).

network_faults_check.py covers a stalled server, a sign-in page and a name that does not resolve; there the runner's
network stays up. Here the kernel drops the packets (iptables, as a pulled cable or a Wi-Fi that went away does:
nothing answers, nothing says no), and then lets them through again:

- the planner road (pilot.models.cloud.CloudClient, Claude): a request in flight when the packets stop must give up
  within CLOUD_GENERATION_BUDGET_SECONDS (35 s) with its sentence; a request started while they are dropped too; the
  next request after the network is back must work on the same client, whose pooled connection died meanwhile;
- the conversation road (ConversationLayer with the real HttpFastModel): the holding phrase at 2 s and the fallback
  sentence by FIRST_TEXT_DEADLINE (4.5 s) while the packets are dropped; the next turn after they are back answers.

A local server plays the provider and keeps its connections open (keep-alive), so the client's pool holds a
connection the drop kills. The rules are removed in a finally path, whatever happened. Linux only (iptables with
sudo on the runner); off a runner it changes nothing (ci_only). Prints CHECK lines; exit code 1 if one fails.

What this cannot show: a real Wi-Fi or cable going down (the routes and the interface stay up here), DNS answers
cached across the drop, macOS (pfctl) and Windows (the firewall).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from ci_only import require_runner

require_runner("adds and removes firewall rules on this machine")
if not sys.platform.startswith("linux"):
    print("NOT RUN: the packet drop uses iptables, on Linux only")
    sys.exit(3)

MARGIN = 5.0
DELAY = 2.0  # the provider takes this long to answer, so a request is in flight when the packets stop
OK_BODY = {"content": [{"type": "text", "text": "a plan"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3}}
STREAM = (
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta",'
    b'"text":"Hello there. "}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
)
rules: list[list[str]] = []


def drop(port: int) -> None:
    """Drop every packet to and from the provider's port, as a network that went away: no reset, no answer."""
    for rule in (["OUTPUT", "-p", "tcp", "--dport", str(port)], ["OUTPUT", "-p", "tcp", "--sport", str(port)]):
        subprocess.run(["sudo", "iptables", "-I", *rule, "-j", "DROP"], check=True)
        rules.append(rule)


def restore() -> None:
    while rules:
        rule = rules.pop()
        subprocess.run(["sudo", "iptables", "-D", *rule, "-j", "DROP"], check=False)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_daemon  # noqa: E402

real_daemon.time_limit("network-drop", 420, restore)  # no check may hang; the packets flow again first


async def serve() -> tuple[asyncio.AbstractServer, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:  # keep-alive: the client's pool keeps this connection
                head = await reader.readuntil(b"\r\n\r\n")
                length = next(
                    (
                        int(line.split(b":", 1)[1])
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length:")
                    ),
                    0,
                )
                body_in = await reader.readexactly(length) if length else b""
                await asyncio.sleep(DELAY)
                streaming = b'"stream": true' in body_in or b'"stream":true' in body_in
                body = STREAM if streaming else json.dumps(OK_BODY).encode()
                kind = b"text/event-stream" if streaming else b"application/json"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: " + kind + b"\r\nContent-Length: " + str(len(body)).encode()
                    + b"\r\nConnection: keep-alive\r\n\r\n" + body
                )  # fmt: skip
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


class _Vault:
    async def get_key(self, provider: str):
        return "sk-check" if provider == "claude" else None


async def planner_road(port: int, results: list[tuple[bool, str]]) -> None:
    import pilot.models.cloud as cloud
    from pilot.config import PilotConfig

    config = PilotConfig()
    config.model.cloud_provider = "claude"
    client = cloud.CloudClient(config, _Vault())
    bound = cloud.CLOUD_GENERATION_BUDGET_SECONDS
    cloud.PROVIDER_ENDPOINTS["claude"] = f"http://127.0.0.1:{port}/v1/messages"
    first = await asyncio.wait_for(client.generate("plan this"), 60)  # the pool now holds a live connection
    results.append(
        (first == "a plan", f"CHECK network-drop/planner/before: {'PASS' if first == 'a plan' else 'FAIL'} - {first!r}")
    )

    async def timed(label: str) -> None:
        started = time.monotonic()
        try:
            answer = await asyncio.wait_for(client.generate("plan this"), bound + 60)
            message = f"answered {answer!r} (it should not have)"
            ok = False
        except Exception as error:
            message = str(error)
            ok = time.monotonic() - started <= bound + MARGIN and "Claude API unavailable" in message
        took = time.monotonic() - started
        results.append(
            (ok, f"CHECK network-drop/planner/{label}: {'PASS' if ok else 'FAIL'} - {took:.1f} s (stated bound {bound:g} s): {message[:200]!r}")
        )  # fmt: skip

    in_flight = asyncio.create_task(timed("in-flight"))
    await asyncio.sleep(DELAY / 4)  # the request is out; its answer is not
    drop(port)
    await in_flight
    await timed("started-while-down")
    restore()
    started = time.monotonic()
    try:
        answer = await asyncio.wait_for(client.generate("plan this"), 60)
    except Exception as error:
        answer = f"failed: {error}"
    ok = answer == "a plan"
    results.append(
        (ok, f"CHECK network-drop/planner/after-it-is-back: {'PASS' if ok else 'FAIL'} - the same client, {time.monotonic() - started:.1f} s: {answer!r}")
    )  # fmt: skip


async def conversation_road(port: int, results: list[tuple[bool, str]]) -> None:
    import httpx

    import pilot.models.cloud as cloud
    from pilot.conversation.fast_model import HttpFastModel
    from pilot.conversation.layer import FALLBACK, FIRST_TEXT_DEADLINE, HOLD_PHRASES, ConversationLayer

    async def key():
        return "sk-check"

    cloud.PROVIDER_ENDPOINTS["claude"] = f"http://127.0.0.1:{port}/v1/messages"
    model = HttpFastModel(provider="claude", name="claude-haiku-4-5", client_factory=httpx.AsyncClient, api_key=key)

    async def turn(label: str, *, expect_answer: bool) -> None:
        said: list[tuple[float, str]] = []
        started = time.monotonic()

        async def sink(text, channel, key_, said=said, started=started):
            said.append((time.monotonic() - started, text))
            return "spoken"

        layer = ConversationLayer(sink=sink, model=lambda: model)
        await asyncio.wait_for(layer.take("What do you think?"), 120)
        took = time.monotonic() - started
        texts = [text for _at, text in said]
        fell_back = any(text in FALLBACK.values() for text in texts)
        held = [at for at, text in said if text in HOLD_PHRASES]
        if expect_answer:
            ok = any("Hello there" in text for text in texts)
        else:
            ok = fell_back and took <= FIRST_TEXT_DEADLINE + MARGIN
        hold = f", holding phrase at {held[0]:.1f} s" if held else ""
        results.append(
            (ok, f"CHECK network-drop/conversation/{label}: {'PASS' if ok else 'FAIL'} - {took:.1f} s{hold}; said: {texts!r}")
        )  # fmt: skip

    await turn("before", expect_answer=True)
    drop(port)
    await turn("while-down", expect_answer=False)
    restore()
    await turn("after-it-is-back", expect_answer=True)


async def run() -> int:
    import pilot.models.cloud as cloud

    original = cloud.PROVIDER_ENDPOINTS["claude"]
    server, port = await serve()
    results: list[tuple[bool, str]] = []
    try:
        await planner_road(port, results)
        await conversation_road(port, results)
    finally:
        restore()
        cloud.PROVIDER_ENDPOINTS["claude"] = original
        server.close()
    for _ok, line in results:
        print(line, flush=True)
    return 0 if results and all(ok for ok, _line in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run()))
    finally:
        restore()
