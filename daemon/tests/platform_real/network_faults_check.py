"""Real network faults in front of Heliox's real HTTP clients (CI runners: Windows, macOS, Linux).

Local servers on this machine play the provider: one accepts the connection and never answers (a stalled proxy),
one answers 200 with an HTML sign-in page (a hotel Wi-Fi portal), and one answers properly. A name that cannot
resolve plays a DNS failure. Both roads are checked with real sockets and the real clients:

- the planner road: pilot.models.cloud.CloudClient (Claude) — the stated bound is CLOUD_GENERATION_BUDGET_SECONDS
  (35 s); the sentence for each fault; the next request works on the same client, without a restart;
- the conversation road: pilot.conversation.layer.ConversationLayer with the real HttpFastModel (Claude) — the
  stated bound is FIRST_TEXT_DEADLINE (4.5 s) with a holding phrase at 2 s, then a fallback sentence.

What this cannot show: a real network link going down (the runner's network stays up), a real captive portal's
redirects, and a real provider's own error bodies. Prints CHECK lines and the measured times; exit code 1 if one
fails.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

MARGIN = 5.0  # seconds over a stated bound that still pass on a slow runner
OK_BODY = {"content": [{"type": "text", "text": "a plan"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3}}
PORTAL = b"<html><body><h1>Sign in to the hotel Wi-Fi</h1></body></html>"


async def _serve(kind: str) -> tuple[asyncio.AbstractServer, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            if kind == "stall":
                await asyncio.sleep(3600)  # accepted, never answered
                return
            if kind == "portal":
                body, kind_header = PORTAL, b"text/html"
            elif kind == "stream":
                body = (
                    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta",'
                    b'"text":"Hello there. "}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
                )
                kind_header = b"text/event-stream"
            else:
                body, kind_header = json.dumps(OK_BODY).encode(), b"application/json"
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: "
                + kind_header
                + b"\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
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


async def planner_road(ports: dict[str, int]) -> list[tuple[bool, str]]:
    import pilot.models.cloud as cloud
    from pilot.config import PilotConfig

    config = PilotConfig()
    config.model.cloud_provider = "claude"
    client = cloud.CloudClient(config, _Vault())
    results = []
    cases = [
        ("stall", f"http://127.0.0.1:{ports['stall']}/v1/messages", "timed out", cloud.CLOUD_GENERATION_BUDGET_SECONDS),
        ("portal", f"http://127.0.0.1:{ports['portal']}/v1/messages", "not from the API (text/html)", 15.0),
        ("dns", "http://heliox-check.invalid/v1/messages", "could not connect", 15.0),
    ]
    original = cloud.PROVIDER_ENDPOINTS["claude"]
    try:
        for name, url, words, bound in cases:
            cloud.PROVIDER_ENDPOINTS["claude"] = url
            started = time.monotonic()
            try:
                await asyncio.wait_for(client.generate("plan this"), bound + 60)
                message, took = "answered (it should not have)", time.monotonic() - started
            except Exception as error:
                message, took = str(error), time.monotonic() - started
            ok = words in message and took <= bound + MARGIN
            results.append(
                (
                    ok,
                    f"CHECK network/planner/{name}: {'PASS' if ok else 'FAIL'} - {took:.1f} s (stated bound {bound:g} s): {message[:200]!r}",
                )
            )
        cloud.PROVIDER_ENDPOINTS["claude"] = f"http://127.0.0.1:{ports['ok']}/v1/messages"
        try:
            answer = await asyncio.wait_for(client.generate("plan this"), 60)
        except Exception as error:
            answer = f"failed: {error}"
        ok = answer == "a plan"
        results.append(
            (
                ok,
                f"CHECK network/planner/next-request-works: {'PASS' if ok else 'FAIL'} - the same client, no restart: {answer!r}",
            )
        )
    finally:
        cloud.PROVIDER_ENDPOINTS["claude"] = original
    return results


async def conversation_road(ports: dict[str, int]) -> list[tuple[bool, str]]:
    import httpx

    import pilot.models.cloud as cloud
    from pilot.conversation.fast_model import HttpFastModel
    from pilot.conversation.layer import FALLBACK, FIRST_TEXT_DEADLINE, HOLD_PHRASES, ConversationLayer

    async def key():
        return "sk-check"

    results = []
    original = cloud.PROVIDER_ENDPOINTS["claude"]
    try:
        for name, url in (
            ("stall", f"http://127.0.0.1:{ports['stall']}/v1/messages"),
            ("portal", f"http://127.0.0.1:{ports['portal']}/v1/messages"),
            ("dns", "http://heliox-check.invalid/v1/messages"),
        ):
            cloud.PROVIDER_ENDPOINTS["claude"] = url
            said: list[tuple[float, str]] = []
            started = time.monotonic()

            async def sink(text, channel, key_, said=said, started=started):
                said.append((time.monotonic() - started, text))
                return "spoken"

            model = HttpFastModel(
                provider="claude", name="claude-haiku-4-5", client_factory=httpx.AsyncClient, api_key=key
            )
            layer = ConversationLayer(sink=sink, model=lambda model=model: model)
            await asyncio.wait_for(layer.take("What do you think?"), 120)
            took = time.monotonic() - started
            texts = [text for _at, text in said]
            fell_back = any(text in FALLBACK.values() for text in texts)
            held = [at for at, text in said if text in HOLD_PHRASES]
            bound = FIRST_TEXT_DEADLINE if name == "stall" else 15.0
            ok = fell_back and took <= bound + MARGIN
            hold_note = f", holding phrase at {held[0]:.1f} s" if held else ""
            results.append(
                (
                    ok,
                    f"CHECK network/conversation/{name}: {'PASS' if ok else 'FAIL'} - {took:.1f} s (stated bound "
                    f"{bound:g} s){hold_note}; said: {texts!r}",
                )
            )
    finally:
        cloud.PROVIDER_ENDPOINTS["claude"] = original
    return results


async def run() -> int:
    servers = {kind: await _serve(kind) for kind in ("stall", "portal", "ok")}
    ports = {kind: port for kind, (_server, port) in servers.items()}
    try:
        results = await planner_road(ports) + await conversation_road(ports)
    finally:
        for server, _port in servers.values():
            server.close()
    for _ok, line in results:
        print(line)
    return 0 if all(ok for ok, _line in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
