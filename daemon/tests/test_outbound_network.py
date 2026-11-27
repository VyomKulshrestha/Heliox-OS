"""Heliox's own HTTP senders reach only public addresses unless the person approved a local one
(pilot.system.outbound).

Fakes only: a fake resolver and a fake transport (tests/fake_outbound.py). No request, name lookup, browser or
download leaves the test.
"""

from __future__ import annotations

import ipaddress
import ssl
from unittest.mock import AsyncMock

import httpx
import pytest

from pilot.actions import (
    Action,
    ActionType,
    ApiRequestParams,
    BrowserParams,
    DownloadParams,
    OpenUrlParams,
    PermissionTier,
)
from pilot.security import addresses
from pilot.system import outbound
from tests import fake_outbound

REFUSAL = "I don't fetch from addresses on this computer or your own network unless you ask for that address yourself."

# ── The address-form table ────────────────────────────────────────────
# (host as written, refused when nobody approved it, why)
ADDRESS_FORMS = [
    ("127.0.0.1", True, "loopback, this computer"),
    ("127.1", True, "short form of 127.0.0.1 that resolvers accept"),
    ("0x7f.1", True, "hex and short form of 127.0.0.1"),
    ("0x7f000001", True, "one hex number for 127.0.0.1"),
    ("2130706433", True, "one decimal number for 127.0.0.1"),
    ("017700000001", True, "one octal number for 127.0.0.1"),
    ("0177.0.0.01", True, "octal parts for 127.0.0.1"),
    ("0.0.0.0", True, "unspecified, connects to this computer"),
    ("0", True, "single number 0, the unspecified address"),
    ("10.1.2.3", True, "private network"),
    ("172.16.0.9", True, "private network"),
    ("192.168.1.1", True, "private network, a home router"),
    ("100.64.0.1", True, "carrier-grade NAT"),
    ("169.254.10.10", True, "IPv4 link-local"),
    ("169.254.169.254", True, "cloud metadata"),
    ("100.100.100.200", True, "Alibaba Cloud metadata"),
    ("192.0.0.192", True, "Oracle Cloud metadata"),
    ("224.0.0.251", True, "multicast"),
    ("255.255.255.255", True, "broadcast, reserved"),
    ("198.18.0.1", True, "benchmarking range used on local networks"),
    ("::1", True, "IPv6 loopback"),
    ("::", True, "IPv6 unspecified"),
    ("fe80::1", True, "IPv6 link-local"),
    ("fe80::1%eth0", True, "IPv6 link-local with a zone"),
    ("fd12:3456::1", True, "IPv6 unique local"),
    ("fec0::1", True, "deprecated site-local"),
    ("fd00:ec2::254", True, "AWS IPv6 metadata"),
    ("::ffff:127.0.0.1", True, "IPv4-mapped loopback"),
    ("::ffff:169.254.169.254", True, "IPv4-mapped metadata"),
    ("::ffff:0:10.0.0.1", True, "IPv4-translated private address"),
    ("::10.0.0.1", True, "IPv4-compatible private address"),
    ("::8.8.8.8", True, "IPv4-compatible form is retired, never public"),
    ("64:ff9b::7f00:1", True, "well-known NAT64 prefix carrying 127.0.0.1"),
    ("64:ff9b::a9fe:a9fe", True, "well-known NAT64 prefix carrying the metadata address"),
    ("64:ff9b:1::a00:1", True, "local-use NAT64 prefix, /96 layout, carrying 10.0.0.1"),
    ("64:ff9b:1:a00:1::", True, "local-use NAT64 prefix, /48 layout, carrying 10.0.1.0"),
    ("2002:c0a8:101::1", True, "6to4 carrying 192.168.1.1"),
    ("2001:0:4136:e378:8000:63bf:f5ff:fffe", True, "Teredo whose client is 10.0.0.1"),
    ("2a00:1450::5efe:a00:1", True, "ISATAP interface carrying 10.0.0.1"),
    ("2001:db8::1", True, "IPv6 documentation"),
    ("localhost", True, "the local host name"),
    ("localhost.", True, "the local host name with the root dot"),
    ("api.localhost", True, "any name under .localhost"),
    ("printer.local", True, "multicast DNS name"),
    ("metadata.google.internal", True, "Google Cloud metadata name"),
    ("nas.home.arpa", True, "home network name"),
    ("router.lan", True, "common home network suffix"),
    ("intranet", True, "a name without a dot only resolves through local search lists"),
    ("%31%32%37.0.0.1", True, "percent-escaped 127.0.0.1"),
    ("１２７.０.０.１", True, "full-width digits for 127.0.0.1"),
    ("127。0。0。1", True, "ideographic dots for 127.0.0.1"),
    ("8.8.8.8", False, "public IPv4"),
    ("1.1.1.1", False, "public IPv4"),
    ("2606:4700:4700::1111", False, "public IPv6"),
    ("::ffff:8.8.8.8", False, "IPv4-mapped public address"),
    ("64:ff9b::808:808", False, "NAT64 carrying a public address: an IPv6-only network reaches public sites"),
    ("64:ff9b:1::808:808", False, "local-use NAT64 carrying a public address"),
    ("2002:808:808::1", False, "6to4 carrying a public address"),
    ("example.com", False, "a public name; its addresses are judged after the one lookup"),
    ("1.2.3.4.5", False, "not an address in any form: a name, judged by its lookup"),
]


@pytest.mark.parametrize(("host", "refused", "why"), ADDRESS_FORMS, ids=[row[0] for row in ADDRESS_FORMS])
def test_every_address_form_gets_its_verdict(host: str, refused: bool, why: str) -> None:
    reason = addresses.host_refusal(host)
    assert (reason is not None) is refused, f"{host}: {why} (got {reason!r})"


@pytest.mark.parametrize(
    ("host", "refused", "why"),
    [row for row in ADDRESS_FORMS if ":" in row[0] and "%" not in row[0]],
)
async def test_a_name_resolving_to_each_ipv6_form_gets_the_same_verdict(monkeypatch, host, refused, why) -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, text="ok")

    fake_outbound.install(monkeypatch, handler=handler, answers={"site.example": [host]})

    async def run() -> None:
        await outbound.fetch("GET", "https://site.example/")

    if refused:
        with pytest.raises(outbound.OutboundRefused):
            await run()
        assert sent == [], why
    else:
        await run()
        assert len(sent) == 1, why


@pytest.mark.parametrize("host", [row[0] for row in ADDRESS_FORMS if row[1] and "%" not in row[0]][:20])
async def test_a_url_with_a_local_address_is_refused_with_the_plain_sentence(monkeypatch, host) -> None:
    calls: list[str] = []
    sent: list[str] = []
    fake_outbound.install(monkeypatch, handler=lambda r: sent.append(str(r.url)) or httpx.Response(200), calls=calls)
    shown = f"[{host}]" if ":" in host else host

    with pytest.raises(ValueError) as refused:
        await outbound.fetch("GET", f"http://{shown}/admin")

    # A spelling the HTTP client cannot read at all is refused as an invalid address; every other one with the
    # sentence and the failure kind.
    if "not a valid address" not in str(refused.value):
        assert isinstance(refused.value, outbound.OutboundRefused)
        assert str(refused.value).startswith(REFUSAL)
        assert refused.value.failure_kind == "local_address_refused"
    assert sent == [] and calls == []


# ── One lookup, and the connection goes to the judged address ───────────


async def test_the_connection_goes_to_the_judged_address_with_the_name_in_host_and_tls(monkeypatch) -> None:
    seen: list[tuple[str, str, str]] = []
    kwargs: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["host"], request.extensions["sni_hostname"]))
        return httpx.Response(200, text="hello")

    fake_outbound.install(monkeypatch, handler=handler, answers={"site.example": ["8.8.4.4"]}, client_kwargs=kwargs)

    response = await outbound.fetch("GET", "https://site.example:8443/page?q=1")

    assert seen == [("https://8.8.4.4:8443/page?q=1", "site.example:8443", "site.example")]
    assert str(response.url) == "https://site.example:8443/page?q=1"
    context = kwargs[0]["verify"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    assert kwargs[0]["trust_env"] is False and kwargs[0]["follow_redirects"] is False


async def test_a_name_whose_second_answer_differs_is_looked_up_once_and_connects_to_the_first(monkeypatch) -> None:
    calls: list[str] = []
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/next"})
        return httpx.Response(200, text="done")

    fake_outbound.install(
        monkeypatch,
        handler=handler,
        answers={"rebind.example": [["8.8.4.4"], ["127.0.0.1"]]},
        calls=calls,
    )

    await outbound.fetch("GET", "http://rebind.example/start")

    assert calls == ["rebind.example"]
    assert seen == ["http://8.8.4.4/start", "http://8.8.4.4/next"]


async def test_a_name_with_one_private_answer_among_public_ones_is_refused(monkeypatch) -> None:
    fake_outbound.install(
        monkeypatch, handler=lambda r: httpx.Response(200), answers={"mixed.example": ["8.8.8.8", "10.0.0.7"]}
    )

    with pytest.raises(outbound.OutboundRefused, match=r"mixed\.example \(10\.0\.0\.7\) is on the local network"):
        await outbound.fetch("GET", "https://mixed.example/")


async def test_the_next_judged_address_is_tried_when_the_first_does_not_connect(monkeypatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "2606:4700:4700::1111":
            raise httpx.ConnectError("unreachable", request=request)
        return httpx.Response(200)

    fake_outbound.install(monkeypatch, handler=handler, answers={"dual.example": ["2606:4700:4700::1111", "1.1.1.1"]})

    await outbound.fetch("GET", "https://dual.example/")

    assert seen == ["2606:4700:4700::1111", "1.1.1.1"]


async def test_an_ipv6_only_network_reaches_public_sites_through_nat64(monkeypatch) -> None:
    seen: list[str] = []
    fake_outbound.install(
        monkeypatch,
        handler=lambda r: seen.append(str(r.url)) or httpx.Response(200),
        answers={"public.example": ["64:ff9b::5db8:d822"], "inside.example": ["64:ff9b::c0a8:101"]},
    )

    await outbound.fetch("GET", "https://public.example/")
    assert seen == ["https://[64:ff9b::5db8:d822]/"]

    with pytest.raises(outbound.OutboundRefused, match=r"192\.168\.1\.1, carried inside an IPv6 address"):
        await outbound.fetch("GET", "https://inside.example/")


async def test_a_nat64_prefix_the_network_announces_is_judged_by_the_address_it_carries(monkeypatch) -> None:
    calls: list[str] = []
    seen: list[str] = []
    fake_outbound.install(
        monkeypatch,
        handler=lambda r: seen.append(str(r.url)) or httpx.Response(200),
        answers={
            "ipv4only.arpa": ["2001:db8:64::c000:aa", "2001:db8:64::c000:ab"],
            "public.example": ["2001:db8:64::808:808"],
            "inside.example": ["2001:db8:64::a00:5"],
        },
        calls=calls,
    )

    # 2001:db8::/32 is a documentation range, refused on its own; inside the announced prefix it is judged by the
    # IPv4 address it carries.
    await outbound.fetch("GET", "https://public.example/")
    with pytest.raises(outbound.OutboundRefused, match=r"10\.0\.0\.5"):
        await outbound.fetch("GET", "https://inside.example/")
    assert calls.count("ipv4only.arpa") == 1


async def test_a_name_that_does_not_resolve_is_not_fetched(monkeypatch) -> None:
    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200), default=[])

    with pytest.raises(ValueError, match="Could not resolve outbound hostname"):
        await outbound.fetch("GET", "https://nowhere.example/")


async def test_a_proxy_connects_by_name_after_the_name_is_judged(monkeypatch) -> None:
    seen: list[httpx.Request] = []
    kwargs: list[dict] = []
    fake_outbound.install(
        monkeypatch,
        handler=lambda r: seen.append(r) or httpx.Response(200),
        answers={"site.example": ["8.8.4.4"], "inside.example": ["10.9.9.9"]},
        client_kwargs=kwargs,
    )
    monkeypatch.setattr(outbound, "proxy_for", lambda _config, _scheme, _host: "http://proxy.example:3128")

    await outbound.fetch("GET", "https://site.example/")
    assert kwargs[0]["proxy"] == "http://proxy.example:3128"
    assert str(seen[0].url) == "https://site.example/"

    with pytest.raises(outbound.OutboundRefused):
        await outbound.fetch("GET", "https://inside.example/")
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("ftp://example.com/file", "must use http:// or https://"),
        ("file:///etc/passwd", "must use http:// or https://"),
        ("http://user:secret@example.com/", "Credentials must be supplied"),
        ("http:///nohost", "must include a hostname"),
    ],
)
async def test_addresses_that_are_not_plain_http_are_refused_before_any_lookup(monkeypatch, url, message) -> None:
    calls: list[str] = []
    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200), calls=calls)

    with pytest.raises(ValueError, match=message):
        await outbound.fetch("GET", url)
    assert calls == []


# ── Redirects, one hop at a time ──────────────────────────────────────


def _chain(routes: dict[str, httpx.Response], seen: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        key = f"{request.headers.get('host', request.url.netloc.decode())}{request.url.path}"
        seen.append(key)
        return routes[key]

    return handler


async def test_a_redirect_from_a_public_site_to_a_local_address_is_not_followed(monkeypatch) -> None:
    seen: list[str] = []
    routes = {"public.example/start": httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})}
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen), answers={"public.example": ["8.8.4.4"]})

    with pytest.raises(outbound.OutboundRefused, match="cloud metadata"):
        await outbound.fetch("GET", "https://public.example/start")
    assert seen == ["public.example/start"]


async def test_a_redirect_to_a_name_that_resolves_locally_is_not_followed(monkeypatch) -> None:
    seen: list[str] = []
    routes = {"public.example/start": httpx.Response(307, headers={"location": "https://sneaky.example/x"})}
    fake_outbound.install(
        monkeypatch,
        handler=_chain(routes, seen),
        answers={"public.example": ["8.8.4.4"], "sneaky.example": ["::ffff:10.0.0.2"]},
    )

    with pytest.raises(outbound.OutboundRefused, match="10.0.0.2"):
        await outbound.fetch("GET", "https://public.example/start")
    assert seen == ["public.example/start"]


async def test_an_approved_local_address_is_followed_only_on_the_same_host(monkeypatch) -> None:
    seen: list[str] = []
    routes = {
        "127.0.0.1:8080/a": httpx.Response(302, headers={"location": "/b"}),
        "127.0.0.1:8080/b": httpx.Response(200, text="local page"),
    }
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen))

    response = await outbound.fetch("GET", "http://127.0.0.1:8080/a", allow_private_network=True)

    assert response.text == "local page"
    assert seen == ["127.0.0.1:8080/a", "127.0.0.1:8080/b"]


@pytest.mark.parametrize(
    "location",
    [
        "http://10.0.0.5/admin",
        "http://127.0.0.1:9000/other-service",
        "http://192.168.1.1/",
        "https://public.example/",
    ],
)
async def test_an_approved_local_address_never_sends_the_request_on_to_another_host(monkeypatch, location) -> None:
    seen: list[str] = []
    routes = {"127.0.0.1:8080/a": httpx.Response(302, headers={"location": location})}
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen))

    with pytest.raises(outbound.OutboundRefused, match="which you did not approve, so I did not follow it"):
        await outbound.fetch("GET", "http://127.0.0.1:8080/a", allow_private_network=True)
    assert seen == ["127.0.0.1:8080/a"]


async def test_a_redirect_loop_stops(monkeypatch) -> None:
    seen: list[str] = []
    routes = {
        "public.example/a": httpx.Response(302, headers={"location": "/b"}),
        "public.example/b": httpx.Response(302, headers={"location": "/a"}),
    }
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen))

    with pytest.raises(ValueError, match="redirect loop"):
        await outbound.fetch("GET", "https://public.example/a")
    assert seen == ["public.example/a", "public.example/b"]


async def test_a_long_redirect_chain_stops_after_five_hops(monkeypatch) -> None:
    seen: list[str] = []
    routes = {f"public.example/{n}": httpx.Response(302, headers={"location": f"/{n + 1}"}) for n in range(10)}
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen))

    with pytest.raises(ValueError, match="exceeded 5 redirects"):
        await outbound.fetch("GET", "https://public.example/0")
    assert len(seen) == 6


async def test_a_redirect_from_https_to_http_is_not_followed(monkeypatch) -> None:
    seen: list[str] = []
    routes = {"public.example/a": httpx.Response(301, headers={"location": "http://public.example/a"})}
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen))

    with pytest.raises(ValueError, match=r"from a secure \(https\) address to an insecure \(http\) one"):
        await outbound.fetch("GET", "https://public.example/a")
    assert seen == ["public.example/a"]


async def test_a_redirect_to_another_host_drops_credentials_and_cookies(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "https://other.example/next"})
        return httpx.Response(200)

    fake_outbound.install(monkeypatch, handler=handler)

    await outbound.fetch(
        "GET",
        "https://public.example/start",
        headers={"Authorization": "Bearer secret", "Cookie": "session=secret", "Accept": "application/json"},
        auth=("user", "secret"),
    )

    assert "authorization" not in seen[1].headers and "cookie" not in seen[1].headers
    assert seen[1].headers["accept"] == "application/json"
    assert seen[1].headers["host"] == "other.example"


async def test_a_host_header_from_the_step_is_replaced_by_the_name_it_connects_to(monkeypatch) -> None:
    seen: list[httpx.Request] = []
    fake_outbound.install(monkeypatch, handler=lambda r: seen.append(r) or httpx.Response(200))

    await outbound.fetch("GET", "https://public.example/", headers={"host": "internal.example"})

    assert seen[0].headers.get_list("host") == ["public.example"]


# ── Each action that fetches ──────────────────────────────────────────


async def test_api_request_refuses_a_local_address(monkeypatch) -> None:
    from pilot.system import api_client

    sent: list[str] = []
    fake_outbound.install(monkeypatch, handler=lambda r: sent.append(str(r.url)) or httpx.Response(200))

    with pytest.raises(outbound.OutboundRefused, match="I don't fetch from addresses"):
        await api_client.api_request("GET", "http://192.168.0.10/api")
    assert sent == []


async def test_api_scrape_refuses_a_name_that_resolves_locally(monkeypatch) -> None:
    from pilot.system import api_client

    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200), answers={"x.example": ["127.0.0.1"]})

    with pytest.raises(outbound.OutboundRefused):
        await api_client.scrape_url("https://x.example/")


async def test_a_webhook_send_refuses_a_local_address(monkeypatch) -> None:
    from pilot.system import api_client

    sent: list[str] = []
    fake_outbound.install(monkeypatch, handler=lambda r: sent.append(str(r.url)) or httpx.Response(200))

    with pytest.raises(outbound.OutboundRefused):
        await api_client.send_webhook("http://10.0.0.1/hook", {"a": 1})
    assert sent == []


async def test_download_file_refuses_a_local_address_and_writes_nothing(monkeypatch, tmp_path) -> None:
    from pilot.system import download

    sent: list[str] = []
    fake_outbound.install(monkeypatch, handler=lambda r: sent.append(str(r.url)) or httpx.Response(200, content=b"x"))
    target = tmp_path / "file.bin"

    with pytest.raises(outbound.OutboundRefused):
        await download.download_file("http://169.254.169.254/latest/meta-data/iam", str(target))

    assert sent == [] and not target.exists()
    # No other download program is ever started: the module has no way to start one.
    assert not hasattr(download, "run_command") and not hasattr(download, "run_powershell")


async def test_download_file_follows_a_redirect_only_through_the_checks(monkeypatch, tmp_path) -> None:
    from pilot.system import download

    seen: list[str] = []
    routes = {"files.example/report.pdf": httpx.Response(302, headers={"location": "http://10.1.1.1/report.pdf"})}
    fake_outbound.install(monkeypatch, handler=_chain(routes, seen))

    with pytest.raises(outbound.OutboundRefused):
        await download.download_file("https://files.example/report.pdf", str(tmp_path / "r.pdf"))
    assert seen == ["files.example/report.pdf"] and not (tmp_path / "r.pdf").exists()


async def test_download_file_reaches_an_approved_local_address(monkeypatch, tmp_path) -> None:
    from pilot.system import download

    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200, content=b"local bytes"))
    target = tmp_path / "local.bin"

    output = await download.download_file("http://127.0.0.1:8000/f.bin", str(target), allow_private_network=True)

    assert target.read_bytes() == b"local bytes"
    assert "[DOWNLOAD]" in output


async def test_the_messaging_read_back_refuses_a_local_address(monkeypatch) -> None:
    from pilot.agents.postconditions import messaging

    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200, json={}))

    with pytest.raises(messaging._Unreachable):
        await messaging._http_get("http://127.0.0.1/api/messages")


async def test_a_feed_on_a_local_address_is_refused(monkeypatch) -> None:
    from types import SimpleNamespace

    from pilot.agents.rss_agent import RssAgent

    fake_outbound.install(monkeypatch, handler=lambda r: httpx.Response(200, text="<rss/>"))
    agent = object.__new__(RssAgent)
    agent._config = SimpleNamespace(rss=SimpleNamespace(local_feeds=[]))

    with pytest.raises(outbound.OutboundRefused):
        await agent._fetch_feed("http://router.lan/feed.xml")


# ── Which steps need the person's approval ────────────────────────────


def test_download_file_with_allow_private_network_needs_approval_every_time() -> None:
    action = Action(
        action_type=ActionType.DOWNLOAD_FILE,
        parameters=DownloadParams(url="http://127.0.0.1/f", output_path="f", allow_private_network=True),
    )
    assert action.permission_tier == PermissionTier.SYSTEM_MODIFY
    assert action.requires_confirmation is True and action.is_irreversible is True


def test_download_file_of_a_public_address_needs_no_approval() -> None:
    action = Action(
        action_type=ActionType.DOWNLOAD_FILE, parameters=DownloadParams(url="https://example.com/f", output_path="f")
    )
    assert action.permission_tier == PermissionTier.USER_WRITE
    assert action.requires_confirmation is False


@pytest.mark.parametrize(
    ("action_type", "params"),
    [
        (ActionType.OPEN_URL, OpenUrlParams(url="http://192.168.1.1")),
        (ActionType.OPEN_URL, OpenUrlParams(url="localhost:3000")),
        (ActionType.OPEN_URL, OpenUrlParams(url="http://169.254.169.254/latest/meta-data/")),
        (ActionType.OPEN_URL, OpenUrlParams(url="http://0x7f.1/")),
        (ActionType.OPEN_URL, OpenUrlParams(url="http://example.com\\@127.0.0.1/")),
        (ActionType.OPEN_URL, OpenUrlParams(url="http://127.0.0.1\\@example.com/")),
        (ActionType.OPEN_URL, OpenUrlParams(url="https://nas.local/")),
        (ActionType.BROWSER_NEW_TAB, BrowserParams(url="http://[::ffff:10.0.0.1]/")),
        (ActionType.BROWSER_NAVIGATE, BrowserParams(url="http://metadata.google.internal/")),
    ],
)
def test_a_browser_step_to_a_local_address_is_asked_once_with_the_reason(action_type, params) -> None:
    action = Action(action_type=action_type, parameters=params)
    reason = action.local_address_reason
    assert reason
    assert action.permission_tier == PermissionTier.SYSTEM_MODIFY
    assert action.requires_confirmation is True and action.is_irreversible is True


@pytest.mark.parametrize(
    ("action_type", "params", "tier"),
    [
        (ActionType.OPEN_URL, OpenUrlParams(url="https://github.com"), PermissionTier.USER_WRITE),
        (ActionType.OPEN_URL, OpenUrlParams(url="youtube.com/watch?v=1"), PermissionTier.USER_WRITE),
        (ActionType.BROWSER_NEW_TAB, BrowserParams(url="https://example.com"), PermissionTier.USER_WRITE),
        (ActionType.BROWSER_NEW_TAB, BrowserParams(), PermissionTier.USER_WRITE),
        (ActionType.BROWSER_NAVIGATE, BrowserParams(url="file:///tmp/page.html"), PermissionTier.SYSTEM_MODIFY),
    ],
)
def test_a_browser_step_to_a_public_address_keeps_its_tier(action_type, params, tier) -> None:
    action = Action(action_type=action_type, parameters=params)
    assert action.local_address_reason is None
    assert action.permission_tier == tier
    assert action.is_irreversible is False


async def test_the_executor_passes_the_download_approval_on(monkeypatch) -> None:
    from pilot.agents.executor import Executor
    from pilot.system import download

    action = Action(
        action_type=ActionType.DOWNLOAD_FILE,
        parameters=DownloadParams(url="http://127.0.0.1/f", output_path="out.bin", allow_private_network=True),
    )
    fake = AsyncMock(return_value="ok")
    monkeypatch.setattr(download, "download_file", fake)

    assert await Executor._exec_download_file(object.__new__(Executor), action) == "ok"
    fake.assert_awaited_once_with("http://127.0.0.1/f", "out.bin", False, allow_private_network=True, max_bytes=None)


def test_nat64_extraction_follows_rfc_6052_for_every_prefix_length() -> None:
    # RFC 6052 section 2.4: 192.0.2.33 under 2001:db8::/32 ... /96.
    examples = {
        32: "2001:db8:c000:221::",
        40: "2001:db8:1c0:2:21::",
        48: "2001:db8:122:c000:2:2100::",
        56: "2001:db8:122:3c0:0:221::",
        64: "2001:db8:122:344:c0:2:2100:0",
        96: "2001:db8:122:344::192.0.2.33",
    }
    for length, text in examples.items():
        address = ipaddress.IPv6Address(text)
        prefix = ipaddress.IPv6Network((int(address) >> (128 - length) << (128 - length), length))
        assert str(addresses.nat64_ipv4(address, prefix)) == "192.0.2.33", length
