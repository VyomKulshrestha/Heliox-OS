"""Which network addresses Heliox's own HTTP senders (pilot.system.outbound) may reach without the person asking for
that address, and the hosts a person's words name.

``address_refusal`` judges an address a connection would go to. ``host_refusal`` judges a host as a URL writes it,
before any name lookup: every spelling a resolver or a browser reads as an address (``127.1``, ``0x7f.1``,
``2130706433``, ``017700000001``, ``[::ffff:127.0.0.1]``) is read as that address, and names that only this computer
or the local network can answer (``localhost``, ``*.local``, ``*.internal``, ``*.home.arpa``, a name without a dot)
are local. Both give None for a public address, or a short reason that completes the sentence "<host> is <reason>".

An IPv6 address that carries an IPv4 address (IPv4-mapped and IPv4-compatible forms, NAT64 in the well-known
``64:ff9b::/96`` and local-use ``64:ff9b:1::/48`` prefixes and in any prefix the network announces, 6to4, Teredo,
ISATAP) is judged by the IPv4 address it carries: a NAT64 gateway sends a connection on to that address. The NAT64
prefix itself is never refused, so a computer on an IPv6-only network still reaches public sites.

The tables are written out here instead of read from ``ipaddress``'s ``is_global``, which changed between Python
3.11 and 3.13; every range below is refused on every version.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import unquote, urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

THIS_COMPUTER = "on this computer"
LOCAL_NETWORK = "on the local network"
LINK_LOCAL = "a link-local address on the local network"
METADATA = "a cloud metadata address, which hands out this machine's credentials"
LOCAL_NAME = "a name that only this computer or the local network answers"
UNREADABLE = "an address I cannot read"

_V4_METADATA = frozenset(
    ipaddress.IPv4Address(text)
    for text in (
        "169.254.169.254",  # AWS, Azure, Google Cloud, DigitalOcean, OpenStack
        "169.254.170.2",  # AWS container credentials
        "169.254.170.23",  # AWS EKS pod identity
        "100.100.100.200",  # Alibaba Cloud
        "192.0.0.192",  # Oracle Cloud
    )
)
_V6_METADATA = frozenset(ipaddress.IPv6Address(text) for text in ("fd00:ec2::254", "fd00:ec2::23"))

_V4_RANGES: tuple[tuple[ipaddress.IPv4Network, str], ...] = tuple(
    (ipaddress.IPv4Network(network), reason)
    for network, reason in (
        ("0.0.0.0/8", "an unspecified address, which reaches this computer"),
        ("127.0.0.0/8", THIS_COMPUTER),
        ("10.0.0.0/8", LOCAL_NETWORK),
        ("172.16.0.0/12", LOCAL_NETWORK),
        ("192.168.0.0/16", LOCAL_NETWORK),
        ("100.64.0.0/10", "a carrier-grade NAT address inside a provider's network"),
        ("169.254.0.0/16", LINK_LOCAL),
        ("192.0.0.0/24", "a protocol address that is not on the public internet"),
        ("192.0.2.0/24", "a documentation address"),
        ("198.51.100.0/24", "a documentation address"),
        ("203.0.113.0/24", "a documentation address"),
        ("198.18.0.0/15", "a benchmarking address used on local networks"),
        ("192.88.99.0/24", "a retired relay address"),
        ("224.0.0.0/4", "a multicast address"),
        ("240.0.0.0/4", "a reserved address"),
    )
)

_V6_RANGES: tuple[tuple[ipaddress.IPv6Network, str], ...] = tuple(
    (ipaddress.IPv6Network(network), reason)
    for network, reason in (
        ("::/128", "an unspecified address, which reaches this computer"),
        ("::1/128", THIS_COMPUTER),
        ("fe80::/10", LINK_LOCAL),
        ("fec0::/10", "a site-local address on the local network"),
        ("fc00::/7", "a unique local address on the local network"),
        ("ff00::/8", "a multicast address"),
        ("2001:db8::/32", "a documentation address"),
        ("3fff::/20", "a documentation address"),
        ("100::/64", "a discard address"),
        ("2001:2::/48", "a benchmarking address used on local networks"),
        ("2001:10::/28", "an identifier that is not on the public internet"),
        ("5f00::/16", "a segment routing address that is not on the public internet"),
    )
)

NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
NAT64_LOCAL_USE = ipaddress.IPv6Network("64:ff9b:1::/48")
_MAPPED = ipaddress.IPv6Network("::ffff:0:0/96")
_TRANSLATED = ipaddress.IPv6Network("::ffff:0:0:0/96")
_COMPATIBLE = ipaddress.IPv6Network("::/96")
_SIX_TO_FOUR = ipaddress.IPv6Network("2002::/16")
_TEREDO = ipaddress.IPv6Network("2001::/32")


def _v4_refusal(address: ipaddress.IPv4Address) -> str | None:
    if address in _V4_METADATA:
        return METADATA
    for network, reason in _V4_RANGES:
        if address in network:
            return reason
    return None


def nat64_ipv4(address: ipaddress.IPv6Address, prefix: ipaddress.IPv6Network) -> ipaddress.IPv4Address:
    """The IPv4 address an RFC 6052 NAT64 address in ``prefix`` carries (prefix lengths 32, 40, 48, 56, 64 or 96).

    Bits 64 to 71 (the "u" octet) never hold address bits, so for the shorter prefixes the IPv4 address is split
    around them."""
    packed = address.packed
    length = prefix.prefixlen
    if length == 96:
        return ipaddress.IPv4Address(packed[12:16])
    start = length // 8
    octets = [packed[i] for i in range(start, 16) if i != 8][:4]
    return ipaddress.IPv4Address(bytes(octets))


def _local_use_candidates(address: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """The local-use NAT64 prefix may be any length from /48 to /96: a /96 layout leaves bits 48 to 95 zero; any
    other layout is judged by every reading it could have."""
    if int(address) >> 32 & ((1 << 48) - 1) == 0:
        return [nat64_ipv4(address, ipaddress.IPv6Network("64:ff9b:1::/96"))]
    return [
        nat64_ipv4(address, ipaddress.IPv6Network((int(address) >> (128 - n) << (128 - n), n))) for n in (48, 56, 64)
    ]


def embedded_ipv4(
    address: ipaddress.IPv6Address, nat64_prefixes: Iterable[ipaddress.IPv6Network] = ()
) -> list[ipaddress.IPv4Address] | None:
    """The IPv4 addresses an IPv6 address carries and sends a connection on to, or None when it carries none."""
    if address in _MAPPED or address in _TRANSLATED:
        return [ipaddress.IPv4Address(address.packed[12:16])]
    if address in NAT64_WELL_KNOWN:
        return [nat64_ipv4(address, NAT64_WELL_KNOWN)]
    if address in NAT64_LOCAL_USE:
        return _local_use_candidates(address)
    for prefix in nat64_prefixes:
        if address in prefix:
            return [nat64_ipv4(address, prefix)]
    if address in _COMPATIBLE and int(address) > 1:
        return [ipaddress.IPv4Address(address.packed[12:16])]
    if address in _SIX_TO_FOUR:
        return [ipaddress.IPv4Address(address.packed[2:6])]
    if address in _TEREDO:
        server = ipaddress.IPv4Address(address.packed[4:8])
        client = ipaddress.IPv4Address(bytes(b ^ 0xFF for b in address.packed[12:16]))
        return [server, client]
    if address.packed[8:12] in (b"\x00\x00\x5e\xfe", b"\x02\x00\x5e\xfe"):  # ISATAP interface identifier
        return [ipaddress.IPv4Address(address.packed[12:16])]
    return None


def address_refusal(address: IPAddress, nat64_prefixes: Iterable[ipaddress.IPv6Network] = ()) -> str | None:
    """None when a connection to ``address`` reaches the public internet; otherwise why it does not."""
    if isinstance(address, ipaddress.IPv4Address):
        return _v4_refusal(address)
    if address.scope_id:
        return LINK_LOCAL
    if address in _V6_METADATA:
        return METADATA
    for network, reason in _V6_RANGES[:2]:  # :: and ::1 before the IPv4-compatible form
        if address in network:
            return reason
    carried = embedded_ipv4(address, nat64_prefixes)
    if carried is not None:
        for inner in carried:
            reason = _v4_refusal(inner)
            if reason:
                return f"{reason} ({inner}, carried inside an IPv6 address)"
        if address in _COMPATIBLE or address in _TRANSLATED:
            return "a retired IPv6 form of an IPv4 address that is not used on the public internet"
        if address.packed[8:12] not in (b"\x00\x00\x5e\xfe", b"\x02\x00\x5e\xfe"):
            return None
    for network, reason in _V6_RANGES:
        if address in network:
            return reason
    return None


_PART = re.compile(r"^(?:0x[0-9a-f]*|0[0-7]*|[1-9][0-9]*)$")


def _loose_ipv4(name: str) -> ipaddress.IPv4Address | None:
    """An IPv4 address in any spelling ``inet_aton`` and browsers accept: 1 to 4 parts, each decimal, octal (a leading
    0) or hex (0x), the last part filling the remaining bytes (``127.1``, ``0x7f000001``, ``2130706433``)."""
    parts = name.split(".")
    if not 1 <= len(parts) <= 4 or not all(_PART.match(part) for part in parts):
        return None
    values = []
    for part in parts:
        if part.startswith("0x"):
            values.append(int(part[2:] or "0", 16))
        elif part.startswith("0") and len(part) > 1:
            values.append(int(part, 8))
        else:
            values.append(int(part))
    *head, last = values
    if any(value > 255 for value in head) or last >= 256 ** (4 - len(head)):
        return None
    number = 0
    for value in head:
        number = number * 256 + value
    return ipaddress.IPv4Address(number * 256 ** (4 - len(head)) + last)


def normalize_host(host: str) -> str:
    """A host the way a resolver or a browser reads it: percent escapes decoded, full-width and other compatibility
    characters folded (NFKC), the ideographic dots read as dots, brackets and trailing dots removed, lower case."""
    text = unquote(str(host or ""))
    text = unicodedata.normalize("NFKC", text)
    for dot in ("。", "．", "｡"):
        text = text.replace(dot, ".")
    text = text.strip().strip("[]").lower()
    return text.rstrip(".")


def parse_address(host: str) -> IPAddress | None:
    """The address a host spells, in any form a resolver or browser accepts, or None when it is a name."""
    name = normalize_host(host)
    if ":" in name:
        bare, _, zone = name.partition("%")
        try:
            address = ipaddress.IPv6Address(bare)
        except ValueError:
            return None
        return ipaddress.IPv6Address(f"{bare}%{zone}") if zone else address
    return _loose_ipv4(name)


_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
_LOCAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".localdomain",
    ".internal",
    ".home.arpa",
    ".lan",
    ".home",
    ".intranet",
    ".corp",
    ".private",
    ".in-addr.arpa",
    ".ip6.arpa",
)


def host_refusal(host: str, nat64_prefixes: Iterable[ipaddress.IPv6Network] = ()) -> str | None:
    """None when ``host`` as written may be public; otherwise why it is not. A name still has to be looked up and its
    addresses judged with ``address_refusal`` before anything connects."""
    name = normalize_host(host)
    if not name:
        return UNREADABLE
    address = parse_address(name)
    if address is not None:
        return address_refusal(address, nat64_prefixes)
    if any(mark in name for mark in ":%@\\/[]") or any(ch.isspace() for ch in name):
        return UNREADABLE
    if name in _LOCAL_NAMES or name.endswith(_LOCAL_SUFFIXES) or "." not in name:
        return LOCAL_NAME
    return None


def url_host(url: str, *, as_browser: bool = True) -> str:
    """The host this address names. A missing scheme is read as https (as open_url does) and tabs and line breaks are
    dropped. ``as_browser`` reads backslashes as slashes, as browsers do for http and https; without it the address
    is read as Python's and many system launchers' parsers read it."""
    text = "".join(ch for ch in str(url or "").strip() if ch not in "\t\r\n")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        text = "https://" + text.lstrip("/")
    scheme, _, rest = text.partition("://")
    if as_browser and scheme.lower() in {"http", "https"}:
        rest = rest.replace("\\", "/")
    try:
        parsed = urlsplit(f"{scheme}://{rest}")
        netloc = parsed.netloc
    except ValueError:
        return "[" + rest
    host = netloc.rpartition("@")[2]
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+")
_BRACKETED = re.compile(r"\[([0-9A-Za-z:.%]+)\]")
_TOKEN_EDGES = ".,;:!?'\"()<>{}"


def hosts_named(text: str) -> frozenset[str]:
    """Every host ``text`` names, each in the form ``same_host`` compares: the host of each address with a scheme, each
    bracketed IPv6 address, and each bare word that reads as an address or a name with a port or without. Whole
    hosts only: "10.0.0.15" names 10.0.0.15, never 10.0.0.1."""
    words = str(text or "")
    found: set[str] = set()
    for match in _URL_IN_TEXT.finditer(words):
        found.add(host_key(url_host(match.group(0))))
    for match in _BRACKETED.finditer(words):
        found.add(host_key(match.group(1)))
    for raw in re.split(r"\s+", _URL_IN_TEXT.sub(" ", words)):
        token = raw.strip(_TOKEN_EDGES)
        if not token or "/" in token or "@" in token:
            continue
        if token.count(":") == 1 and token.rpartition(":")[2].isdigit():
            token = token.rpartition(":")[0]  # host:port
        found.add(host_key(token))
    found.discard("")
    return frozenset(found)


def host_key(host: str) -> str:
    """One spelling for a host: an address in its standard form (any spelling a resolver or browser accepts, such as
    127.1 or 0x7f000001 for 127.0.0.1), a name normalised as ``normalize_host`` does."""
    address = parse_address(host)
    return str(address) if address is not None else normalize_host(host)


def same_host(first: str, second: str) -> bool:
    """Whether two hosts are the same host, whole: equal addresses, or equal names after normalisation."""
    key = host_key(first)
    return bool(key) and key == host_key(second)


def browser_target_refusal(url: str) -> str | None:
    """Why an address handed to a browser (open_url, browser_navigate, browser_new_tab) needs the person's approval:
    a literal address on this computer or the local network, a metadata address or a local-only name, in either way
    the address can be read (a browser's, or a launcher's that keeps backslashes). None for anything else, including
    schemes other than http and https, which are not network fetches."""
    text = str(url or "").strip()
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):", text)
    if match and match.group(1).lower() not in {"http", "https"} and "://" in text:
        return None
    return host_refusal(url_host(text)) or host_refusal(url_host(text, as_browser=False))
