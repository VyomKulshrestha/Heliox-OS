"""
tools/log_tools.py
------------------
Stateless helper tools used by ForensicsAgent's ReAct loop.

Each tool is a plain callable that accepts typed arguments and returns a
plain Python dict so results are trivially JSON-serialisable for the LLM.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# RFC 3164 / traditional syslog  e.g.  Jun 14 15:16:01 hostname sshd[1234]: msg
_SYSLOG_RE = re.compile(
    r"(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})"
    r"\s+(?P<host>\S+)\s+(?P<process>[^\[:]+)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.+)"
)

# ISO-8601 / systemd-style  e.g.  2024-06-14T15:16:01.123456+00:00 hostname ...
_ISO_SYSLOG_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s+(?P<host>\S+)\s+(?P<process>[^\[:]+)(?:\[(?P<pid>\d+)\])?:\s+(?P<message>.+)"
)

# Common web / nginx / apache combined log
_ACCESS_LOG_RE = re.compile(
    r"(?P<ip>\S+)\s+-\s+(?P<user>\S+)\s+\[(?P<time>[^\]]+)\]\s+"
    r'"(?P<method>\S+)\s+(?P<path>\S+)\s+(?P<proto>[^"]+)"\s+'
    r"(?P<status>\d{3})\s+(?P<size>\S+)"
)

# Auth / security keywords
_AUTH_FAIL_RE = re.compile(
    r"(?i)(failed\s+password|authentication\s+failure|invalid\s+user"
    r"|permission\s+denied|access\s+denied|unauthorized|login\s+failed"
    r"|bad\s+password|incorrect\s+password)"
)
_BRUTE_USER_RE = re.compile(r"(?i)(?:invalid user|failed password for(?: invalid user)?)\s+(\S+)")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PRIV_ESC_RE = re.compile(r"(?i)(sudo|su |setuid|privilege\s+escalat|root\s+shell|NOPASSWD)")
_CRASH_RE = re.compile(r"(?i)(segfault|kernel\s+panic|oom[- ]kill|out\s+of\s+memory|core\s+dump|fatal\s+error)")
_PORT_SCAN_RE = re.compile(r"(?i)(nmap|port\s+scan|syn\s+flood|connection\s+refused.*\d{2,5})")


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def parse_syslog(raw_lines: list[str]) -> dict[str, Any]:
    """
    Parse a list of raw log strings into structured records.

    Supports:
      * RFC 3164 syslog
      * ISO-8601 / systemd journal export
      * Common access log (nginx/apache)
      * Falls back to storing the raw line if no format matches.

    Returns
    -------
    dict with keys:
        parsed   – list of structured log record dicts
        unparsed – list of raw lines that matched no known format
        stats    – summary counts
    """
    parsed: list[dict[str, Any]] = []
    unparsed: list[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        m = _ISO_SYSLOG_RE.match(line) or _SYSLOG_RE.match(line) or _ACCESS_LOG_RE.match(line)
        if m:
            record = m.groupdict()
            record["raw"] = line
            record["_format"] = (
                "iso_syslog" if "timestamp" in record
                else "access_log" if "method" in record
                else "rfc3164"
            )
            parsed.append(record)
        else:
            unparsed.append(line)

    return {
        "parsed": parsed,
        "unparsed": unparsed,
        "stats": {
            "total_lines": len(raw_lines),
            "parsed_count": len(parsed),
            "unparsed_count": len(unparsed),
        },
    }


def extract_timestamps(parsed_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Normalise timestamps from parsed log records into ISO-8601 UTC strings
    and compute a basic time-density histogram (events per minute).

    Returns
    -------
    dict with keys:
        records_with_ts   – original records annotated with 'ts_iso' field
        first_event       – earliest ISO timestamp seen
        last_event        – latest ISO timestamp seen
        density_histogram – {minute_bucket: count}
        parse_errors      – count of records where timestamp couldn't be parsed
    """
    current_year = datetime.now(timezone.utc).replace(tzinfo=None).year
    annotated: list[dict[str, Any]] = []
    histogram: dict[str, int] = defaultdict(int)
    errors = 0

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }

    timestamps: list[datetime] = []

    for rec in parsed_records:
        ts: datetime | None = None
        fmt = rec.get("_format", "")

        try:
            if fmt == "iso_syslog":
                raw_ts = rec["timestamp"].replace("Z", "+00:00")
                ts = datetime.fromisoformat(raw_ts.replace("+00:00", ""))
            elif fmt == "rfc3164":
                month = month_map.get(rec.get("month", ""), 1)
                day = int(rec.get("day", 1))
                h, m, s = rec.get("time", "00:00:00").split(":")
                ts = datetime(current_year, month, day, int(h), int(m), int(s))
            elif fmt == "access_log":
                # e.g. 14/Jun/2024:15:16:01 +0000
                raw = rec.get("time", "")
                ts = datetime.strptime(raw[:20], "%d/%b/%Y:%H:%M:%S")
        except Exception:
            errors += 1

        rec_copy = dict(rec)
        if ts:
            rec_copy["ts_iso"] = ts.isoformat()
            timestamps.append(ts)
            bucket = ts.strftime("%Y-%m-%dT%H:%M")
            histogram[bucket] += 1
        annotated.append(rec_copy)

    return {
        "records_with_ts": annotated,
        "first_event": min(timestamps).isoformat() if timestamps else None,
        "last_event": max(timestamps).isoformat() if timestamps else None,
        "density_histogram": dict(sorted(histogram.items())),
        "parse_errors": errors,
    }


def scan_auth_anomalies(parsed_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Detect authentication-related anomalies:
      * Brute-force attempts (≥5 failures from the same IP within the window)
      * Invalid / unknown user logins
      * Privilege escalation attempts

    Returns a structured findings dict.
    """
    failures: dict[str, list[dict]] = defaultdict(list)  # ip -> events
    invalid_users: list[dict] = []
    priv_esc: list[dict] = []

    for rec in parsed_records:
        msg = rec.get("message", rec.get("raw", ""))

        if _AUTH_FAIL_RE.search(msg):
            ips = _IP_RE.findall(msg)
            src_ip = ips[0] if ips else "unknown"
            entry = {"ts": rec.get("ts_iso", rec.get("time", "")), "message": msg, "ip": src_ip}
            failures[src_ip].append(entry)

            user_m = _BRUTE_USER_RE.search(msg)
            if user_m:
                entry["username"] = user_m.group(1)
                invalid_users.append(entry)

        if _PRIV_ESC_RE.search(msg):
            priv_esc.append({"ts": rec.get("ts_iso", rec.get("time", "")), "message": msg})

    brute_force_ips = {ip: evts for ip, evts in failures.items() if len(evts) >= 5}

    return {
        "brute_force_suspects": [
            {"ip": ip, "attempt_count": len(evts), "events": evts[:10]}
            for ip, evts in brute_force_ips.items()
        ],
        "invalid_user_attempts": invalid_users[:20],
        "privilege_escalation_attempts": priv_esc[:20],
        "total_auth_failures": sum(len(v) for v in failures.values()),
        "unique_source_ips": len(failures),
    }


def scan_system_anomalies(parsed_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Detect OS-level anomalies:
      * Kernel panics / OOM kills / segfaults
      * Port scans / SYN floods
      * Repeated process crashes

    Returns a structured findings dict.
    """
    crashes: list[dict] = []
    network_attacks: list[dict] = []
    process_crash_counts: dict[str, int] = defaultdict(int)

    for rec in parsed_records:
        msg = rec.get("message", rec.get("raw", ""))
        proc = rec.get("process", "unknown")
        ts = rec.get("ts_iso", rec.get("time", ""))

        if _CRASH_RE.search(msg):
            crashes.append({"ts": ts, "process": proc, "message": msg})
            process_crash_counts[proc] += 1

        if _PORT_SCAN_RE.search(msg):
            ips = _IP_RE.findall(msg)
            network_attacks.append({
                "ts": ts,
                "message": msg,
                "source_ips": ips,
            })

    return {
        "system_crashes": crashes[:20],
        "network_attacks": network_attacks[:20],
        "repeated_crashes": {
            proc: count for proc, count in process_crash_counts.items() if count >= 3
        },
        "total_crash_events": len(crashes),
        "total_network_attack_events": len(network_attacks),
    }


def compute_risk_score(
    auth_findings: dict[str, Any],
    system_findings: dict[str, Any],
) -> dict[str, Any]:
    """
    Derive a simple 0-100 risk score and severity label from findings.

    Scoring heuristics (additive, capped at 100):
      +30  any brute-force IP found
      +10  per brute-force IP beyond the first (max +30 extra)
      +20  privilege escalation detected
      +15  system crash / OOM event
      +10  network attack / port scan detected
      +5   > 20 total auth failures
    """
    score = 0

    bf = auth_findings.get("brute_force_suspects", [])
    if bf:
        score += 30
        score += min(len(bf) - 1, 3) * 10

    if auth_findings.get("privilege_escalation_attempts"):
        score += 20

    if system_findings.get("total_crash_events", 0) > 0:
        score += 15

    if system_findings.get("total_network_attack_events", 0) > 0:
        score += 10

    if auth_findings.get("total_auth_failures", 0) > 20:
        score += 5

    score = min(score, 100)

    if score >= 70:
        severity = "CRITICAL"
    elif score >= 40:
        severity = "HIGH"
    elif score >= 20:
        severity = "MEDIUM"
    elif score > 0:
        severity = "LOW"
    else:
        severity = "NONE"

    return {"risk_score": score, "severity": severity}