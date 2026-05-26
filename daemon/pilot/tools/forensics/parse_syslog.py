import datetime
import os
import re
from typing import Any, Dict, List, Optional

# RFC3164 format: Jan 22 14:32:01 host process[123]: msg
# or: Jan 22 14:32:01 host process: msg
RFC3164_PATTERN = re.compile(
    r'^([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^\[:]+)(?:\[(\d+)\])?:\s+(.*)$'
)

# RFC5424 format: <13>1 2026-05-25T14:32:01.000Z host process 123 msgid - msg
RFC5424_PATTERN = re.compile(
    r'^<\d+>1\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(?:-|\S+)\s+(.*)$'
)

def parse_syslog_line(line: str) -> dict[str, Any] | None:
    """Parse a single syslog line into a structured dictionary."""
    line = line.strip()
    if not line:
        return None

    # Try RFC3164
    m3164 = RFC3164_PATTERN.match(line)
    if m3164:
        timestamp_str, hostname, process, pid, message = m3164.groups()
        year = datetime.datetime.now().year
        try:
            dt = datetime.datetime.strptime(f"{year} {timestamp_str}", "%Y %b %d %H:%M:%S")
            timestamp = dt.isoformat()
        except Exception:
            timestamp = timestamp_str

        return {
            "timestamp": timestamp,
            "hostname": hostname,
            "process": process.strip(),
            "pid": int(pid) if pid else None,
            "message": message,
            "raw": line
        }

    # Try RFC5424
    m5424 = RFC5424_PATTERN.match(line)
    if m5424:
        timestamp_str, hostname, process, pid, msgid, message = m5424.groups()
        return {
            "timestamp": timestamp_str,
            "hostname": hostname,
            "process": process,
            "pid": int(pid) if pid and pid.isdigit() else None,
            "message": message,
            "raw": line
        }

    # Fallback/Generic parser (just extract timestamp if possible)
    iso_match = re.match(r'^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+(.*)$', line)
    if iso_match:
        timestamp_str, message = iso_match.groups()
        return {
            "timestamp": timestamp_str,
            "hostname": "localhost",
            "process": "unknown",
            "pid": None,
            "message": message,
            "raw": line
        }

    # Generic fallback
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": "localhost",
        "process": "unknown",
        "pid": None,
        "message": line,
        "raw": line
    }

def _parse_time_window(window_str: str) -> datetime.timedelta | None:
    if not window_str:
        return None
    match = re.match(r'^(\d+)\s*([smhd])$', window_str.strip().lower())
    if not match:
        return None
    value, unit = match.groups()
    val = int(value)
    if unit == 's':
        return datetime.timedelta(seconds=val)
    elif unit == 'm':
        return datetime.timedelta(minutes=val)
    elif unit == 'h':
        return datetime.timedelta(hours=val)
    elif unit == 'd':
        return datetime.timedelta(days=val)
    return None

def parse_syslog_file(filepath: str, query: str | None = None, time_window: str | None = None) -> list[dict[str, Any]]:
    """Safely parse a syslog file and apply time window and search query filters."""
    if not os.path.exists(filepath):
        return []

    events = []
    limit_dt = None
    if time_window:
        delta = _parse_time_window(time_window)
        if delta:
            # handle timezone naive datetime
            limit_dt = datetime.datetime.now()

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parsed = parse_syslog_line(line)
                if not parsed:
                    continue

                # Filter by time window
                if limit_dt and time_window:
                    try:
                        # simple comparison by converting ISO timestamp
                        dt_str = parsed["timestamp"].replace("Z", "").split(".")[0].split("+")[0]
                        # Replace T with space if needed
                        dt_str = dt_str.replace("T", " ")
                        dt = datetime.datetime.fromisoformat(dt_str)
                        delta = _parse_time_window(time_window)
                        if delta and (datetime.datetime.now() - dt) > delta:
                            continue
                    except Exception:
                        pass

                # Filter by query
                if query:
                    q_lower = query.lower()
                    if q_lower not in parsed["message"].lower() and q_lower not in parsed["process"].lower():
                        continue

                events.append(parsed)
    except Exception:
        pass

    return events
