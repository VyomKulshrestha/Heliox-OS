import datetime
from typing import Any, Dict, List


def detect_spikes(
    events: list[dict[str, Any]],
    window_minutes: int = 10,
    threshold_count: int = 5
) -> list[dict[str, Any]]:
    """Detect rate spikes or frequency anomalies within a rolling time window."""
    anomalies = []

    # 1. Detect restart loops (same process restarting repeatedly)
    restart_events = [
        e for e in events
        if "restart" in e["message"].lower() or "started" in e["message"].lower()
    ]
    process_restarts: dict[str, list[dict[str, Any]]] = {}
    for ev in restart_events:
        proc = ev.get("process") or "unknown"
        if proc == "unknown":
            continue
        process_restarts.setdefault(proc, []).append(ev)

    for proc, evs in process_restarts.items():
        evs_sorted = sorted(evs, key=lambda x: x["timestamp"])
        for i in range(len(evs_sorted)):
            window_end = evs_sorted[i]
            try:
                dt_str = window_end["timestamp"].replace("Z", "").split(".")[0].split("+")[0].replace("T", " ")
                end_dt = datetime.datetime.fromisoformat(dt_str)
            except Exception:
                continue

            count = 1
            for j in range(i - 1, -1, -1):
                try:
                    dt_start_str = evs_sorted[j]["timestamp"].replace("Z", "").split(".")[0].split("+")[0].replace("T", " ")
                    start_dt = datetime.datetime.fromisoformat(dt_start_str)
                    if (end_dt - start_dt).total_seconds() <= window_minutes * 60:
                        count += 1
                    else:
                        break
                except Exception:
                    continue

            if count >= threshold_count:
                anomalies.append({
                    "severity": "high",
                    "anomaly_type": "restart_loop",
                    "resource": proc,
                    "summary": f"Service restart loop detected: process '{proc}' restarted {count} times within {window_minutes} minutes.",
                    "details": {
                        "process": proc,
                        "restarts_count": count,
                        "time_window_seconds": window_minutes * 60
                    },
                    "timestamp": window_end["timestamp"]
                })
                break  # only report once per loop cluster

    # 2. Detect failed login spikes (brute-force attempts)
    failed_logins = [
        e for e in events
        if "login_failed" in e.get("event_type", "") or "ssh_login_failed" in e.get("event_type", "")
    ]
    ip_failed_logins: dict[str, list[dict[str, Any]]] = {}
    for ev in failed_logins:
        ip = ev.get("ip_address") or "local"
        ip_failed_logins.setdefault(ip, []).append(ev)

    for ip, evs in ip_failed_logins.items():
        evs_sorted = sorted(evs, key=lambda x: x["timestamp"])
        for i in range(len(evs_sorted)):
            window_end = evs_sorted[i]
            try:
                dt_str = window_end["timestamp"].replace("Z", "").split(".")[0].split("+")[0].replace("T", " ")
                end_dt = datetime.datetime.fromisoformat(dt_str)
            except Exception:
                continue

            count = 1
            for j in range(i - 1, -1, -1):
                try:
                    dt_start_str = evs_sorted[j]["timestamp"].replace("Z", "").split(".")[0].split("+")[0].replace("T", " ")
                    start_dt = datetime.datetime.fromisoformat(dt_start_str)
                    if (end_dt - start_dt).total_seconds() <= window_minutes * 60:
                        count += 1
                    else:
                        break
                except Exception:
                    continue

            if count >= threshold_count:
                anomalies.append({
                    "severity": "medium" if ip == "local" else "high",
                    "anomaly_type": "login_spike",
                    "resource": ip,
                    "summary": f"Multiple failed login attempts from IP '{ip}': {count} failures within {window_minutes} minutes.",
                    "details": {
                        "ip_address": ip,
                        "failures_count": count,
                        "time_window_seconds": window_minutes * 60
                    },
                    "timestamp": window_end["timestamp"]
                })
                break

    # 3. Detect generic error spikes
    error_events = [
        e for e in events
        if any(err in e["message"].lower() for err in ["error", "fail", "critical", "panic", "crash"])
    ]
    if len(error_events) >= threshold_count * 2:
        anomalies.append({
            "severity": "medium",
            "anomaly_type": "error_spike",
            "resource": "system",
            "summary": f"High frequency of system errors detected: {len(error_events)} errors in current scope.",
            "details": {
                "errors_count": len(error_events)
            },
            "timestamp": datetime.datetime.now().isoformat()
        })

    return anomalies
