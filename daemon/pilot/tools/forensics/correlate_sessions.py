from typing import Any, Dict, List


def correlate_sessions(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Correlate and group extracted security events by IP, username, or system activity."""
    correlations = {}

    for ev in events:
        ip = ev.get("ip_address")
        user = ev.get("username")

        keys = []
        if ip and ip != "local":
            keys.append(f"ip:{ip}")
        if user and user != "unknown":
            keys.append(f"user:{user}")

        if not keys:
            keys.append("system:general")

        for k in keys:
            correlations.setdefault(k, []).append({
                "timestamp": ev["timestamp"],
                "event_type": ev.get("event_type", "general"),
                "description": ev.get("description", ev["message"]),
                "process": ev.get("original_event", {}).get("process", ev.get("process", "unknown"))
            })

    # Chronologically sort timelines for each correlation key
    for k in correlations:
        correlations[k] = sorted(correlations[k], key=lambda x: x["timestamp"])

    return correlations
