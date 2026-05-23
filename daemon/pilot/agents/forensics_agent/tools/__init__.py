"""Heliox ForensicsAgent – tool registry."""

from .log_tools import (
    compute_risk_score,
    extract_timestamps,
    parse_syslog,
    scan_auth_anomalies,
    scan_system_anomalies,
)

__all__ = [
    "parse_syslog",
    "extract_timestamps",
    "scan_auth_anomalies",
    "scan_system_anomalies",
    "compute_risk_score",
]