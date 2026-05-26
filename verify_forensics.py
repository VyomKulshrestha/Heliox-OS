import sys
import os

# Add daemon directory to path so imports work correctly
daemon_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "daemon"))
if daemon_dir not in sys.path:
    sys.path.insert(0, daemon_dir)

import json
from pilot.tools.forensics import parse_syslog_line, parse_syslog_file, extract_auth_events, detect_spikes, correlate_sessions
from pilot.agents.registry import AgentRegistry

print("=== STARTING FORENSICS INTEGRATION TEST ===")

# Test 1: Parser
print("\n--- Test 1: Parsing log lines ---")
line1 = "May 25 14:01:05 server sshd[1234]: Accepted publickey for admin from 192.168.1.50 port 55670 ssh2"
parsed1 = parse_syslog_line(line1)
print(f"RFC3164 Line: {line1}")
print(f"Parsed Result: {parsed1}")
assert parsed1 is not None
assert parsed1["hostname"] == "server"
assert parsed1["process"] == "sshd"
assert parsed1["pid"] == 1234
assert "Accepted publickey" in parsed1["message"]

# Test 2: Auth Extractor
print("\n--- Test 2: Extracting auth events ---")
events = [parsed1]
extracted = extract_auth_events(events)
print(f"Extracted Auth Event: {extracted}")
assert len(extracted) == 1
assert extracted[0]["event_type"] == "ssh_login_success"
assert extracted[0]["username"] == "admin"
assert extracted[0]["ip_address"] == "192.168.1.50"

# Test 3: Spike Detection
print("\n--- Test 3: Anomaly / Spike Detection ---")
# Simulate restart loop logs
restart_logs = [
    {"timestamp": "2026-05-25T14:05:00", "process": "nginx", "message": "nginx.service: Scheduled restart job, restart counter is at 1."},
    {"timestamp": "2026-05-25T14:05:02", "process": "nginx", "message": "nginx.service: Scheduled restart job, restart counter is at 2."},
    {"timestamp": "2026-05-25T14:05:04", "process": "nginx", "message": "nginx.service: Scheduled restart job, restart counter is at 3."},
    {"timestamp": "2026-05-25T14:05:06", "process": "nginx", "message": "nginx.service: Scheduled restart job, restart counter is at 4."},
    {"timestamp": "2026-05-25T14:05:08", "process": "nginx", "message": "nginx.service: Scheduled restart job, restart counter is at 5."},
]
anomalies = detect_spikes(restart_logs, window_minutes=1, threshold_count=3)
print(f"Detected Anomalies: {anomalies}")
assert len(anomalies) > 0
assert anomalies[0]["anomaly_type"] == "restart_loop"

# Test 4: Dynamic Registry Discovery
print("\n--- Test 4: Dynamic Registry Agent Discovery ---")
AgentRegistry.discover_agents()
all_agents = AgentRegistry.get_all_agents()
print(f"Registered Agents: {list(all_agents.keys())}")
assert "ForensicsAgent" in all_agents
print("ForensicsAgent successfully registered dynamically!")

print("\n=== ALL FORENSICS UNIT TESTS PASSED SUCCESSFULLY ===")
