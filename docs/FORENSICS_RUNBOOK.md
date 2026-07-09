# Forensics Agent Runbook

Welcome to the official administrator guide for the **Forensics Agent** in Heliox OS. The Forensics Agent is responsible for parsing system logs, identifying potential security incidents, generating structured incident reports, and communicating with the ThreatContainmentBridge to mitigate active threats.

## 1. Triggering the Agent
The Multi-Agent Orchestrator automatically invokes the forensics workflow when it detects suspicious activities or when the user explicitly requests a security audit (e.g., "Check my system logs for any unauthorized access attempts").
During an automated trigger:
- The Orchestrator routes the log analysis request to the **Forensics Agent**.
- The Agent pulls relevant system logs (e.g., `/var/log/auth.log`, Windows Event Logs).
- The Agent applies pattern matching and LLM-assisted analysis to detect anomalies.

## 2. ThreatContainmentBridge & PID Translation
When a malicious process is identified, the Forensics Agent uses the **ThreatContainmentBridge** to handle mitigation.
The bridge includes a **rule-based PID translation helper** which safely translates service names, ports, or application signatures into precise Process IDs (PIDs). This ensures the correct process is targeted without affecting legitimate system operations. 

## 3. Tier 3/4 Security Gate Confirmation Loop
The Forensics Agent cannot autonomously destroy system components. Any containment action (such as killing a PID or modifying firewall rules) requires strict user confirmation.
- **Tier 3 (Destructive):** Actions like killing a user-level process require explicit confirmation.
- **Tier 4 (Root Critical):** Actions like modifying iptables, changing system configurations, or killing root processes trigger the maximum security gate confirmation loop, often requiring elevated credentials.

## 4. The JSON Schema
The Forensics Agent outputs incident reports in a structured JSON schema. This allows seamless integration with the Heliox OS dashboard and the Orchestrator.

```json
{
  "incident_id": "INC-20260709-001",
  "timestamp": "2026-07-09T10:00:00Z",
  "incident_type": "brute_force_ssh",
  "severity": "CRITICAL",
  "source_ip": "192.168.1.100",
  "target_service": "sshd",
  "description": "Detected 50 failed login attempts within 2 minutes.",
  "proposed_resolution": {
    "action": "block_ip",
    "parameters": {
      "ip": "192.168.1.100",
      "duration": "24h"
    }
  }
}
```

### Schema Fields
- `incident_type`: A categorized string identifying the threat (e.g., `brute_force_ssh`, `malware_detected`, `unauthorized_access`).
- `severity`: The calculated threat level (see Severity Matrix below).
- `proposed_resolution`: The actionable steps the ThreatContainmentBridge will take if approved by the user.

## 5. Severity Matrix

| Severity Level | Definition | Automated Action | Examples |
|----------------|------------|------------------|----------|
| **INFO** | Normal operations, audits, or minor anomalies with no immediate threat. | Logged only, no alerts. | Scheduled security scans, routine user logins. |
| **WARNING** | Suspicious activity that warrants administrator attention but is not immediately destructive. | Prompts a low-priority notification. | Failed login attempts (low frequency), unknown USB device insertion. |
| **CRITICAL** | Active threat that could lead to system compromise, data loss, or significant downtime. | Triggers Tier 3/4 Security Gate and immediate lockdown proposal. | Brute-force SSH attack, ransomware activity detected, unauthorized root escalation. |

## 6. Example Workflow: SSH Brute-Force Containment

Here is a step-by-step example of how the Forensics Agent handles a brute-force SSH attack:

1. **Detection:** The Monitor Agent notices a spike in CPU usage from `sshd` and alerts the Orchestrator.
2. **Analysis:** The Orchestrator routes the alert to the Forensics Agent, which parses `/var/log/auth.log` and detects 100 failed login attempts from IP `203.0.113.42`.
3. **Report Generation:** The Forensics Agent generates a JSON incident report with `severity: CRITICAL` and `incident_type: brute_force_ssh`.
4. **Resolution Proposal:** The proposed resolution is to block the IP via the ThreatContainmentBridge.
5. **Security Gate:** The system triggers a Tier 4 confirmation loop: *"CRITICAL THREAT DETECTED: Brute-force SSH attack from 203.0.113.42. Block IP in firewall? (Y/N)"*.
6. **Execution:** The administrator approves the action. The ThreatContainmentBridge adds the firewall rule and terminates any active connections from that IP.
