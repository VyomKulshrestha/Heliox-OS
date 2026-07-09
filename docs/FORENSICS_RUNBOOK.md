# Forensics Agent Runbook

This runbook outlines the operational architecture, containment workflows, and schema definitions for the **Forensics Agent** and its autonomous integration with the **Orchestrator** and **System Agent** within Heliox-OS. It serves as an authoritative manual for system administrators and security operators.

---

## Overview

The **Forensics Agent** is a specialized, autonomous threat-detection component designed to maintain real-time visibility across Heliox-OS. It continuously inspects system logs, monitors session timelines, correlates cross-service events, and detects active operational anomalies.

By integrating directly with the **Agent Orchestrator** and utilizing the **Threat Containment Bridge**, the Forensics Agent acts as the telemetry provider in an autonomous loop. It not only detects threats but also triggers swift, structured containment plans to neutralize active hazards before they compromise system integrity.

```
+--------------------+      Log Analysis      +------------------+
|   System Logs /    | ---------------------> |  Forensics Agent |
|    Auth Streams    |                        +------------------+
+--------------------+                                 |
                                             Generates CRITICAL Report
                                                       v
+--------------------+     Routes Actions     +------------------+
|    System Agent    | <--------------------- |  Threat Contain- |
| (Executes Kill/OS) |                        |   ment Bridge    |
+--------------------+                        +------------------+
                                                       |
                                            Audits & Broadcasts Alert
                                                       v
                                              [WebSocket / UI Gate]
```

---

## Triggering & Containment

The containment pipeline is engineered to react to log anomalies without human delay while maintaining a fail-safe manual confirmation policy for destructive actions.

### 1. Ingestion and Triggering
When system events occur, the **Orchestrator** dispatches tasks with the `ActionType.LOG_ANALYZE` capability to the **Forensics Agent**. The agent parses the logs (e.g., `syslog`, `/var/log/auth.log`, Nginx traffic logs) and chronologically correlates session activities using the underlying LLM capability.

### 2. Autonomous Containment Loop (The Bridge)
When a log analysis task returns an incident report with a `CRITICAL` severity level:
1. **Interception**: The `ForensicsAgent` intercepts the result in a non-blocking background thread (`_intercept_critical_threats`) and passes it directly to the `ThreatContainmentBridge`.
2. **Deterministic Translation**: Rather than relying on LLM-based commands—which present hallucination risks—the `ThreatContainmentBridge` uses a **deterministic, rule-based translator** (`translate_resolution`). It scans the `proposed_resolution` text via regular expressions (matching patterns like `kill pid \d+` or `process \d+`) or directly checks the `affected_pids` array to isolate the offending Process IDs (PIDs).
3. **Action Generation**: If PIDs are successfully isolated, the bridge generates one or more `PROCESS_KILL` actions targeted at the malicious processes. If no PIDs are found, it falls back to a destructive `SHELL_COMMAND` mapping the `proposed_resolution` command.
4. **Security Gate Escalation**: To safeguard the operating system, all translated containment actions are forced to carry `destructive=True`. This automatically classifies them under **Permission Tier 3 (DESTRUCTIVE)** (or Tier 4 `ROOT_CRITICAL` if root-level escalation is required).

### 3. Tier 3/4 Security Gate Confirmation
Because the generated plan is classified as destructive, it triggers an operator confirmation workflow:
* **Auditory Hold & WebSocket Alert**: The bridge registers a `PendingConfirmation` in the server's global registry and broadcasts a `threat_confirmation_required` notification containing the incident details, PIDs, and proposed actions.
* **60-Second Timeout Window**: The system enters a 60-second window. During this time, the operator must explicitly approve or deny the action in the UI. If the timer expires without a response, the containment plan is aborted, and a `threat_containment_timeout` event is broadcast.
* **Orchestration to System Agent**: If the operator approves, the Orchestrator receives the confirmed plan and routes it directly to the `System Agent` via `execute_plan()`. The `System Agent` executes the system calls (e.g., sending `SIGKILL` to target processes).
* **Immutable Security Auditing**: Every stage of this containment pipeline is logged to the immutable security audit log (`AuditLogger`). Events are labeled `threat_contained`, `threat_containment_denied`, or `threat_containment_failed` along with execution stats and targeted PIDs.

---

## JSON Output Schema

When the Forensics Agent identifies a `CRITICAL` threat, it must encapsulate its findings within a structured JSON block inside its output. 

### Schema Code Block
```json
{
  "severity": "CRITICAL",
  "incident_type": "brute_force",
  "summary": "Suspicious process spawning under non-privileged account. Multiple failed authorization attempts detected.",
  "proposed_resolution": "Kill malicious process 1042",
  "affected_pids": [1042],
  "timestamp": "2026-06-11T12:00:00.000000Z"
}
```

### Schema Fields Definition

| Field Name | Type | Description | Required / Optional |
| :--- | :--- | :--- | :--- |
| `severity` | `string` | The threat severity classification. Must be strictly normalized to `CRITICAL` to trigger the autonomous containment loop. | Required |
| `incident_type` | `string` | The classification category of the security event (e.g., `brute_force`, `malware_process`, `privilege_escalation`, `unauthorized_access`). | Required |
| `summary` | `string` | A detailed, human-readable description of the log anomaly, timeline mapping, and forensic findings. | Required |
| `proposed_resolution` | `string` | A clear, natural-language action statement explaining how to resolve the incident. Must include the target PID for regex parsing. | Required for `CRITICAL` |
| `affected_pids` | `array of integers` | A list of PIDs identified as directly involved in or spawned by the incident. | Required for `CRITICAL` (can be empty if no process is targeted) |
| `timestamp` | `string` | An ISO 8601 formatted UTC timestamp marking when the log event was analyzed. | Optional |

---

## Severity Matrix

The Forensics Agent categorizes log anomalies based on threat potential, which determines whether the containment bridge triggers.

| Severity | Threat Description / Trigger Criteria | Containment Pipeline Action | Permission Tier | Action Owner |
| :--- | :--- | :--- | :--- | :--- |
| **INFO** | Normal operational events, successful service status changes, user logouts, expected daemon cycles, or standard audit log generation. | No containment. Logs are recorded to standard history. | **Tier 0/1 (READ_ONLY / USER_WRITE)** | Forensics Agent |
| **WARNING** | Anomalous activities presenting no immediate system threat, e.g., failed login attempts below lock threshold, single CPU/RAM spikes, or minor service restart loops. | Generates alert notifications to dashboard. No mitigation actions are created. | **Tier 1/2 (USER_WRITE / SYSTEM_MODIFY)** | Forensics Agent / Orchestrator |
| **CRITICAL** | Clear indicators of compromise, including brute-force authentication attacks, active privilege escalation events, unauthorized reverse shells, or rogue processes. | **Triggers Autonomous Containment Loop**. Generates immediate `PROCESS_KILL` plan and initiates Security Gate hold. | **Tier 3 (DESTRUCTIVE)** (Or Tier 4 `ROOT_CRITICAL` if root access is required) | Threat Containment Bridge, Orchestrator, & System Agent |

---

## Example Workflow (SSH Brute-Force)

This section demonstrates the progression of a live SSH brute-force containment flow from log discovery to mitigation.

### 1. The Incident Logs (`/var/log/auth.log`)
The syslog stream records multiple rapid authentication failures from an external IP targeting the administrative shell:

```text
Jun 11 12:28:10 heliox-os sshd[1042]: Failed password for invalid user admin from 192.168.1.150 port 49210 ssh2
Jun 11 12:28:12 heliox-os sshd[1042]: Failed password for invalid user admin from 192.168.1.150 port 49214 ssh2
Jun 11 12:28:15 heliox-os sshd[1042]: Failed password for invalid user root from 192.168.1.150 port 49218 ssh2
Jun 11 12:28:18 heliox-os sshd[1042]: Failed password for invalid user database from 192.168.1.150 port 49222 ssh2
Jun 11 12:28:20 heliox-os sshd[1042]: Maximum login attempts exceeded for invalid user admin from 192.168.1.150 port 49226 ssh2
```

### 2. Forensics Agent Report Output
The Forensics Agent analyzes this pattern, recognizes an active brute-force threat, and outputs the following JSON report block:

```json
{
  "severity": "CRITICAL",
  "incident_type": "brute_force",
  "summary": "SSH brute-force attack detected from source 192.168.1.150. Over 5 authentication failures in a 10-second window targeting administrative users.",
  "proposed_resolution": "Kill malicious process 1042",
  "affected_pids": [1042],
  "timestamp": "2026-06-11T12:28:21+00:00"
}
```

### 3. Containment Bridge Translation
The Threat Containment Bridge intercepts this report, inspects the JSON, and identifies the `CRITICAL` severity tag. It executes the translation logic:
1. Extract PID `1042` from `affected_pids` and double-checks the `proposed_resolution` string via regex.
2. Constructs a `PROCESS_KILL` Action targeted at PID `1042` with signal `SIGKILL`.
3. Marks the action with `destructive=True` to elevate it to **Permission Tier 3 (DESTRUCTIVE)**.
4. Pauses the pipeline and broadcasts a `threat_confirmation_required` notification.

### 4. Mitigation Command Execution
Upon receiving explicit approval from the operator via the WebSocket gate, the Orchestrator routes the action to the **System Agent**, which executes the target system call:

```bash
# Executed by System Agent via system-level process controller
kill -9 1042
```

The attack vector is closed, and an audit log event `threat_contained` is appended to the JSONL log stream:

```json
{"event": "threat_contained", "severity": "CRITICAL", "incident_type": "brute_force", "proposed_resolution": "Kill malicious process 1042", "affected_pids": [1042], "user_confirmed": true, "execution_success": true, "pids_killed": [1042], "action_count": 1, "timestamp": 1781267301.2}
```
