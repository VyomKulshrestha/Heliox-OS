import re
from typing import Any, Dict, List

# SSH regexes
SSH_FAILED_PATTERN = re.compile(
    r'Failed\s+(?:password|publickey)\s+for\s+(invalid\s+user\s+)?(\S+)\s+from\s+(\S+)\s+port\s+(\d+)\s+ssh2',
    re.IGNORECASE
)
SSH_ACCEPTED_PATTERN = re.compile(
    r'Accepted\s+(?:password|publickey)\s+for\s+(\S+)\s+from\s+(\S+)\s+port\s+(\d+)\s+ssh2',
    re.IGNORECASE
)

# Sudo regexes
SUDO_PATTERN = re.compile(
    r'(\S+)\s+:\s+TTY=\S+\s+;\s+PWD=\S+\s+;\s+USER=(\S+)\s+;\s+COMMAND=(.*)',
    re.IGNORECASE
)

def extract_auth_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan parsed events for authentications, authorization, and privilege events."""
    auth_events = []
    for ev in events:
        msg = ev["message"]

        # SSH failed password/publickey
        ssh_fail_match = SSH_FAILED_PATTERN.search(msg)
        if ssh_fail_match:
            is_invalid, username, ip_address, port = ssh_fail_match.groups()
            auth_events.append({
                "timestamp": ev["timestamp"],
                "event_type": "ssh_login_failed",
                "username": username,
                "ip_address": ip_address,
                "port": int(port),
                "is_invalid_user": bool(is_invalid),
                "description": f"Failed SSH login attempt for {username} from {ip_address}",
                "original_event": ev
            })
            continue

        # SSH accepted login
        ssh_acc_match = SSH_ACCEPTED_PATTERN.search(msg)
        if ssh_acc_match:
            username, ip_address, port = ssh_acc_match.groups()
            auth_events.append({
                "timestamp": ev["timestamp"],
                "event_type": "ssh_login_success",
                "username": username,
                "ip_address": ip_address,
                "port": int(port),
                "description": f"Successful SSH login for {username} from {ip_address}",
                "original_event": ev
            })
            continue

        # Sudo execution
        sudo_match = SUDO_PATTERN.search(msg)
        if sudo_match:
            tty_user, target_user, command = sudo_match.groups()
            auth_events.append({
                "timestamp": ev["timestamp"],
                "event_type": "sudo_execution",
                "username": tty_user,
                "target_user": target_user,
                "command": command,
                "description": f"User {tty_user} executed command via sudo as {target_user}: {command}",
                "original_event": ev
            })
            continue

        # Generic auth failure / PAM failure
        msg_lower = msg.lower()
        if "authentication failure" in msg_lower or "failed password" in msg_lower or "permission denied" in msg_lower:
            auth_events.append({
                "timestamp": ev["timestamp"],
                "event_type": "auth_failure_generic",
                "username": ev.get("process", "unknown"),
                "description": f"Generic auth failure/denial: {msg}",
                "original_event": ev
            })

    return auth_events
