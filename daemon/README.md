# Heliox OS Daemon

Python backend for the Heliox OS AI System Control Agent. Provides the agent system (Planner, Executor, Verifier, Code Sanitizer), security layer, and system interface modules.

See the [main README](../README.md) for full project documentation.

## SSH Agent (Remote Host Execution)

Heliox OS includes an optional `SshAgent` that can execute `ssh_command` / `ssh_script` actions on **pre-configured** remote hosts using Paramiko.

### Install

Install the optional SSH dependency set:

```bash
pip install "pilot-daemon[ssh]"
```

### Configure allowed hosts

In your `config.toml`, enable SSH and define allowed destinations (aliases only):

```toml
[ssh]
enabled = true
connect_timeout_seconds = 10
allowed_hosts = [
  { name = "prod-1", hostname = "10.0.0.10", port = 22, username = "ubuntu", private_key_provider = "ssh_prod_1_key", strict_host_key_checking = true },
]
```

### Store SSH keys in the KeyVault

Store the private key PEM in the encrypted vault using the JSON-RPC method `store_api_key` with `provider=<private_key_provider>`.

Notes:
- The vault stores arbitrary provider→secret strings; SSH keys are treated the same as API keys.
- Keys/passphrases are never logged and are only decrypted at call time.
