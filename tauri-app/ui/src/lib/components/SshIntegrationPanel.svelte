<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type Host = {
    name: string;
    hostname: string;
    port: number;
    username: string;
    strict_host_key_checking: boolean;
    has_private_key: boolean;
    has_passphrase: boolean;
  };

  let hosts = $state<Host[]>([]);
  let enabled = $state(false);
  let name = $state("");
  let hostname = $state("");
  let port = $state(22);
  let username = $state("");
  let privateKey = $state("");
  let passphrase = $state("");
  let strictHostKeyChecking = $state(true);
  let editingExisting = $state(false);
  let busy = $state(false);
  let message = $state("");
  let messageType = $state<"ok" | "error" | "">("");
  let confirmDelete = $state("");

  onMount(() => void refresh());

  async function refresh() {
    try {
      const result = await call<{ status: string; enabled: boolean; hosts: Host[]; message?: string }>(
        "ssh_list_hosts",
      );
      if (result.status !== "ok") throw new Error(result.message || "SSH hosts could not be loaded.");
      if (!Array.isArray(result.hosts)) throw new Error("The daemon's SSH host list is incomplete.");
      enabled = Boolean(result.enabled);
      hosts = result.hosts;
    } catch (error) {
      message = error instanceof Error ? error.message : "SSH hosts could not be loaded.";
      messageType = "error";
    }
  }

  async function setEnabled(next: boolean) {
    if (busy) return;
    busy = true;
    try {
      const result = await call<{ status: string; enabled?: boolean; message?: string }>("ssh_set_enabled", {
        enabled: next,
      });
      if (result.status !== "ok") throw new Error(result.message || "SSH state was not saved.");
      enabled = result.enabled ?? next;
      message = enabled ? "SSH integration enabled." : "SSH integration disabled; saved hosts were kept.";
      messageType = "ok";
    } catch (error) {
      message = error instanceof Error ? error.message : "SSH state was not saved.";
      messageType = "error";
    } finally {
      busy = false;
    }
  }

  function clearEditor() {
    name = "";
    hostname = "";
    port = 22;
    username = "";
    privateKey = "";
    passphrase = "";
    strictHostKeyChecking = true;
    editingExisting = false;
    confirmDelete = "";
  }

  function edit(host: Host) {
    name = host.name;
    hostname = host.hostname;
    port = host.port;
    username = host.username;
    privateKey = "";
    passphrase = "";
    strictHostKeyChecking = host.strict_host_key_checking;
    editingExisting = true;
    message = "";
    messageType = "";
    confirmDelete = "";
  }

  async function save() {
    if (busy) return;
    if (!name.trim() || !hostname.trim() || !username.trim() || (!editingExisting && !privateKey.trim())) {
      message = "Alias, hostname, username, and a private key are required for a new host.";
      messageType = "error";
      return;
    }
    busy = true;
    message = "Saving the host and key securely…";
    messageType = "";
    try {
      const result = await call<{ status: string; message?: string }>("ssh_save_host", {
        name: name.trim(),
        hostname: hostname.trim(),
        port,
        username: username.trim(),
        private_key: privateKey,
        passphrase,
        strict_host_key_checking: strictHostKeyChecking,
        enabled,
      });
      if (result.status !== "ok") throw new Error(result.message || "The SSH host was not saved.");
      privateKey = "";
      passphrase = "";
      editingExisting = true;
      message = "SSH host saved. Test authentication before asking Heliox to run a remote command.";
      messageType = "ok";
      await refresh();
    } catch (error) {
      message = error instanceof Error ? error.message : "The SSH host was not saved.";
      messageType = "error";
    } finally {
      busy = false;
    }
  }

  async function testHost(hostName: string) {
    if (busy) return;
    busy = true;
    message = `Authenticating to ${hostName} without running a command…`;
    messageType = "";
    try {
      const result = await call<{ status: string; message?: string }>("ssh_test_connection", { name: hostName });
      if (result.status !== "ok") throw new Error(result.message || "SSH authentication failed.");
      message = `Connected to ${hostName}. No remote command was executed.`;
      messageType = "ok";
    } catch (error) {
      message = error instanceof Error ? error.message : "SSH authentication failed.";
      messageType = "error";
    } finally {
      busy = false;
    }
  }

  async function remove(hostName: string) {
    if (confirmDelete !== hostName) {
      confirmDelete = hostName;
      message = `Press Remove ${hostName} again to delete its alias and saved credentials.`;
      messageType = "error";
      return;
    }
    busy = true;
    try {
      const result = await call<{ status: string; message?: string }>("ssh_delete_host", { name: hostName });
      if (result.status !== "ok") throw new Error(result.message || "The SSH host was not removed.");
      if (name === hostName) clearEditor();
      message = `${hostName} and its saved SSH credentials were removed.`;
      messageType = "ok";
      await refresh();
    } catch (error) {
      message = error instanceof Error ? error.message : "The SSH host was not removed.";
      messageType = "error";
    } finally {
      busy = false;
      confirmDelete = "";
    }
  }
</script>

<section class="ssh-card" aria-labelledby="ssh-heading">
  <div class="heading">
    <div>
      <span class="eyebrow">Remote systems</span>
      <h3 id="ssh-heading">SSH hosts</h3>
      <p>Allowlist named hosts. Every remote command still requires approval.</p>
    </div>
    <button
      class="switch"
      class:on={enabled}
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label="Toggle SSH integration"
      disabled={busy}
      onclick={() => setEnabled(!enabled)}
    >
      <span></span>
    </button>
  </div>

  <div class="security-note">
    <strong>Fail-closed security</strong>
    <span>Private keys and passphrases live only in the OS keyring. Host-key verification stays on by default.</span>
  </div>

  {#if hosts.length}
    <div class="host-list">
      {#each hosts as host}
        <div class="host-row">
          <button class="host-name" type="button" onclick={() => edit(host)}>
            <strong>{host.name}</strong>
            <span>{host.username}@{host.hostname}:{host.port}</span>
          </button>
          <span class:ready={host.has_private_key} class="key-state"
            >{host.has_private_key ? "Key ready" : "No key"}</span
          >
          <button class="secondary" type="button" disabled={busy || !enabled} onclick={() => testHost(host.name)}
            >Test</button
          >
          <button class="danger" type="button" disabled={busy} onclick={() => remove(host.name)}>
            {confirmDelete === host.name ? `Confirm remove` : "Remove"}
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <div class="editor">
    <div class="editor-title">
      <strong>{editingExisting ? `Edit ${name}` : "Add an SSH host"}</strong>
      {#if editingExisting}<button class="link" type="button" onclick={clearEditor}>Add another</button>{/if}
    </div>
    <div class="fields">
      <label><span>Alias</span><input bind:value={name} disabled={editingExisting} placeholder="build-box" /></label>
      <label><span>Hostname or IP</span><input bind:value={hostname} placeholder="10.0.0.7" /></label>
      <label><span>Port</span><input type="number" bind:value={port} min="1" max="65535" /></label>
      <label><span>Username</span><input bind:value={username} autocomplete="username" placeholder="builder" /></label>
    </div>
    <label class="wide">
      <span>Private key {editingExisting ? "(leave blank to keep saved key)" : ""}</span>
      <textarea
        bind:value={privateKey}
        rows="4"
        spellcheck="false"
        autocomplete="off"
        placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"></textarea>
    </label>
    <label class="wide">
      <span>Key passphrase {editingExisting ? "(leave blank to keep saved passphrase)" : "(optional)"}</span>
      <input type="password" bind:value={passphrase} autocomplete="new-password" />
    </label>
    <label class="check"><input type="checkbox" bind:checked={strictHostKeyChecking} /> Require a known host key</label>
    <div class="actions">
      <button class="primary" type="button" disabled={busy} onclick={save}>{busy ? "Working…" : "Save host"}</button>
    </div>
  </div>

  {#if message}<p class:ok={messageType === "ok"} class:error={messageType === "error"} class="message">
      {message}
    </p>{/if}
</section>

<style>
  .ssh-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 18px 20px;
  }
  h3 {
    margin: 3px 0 5px;
    font-size: 16px;
  }
  p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 12px;
  }
  .eyebrow {
    color: var(--accent);
    font: 700 10px/1 var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .switch {
    width: 46px;
    height: 26px;
    padding: 3px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--bg-primary);
    cursor: pointer;
  }
  .switch span {
    display: block;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--text-secondary);
    transition:
      transform 0.2s,
      background 0.2s;
  }
  .switch.on {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 18%, var(--bg-primary));
  }
  .switch.on span {
    transform: translateX(20px);
    background: var(--accent);
  }
  .security-note {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 16px;
    padding: 12px 20px;
    border-block: 1px solid var(--border);
    background: color-mix(in srgb, #18d6b0 7%, transparent);
    font-size: 11px;
  }
  .security-note span,
  .host-name span {
    color: var(--text-secondary);
  }
  .host-list {
    display: grid;
    gap: 8px;
    padding: 14px 20px 0;
  }
  .host-row {
    display: grid;
    grid-template-columns: minmax(180px, 1fr) auto auto auto;
    gap: 8px;
    align-items: center;
    padding: 9px;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .host-name {
    display: grid;
    gap: 2px;
    padding: 0;
    border: 0;
    text-align: left;
    background: none;
    color: var(--text-primary);
    cursor: pointer;
  }
  .key-state {
    color: var(--danger);
    font: 700 10px/1 var(--font-mono);
  }
  .key-state.ready {
    color: var(--success);
  }
  .editor {
    display: grid;
    gap: 12px;
    padding: 16px 20px;
  }
  .editor-title {
    display: flex;
    justify-content: space-between;
  }
  .fields {
    display: grid;
    grid-template-columns: 1fr 1.5fr 0.6fr 1fr;
    gap: 10px;
  }
  label {
    display: grid;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 11px;
  }
  input,
  textarea {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    padding: 9px;
    font: 12px var(--font-mono);
  }
  textarea {
    resize: vertical;
  }
  .wide {
    width: 100%;
  }
  .check {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .check input {
    width: auto;
  }
  .actions {
    display: flex;
    gap: 8px;
  }
  button.primary,
  button.secondary,
  button.danger {
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 12px;
    cursor: pointer;
    font-weight: 700;
  }
  button.primary {
    border-color: var(--accent);
    background: var(--accent);
    color: #080914;
  }
  button.secondary {
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  button.danger {
    background: color-mix(in srgb, var(--danger) 12%, var(--bg-primary));
    color: var(--danger);
  }
  button.link {
    border: 0;
    background: none;
    color: var(--accent);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .message {
    margin: 0 20px 16px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 7px;
  }
  .message.ok {
    border-color: color-mix(in srgb, var(--success) 45%, var(--border));
    color: var(--success);
  }
  .message.error {
    border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
    color: var(--danger);
  }
  @media (max-width: 900px) {
    .fields {
      grid-template-columns: 1fr 1fr;
    }
    .host-row {
      grid-template-columns: 1fr auto;
    }
  }
</style>
