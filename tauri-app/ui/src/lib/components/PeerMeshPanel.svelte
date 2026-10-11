<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { call } from "../api/daemon";

  type MeshStatus = {
    status: string;
    enabled: boolean;
    configured_enabled: boolean;
    authenticated: boolean;
    secret_configured: boolean;
    reason?: string;
    instance_id?: string;
    peer_count?: number;
    skill_sync_enabled: boolean;
    collab_exec_enabled: boolean;
    port: number;
  };

  let status = $state<MeshStatus | null>(null);
  let secretInput = $state("");
  let skillSyncEnabled = $state(false);
  let collabExecEnabled = $state(false);
  let busy = $state(false);
  let error = $state("");
  let notice = $state("");
  let refreshTimer: ReturnType<typeof setInterval> | null = null;

  // The sharing switches follow the daemon (another window or a reset can change them)
  // until the person flips one here without saving.
  const draftEdited = $derived(
    status !== null &&
      (skillSyncEnabled !== status.skill_sync_enabled || collabExecEnabled !== status.collab_exec_enabled),
  );

  async function refresh(syncDraft = true) {
    try {
      const next = await call<MeshStatus>("mesh_status");
      if (next.status !== "ok") throw new Error(next.reason || "Could not read Peer Mesh status.");
      const followDaemon = syncDraft || !draftEdited;
      status = next;
      if (followDaemon && !busy) {
        skillSyncEnabled = next.skill_sync_enabled;
        collabExecEnabled = next.collab_exec_enabled;
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "Could not read Peer Mesh status.";
    }
  }

  onMount(() => {
    void refresh();
    refreshTimer = setInterval(() => void refresh(false), 3000);
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
  });

  async function configure(enabled = status?.configured_enabled ?? false) {
    busy = true;
    error = "";
    notice = "";
    try {
      // Send only what changed here so a switch set elsewhere is not reverted.
      const result = await call<MeshStatus & { status: string; message?: string }>("mesh_configure", {
        enabled,
        ...(status === null || skillSyncEnabled !== status.skill_sync_enabled
          ? { skill_sync_enabled: skillSyncEnabled }
          : {}),
        ...(status === null || collabExecEnabled !== status.collab_exec_enabled
          ? { collab_exec_enabled: collabExecEnabled }
          : {}),
        ...(secretInput ? { shared_secret: secretInput } : {}),
      });
      if (result.status !== "ok") throw new Error(result.message || "The daemon rejected Peer Mesh settings.");
      secretInput = "";
      notice = enabled ? "Authenticated Peer Mesh is running with the confirmed settings." : "Peer Mesh is disabled.";
      await refresh();
    } catch (err) {
      error = err instanceof Error ? err.message : "Peer Mesh settings were not changed.";
      await refresh(false);
    } finally {
      busy = false;
    }
  }

  async function generateSecret() {
    error = "";
    notice = "";
    try {
      const result = await call<{ status: string; shared_secret?: string; message?: string }>("mesh_generate_secret");
      if (result.status !== "ok" || !result.shared_secret) {
        throw new Error(result.message || "The daemon did not generate a secret.");
      }
      secretInput = result.shared_secret;
      notice = "Generated locally. Copy this once to every trusted Heliox peer, then save on each device.";
    } catch (err) {
      error = err instanceof Error ? err.message : "Could not generate a mesh secret.";
    }
  }

  async function copySecret() {
    if (!secretInput) return;
    try {
      await navigator.clipboard.writeText(secretInput);
      notice = "Secret copied. Treat it like a password and paste it only into trusted Heliox peers.";
    } catch {
      error = "Clipboard access failed. Select and copy the secret manually.";
    }
  }
</script>

<section class="mesh-card" aria-labelledby="peer-mesh-heading">
  <div class="section-title">
    <div>
      <span class="eyebrow">Authenticated collaboration</span>
      <h3 id="peer-mesh-heading">Peer Mesh</h3>
      <p>Share optional compute and plugin source between trusted Heliox computers on the same LAN.</p>
    </div>
    <button
      class="switch"
      class:on={status?.configured_enabled}
      type="button"
      role="switch"
      aria-checked={status?.configured_enabled ?? false}
      aria-label="Toggle Peer Mesh"
      disabled={busy || !status}
      onclick={() => configure(!(status?.configured_enabled ?? false))}><span></span></button
    >
  </div>

  <div class="security-note">
    <strong>Fail-closed trust</strong>
    <span
      >Off by default. Every message is HMAC-authenticated, timestamped, and replay-protected. The shared secret is
      stored only in the operating-system credential vault and must match on every peer. Plugin sync and remote task
      execution are separate opt-ins.</span
    >
  </div>

  {#if error}<div class="message error" role="alert">{error}</div>{/if}
  {#if notice}<div class="message notice" role="status">{notice}</div>{/if}

  <div class="runtime-row">
    <div>
      <span class="label">Runtime</span>
      <strong>{status?.enabled ? "Authenticated and listening" : status?.reason || "Loading..."}</strong>
    </div>
    <span class:online={status?.enabled} class="status-badge">
      {status?.enabled ? `${status.peer_count ?? 0} peer${status.peer_count === 1 ? "" : "s"}` : "Offline"}
    </span>
  </div>

  <div class="secret-row">
    <div class="secret-copy">
      <span class="label">Shared peer secret</span>
      <small
        >{status?.secret_configured
          ? "A secret is stored securely. Enter another only to rotate it."
          : "Generate or enter at least 32 bytes, then save it on every peer."}</small
      >
    </div>
    <div class="secret-controls">
      <input
        type="password"
        autocomplete="new-password"
        bind:value={secretInput}
        placeholder={status?.secret_configured ? "Stored in OS vault" : "Enter a shared secret"}
        aria-label="Peer Mesh shared secret"
      />
      <button type="button" class="secondary" disabled={busy} onclick={generateSecret}>Generate</button>
      <button type="button" class="secondary" disabled={!secretInput || busy} onclick={copySecret}>Copy</button>
    </div>
  </div>

  <div class="options">
    <label>
      <input type="checkbox" bind:checked={skillSyncEnabled} />
      <span
        ><strong>Plugin source sync</strong><small>Install syntax-checked source from authenticated peers.</small></span
      >
    </label>
    <label>
      <input type="checkbox" bind:checked={collabExecEnabled} />
      <span
        ><strong>Collaborative execution</strong><small
          >Delegate portable system telemetry; web access, UI, files, and commands stay local.</small
        ></span
      >
    </label>
    <button type="button" class="primary" disabled={busy || !status} onclick={() => configure()}>
      {busy ? "Applying..." : "Save and reconcile"}
    </button>
  </div>
</section>

<style>
  .mesh-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title,
  .runtime-row,
  .secret-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 20px;
  }
  h3 {
    margin: 3px 0 5px;
    font-size: 16px;
  }
  p,
  small {
    display: block;
    margin: 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .eyebrow,
  .label {
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
  .message {
    padding: 10px 20px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
  }
  .error {
    color: #ff8b94;
    background: rgba(255, 70, 85, 0.08);
  }
  .notice {
    color: #8ee8d5;
    background: rgba(24, 214, 176, 0.06);
  }
  .runtime-row strong,
  .secret-copy small {
    display: block;
    margin-top: 7px;
    max-width: 620px;
    font-size: 12px;
  }
  .status-badge {
    padding: 5px 9px;
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-secondary);
    font: 700 10px/1 var(--font-mono);
  }
  .status-badge.online {
    border-color: #18d6b0;
    color: #18d6b0;
  }
  .secret-row {
    align-items: flex-end;
    border-top: 1px solid var(--border);
  }
  .secret-controls {
    display: flex;
    gap: 8px;
    min-width: min(560px, 55vw);
  }
  .secret-controls input {
    min-width: 0;
    flex: 1;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  .options {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    align-items: center;
    gap: 16px;
    padding: 15px 20px 18px;
    border-top: 1px solid var(--border);
  }
  .options label {
    display: flex;
    align-items: flex-start;
    gap: 9px;
  }
  .options strong {
    display: block;
    font-size: 12px;
  }
  button.primary,
  button.secondary {
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 7px;
    cursor: pointer;
  }
  button.primary {
    border-color: var(--accent);
    background: var(--accent);
    color: #090b12;
    font-weight: 700;
  }
  button.secondary {
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  button:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }
  @media (max-width: 900px) {
    .section-title,
    .runtime-row,
    .secret-row {
      align-items: flex-start;
      flex-direction: column;
    }
    .secret-controls {
      width: 100%;
      min-width: 0;
      flex-wrap: wrap;
    }
    .options {
      grid-template-columns: 1fr;
    }
    .security-note {
      grid-template-columns: 1fr;
    }
  }
</style>
