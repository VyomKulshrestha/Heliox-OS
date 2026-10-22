<script lang="ts">
  import QRCode from "qrcode";
  import { onDestroy, onMount } from "svelte";
  import { airHandoff } from "../stores/airHandoff";
  import ConfirmPrompt from "./ConfirmPrompt.svelte";

  let qrDataUrl = $state("");
  let refreshTimer: ReturnType<typeof setInterval> | null = null;
  let pendingRevoke = $state<{ deviceId: string; name: string } | null>(null);

  $effect(() => {
    const pairingUrl = $airHandoff.pairing?.pairing_url || "";
    qrDataUrl = "";
    if (pairingUrl) {
      void QRCode.toDataURL(pairingUrl, {
        width: 260,
        margin: 1,
        color: { dark: "#090b12", light: "#f3f6ff" },
        errorCorrectionLevel: "M",
      }).then((value) => {
        if ($airHandoff.pairing?.pairing_url === pairingUrl) qrDataUrl = value;
      });
    }
  });

  onMount(() => {
    void airHandoff.refresh();
    refreshTimer = setInterval(() => void airHandoff.refresh(), 2500);
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
  });

  async function toggleEnabled() {
    try {
      await airHandoff.setEnabled(!$airHandoff.enabled);
    } catch {
      // The store surfaces the daemon error in the panel.
    }
  }

  async function startPairing() {
    try {
      await airHandoff.startPairing();
    } catch {
      // The store surfaces the daemon error in the panel.
    }
  }

  async function cancelPairing() {
    try {
      await airHandoff.cancelPairing();
    } catch {
      // The store surfaces the daemon error in the panel.
    }
  }

  function requestRevokeDevice(deviceId: string, name: string) {
    pendingRevoke = { deviceId, name };
  }

  async function confirmRevokeDevice() {
    const device = pendingRevoke;
    pendingRevoke = null;
    if (!device) return;
    try {
      await airHandoff.revokeDevice(device.deviceId);
    } catch {
      // The store surfaces the daemon error in the panel.
    }
  }

  function formatSeen(timestamp: number): string {
    if (!timestamp) return "Never";
    return new Date(timestamp * 1000).toLocaleString();
  }
</script>

{#if pendingRevoke}
  <ConfirmPrompt
    message={`REVOKE ${pendingRevoke.name}?\n\nThis phone will immediately lose access to future Air Handoff transfers. Pair it again to restore access.`}
    confirmLabel="Revoke phone"
    danger={true}
    onconfirm={() => void confirmRevokeDevice()}
    oncancel={() => (pendingRevoke = null)}
  />
{/if}

<section class="handoff-card" aria-labelledby="air-handoff-heading">
  <div class="section-title">
    <div>
      <span class="eyebrow">Cross-device</span>
      <h3 id="air-handoff-heading">Air Handoff</h3>
      <p>Grab the current screen with a fist, then push an open palm to send it to one paired phone.</p>
    </div>
    <button
      class="switch"
      class:on={$airHandoff.enabled}
      type="button"
      role="switch"
      aria-checked={$airHandoff.enabled}
      aria-label="Toggle Air Handoff"
      disabled={$airHandoff.busy}
      onclick={toggleEnabled}><span></span></button
    >
  </div>

  <div class="security-note">
    <strong>Private by design</strong>
    <span
      >Off by default. The desktop credential stays in the OS keyring; the paired phone credential stays in that
      browser's local storage. Transfer metadata and content are encrypted for one selected device. The receiver cannot
      control Heliox. Use only a trusted LAN.</span
    >
  </div>

  {#if !$airHandoff.secure_storage_available && $airHandoff.enabled}
    <div class="error">Secure OS credential storage is unavailable. Pairing is blocked.</div>
  {/if}
  {#if $airHandoff.error}<div class="error">{$airHandoff.error}</div>{/if}

  {#if $airHandoff.enabled}
    <div class="receiver-row">
      <div>
        <span class="label">Phone receiver</span>
        <code>{$airHandoff.receiver_url || "Starting..."}</code>
      </div>
      <span class:online={$airHandoff.running} class="status">
        {$airHandoff.running ? "LAN receiver online" : "Receiver offline"}
      </span>
    </div>

    <div class="pairing-actions">
      {#if $airHandoff.pairing}
        <button type="button" class="secondary" onclick={cancelPairing}>Cancel pairing</button>
      {:else}
        <button
          type="button"
          class="primary"
          disabled={!$airHandoff.running || $airHandoff.busy}
          onclick={startPairing}
        >
          Pair a phone
        </button>
      {/if}
    </div>

    {#if $airHandoff.pairing}
      <div class="pairing-grid">
        <div class="qr-shell">
          {#if qrDataUrl}<img src={qrDataUrl} alt="Air Handoff phone pairing QR code" />{/if}
        </div>
        <div class="pairing-copy">
          <strong>Scan this QR with the phone</strong>
          <ol>
            <li>Keep the phone and computer on the same trusted Wi-Fi.</li>
            <li>Open the QR link and name the phone.</li>
            <li>Tap <em>Pair securely</em> on the phone.</li>
          </ol>
          <small>The one-time pairing link expires in five minutes.</small>
        </div>
      </div>
    {/if}

    <div class="requests" data-testid="phone-requests">
      <label class="request-option">
        <input
          type="checkbox"
          checked={$airHandoff.accept_requests}
          disabled={$airHandoff.busy}
          onchange={(event) =>
            void airHandoff.configureRequests(event.currentTarget.checked, $airHandoff.run_read_only_requests)}
        />
        <span>
          <strong>Accept requests from paired phones</strong>
          <small>A request waits here until you press Run, and every action still asks for approval here.</small>
        </span>
      </label>
      <label class="request-option">
        <input
          type="checkbox"
          checked={$airHandoff.run_read_only_requests}
          disabled={$airHandoff.busy || !$airHandoff.accept_requests}
          onchange={(event) => void airHandoff.configureRequests(true, event.currentTarget.checked)}
        />
        <span>
          <strong>Start status checks right away</strong>
          <small
            >Runs a request at once, limited to status and file names: no file contents, clipboard, email, screen or
            changes. The answer goes back to the phone.</small
          >
        </span>
      </label>
      {#each ($airHandoff.requests ?? []).filter((item) => item.status === "waiting") as request (request.request_id)}
        <div class="request">
          <span><strong>{request.device_name}</strong> asks: {request.text}</span>
          <span class="request-buttons">
            <button type="button" class="primary" onclick={() => void airHandoff.runRequest(request.request_id)}
              >Run</button
            >
            <button type="button" class="revoke" onclick={() => void airHandoff.dismissRequest(request.request_id)}
              >Dismiss</button
            >
          </span>
        </div>
      {/each}
    </div>

    <div class="devices">
      <div class="devices-heading">
        <strong>Paired phones</strong>
        <span>{$airHandoff.paired_devices.length}</span>
      </div>
      {#if $airHandoff.paired_devices.length === 0}
        <p class="empty">No phone is paired yet.</p>
      {:else}
        {#each $airHandoff.paired_devices as device (device.device_id)}
          <div class="device" class:selected={$airHandoff.selectedDeviceId === device.device_id}>
            <label>
              <input
                type="radio"
                name="air-handoff-target"
                value={device.device_id}
                checked={$airHandoff.selectedDeviceId === device.device_id}
                onchange={() => airHandoff.selectDevice(device.device_id)}
              />
              <span>
                <strong>{device.name}</strong>
                <small>Last seen {formatSeen(device.last_seen_at)}</small>
              </span>
            </label>
            <button
              type="button"
              class="revoke"
              disabled={$airHandoff.busy}
              onclick={() => requestRevokeDevice(device.device_id, device.name)}>Revoke</button
            >
          </div>
        {/each}
      {/if}
    </div>
  {/if}
</section>

<style>
  .requests {
    display: grid;
    gap: 10px;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }
  .request-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 12px;
  }
  .request-option small {
    display: block;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .request {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 9px;
    font-size: 12px;
    overflow-wrap: anywhere;
  }
  .request-buttons {
    display: flex;
    flex-shrink: 0;
    gap: 6px;
  }
  .handoff-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title,
  .receiver-row,
  .devices-heading,
  .device {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
  }
  .section-title {
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
  .security-note span {
    color: var(--text-secondary);
    line-height: 1.5;
  }
  .receiver-row,
  .pairing-actions,
  .devices {
    padding: 15px 20px;
  }
  .receiver-row code {
    display: block;
    margin-top: 7px;
    color: var(--text-primary);
  }
  .status {
    color: var(--danger);
    font: 700 10px var(--font-mono);
    text-transform: uppercase;
  }
  .status.online {
    color: var(--success);
  }
  button.primary,
  button.secondary,
  button.revoke {
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 12px;
    color: var(--text-primary);
    cursor: pointer;
  }
  button.primary {
    border-color: var(--accent);
    background: var(--accent);
    color: #080a12;
    font-weight: 800;
  }
  button.secondary,
  button.revoke {
    background: var(--bg-primary);
  }
  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .pairing-grid {
    display: grid;
    grid-template-columns: 180px 1fr;
    gap: 20px;
    margin: 0 20px 16px;
    padding: 18px;
    border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--border));
    border-radius: 10px;
    background: var(--bg-primary);
  }
  .qr-shell {
    padding: 8px;
    border-radius: 8px;
    background: #f3f6ff;
  }
  .qr-shell img {
    display: block;
    width: 100%;
    height: auto;
  }
  .pairing-copy ol {
    margin: 12px 0;
    padding-left: 18px;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.7;
  }
  .pairing-copy small,
  .device small {
    display: block;
    color: var(--text-secondary);
  }
  .devices {
    border-top: 1px solid var(--border);
  }
  .devices-heading span {
    color: var(--accent);
    font: 700 11px var(--font-mono);
  }
  .device {
    margin-top: 10px;
    padding: 11px;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .device.selected {
    border-color: color-mix(in srgb, var(--accent) 65%, var(--border));
  }
  .device label {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
  }
  .device small {
    margin-top: 3px;
    font-size: 10px;
  }
  button.revoke {
    padding: 5px 9px;
    color: var(--danger);
    font-size: 10px;
  }
  .empty {
    padding: 18px 0 3px;
    text-align: center;
  }
  .error {
    margin: 12px 20px;
    padding: 10px 12px;
    border: 1px solid color-mix(in srgb, var(--danger) 55%, var(--border));
    border-radius: 7px;
    color: var(--danger);
    background: color-mix(in srgb, var(--danger) 8%, transparent);
    font-size: 11px;
  }
  @media (max-width: 720px) {
    .pairing-grid {
      grid-template-columns: 1fr;
    }
    .qr-shell {
      max-width: 220px;
    }
    .security-note {
      grid-template-columns: 1fr;
      gap: 5px;
    }
  }
</style>
