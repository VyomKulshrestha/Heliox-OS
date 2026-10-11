<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  let enabled = $state(false);
  let imapHost = $state("");
  let smtpHost = $state("");
  let smtpPort = $state(587);
  let username = $state("");
  let password = $state("");
  let passwordSaved = $state(false);
  let edited = $state(false);
  let busy = $state(false);
  let message = $state("");
  let failed = $state(false);

  $effect(() => {
    if (!edited) {
      enabled = $settings.email?.enabled ?? false;
      imapHost = $settings.email?.imap_host ?? "";
      smtpHost = $settings.email?.smtp_host ?? "";
      smtpPort = $settings.email?.smtp_port ?? 587;
      username = $settings.email?.username ?? "";
    }
  });

  onMount(() => void refreshCredentialStatus());

  function change() {
    edited = true;
    message = "";
    failed = false;
  }

  async function refreshCredentialStatus() {
    try {
      const result = await call<{ status: string; providers?: string[]; available?: boolean; message?: string }>(
        "list_api_keys",
      );
      if (result.status !== "ok" || result.available === false) {
        throw new Error(result.message || "Secure credential storage is unavailable.");
      }
      passwordSaved = result.providers?.includes("email") ?? false;
    } catch (error) {
      passwordSaved = false;
      failed = true;
      message = error instanceof Error ? error.message : "Secure credential storage is unavailable.";
    }
  }

  async function save() {
    if (busy) return;
    if (enabled && (!imapHost.trim() || !smtpHost.trim() || !username.trim() || (!password.trim() && !passwordSaved))) {
      message = "IMAP host, SMTP host, username, and a securely saved app password are required.";
      failed = true;
      return;
    }
    if (!Number.isInteger(smtpPort) || smtpPort < 1 || smtpPort > 65535) {
      message = "SMTP port must be from 1 to 65535.";
      failed = true;
      return;
    }

    busy = true;
    message = "";
    failed = false;
    try {
      // Save the account first so a refused account never leaves a mismatched password
      // behind; if storing the new password fails, put the previous account back.
      const previous = { ...$settings.email };
      const saved = await settings.updateSectionDetailed("email", {
        enabled,
        imap_host: imapHost.trim(),
        smtp_host: smtpHost.trim(),
        smtp_port: smtpPort,
        username: username.trim(),
        password_provider: "email",
      });
      if (!saved.ok) throw new Error(saved.message || "The daemon did not confirm the email configuration.");
      if (password.trim()) {
        let storeError = "";
        try {
          const stored = await call<{ status: string; message?: string }>("store_api_key", {
            provider: "email",
            api_key: password.trim(),
          });
          if (stored.status !== "ok") storeError = stored.message || "The app password was not stored securely.";
        } catch (error) {
          storeError = error instanceof Error ? error.message : "The app password was not stored securely.";
        }
        if (storeError) {
          await settings.updateSectionDetailed("email", previous);
          throw new Error(`${storeError} The previous email settings were kept.`);
        }
        password = "";
        passwordSaved = true;
      }
      edited = false;
      message = enabled ? "Email settings saved. Test IMAP before using mail actions." : "Email integration disabled.";
    } catch (error) {
      failed = true;
      message = error instanceof Error ? error.message : "Email settings could not be saved.";
    } finally {
      busy = false;
    }
  }

  async function testConnection() {
    if (busy) return;
    busy = true;
    failed = false;
    message = "Authenticating without reading messages…";
    try {
      const result = await call<{ status: string; message?: string }>("email_test_connection");
      if (result.status !== "ok") throw new Error(result.message || "The email account could not be reached.");
      message = "Connected securely to IMAP.";
    } catch (error) {
      failed = true;
      message = error instanceof Error ? error.message : "The email account could not be reached.";
    } finally {
      busy = false;
    }
  }
</script>

<section class="card" aria-labelledby="email-heading">
  <div class="heading">
    <div>
      <span class="eyebrow">Communication</span>
      <h3 id="email-heading">Email integration</h3>
      <p>Fetch and summarize through IMAP; approved sends use SMTP with STARTTLS.</p>
    </div>
    <button
      class="switch"
      class:on={enabled}
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label="Toggle email integration"
      onclick={() => {
        enabled = !enabled;
        change();
      }}><span></span></button
    >
  </div>
  <div class="security-note">
    <strong>Use an app password</strong>
    <span>The secret stays in the OS keyring and is injected only at connection time, never into an AI plan.</span>
  </div>
  <div class="fields">
    <label><span>IMAP host</span><input bind:value={imapHost} oninput={change} placeholder="imap.example.com" /></label>
    <label><span>SMTP host</span><input bind:value={smtpHost} oninput={change} placeholder="smtp.example.com" /></label>
    <label
      ><span>SMTP port</span><input type="number" min="1" max="65535" bind:value={smtpPort} oninput={change} /></label
    >
    <label><span>Username</span><input bind:value={username} oninput={change} autocomplete="username" /></label>
    <label class="password"
      ><span>App password {passwordSaved ? "(saved)" : ""}</span><input
        type="password"
        bind:value={password}
        oninput={change}
        autocomplete="new-password"
        placeholder={passwordSaved ? "Leave blank to keep saved password" : "Required"}
      /></label
    >
  </div>
  <div class="actions">
    <button class="primary" type="button" disabled={busy} onclick={save}>{busy ? "Working…" : "Save"}</button>
    <button class="secondary" type="button" disabled={busy || !enabled || edited} onclick={testConnection}
      >Test IMAP</button
    >
  </div>
  {#if message}<p class:error={failed} class:ok={!failed} class="message">{message}</p>{/if}
</section>

<style>
  .card {
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
  .security-note span {
    color: var(--text-secondary);
  }
  .fields {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    padding: 16px 20px 10px;
  }
  label {
    display: grid;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 11px;
  }
  label.password {
    grid-column: span 2;
  }
  input {
    min-width: 0;
    padding: 9px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  .actions {
    display: flex;
    gap: 8px;
    padding: 6px 20px 16px;
  }
  button.primary,
  button.secondary {
    padding: 8px 12px;
    border: 1px solid var(--accent);
    border-radius: 7px;
    cursor: pointer;
  }
  .primary {
    background: var(--accent);
    color: var(--bg-primary);
  }
  .secondary {
    background: transparent;
    color: var(--text-primary);
  }
  button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }
  .message {
    padding: 0 20px 16px;
  }
  .message.ok {
    color: var(--success, #10b981);
  }
  .message.error {
    color: var(--danger, #ef4444);
  }
  @media (max-width: 900px) {
    .fields {
      grid-template-columns: 1fr;
    }
    label.password {
      grid-column: auto;
    }
  }
</style>
