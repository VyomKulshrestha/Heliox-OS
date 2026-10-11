<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  let enabled = $state(false);
  let caldavUrl = $state("");
  let username = $state("");
  let icsFiles = $state("");
  let password = $state("");
  let passwordSaved = $state(false);
  let edited = $state(false);
  let busy = $state(false);
  let message = $state("");
  let messageType = $state<"ok" | "error" | "">("");

  $effect(() => {
    if (!edited) {
      enabled = $settings.calendar?.enabled ?? false;
      caldavUrl = $settings.calendar?.caldav_url ?? "";
      username = $settings.calendar?.caldav_username ?? "";
      icsFiles = ($settings.calendar?.ics_files ?? []).join("\n");
    }
  });

  onMount(() => {
    void refreshCredentialStatus();
  });

  function markEdited() {
    edited = true;
    message = "";
    messageType = "";
  }

  async function refreshCredentialStatus() {
    try {
      const result = await call<{ status: string; providers?: string[]; available?: boolean; message?: string }>(
        "list_api_keys",
      );
      if (result.status !== "ok" || result.available === false) {
        throw new Error(result.message || "Secure credential storage is unavailable.");
      }
      passwordSaved = result.providers?.includes("caldav") ?? false;
    } catch (error) {
      passwordSaved = false;
      message = error instanceof Error ? error.message : "Secure credential storage is unavailable.";
      messageType = "error";
    }
  }

  async function save() {
    if (busy) return;
    if (enabled && (!caldavUrl.trim() || !username.trim() || (!password.trim() && !passwordSaved))) {
      message = "URL, username, and a securely saved app password are required when CalDAV is enabled.";
      messageType = "error";
      return;
    }

    busy = true;
    message = "";
    messageType = "";
    try {
      const localSources = icsFiles
        .split(/\r?\n/)
        .map((path) => path.trim())
        .filter(Boolean);
      // Save the account first: if the daemon refuses it, the stored password still matches
      // the account it was saved for. If storing the new password then fails, put the
      // previous account back so the two never disagree.
      const previous = { ...$settings.calendar };
      const saved = await settings.updateSectionDetailed("calendar", {
        enabled,
        caldav_url: caldavUrl.trim(),
        caldav_username: username.trim(),
        caldav_password_provider: "caldav",
        ics_files: localSources,
      });
      if (!saved.ok) throw new Error(saved.message || "The daemon did not confirm the calendar configuration.");

      if (password.trim()) {
        let storeError = "";
        try {
          const stored = await call<{ status: string; message?: string }>("store_api_key", {
            provider: "caldav",
            api_key: password.trim(),
          });
          if (stored.status !== "ok") storeError = stored.message || "The password was not stored securely.";
        } catch (error) {
          storeError = error instanceof Error ? error.message : "The password was not stored securely.";
        }
        if (storeError) {
          await settings.updateSectionDetailed("calendar", previous);
          throw new Error(`${storeError} The previous calendar settings were kept.`);
        }
        password = "";
        passwordSaved = true;
      }
      edited = false;
      message = enabled
        ? `Calendar settings saved with ${localSources.length} local source${localSources.length === 1 ? "" : "s"}. Test the CalDAV connection before using remote actions.`
        : localSources.length > 0
          ? `Saved ${localSources.length} local calendar source${localSources.length === 1 ? "" : "s"}; CalDAV remains disabled.`
          : "CalDAV disabled.";
      messageType = "ok";
    } catch (error) {
      message = error instanceof Error ? error.message : "Calendar settings could not be saved.";
      messageType = "error";
    } finally {
      busy = false;
    }
  }

  async function testConnection() {
    if (busy) return;
    busy = true;
    message = "Checking the saved account without changing events…";
    messageType = "";
    try {
      const result = await call<{ status: string; message?: string; calendars?: string[] }>("calendar_test_connection");
      if (result.status !== "ok") throw new Error(result.message || "The account could not be reached.");
      const count = result.calendars?.length ?? 0;
      message = `Connected. Heliox found ${count} calendar${count === 1 ? "" : "s"}.`;
      messageType = "ok";
    } catch (error) {
      message = error instanceof Error ? error.message : "The account could not be reached.";
      messageType = "error";
    } finally {
      busy = false;
    }
  }
</script>

<section class="calendar-card" aria-labelledby="calendar-heading">
  <div class="heading">
    <div>
      <span class="eyebrow">Productivity</span>
      <h3 id="calendar-heading">Calendar integration</h3>
      <p>Connect a CalDAV account for real list, create, and explicitly approved delete actions.</p>
    </div>
    <button
      class="switch"
      class:on={enabled}
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label="Toggle CalDAV integration"
      onclick={() => {
        enabled = !enabled;
        markEdited();
      }}><span></span></button
    >
  </div>

  <div class="security-note">
    <strong>Credential safety</strong>
    <span>The password is stored only in the operating-system keyring. Remote accounts must use HTTPS.</span>
  </div>

  <div class="fields">
    <label>
      <span>CalDAV URL</span>
      <input
        type="url"
        value={caldavUrl}
        placeholder="https://calendar.example.com/dav"
        autocomplete="url"
        oninput={(event) => {
          caldavUrl = event.currentTarget.value;
          markEdited();
        }}
      />
    </label>
    <label>
      <span>Username</span>
      <input
        value={username}
        placeholder="you@example.com"
        autocomplete="username"
        oninput={(event) => {
          username = event.currentTarget.value;
          markEdited();
        }}
      />
    </label>
    <label>
      <span>App password {passwordSaved ? "(saved)" : ""}</span>
      <input
        type="password"
        value={password}
        placeholder={passwordSaved ? "Leave blank to keep saved password" : "Required"}
        autocomplete="new-password"
        oninput={(event) => {
          password = event.currentTarget.value;
          markEdited();
        }}
      />
    </label>
  </div>

  <label class="local-sources">
    <span>Local .ics files <small>(one absolute path per line)</small></span>
    <textarea
      value={icsFiles}
      rows="3"
      placeholder="C:\Users\you\Calendar\personal.ics"
      oninput={(event) => {
        icsFiles = event.currentTarget.value;
        markEdited();
      }}></textarea>
    <small
      >Local files remain available when CalDAV is off. Invalid or unreadable sources are reported without suppressing
      other calendars.</small
    >
  </label>

  <div class="actions">
    <button class="primary" type="button" disabled={busy} onclick={save}>{busy ? "Working…" : "Save"}</button>
    <button class="secondary" type="button" disabled={busy || !enabled || edited} onclick={testConnection}
      >Test connection</button
    >
  </div>
  {#if message}<p class:ok={messageType === "ok"} class:error={messageType === "error"} class="message">
      {message}
    </p>{/if}
</section>

<style>
  .calendar-card {
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
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    padding: 16px 20px 10px;
  }
  label {
    display: grid;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 11px;
  }
  input,
  textarea {
    min-width: 0;
    padding: 9px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  textarea {
    resize: vertical;
    font-family: var(--font-mono);
  }
  .local-sources {
    padding: 0 20px 10px;
  }
  .local-sources small {
    color: var(--text-tertiary, var(--text-secondary));
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
  @media (max-width: 850px) {
    .fields {
      grid-template-columns: 1fr;
    }
  }
</style>
