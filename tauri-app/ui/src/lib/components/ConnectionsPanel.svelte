<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type Service = "mail" | "slack" | "discord" | "github";
  type Hint = { host: string; last4: string } | null;
  type Connection = {
    connected: boolean;
    source: "settings" | "environment" | "";
    environment_also_set: boolean;
    account?: string;
    webhook?: Hint;
    read_back?: boolean;
    bot_token?: boolean;
    channel_id?: string;
  };
  type Status = {
    status: string;
    available?: boolean;
    message?: string;
    field?: string;
    connections?: Record<Service, Connection>;
  };

  const titles: Record<Service, string> = { mail: "Mail", slack: "Slack", discord: "Discord", github: "GitHub" };
  const services: Service[] = ["mail", "slack", "discord", "github"];

  let available = $state(true);
  let storeMessage = $state("");
  let current = $state<Partial<Record<Service, Connection>>>({});
  let busy = $state<Service | "">("");
  let confirming = $state<Service | "">("");
  let notes = $state<Partial<Record<Service, { text: string; failed: boolean }>>>({});
  // Write-only: what the person types is sent once to the daemon and cleared; nothing saved is ever read back.
  let slackWebhook = $state("");
  let slackToken = $state("");
  let slackChannel = $state("");
  let discordWebhook = $state("");
  let githubToken = $state("");

  onMount(() => void refresh());

  function apply(result: Status) {
    if (result.connections) current = result.connections;
    if (result.available !== undefined) available = result.available;
    storeMessage = result.available === false ? result.message || "" : "";
  }

  function note(service: Service, text: string, failed: boolean) {
    notes = { ...notes, [service]: { text, failed } };
  }

  async function refresh() {
    try {
      apply(await call<Status>("connections_status"));
    } catch (error) {
      storeMessage = error instanceof Error ? error.message : "Heliox could not read the connections.";
    }
  }

  function describe(connection: Connection | undefined): string {
    if (!connection?.connected) return "Not connected";
    const from =
      connection.source === "settings" ? "Connected in Settings" : "Connected through an environment variable";
    const hint = connection.webhook ? ` · ${connection.webhook.host}, ends in ${connection.webhook.last4}` : "";
    const account = connection.account ? ` · ${connection.account}` : "";
    return from + hint + account;
  }

  async function save(service: Service, values: Record<string, string>, clear: () => void) {
    if (busy) return;
    const given = Object.fromEntries(Object.entries(values).filter(([, value]) => value.trim()));
    if (Object.keys(given).length === 0) {
      note(service, "Enter a value first.", true);
      return;
    }
    busy = service;
    try {
      const result = await call<Status>("connection_save", { service, values: given });
      if (result.status !== "ok") throw new Error(result.message || "The connection was not saved.");
      apply(result);
      clear();
      note(service, `${titles[service]} saved in the credential store.`, false);
    } catch (error) {
      note(service, error instanceof Error ? error.message : "The connection was not saved.", true);
    } finally {
      busy = "";
    }
  }

  async function remove(service: Service) {
    if (busy) return;
    busy = service;
    confirming = "";
    try {
      const result = await call<Status>("connection_remove", { service });
      if (result.status !== "ok") throw new Error(result.message || "The connection was not removed.");
      apply(result);
      note(
        service,
        result.connections?.[service]?.connected
          ? "Removed from Settings. An environment variable still connects it."
          : `${titles[service]} removed.`,
        false,
      );
    } catch (error) {
      note(service, error instanceof Error ? error.message : "The connection was not removed.", true);
    } finally {
      busy = "";
    }
  }

  async function test(service: Service) {
    if (busy) return;
    busy = service;
    confirming = "";
    note(
      service,
      service === "github" ? "Asking GitHub who this token belongs to…" : "Posting the test message…",
      false,
    );
    try {
      const result = await call<Status>("connection_test", { service, confirm_post: service !== "github" });
      if (result.status !== "ok") throw new Error(result.message || "The test failed.");
      note(service, result.message || "The test passed.", false);
    } catch (error) {
      note(service, error instanceof Error ? error.message : "The test failed.", true);
    } finally {
      busy = "";
    }
  }
</script>

<section class="card" aria-labelledby="connections-heading">
  <div class="heading">
    <span class="eyebrow">Communication</span>
    <h3 id="connections-heading">Connections</h3>
    <p>
      Mail, Slack, Discord and GitHub for Heliox's actions. Secrets go to the operating system's credential store and
      are never shown again; plans never hold them.
    </p>
  </div>
  {#if storeMessage}<p class="message error store">{storeMessage}</p>{/if}

  {#each services as service (service)}
    {@const connection = current[service]}
    <div class="row" data-testid={`connection-${service}`}>
      <div class="row-head">
        <strong>{titles[service]}</strong>
        <span class="state" class:on={connection?.connected}>{describe(connection)}</span>
      </div>
      {#if connection?.environment_also_set}
        <p class="hint">An environment variable is also set; Heliox uses the one saved in Settings.</p>
      {/if}

      {#if service === "mail"}
        <p class="hint">
          Set up the mail account and its app password under Email integration. Both ways Heliox sends mail use that one
          account.
        </p>
      {:else if service === "slack"}
        <div class="fields">
          <label
            ><span>Webhook address</span><input
              type="password"
              autocomplete="off"
              bind:value={slackWebhook}
              placeholder={connection?.source === "settings"
                ? "Leave blank to keep the saved one"
                : "https://hooks.slack.com/services/…"}
            /></label
          >
          <label
            ><span>Bot token, to read messages back (optional)</span><input
              type="password"
              autocomplete="off"
              bind:value={slackToken}
              placeholder={connection?.bot_token ? "Saved; leave blank to keep it" : "xoxb-…"}
            /></label
          >
          <label
            ><span>Channel ID (optional)</span><input
              bind:value={slackChannel}
              placeholder={connection?.channel_id || "C0123ABCD"}
            /></label
          >
        </div>
      {:else if service === "discord"}
        <div class="fields">
          <label
            ><span>Webhook address</span><input
              type="password"
              autocomplete="off"
              bind:value={discordWebhook}
              placeholder={connection?.source === "settings"
                ? "Leave blank to keep the saved one"
                : "https://discord.com/api/webhooks/…"}
            /></label
          >
        </div>
      {:else}
        <div class="fields">
          <label
            ><span>Personal access token</span><input
              type="password"
              autocomplete="off"
              bind:value={githubToken}
              placeholder={connection?.source === "settings"
                ? "Leave blank to keep the saved one"
                : "github_pat_… or ghp_…"}
            /></label
          >
        </div>
      {/if}

      <div class="actions">
        {#if service === "slack"}
          <button
            class="primary"
            type="button"
            disabled={!!busy || !available}
            onclick={() =>
              save("slack", { webhook_url: slackWebhook, bot_token: slackToken, channel_id: slackChannel }, () => {
                slackWebhook = slackToken = slackChannel = "";
              })}>Save</button
          >
        {:else if service === "discord"}
          <button
            class="primary"
            type="button"
            disabled={!!busy || !available}
            onclick={() => save("discord", { webhook_url: discordWebhook }, () => (discordWebhook = ""))}>Save</button
          >
        {:else if service === "github"}
          <button
            class="primary"
            type="button"
            disabled={!!busy || !available}
            onclick={() => save("github", { token: githubToken }, () => (githubToken = ""))}>Save</button
          >
        {/if}
        {#if connection?.source === "settings"}
          <button class="secondary" type="button" disabled={!!busy} onclick={() => remove(service)}
            >{service === "mail" ? "Remove saved app password" : "Remove"}</button
          >
        {/if}
        {#if connection?.connected && (service === "slack" || service === "discord")}
          {#if confirming === service}
            <button class="secondary" type="button" disabled={!!busy} onclick={() => test(service)}
              >Yes, post a test message to the connected {titles[service]} channel</button
            >
            <button class="secondary" type="button" onclick={() => (confirming = "")}>Cancel</button>
          {:else}
            <button class="secondary" type="button" disabled={!!busy} onclick={() => (confirming = service)}
              >Post a test message…</button
            >
          {/if}
        {:else if connection?.connected && service === "github"}
          <button class="secondary" type="button" disabled={!!busy} onclick={() => test("github")}
            >Check the token (reads its own user)</button
          >
        {/if}
      </div>
      {#if notes[service]}
        <p class="message" class:error={notes[service]?.failed} class:ok={!notes[service]?.failed}>
          {notes[service]?.text}
        </p>
      {/if}
    </div>
  {/each}
</section>

<style>
  .card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .heading {
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
  .row {
    display: grid;
    gap: 8px;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }
  .row-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 10px;
  }
  .state {
    color: var(--text-secondary);
    font-size: 12px;
  }
  .state.on {
    color: var(--success);
  }
  .hint {
    font-size: 11px;
  }
  .fields {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }
  label {
    display: grid;
    gap: 6px;
    color: var(--text-secondary);
    font-size: 11px;
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
    flex-wrap: wrap;
    gap: 8px;
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
    color: var(--on-accent);
  }
  .secondary {
    background: transparent;
    color: var(--text-primary);
  }
  button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }
  .message.ok {
    color: var(--success);
  }
  .message.error {
    color: var(--danger);
  }
  .store {
    padding: 0 20px 14px;
  }
  @media (max-width: 900px) {
    .fields {
      grid-template-columns: 1fr;
    }
  }
</style>
