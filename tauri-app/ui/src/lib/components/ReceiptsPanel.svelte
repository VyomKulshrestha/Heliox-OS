<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type ReceiptRow = {
    receipt_id: string;
    finished_at: string;
    status: string;
    summary: string;
    request: string;
    source: string;
    action_count: number;
    signed: boolean;
  };
  type KeyInfo = { available: boolean; fingerprint: string | null; reason?: string };
  type Check = { valid: boolean; this_computer?: boolean; reasons: string[] };
  type ListResult = { status: string; message?: string; receipts?: ReceiptRow[]; key?: KeyInfo };

  let receipts = $state<ReceiptRow[]>([]);
  let key = $state<KeyInfo | null>(null);
  let error = $state("");
  let notice = $state("");
  let checks = $state<Record<string, Check>>({});
  let busy = $state<Record<string, boolean>>({});
  let chainNote = $state("");

  async function load() {
    error = "";
    try {
      const result = await call<ListResult>("receipts_list", { limit: 20 });
      if (result.status !== "ok") throw new Error(result.message || "Receipts are not available.");
      receipts = result.receipts ?? [];
      key = result.key ?? null;
    } catch (err) {
      error = err instanceof Error ? err.message : "Receipts are not available.";
    }
  }

  async function verify(row: ReceiptRow) {
    busy = { ...busy, [row.receipt_id]: true };
    try {
      const result = await call<{ status: string; message?: string; result?: Check }>("receipt_verify", {
        receipt_id: row.receipt_id,
      });
      if (result.status !== "ok" || !result.result) throw new Error(result.message || "Verification failed.");
      checks = { ...checks, [row.receipt_id]: result.result };
    } catch (err) {
      error = err instanceof Error ? err.message : "Verification failed.";
    } finally {
      busy = { ...busy, [row.receipt_id]: false };
    }
  }

  async function exportReceipt(row: ReceiptRow) {
    busy = { ...busy, [row.receipt_id]: true };
    try {
      const result = await call<{ status: string; message?: string; path?: string }>("receipt_get", {
        receipt_id: row.receipt_id,
        export: true,
      });
      if (result.status !== "ok") throw new Error(result.message || "Export failed.");
      notice = `Saved to ${result.path}`;
    } catch (err) {
      error = err instanceof Error ? err.message : "Export failed.";
    } finally {
      busy = { ...busy, [row.receipt_id]: false };
    }
  }

  async function verifyAll() {
    chainNote = "Checking...";
    try {
      const result = await call<{
        status: string;
        message?: string;
        result?: { valid: boolean; count: number; problems: { file: string; reason: string }[] };
      }>("receipts_verify_chain");
      if (result.status !== "ok" || !result.result) throw new Error(result.message || "Check failed.");
      chainNote = result.result.valid
        ? `All ${result.result.count} receipts are intact and linked.`
        : `${result.result.problems.length} problem(s): ${result.result.problems[0]?.reason ?? ""}`;
    } catch (err) {
      chainNote = err instanceof Error ? err.message : "Check failed.";
    }
  }

  function describe(check: Check): string {
    if (!check.valid) return `Not valid: ${check.reasons.join(" ")}`;
    return check.this_computer ? "Valid, signed by this computer" : "Valid, signed by another key";
  }

  onMount(load);
</script>

<section class="receipts-card" aria-labelledby="receipts-heading" data-testid="receipts-panel">
  <div class="section-title">
    <div>
      <span class="eyebrow">Accountability</span>
      <h3 id="receipts-heading">Task receipts</h3>
      <p>
        After each task that ran actions, Heliox signs a receipt of what it did, what you approved and what happened.
        File contents and outputs appear only as hashes.
      </p>
      {#if key?.available}
        <small>This computer's key: <code>{key.fingerprint}</code></small>
      {:else if key}
        <small>Receipts are not being signed: {key.reason || "no signing key is available."}</small>
      {/if}
    </div>
    <button type="button" onclick={verifyAll}>Check all</button>
  </div>

  {#if chainNote}<div class="note" role="status">{chainNote}</div>{/if}
  {#if notice}<div class="note" role="status">{notice}</div>{/if}
  {#if error}<div class="message" role="alert">{error}</div>{/if}

  {#if receipts.length === 0}
    <p class="empty">No receipts yet.</p>
  {:else}
    <ul class="rows">
      {#each receipts as row (row.receipt_id)}
        <li>
          <div class="row-text">
            <strong>{row.request || row.summary || "Task"}</strong>
            <small
              >{new Date(row.finished_at).toLocaleString()} · {row.status} · {row.action_count} action{row.action_count ===
              1
                ? ""
                : "s"}{row.signed ? "" : " · unsigned"}</small
            >
            {#if checks[row.receipt_id]}
              <small class:bad={!checks[row.receipt_id].valid}>{describe(checks[row.receipt_id])}</small>
            {/if}
          </div>
          <div class="row-buttons">
            <button type="button" disabled={busy[row.receipt_id]} onclick={() => verify(row)}>Verify</button>
            <button type="button" disabled={busy[row.receipt_id]} onclick={() => exportReceipt(row)}>Export</button>
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .receipts-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title {
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
  small.bad {
    color: #ff8b94;
  }
  code {
    font-size: 10px;
    overflow-wrap: anywhere;
  }
  .eyebrow {
    color: var(--accent);
    font: 700 10px/1 var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .message,
  .note {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    font-size: 11px;
  }
  .message {
    color: #ff8b94;
    background: rgba(255, 70, 85, 0.08);
  }
  .note {
    color: var(--text-secondary);
  }
  .empty {
    padding: 12px 20px 16px;
    border-top: 1px solid var(--border);
  }
  .rows {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .rows li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .row-text {
    min-width: 0;
  }
  .row-text strong {
    display: block;
    overflow: hidden;
    font-size: 12px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-buttons {
    display: flex;
    flex-shrink: 0;
    gap: 6px;
  }
  button {
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
    cursor: pointer;
  }
  button:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>
