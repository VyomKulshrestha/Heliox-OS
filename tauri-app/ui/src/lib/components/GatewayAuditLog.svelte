<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  interface GatewayAuditEvent {
    id: number;
    timestamp: string;
    plan_id: string;
    action_index: number;
    action_type: string;
    action_family: string;
    target: string;
    source_profile: string;
    permission_tier: string;
    override_applied: boolean;
    override_restricted: boolean;
    decision: string;
    denial_reason: string;
    dry_run: boolean;
    execution_success: boolean | null;
    execution_error: string;
  }

  let events: GatewayAuditEvent[] = $state([]);
  let loading = $state(true);
  let verifyResult: { valid: boolean; checked_entries: number; error: string } | null = $state(null);
  let verifying = $state(false);

  let sourceFilter = $state("");
  let familyFilter = $state("");
  let decisionFilter = $state("");

  onMount(loadEvents);

  async function loadEvents() {
    loading = true;
    try {
      const params: Record<string, string | number> = { limit: 100 };
      if (sourceFilter) params.source_profile = sourceFilter;
      if (familyFilter) params.action_family = familyFilter;
      if (decisionFilter) params.decision = decisionFilter;
      const result = (await call("list_gateway_events", params)) as { status: string; events: GatewayAuditEvent[] };
      events = result.events ?? [];
    } catch {
      events = [];
    } finally {
      loading = false;
    }
  }

  async function verifyIntegrity() {
    verifying = true;
    verifyResult = null;
    try {
      const result = (await call("verify_gateway_audit")) as { valid: boolean; checked_entries: number; error: string };
      verifyResult = result;
    } catch (err) {
      verifyResult = { valid: false, checked_entries: 0, error: String(err instanceof Error ? err.message : err) };
    } finally {
      verifying = false;
    }
  }

  function formatTime(iso: string): string {
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  function decisionClass(decision: string): string {
    if (decision === "denied") return "decision-bad";
    if (decision === "confirm_required") return "decision-partial";
    return "decision-ok";
  }
</script>

<div class="audit-log">
  <div class="log-header">
    <h2>Agent Gateway Audit Log</h2>
    <div class="header-right">
      <span class="count">{events.length} events</span>
      <button class="verify-btn" onclick={verifyIntegrity} disabled={verifying}>
        {verifying ? "Verifying..." : "Verify Integrity"}
      </button>
    </div>
  </div>

  <div class="filter-bar">
    <select class="input-sm" bind:value={sourceFilter} onchange={loadEvents}>
      <option value="">All sources</option>
      <option value="interactive">interactive</option>
      <option value="autonomous">autonomous</option>
      <option value="web_agent">web_agent</option>
      <option value="voice">voice</option>
      <option value="gesture">gesture</option>
    </select>
    <select class="input-sm" bind:value={familyFilter} onchange={loadEvents}>
      <option value="">All families</option>
      <option value="shell">shell</option>
      <option value="browsing">browsing</option>
      <option value="system_control">system_control</option>
      <option value="other">other</option>
    </select>
    <select class="input-sm" bind:value={decisionFilter} onchange={loadEvents}>
      <option value="">Any decision</option>
      <option value="allowed">allowed</option>
      <option value="denied">denied</option>
    </select>
  </div>

  {#if verifyResult}
    <div class="verify-banner" class:valid={verifyResult.valid}>
      {#if verifyResult.valid}
        Chain verified — {verifyResult.checked_entries} entries intact, no tampering detected.
      {:else}
        Chain verification FAILED after {verifyResult.checked_entries} entries: {verifyResult.error}
      {/if}
    </div>
  {/if}

  {#if loading}
    <div class="empty">Loading...</div>
  {:else if events.length === 0}
    <div class="empty">No gateway decisions recorded yet.</div>
  {:else}
    <div class="log-list">
      {#each events as event}
        <div class="log-entry">
          <div class="entry-header">
            <span class="decision-tag {decisionClass(event.decision)}">{event.decision}</span>
            <span class="entry-time">{formatTime(event.timestamp)}</span>
          </div>
          <div class="entry-body">
            <strong>{event.action_type}</strong> on <code>{event.target}</code>
            <span class="source-tag">{event.source_profile}</span>
            <span class="family-tag">{event.action_family}</span>
            <span class="tier-tag">{event.permission_tier}</span>
            {#if event.override_restricted}<span class="override-tag">OVERRIDE-NARROWED</span>{/if}
            {#if event.dry_run}<span class="dry-run-tag">DRY RUN</span>{/if}
          </div>
          {#if event.denial_reason}
            <div class="entry-reason">{event.denial_reason}</div>
          {/if}
          {#if event.execution_success !== null}
            <div class="entry-result" class:failed={!event.execution_success}>
              {event.execution_success ? "Executed successfully" : `Execution failed: ${event.execution_error}`}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .audit-log {
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .log-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 16px 0;
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .verify-btn {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    color: var(--text-primary);
    cursor: pointer;
    transition: background 0.15s;
  }

  .verify-btn:hover:not(:disabled) {
    background: var(--accent);
    color: white;
  }

  .verify-btn:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }

  .filter-bar {
    display: flex;
    gap: 8px;
    padding: 10px 16px 0;
  }

  .verify-banner {
    margin: 10px 16px 0;
    padding: 8px 12px;
    font-size: 12px;
    border-radius: var(--radius-sm);
    background: var(--danger-bg);
    color: var(--danger);
  }

  .verify-banner.valid {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .empty {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 13px;
    padding: 20px;
  }

  .log-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 16px;
  }

  .log-entry {
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 13px;
  }

  .entry-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .entry-time {
    font-size: 11px;
    color: var(--text-muted);
  }

  .entry-body {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .entry-body code {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
  }

  .entry-reason {
    margin-top: 4px;
    font-size: 12px;
    color: var(--danger);
  }

  .entry-result {
    margin-top: 4px;
    font-size: 12px;
    color: var(--success);
  }

  .entry-result.failed {
    color: var(--danger);
  }

  .decision-tag {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 8px;
    border-radius: 10px;
  }

  .decision-ok {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .decision-partial {
    background: rgba(251, 191, 36, 0.1);
    color: var(--warning);
  }

  .decision-bad {
    background: var(--danger-bg);
    color: var(--danger);
  }

  .source-tag,
  .family-tag,
  .tier-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .override-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: rgba(251, 191, 36, 0.1);
    color: var(--warning);
  }

  .dry-run-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
  }
</style>
