<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  interface SourceProfile {
    max_tier: Record<string, number>;
    deny_action_types: string[];
    allow_root: boolean;
  }

  const TIER_NAMES = ["READ_ONLY", "USER_WRITE", "SYSTEM_MODIFY", "DESTRUCTIVE", "ROOT_CRITICAL"];
  const FAMILIES = ["shell", "browsing", "system_control", "other"];
  const PROFILE_ORDER = ["interactive", "autonomous", "web_agent", "voice", "gesture"];

  let profiles = $state<Record<string, SourceProfile>>({});
  let loading = $state(true);
  let savingProfile = $state<string | null>(null);
  let savedProfile = $state<string | null>(null);

  onMount(loadPolicy);

  async function loadPolicy() {
    loading = true;
    try {
      const result = (await call("gateway_policy_get")) as { status: string; profiles: Record<string, SourceProfile> };
      profiles = result.profiles ?? {};
    } catch {
      profiles = {};
    } finally {
      loading = false;
    }
  }

  function orderedProfileNames(): string[] {
    const known = PROFILE_ORDER.filter((name) => name in profiles);
    const extra = Object.keys(profiles).filter((name) => !PROFILE_ORDER.includes(name));
    return [...known, ...extra];
  }

  function updateTier(name: string, family: string, e: Event) {
    const tier = Number((e.target as HTMLSelectElement).value);
    profiles = { ...profiles, [name]: { ...profiles[name], max_tier: { ...profiles[name].max_tier, [family]: tier } } };
  }

  function toggleAllowRoot(name: string) {
    profiles = { ...profiles, [name]: { ...profiles[name], allow_root: !profiles[name].allow_root } };
  }

  async function saveProfile(name: string) {
    savingProfile = name;
    savedProfile = null;
    try {
      await call("gateway_policy_update", {
        profile: name,
        max_tier: profiles[name].max_tier,
        deny_action_types: profiles[name].deny_action_types,
        allow_root: profiles[name].allow_root,
      });
      savedProfile = name;
      setTimeout(() => {
        if (savedProfile === name) savedProfile = null;
      }, 2500);
    } finally {
      savingProfile = null;
    }
  }
</script>

<div class="policy-editor">
  <div class="policy-header">
    <h2>Agent Gateway Policy</h2>
    <span class="count">{orderedProfileNames().length} source profiles</span>
  </div>

  <p class="policy-note">
    Each source (interactive, autonomous background jobs, the web agent, voice, gesture) has an enforced ceiling per
    action family. A per-task override supplied by a caller (e.g. an autonomous job) can only narrow this floor
    further — it can never widen it, no matter what it requests.
  </p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    <div class="profile-list">
      {#each orderedProfileNames() as name}
        {@const profile = profiles[name]}
        <div class="profile-card">
          <div class="profile-card-header">
            <strong class="profile-name">{name}</strong>
            <button class="btn-save" onclick={() => saveProfile(name)} disabled={savingProfile === name}>
              {savingProfile === name ? "Saving..." : savedProfile === name ? "✓ Saved" : "Save"}
            </button>
          </div>

          <div class="family-grid">
            {#each FAMILIES as family}
              <div class="family-row">
                <span class="family-label">{family}</span>
                <select
                  class="input-sm"
                  value={profile.max_tier[family] ?? 0}
                  onchange={(e) => updateTier(name, family, e)}
                >
                  {#each TIER_NAMES as tierName, tierValue}
                    <option value={tierValue}>{tierName}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>

          <div class="setting-row">
            <span class="setting-desc">Allow root/Tier-4 actions</span>
            <button
              class="toggle"
              class:active={profile.allow_root}
              onclick={() => toggleAllowRoot(name)}
              aria-label={`Toggle root access for ${name}`}
              title={`Toggle root access for ${name}`}
            >
              <span class="toggle-knob"></span>
            </button>
          </div>

          {#if profile.deny_action_types.length > 0}
            <div class="deny-list">
              <span class="deny-label">Always denied:</span>
              {#each profile.deny_action_types as actionType}
                <code class="deny-tag">{actionType}</code>
              {/each}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .policy-editor {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .policy-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
  }

  .count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .policy-note {
    margin: 0;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
  }

  .empty {
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
  }

  .profile-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .profile-card {
    padding: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .profile-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .profile-name {
    font-size: 13px;
    text-transform: capitalize;
  }

  .family-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 8px;
  }

  .family-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .family-label {
    font-size: 11px;
    color: var(--text-secondary);
    text-transform: capitalize;
  }

  .deny-list {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }

  .deny-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .deny-tag {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    background: var(--danger-bg);
    color: var(--danger);
  }
</style>
