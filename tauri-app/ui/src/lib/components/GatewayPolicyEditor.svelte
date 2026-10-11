<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireOkResult, type DaemonStatusResult } from "../api/daemon";
  import { changedFields, onConfigChange } from "../api/configSync";

  type SourceProfile = {
    max_tier: Record<string, number>;
    deny_action_types: string[];
    allow_root: boolean;
  };

  const TIER_NAMES = ["READ_ONLY", "USER_WRITE", "SYSTEM_MODIFY", "DESTRUCTIVE", "ROOT_CRITICAL"];
  // The gateway enforces ROOT_CRITICAL for a family a profile leaves unset (pilot/security/gateway.py),
  // so the editor shows that instead of guessing a lower ceiling the daemon is not applying.
  const ROOT_CRITICAL = 4;
  const FAMILIES = ["shell", "browsing", "system_control", "other"];
  const PROFILE_ORDER = [
    "interactive",
    "autonomous",
    "web_agent",
    "voice",
    "gesture",
    "self_healing",
    // One profile per AgentOrchestrator specialist role -- see
    // SECURITY.md's "Capability-scoped multi-agent orchestration" section.
    "system_agent",
    "code_agent",
    "comm_agent",
    "forensics_agent",
    "ssh_agent",
    "monitor_agent",
    "rss_agent",
    "calendar_agent",
    "semantic_search_agent",
  ];

  let profiles = $state<Record<string, SourceProfile>>({});
  // The policy as the daemon last reported or accepted it; edits are measured against this.
  let savedProfiles = $state.raw<Record<string, SourceProfile>>({});
  let loading = $state(true);
  let savingProfile = $state<string | null>(null);
  let savedProfile = $state<string | null>(null);
  let error = $state("");

  onMount(() => {
    void loadPolicy();
    // Another window, a voice command or a reset can change the policy. Pick that up, but never
    // over edits the person has not saved yet.
    return onConfigChange({ methods: ["gateway_policy_update"] }, () => {
      if (!hasUnsavedChanges()) void loadPolicy(true);
    });
  });

  function withAllFamilies(profile: SourceProfile): SourceProfile {
    const maxTier: Record<string, number> = Object.fromEntries(FAMILIES.map((family) => [family, ROOT_CRITICAL]));
    return {
      ...profile,
      max_tier: { ...maxTier, ...(profile.max_tier ?? {}) },
      deny_action_types: [...(profile.deny_action_types ?? [])],
    };
  }

  async function loadPolicy(background = false) {
    error = "";
    try {
      const result = requireOkResult(
        (await call("gateway_policy_get")) as DaemonStatusResult & {
          profiles: Record<string, SourceProfile>;
        },
        "The daemon did not return the gateway policy.",
      );
      // An edit started while the reload was on its way wins over the reloaded copy.
      if (background && hasUnsavedChanges()) return;
      const loaded = Object.fromEntries(
        Object.entries(result.profiles ?? {}).map(([name, profile]) => [name, withAllFamilies(profile)]),
      );
      profiles = loaded;
      savedProfiles = $state.snapshot(profiles);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not load the gateway policy.";
    } finally {
      loading = false;
    }
  }

  function orderedProfileNames(): string[] {
    const known = PROFILE_ORDER.filter((name) => name in profiles);
    const extra = Object.keys(profiles).filter((name) => !PROFILE_ORDER.includes(name));
    return [...known, ...extra];
  }

  /** Only what the person changed in this profile since the last load or save. */
  function profileChanges(name: string): Partial<SourceProfile> {
    const saved = savedProfiles[name];
    const next = $state.snapshot(profiles[name]);
    if (!saved) return next;
    const changes = changedFields(saved, next);
    // The daemon merges max_tier per family, so untouched families are left out as well.
    if (changes.max_tier) changes.max_tier = changedFields(saved.max_tier, next.max_tier) as Record<string, number>;
    return changes;
  }

  function hasChanges(name: string): boolean {
    return Object.keys(profileChanges(name)).length > 0;
  }

  function hasUnsavedChanges(): boolean {
    return Object.keys(profiles).some(hasChanges);
  }

  function updateTier(name: string, family: string, e: Event) {
    const tier = Number((e.target as HTMLSelectElement).value);
    profiles = { ...profiles, [name]: { ...profiles[name], max_tier: { ...profiles[name].max_tier, [family]: tier } } };
  }

  function toggleAllowRoot(name: string) {
    profiles = { ...profiles, [name]: { ...profiles[name], allow_root: !profiles[name].allow_root } };
  }

  async function saveProfile(name: string) {
    const sent = $state.snapshot(profiles[name]);
    const changes = profileChanges(name);
    if (Object.keys(changes).length === 0) return;
    savingProfile = name;
    savedProfile = null;
    error = "";
    try {
      // gateway_policy_update accepts any subset of max_tier/deny_action_types/allow_root and keeps
      // the rest, so a change made elsewhere to another field is not overwritten with stale values.
      requireOkResult(
        (await call("gateway_policy_update", { profile: name, ...changes })) as DaemonStatusResult,
        "The daemon did not save the gateway policy.",
      );
      savedProfiles = { ...savedProfiles, [name]: sent };
      savedProfile = name;
      setTimeout(() => {
        if (savedProfile === name) savedProfile = null;
      }, 2500);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not save the gateway policy.";
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
    action family. A per-task override supplied by a caller (e.g. an autonomous job) can only narrow this floor further
    — it can never widen it, no matter what it requests.
  </p>

  {#if loading}
    <div class="empty">Loading...</div>
  {:else}
    {#if error}<div class="error" role="alert">{error}</div>{/if}
    <div class="profile-list">
      {#each orderedProfileNames() as name}
        {@const profile = profiles[name]}
        <div class="profile-card">
          <div class="profile-card-header">
            <strong class="profile-name">{name}</strong>
            <button
              class="btn-save"
              onclick={() => saveProfile(name)}
              disabled={savingProfile === name || !hasChanges(name)}
            >
              {savingProfile === name ? "Saving..." : savedProfile === name ? "✓ Saved" : "Save"}
            </button>
          </div>

          <div class="family-grid">
            {#each FAMILIES as family}
              <div class="family-row">
                <span class="family-label">{family}</span>
                <select
                  class="input-sm"
                  value={profile.max_tier[family] ?? ROOT_CRITICAL}
                  onchange={(e) => updateTier(name, family, e)}
                  aria-label={`${family} ceiling for ${name}`}
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

  .error {
    margin-bottom: 10px;
    padding: 8px 10px;
    color: var(--danger);
    font-size: 11px;
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.35);
    border-radius: var(--radius-sm);
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
