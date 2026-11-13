<script lang="ts">
  // Settings, under the cloud model: the model that talks (conversation.<provider>_model), shown beside the model
  // that plans so the person sees both. Talk never takes the planner's model: when the person has not chosen one it
  // is the provider's fast default, from the daemon's list (model_catalog, pilot.models.catalog.TALK_DEFAULTS).
  import { onMount } from "svelte";
  import { settings } from "../stores/settings";
  import { loadModelCatalog, modelCatalog, withdrawnFor, withdrawnLine } from "../stores/modelCatalog";

  let note = $state("");
  const provider = $derived(
    $settings.model?.provider === "cloud" ? ($settings.model?.cloud_provider as string) || "" : "",
  );
  const key = $derived(`${provider}_model`);
  const chosen = $derived(String((($settings.conversation ?? {}) as Record<string, unknown>)[key] ?? "").trim());
  const models = $derived($modelCatalog?.providers[provider]);
  const fallback = $derived(models?.talk_default ?? "");
  const gone = $derived(withdrawnFor($modelCatalog, provider, chosen));

  onMount(() => {
    void loadModelCatalog();
  });

  async function save(value: string) {
    note = "";
    const result = await settings.updateSectionDetailed("conversation", { [key]: value.trim() });
    note = result.ok ? "" : result.message || "Not changed.";
  }
</script>

{#if provider && fallback}
  <div class="setting-row" data-testid="talk-model">
    <div class="setting-info">
      <span class="setting-label">Model for talking</span>
      <span class="setting-desc">
        Answers you hear come from this model; plans and tasks use the cloud model above. Blank uses {fallback}, a fast
        one, so a slow planning model never makes every reply wait.
      </span>
      <span class="security-setting-status">Talking with {chosen || fallback}</span>
      {#if !chosen && models?.talk_note}<span class="setting-desc">{models.talk_note}</span>{/if}
      {#if gone}
        <span class="setting-desc" data-testid="talk-model-withdrawn">
          {withdrawnLine(provider, chosen, gone)} Suggested: {fallback}.
          <button type="button" class="btn-sm" onclick={() => save("")}>Use {fallback}</button>
        </span>
      {/if}
      {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
    </div>
    <input
      class="input-md"
      list={`talk-models-${provider}`}
      value={chosen}
      placeholder={fallback}
      onchange={(e) => save((e.target as HTMLInputElement).value)}
      aria-label="Model for talking"
    />
    <datalist id={`talk-models-${provider}`}>
      {#each models?.offered ?? [] as option (option)}
        <option value={option}></option>
      {/each}
    </datalist>
  </div>
{/if}

<style>
  /* Model ids run to 21 characters (gemini-3.5-flash-lite); Settings' 160 px input cut their last letters on Linux. */
  .input-md {
    width: 200px;
  }
</style>
