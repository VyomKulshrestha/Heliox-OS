<script lang="ts">
  // Settings, the cloud model that plans (model.cloud_model). The offered models come from the daemon's list
  // (model_catalog), so a retired model is never offered; any model id can still be typed (newer or private
  // models). A saved model the provider has retired or deprecated is named as such with the current suggestion,
  // and stays saved until the person chooses another.
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import { settings } from "../stores/settings";
  import { loadModelCatalog, modelCatalog, withdrawnFor, withdrawnLine } from "../stores/modelCatalog";

  let note = $state("");
  const provider = $derived(($settings.model?.cloud_provider as string) || "");
  const saved = $derived(String($settings.model?.cloud_model ?? "").trim());
  const models = $derived($modelCatalog?.providers[provider]);
  const gone = $derived(withdrawnFor($modelCatalog, provider, saved));

  onMount(() => {
    void loadModelCatalog();
  });

  async function choose(value: string, e?: Event) {
    note = "";
    const result = await settings.updateSectionDetailed("model", { cloud_model: value.trim() });
    note = result.ok ? "" : result.message || "Not changed.";
    if (e && !result.ok) (e.target as HTMLInputElement).value = saved;
  }
</script>

<div class="setting-row" data-testid="planner-model">
  <div class="setting-info">
    <span class="setting-label">{$_("settings.cloud_model")}</span>
    <span class="setting-desc">
      Plans and tasks use this model. Blank uses {models?.planner_default ?? "the provider's default"}; you can also
      type any model id.
    </span>
    {#if !saved && models?.planner_note}<span class="setting-desc">{models.planner_note}</span>{/if}
    {#if gone}
      <span class="security-setting-status" data-testid="planner-model-withdrawn">
        {withdrawnLine(provider, saved, gone)}
        {#if gone.successor}
          Suggested: {gone.successor}.
          <button type="button" class="btn-sm" onclick={() => choose(gone.successor)}>Use {gone.successor}</button>
        {/if}
      </span>
    {/if}
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <input
    class="input-md"
    list={`planner-models-${provider}`}
    value={saved}
    placeholder={models?.planner_default ?? ""}
    onchange={(e) => choose((e.target as HTMLInputElement).value, e)}
    aria-label="Cloud model"
  />
  <datalist id={`planner-models-${provider}`}>
    {#each models?.offered ?? [] as option (option)}
      <option value={option}></option>
    {/each}
  </datalist>
</div>

<style>
  /* Model ids run to 21 characters (gemini-3.5-flash-lite); Settings' 160 px input cut their last letters on Linux. */
  .input-md {
    width: 200px;
  }
</style>
