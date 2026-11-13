<script lang="ts">
  // Settings, beside the model choice: what a model that is not on this computer is told about the person
  // (conversation.share_profile, decided in pilot.conversation.adapters.personal_in_prompt). Turning it off is
  // immediate; turning it back on is a widening and the daemon asks in its own approval dialog.
  import { settings } from "../stores/settings";

  let note = $state("");
  let busy = $state(false);
  const sharing = $derived(($settings.conversation?.share_profile ?? "auto") !== "off");

  async function choose(value: "auto" | "off") {
    busy = true;
    note = "";
    try {
      const result = await settings.updateSectionDetailed("conversation", { share_profile: value });
      if (result.ok) {
        note =
          value === "off"
            ? "Done: nothing about you goes to a cloud model now."
            : "Done: a cloud model gets your name, answer style and what bears on your question again.";
      } else if (result.cancelled) {
        note = "Not changed: it was not approved.";
      } else {
        note = result.message || "Not changed.";
      }
    } finally {
      busy = false;
    }
  }
</script>

<div class="setting-row" data-testid="profile-sharing">
  <div class="setting-info">
    <span class="setting-label">Your profile and the model</span>
    <span class="setting-desc">
      With a model that is not on this computer, Heliox sends your name, your answer style and the parts of your profile
      that bear on what you ask. Never health, money, secrets or other people's private details. A model on this
      computer keeps everything here.
    </span>
    <span class="security-setting-status">
      {sharing
        ? "Sent to a cloud model when it bears on your question"
        : "Off: nothing about you goes to a cloud model"}
    </span>
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <button
    type="button"
    class="toggle"
    class:active={sharing}
    disabled={busy}
    onclick={() => choose(sharing ? "off" : "auto")}
    aria-label="Send your profile to a cloud model"
    title="Send your profile to a cloud model"
  >
    <span class="toggle-knob"></span>
  </button>
</div>
