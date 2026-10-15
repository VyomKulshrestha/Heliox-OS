<script lang="ts">
  /**
   * Menu trial (spec 12.4): a 32 px menu button opens a vertical menu of 6, 12 or 16 items of 24 px. Keyboard
   * navigation (arrows, type-ahead, Enter) always works once the menu is open, also in pointer and hand-pointer
   * sessions, where a key activation is recorded with modality "keyboard". It runs on the shared study surface.
   */
  import type { TrialEvent } from "../../coadaptation/commitRules";
  import type { FallbacksV1, InterruptionReason, OfferResponseV1 } from "../../coadaptation/types";
  import StudyPointingField, {
    type FieldActivation,
    type FieldRegime,
    type FieldTable,
    type FieldTrial,
    type FieldTrialEnd,
    type FieldVeto,
    type InputMode,
    type PosteriorParams,
  } from "./StudyPointingField.svelte";

  interface Props {
    trial: FieldTrial;
    regime: FieldRegime;
    table: FieldTable;
    posteriorParams: PosteriorParams;
    inputMode: InputMode;
    inputPaused?: boolean;
    offers?: OfferResponseV1[];
    onactivation?: (activation: FieldActivation) => void;
    ontrialend: (end: FieldTrialEnd) => void;
    onveto?: (veto: FieldVeto) => void;
    onnever?: (variant: string) => void | Promise<void>;
    onneverstep?: (open: boolean) => void;
    onretract?: () => void;
    onpause?: (reason: InterruptionReason) => void;
    oncue?: () => void;
  }

  let props: Props = $props();

  let field = $state<ReturnType<typeof StudyPointingField> | null>(null);

  export function dispatch(event: TrialEvent) {
    field?.dispatch(event);
  }

  export function refresh(fresh: FallbacksV1) {
    field?.refresh(fresh);
  }

  export function interrupt(reason: InterruptionReason) {
    field?.interrupt(reason);
  }

  export function discard() {
    field?.discard();
  }

  export function inProgress(): boolean {
    return field?.inProgress() ?? false;
  }

  export function adoptTable(next: FieldTable) {
    field?.adoptTable(next);
  }
</script>

<StudyPointingField bind:this={field} {...props} />
