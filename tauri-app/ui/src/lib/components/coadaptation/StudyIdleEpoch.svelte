<script lang="ts">
  /**
   * Rest epoch (spec 12.3): 30 s with the targets visible and no cue, under the preceding block's regime (never a stale
   * one: after a block that ended for a refresh the host moves the epoch behind the next block). Any commit is an
   * unintended activation; ambiguous activations use the fixed L2 rule. The epoch is recorded as one idle trial with
   * its actual rest duration.
   */
  import type { TrialEvent } from "../../coadaptation/commitRules";
  import type { FallbacksV1, InterruptionReason } from "../../coadaptation/types";
  import StudyPointingField, {
    type FieldRegime,
    type FieldTrial,
    type FieldTrialEnd,
    type InputMode,
    type PosteriorParams,
  } from "./StudyPointingField.svelte";

  interface Props {
    /** The idle layout (task "idle", no intended candidate). */
    trial: FieldTrial;
    durationMs: number;
    regime: FieldRegime;
    posteriorParams: PosteriorParams;
    inputMode: InputMode;
    inputPaused?: boolean;
    onend: (end: FieldTrialEnd) => void;
    onpause?: (reason: InterruptionReason) => void;
    oncue?: () => void;
  }

  let {
    trial,
    durationMs,
    regime,
    posteriorParams,
    inputMode,
    inputPaused = false,
    onend,
    onpause,
    oncue,
  }: Props = $props();

  let field = $state<ReturnType<typeof StudyPointingField> | null>(null);

  const idleTrial = $derived<FieldTrial>({
    ...trial,
    task: "idle",
    intendedIndex: null,
    labelWord: null,
    limitMs: 30000,
    runMs: durationMs,
  });
  const idleRegime = $derived<FieldRegime>({ ...regime, chips: false, interruptionsUsed: 0 });

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
</script>

<StudyPointingField
  bind:this={field}
  trial={idleTrial}
  regime={idleRegime}
  table={{ table: null, decisionId: null, fallbackReason: null, fixedRule: true }}
  {posteriorParams}
  {inputMode}
  {inputPaused}
  ontrialend={onend}
  {onpause}
  {oncue}
/>
