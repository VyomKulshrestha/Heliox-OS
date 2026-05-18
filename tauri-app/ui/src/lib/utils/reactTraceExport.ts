/** Types and helpers for exporting the ReAct pipeline trace to JSON. */

export type StageStatus = "idle" | "active" | "success" | "error" | "skipped";

export interface PipelineStageSnapshot {
  id: string;
  label: string;
  status: StageStatus;
  detail: string;
  startTime: number;
  endTime: number;
  timingMs?: number;
  decision?: string;
}

export interface ExecutionActionSnapshot {
  type: string;
  target: string;
  status: string;
}

export interface ThoughtEntrySnapshot {
  seq: number;
  stage: string;
  stageId: string;
  text: string;
  type: string;
}

export interface TraceEventRecord {
  timestamp: number;
  method: string;
  payload: Record<string, unknown>;
}

export interface ReactTraceExportPayload {
  version: 1;
  exportedAt: string;
  summary: {
    progressPercent: number;
    totalDurationMs: number;
    stageCount: number;
    thoughtCount: number;
    actionCount: number;
    eventCount: number;
  };
  agentRouting: {
    assigned_agents: string[];
    is_multi_agent: boolean;
  } | null;
  stages: PipelineStageSnapshot[];
  executionActions: ExecutionActionSnapshot[];
  thoughts: ThoughtEntrySnapshot[];
  events: TraceEventRecord[];
}

export function buildReactTraceExport(input: {
  stages: Array<{
    id: string;
    label: string;
    status: StageStatus;
    detail: string;
    startTime: number;
    endTime: number;
  }>;
  stageTiming: Record<string, number>;
  stageDecisions: Record<string, string>;
  executionActions: ExecutionActionSnapshot[];
  thoughtStream: ThoughtEntrySnapshot[];
  traceEvents: TraceEventRecord[];
  agentRouting: { assigned_agents: string[]; is_multi_agent: boolean } | null;
  progress: number;
  totalDuration: number;
}): ReactTraceExportPayload {
  const stages: PipelineStageSnapshot[] = input.stages.map((stage) => ({
    id: stage.id,
    label: stage.label,
    status: stage.status,
    detail: stage.detail,
    startTime: stage.startTime,
    endTime: stage.endTime,
    ...(input.stageTiming[stage.id] !== undefined
      ? { timingMs: input.stageTiming[stage.id] }
      : {}),
    ...(input.stageDecisions[stage.id]
      ? { decision: input.stageDecisions[stage.id] }
      : {}),
  }));

  return {
    version: 1,
    exportedAt: new Date().toISOString(),
    summary: {
      progressPercent: input.progress,
      totalDurationMs: input.totalDuration,
      stageCount: stages.length,
      thoughtCount: input.thoughtStream.length,
      actionCount: input.executionActions.length,
      eventCount: input.traceEvents.length,
    },
    agentRouting: input.agentRouting,
    stages,
    executionActions: input.executionActions,
    thoughts: input.thoughtStream,
    events: input.traceEvents,
  };
}
