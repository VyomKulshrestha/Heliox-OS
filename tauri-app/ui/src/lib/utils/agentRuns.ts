/**
 * Background work (autonomous jobs, scheduled runs, MCP-submitted tasks) shown in chat.
 * The daemon streams each action as "agent_tool_call"; these helpers fold the events
 * into one growing "agent_run" message per job. Pure functions, no daemon access.
 */
import type { Message } from "../stores/session";

/** One action an autonomous job ran, streamed by the daemon as "agent_tool_call". */
export interface ToolCall {
  callId: string;
  actionType: string;
  target: string;
  arguments: string;
  tier: string;
  step: number;
  round: number;
  status: "running" | "success" | "error" | "skipped";
  output: string;
  error: string;
  durationMs: number;
}

/** Background work (autonomous jobs, scheduled runs, MCP tasks) shown as one growing chat card. */
export interface AgentRun {
  jobId: string;
  goal: string;
  source: string;
  status: string;
  calls: ToolCall[];
}

const MAX_TOOL_CALLS_PER_RUN = 60;

function toolCallFrom(p: Record<string, unknown>): ToolCall {
  const complete = p.phase === "complete";
  const executed = p.executed !== false;
  return {
    callId: String(p.call_id ?? ""),
    actionType: String(p.action_type ?? ""),
    target: String(p.target ?? ""),
    arguments: String(p.arguments ?? ""),
    tier: String(p.tier ?? ""),
    step: Number(p.step ?? 0),
    round: Number(p.round ?? 1),
    status: !complete ? "running" : !executed ? "skipped" : p.success ? "success" : "error",
    output: String(p.output ?? ""),
    error: String(p.error ?? ""),
    durationMs: Number(p.duration_ms ?? 0),
  };
}

/** Insert or update a tool call inside its job's agent-run message, creating the message on first sight. */
export function applyToolCall(messages: Message[], p: Record<string, unknown>): Message[] {
  const jobId = String(p.job_id ?? "");
  if (!jobId) return messages;
  const call = toolCallFrom(p);
  const index = messages.findIndex((message) => message.type === "agent_run" && message.agentRun?.jobId === jobId);
  if (index === -1) {
    const run: AgentRun = {
      jobId,
      goal: String(p.goal ?? ""),
      source: String(p.source ?? ""),
      status: "running",
      calls: [call],
    };
    return [...messages, { type: "agent_run", text: run.goal, timestamp: Date.now(), agentRun: run }];
  }
  const run = messages[index].agentRun as AgentRun;
  const existing = run.calls.findIndex((item) => item.callId === call.callId);
  const calls =
    existing === -1
      ? [...run.calls, call].slice(-MAX_TOOL_CALLS_PER_RUN)
      : run.calls.map((item, i) => (i === existing ? { ...item, ...call } : item));
  const next = [...messages];
  next[index] = { ...messages[index], agentRun: { ...run, calls } };
  return next;
}

/** Record a job's final status on its agent-run card, if the job showed any tool calls. */
export function applyAgentRunStatus(messages: Message[], jobId: string, status: string): Message[] {
  const index = messages.findIndex((message) => message.type === "agent_run" && message.agentRun?.jobId === jobId);
  if (index === -1) return messages;
  const next = [...messages];
  next[index] = { ...messages[index], agentRun: { ...(messages[index].agentRun as AgentRun), status } };
  return next;
}
