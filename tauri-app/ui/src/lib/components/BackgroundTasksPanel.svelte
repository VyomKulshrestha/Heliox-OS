<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";

  type Job = {
    job_id: string;
    goal: string;
    status: string;
    current_step: number;
    total_steps: number;
    result_summary: string;
    resumable?: boolean;
  };

  const ACTIVE = new Set(["queued", "decomposing", "running"]);

  let jobs = $state<Job[]>([]);
  let busyJob = $state("");
  let error = $state("");

  const visible = $derived(jobs.filter((job) => ACTIVE.has(job.status) || job.status === "interrupted"));

  function upsert(job: Job) {
    const index = jobs.findIndex((item) => item.job_id === job.job_id);
    jobs = index === -1 ? [job, ...jobs] : jobs.map((item, i) => (i === index ? job : item));
  }

  const handler = (method: string, params: unknown) => {
    if (!method.startsWith("autonomous_")) return;
    const job = params as Job;
    if (job && typeof job.job_id === "string" && typeof job.status === "string") upsert(job);
  };

  async function load() {
    try {
      const result = await call<{ jobs?: Job[]; error?: string }>("autonomous_jobs");
      if (result.error) throw new Error(result.error);
      jobs = result.jobs ?? [];
    } catch (err) {
      error = err instanceof Error ? err.message : "Could not load background tasks.";
    }
  }

  async function act(job: Job, method: "autonomous_resume" | "autonomous_cancel") {
    busyJob = job.job_id;
    error = "";
    try {
      const result = await call<{ status?: string; message?: string; cancelled?: boolean; job?: Job }>(method, {
        job_id: job.job_id,
      });
      if (method === "autonomous_resume") {
        if (result.status !== "resumed" || !result.job) throw new Error(result.message || "The task did not resume.");
        upsert(result.job);
      } else if (!result.cancelled) {
        throw new Error("The task could not be cancelled.");
      } else {
        upsert({ ...job, status: "cancelled" });
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "The request failed.";
    } finally {
      busyJob = "";
    }
  }

  function progress(job: Job): string {
    if (job.status === "interrupted") return "Interrupted when Heliox stopped";
    if (job.status === "running" && job.total_steps > 1) return `Step ${job.current_step + 1} of ${job.total_steps}`;
    return job.status === "decomposing" ? "Planning steps" : job.status === "queued" ? "Starting" : "Running";
  }

  onMount(() => {
    onNotification(handler);
    void load();
  });

  onDestroy(() => offNotification(handler));
</script>

{#if visible.length > 0 || error}
  <section class="tasks" aria-labelledby="background-tasks-heading">
    <h3 id="background-tasks-heading">Background tasks</h3>
    {#if error}<div class="error" role="alert">{error}</div>{/if}
    <ul>
      {#each visible as job (job.job_id)}
        <li class:interrupted={job.status === "interrupted"}>
          <div class="text">
            <span class="goal">{job.goal}</span>
            <small>{progress(job)}</small>
          </div>
          <div class="buttons">
            {#if job.status === "interrupted"}
              <button
                type="button"
                class="primary"
                disabled={busyJob === job.job_id}
                onclick={() => act(job, "autonomous_resume")}>Resume</button
              >
              <button type="button" disabled={busyJob === job.job_id} onclick={() => act(job, "autonomous_cancel")}
                >Dismiss</button
              >
            {:else}
              <button type="button" disabled={busyJob === job.job_id} onclick={() => act(job, "autonomous_cancel")}
                >Cancel</button
              >
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  </section>
{/if}

<style>
  .tasks {
    max-height: 40%;
    overflow-y: auto;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
  }
  h3 {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
  }
  ul {
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-secondary);
  }
  li.interrupted {
    border-color: var(--warning);
  }
  .text {
    min-width: 0;
  }
  .goal {
    display: block;
    overflow: hidden;
    font-size: 12px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  small {
    color: var(--text-muted);
    font-size: 11px;
  }
  li.interrupted small {
    color: var(--warning);
  }
  .buttons {
    display: flex;
    flex-shrink: 0;
    gap: 6px;
  }
  button {
    padding: 5px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
    cursor: pointer;
  }
  button.primary {
    border-color: var(--accent);
    color: var(--accent);
  }
  button:disabled {
    cursor: default;
    opacity: 0.6;
  }
  .error {
    margin-bottom: 8px;
    color: var(--danger);
    font-size: 11px;
  }
</style>
