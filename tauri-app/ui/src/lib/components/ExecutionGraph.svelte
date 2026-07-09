<script lang="ts">
  import { session } from "../stores/session";
  import { fade, slide } from "svelte/transition";
  import { flip } from "svelte/animate";
  import { Shield, Play, CheckCircle2, XCircle, Clock, Terminal } from "lucide-svelte";
</script>

{#if $session.liveActions.length > 0}
  <div class="execution-graph" transition:slide>
    <div class="header">
      <div class="title">
        <span class="pulse-dot"></span>
        Neural Execution Graph
      </div>
      <div class="phase-badge">{$session.phase.toUpperCase() || "IDLE"}</div>
    </div>

    <div class="nodes-container">
      {#each $session.liveActions as liveAction, idx (`${liveAction.index}-${idx}`)}
        <div 
          class="node-wrapper"
          animate:flip
          in:fade={{ duration: 400 }}
        >
          <!-- The Line connecting nodes -->
          {#if liveAction.index > 0}
            <div class="connection-line" class:active={liveAction.status !== 'pending'}></div>
          {/if}

          <!-- The Node -->
          <div class="node {liveAction.status}">
            <div class="node-icon">
              {#if liveAction.status === 'pending'}
                <Clock size={16} />
              {:else if liveAction.status === 'running'}
                <Play size={16} class="spinning" />
              {:else if liveAction.status === 'success'}
                <CheckCircle2 size={16} />
              {:else if liveAction.status === 'error'}
                <XCircle size={16} />
              {/if}
            </div>

            <div class="node-content">
              <div class="node-title">{liveAction.action.action_type.replace(/_/g, ' ')}</div>
              <div class="node-target">{liveAction.action.target || "System"}</div>
              
              {#if liveAction.action.requires_root || liveAction.action.destructive}
                <div class="node-badges">
                  {#if liveAction.action.requires_root}
                    <span class="badge root"><Shield size={10} /> Root</span>
                  {/if}
                  {#if liveAction.action.destructive}
                    <span class="badge destructive">Destructive</span>
                  {/if}
                </div>
              {/if}

              <!-- Expanded output if running or complete -->
              {#if liveAction.status === 'success' && liveAction.output}
                <div class="terminal-output success" transition:slide>
                  <Terminal size={12} />
                  <span>{liveAction.output.substring(0, 100)}{liveAction.output.length > 100 ? '...' : ''}</span>
                </div>
              {/if}
              {#if liveAction.status === 'error' && liveAction.error}
                <div class="terminal-output error" transition:slide>
                  <Terminal size={12} />
                  <span>{liveAction.error}</span>
                </div>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    </div>
  </div>
{/if}

<style>
  .execution-graph {
    background: rgba(10, 15, 25, 0.4);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
    font-family: 'JetBrains Mono', monospace;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #00f0ff;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background: #00f0ff;
    border-radius: 50%;
    box-shadow: 0 0 10px #00f0ff;
    animation: pulse 2s infinite;
  }

  .phase-badge {
    background: rgba(0, 240, 255, 0.1);
    color: #00f0ff;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    border: 1px solid rgba(0, 240, 255, 0.2);
  }

  .nodes-container {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .node-wrapper {
    display: flex;
    flex-direction: column;
  }

  .connection-line {
    width: 2px;
    height: 20px;
    background: rgba(255, 255, 255, 0.1);
    margin: 0 0 0 19px;
    transition: background 0.3s ease;
  }

  .connection-line.active {
    background: linear-gradient(to bottom, #00f0ff, rgba(0, 240, 255, 0.3));
  }

  .node {
    display: flex;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 1rem;
    transition: all 0.3s ease;
  }

  .node.running {
    background: rgba(0, 240, 255, 0.1);
    border-color: rgba(0, 240, 255, 0.4);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);
  }

  .node.success {
    background: rgba(0, 255, 170, 0.05);
    border-color: rgba(0, 255, 170, 0.2);
  }

  .node.error {
    background: rgba(255, 50, 50, 0.05);
    border-color: rgba(255, 50, 50, 0.2);
  }

  .node-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .node.pending .node-icon { color: rgba(255, 255, 255, 0.5); }
  .node.running .node-icon { color: #00f0ff; }
  .node.success .node-icon { color: #00ffaa; }
  .node.error .node-icon { color: #ff3333; }

  .spinning {
    animation: spin 2s linear infinite;
  }

  .node-content {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    width: 100%;
  }

  .node-title {
    color: white;
    font-weight: 500;
    font-size: 0.9rem;
    text-transform: capitalize;
  }

  .node-target {
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.8rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 90%;
  }

  .node-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .badge.root {
    background: rgba(255, 170, 0, 0.2);
    color: #ffaa00;
    border: 1px solid rgba(255, 170, 0, 0.4);
  }

  .badge.destructive {
    background: rgba(255, 50, 50, 0.2);
    color: #ff3333;
    border: 1px solid rgba(255, 50, 50, 0.4);
  }

  .terminal-output {
    margin-top: 0.5rem;
    padding: 0.5rem;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.3);
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.7);
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    word-break: break-all;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }

  .terminal-output.success { border-left: 2px solid #00ffaa; }
  .terminal-output.error { border-left: 2px solid #ff3333; color: #ff7777; }

  @keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 240, 255, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
  }

  @keyframes spin {
    100% { transform: rotate(360deg); }
  }
</style>
