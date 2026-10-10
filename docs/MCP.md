# Heliox MCP interfaces

Heliox exposes two deliberately separate Model Context Protocol (MCP)
interfaces. They do not share credentials or authority.

## Public documentation MCP

`https://www.helioxos.dev/api/mcp` is a read-only Streamable HTTP endpoint for
documentation and evidence discovery. Its manifest is published at
`https://www.helioxos.dev/.well-known/mcp.json`.

It can search documentation, list published capabilities, explain action
safety, return the latest release and installation instructions, and report
current benchmark evidence. It cannot connect to a user's daemon or control a
computer.

## Local control MCP

`heliox-mcp` is a local stdio server installed with `pilot-daemon`. It connects
to the loopback Heliox daemon with a rotated, private MCP credential. Start the
desktop application before connecting an MCP host.

Use this generic MCP-host configuration:

```json
{
  "mcpServers": {
    "heliox-local": {
      "command": "heliox-mcp"
    }
  }
}
```

For a source checkout, replace the command with the Python environment in
which `daemon` was installed, or run `python -m pilot.mcp_server` from that
environment.

The server exposes nine tools:

| Tool | Effect |
| --- | --- |
| `heliox_health` | Check the daemon connection. |
| `get_heliox_system_status` | Read current daemon system status. |
| `list_heliox_capabilities` | List guarded Heliox actions and specialists. |
| `preview_heliox_task` | Create a non-binding, side-effect-free preview. |
| `score_heliox_typed_decision` | Optional local research-only yes/no, choice, or ordered-option scores; never an approval. |
| `score_heliox_typed_questions` | Score up to 64 independent typed questions against one shared state; never an approval. |
| `submit_heliox_task` | Submit a task asynchronously to the guarded executor. |
| `get_heliox_task_status` | Poll durable task, approval, and terminal state. |
| `cancel_heliox_task` | Cancel an MCP-owned task. |

Returned candidate probabilities are model-assigned distribution mass, not empirically validated
probabilities that the answer is correct. For choice and ordered-score answers, `confidence`
describes how concentrated the returned distribution is; it is not a calibrated correctness
probability. Validate any threshold against representative labeled tasks, and keep execution and
approvals in Heliox.
For reference-style use, retrieve only relevant state, ask one focused judgment per question, and give
the scorer explicit candidates; batch independent questions against the shared state.

Submitting is not the same as completing. The tool returns a task identifier;
the host must poll `get_heliox_task_status` until the task reaches a terminal
state. A preview is advisory and submission replans against current state.
Both typed-scoring tools are disabled until enabled in Settings with explicit
local artifact and report paths. They require the supported CUDA/runtime stack
and already-cached model weights; Heliox does not install or download those
weights.
If needed, install the optional `pilot-daemon[typed-decision]` extra into the
same Python environment that runs the daemon; it is intentionally excluded from
the default `all` install because its inference dependencies are large.
Requests are bounded to 255 options for one choice, or 64 questions and 256
total candidate options for a typed batch, with a 64,000-token local scoring
budget. They are evaluated locally and are not saved or uploaded. Scores never
enter planning, the world-model risk gate, approval, or execution. This is an
experimental research interface, not a reference-model parity or desktop-task-success
claim.

Hosts should send a stable optional `request_id` when retrying a submission;
the daemon returns the original task for the same session and input, and
rejects reuse with different input. Inputs and identifiers are length- and
character-bounded by the published tool schema.

The bridge uses explicit Heliox task handles rather than protocol sessions.
This aligns with the [2026-07-28 MCP specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/),
but the nine tools do not claim the optional MCP Tasks extension. Compatibility
is tested through the official Python MCP client.

## Security boundary

- MCP has its own rotated runtime token and `mcp_local` RPC identity.
- The configured daemon URL must be an unambiguous loopback WebSocket with no
  credentials, query, fragment, or alternate path.
- The identity can call only the ten allow-listed RPC methods needed by the nine tools.
- It cannot call raw `execute`, `confirm`, `update_config`, `store_api_key`, or
  arbitrary daemon methods.
- Every submitted action pauses for visible approval in the Heliox UI, even if
  the same action would normally auto-run for an interactive user.
- The MCP gateway profile denies arbitrary browser JavaScript, registry writes,
  and power shutdown/restart/logout, and it cannot request root authority.
- There is intentionally no MCP approval tool. A model or remote host cannot
  approve its own proposed effects.
- Task status and cancellation are restricted to MCP-owned durable tasks.
- The token remains in the per-user runtime directory and is never published
  by the public documentation MCP.

The MCP bridge therefore adds another client surface without creating another
execution authority: planning, policy, approval, execution, verification,
audit, cancellation, and durable recovery stay in the existing daemon path.
