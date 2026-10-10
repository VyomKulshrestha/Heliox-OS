# Specialist Agent Development Guide

Heliox main uses capability-based discovery rather than a fixed roster. The
current runtime registers 21 concrete specialists across 20 source-scoped
gateway roles and covers all 160 declared actions.

Add an agent only when it provides a distinct capability or measurable
reliability advantage. More classes are not a product goal by themselves.

## Before writing code

Read:

- [Architecture](docs/ARCHITECTURE.md)
- [Security](SECURITY.md)
- [IPC Message Formats](IPC_MESSAGE_FORMATS.md)
- `daemon/pilot/agents/base_agent.py`
- `daemon/pilot/agents/agent_mesh.py`
- `daemon/pilot/agents/domain_agents.py`
- `daemon/pilot/security/gateway.py`

Define the proposed agent's:

1. narrow responsibility and exact `ActionType` set;
2. confirmation and irreversible-action expectations;
3. filesystem, network, process, credential, clipboard, and device resources;
4. token, action, latency, and concurrency budgets;
5. failure boundary, cancellation behavior, and replay scenarios;
6. evidence that it improves routing or reliability over an existing provider.

## Runtime contracts

Every concrete specialist inherits `BaseAgent` and implements:

```python
def get_capabilities(self) -> list[AgentCapability]: ...
def get_system_prompt(self) -> str: ...
def can_handle(self, action_type: ActionType) -> bool: ...

async def handle_task(
    self,
    user_input: str,
    plan: ActionPlan,
    context: dict[str, Any] | None = None,
    scope_override: TaskScopeOverride | None = None,
) -> list[ActionResult]: ...
```

`handle_task` returns one result per executed action. It must preserve action
order unless the orchestrator explicitly supplied independent branches.

### Role and gateway identity

Built-in executor-backed specialists need:

- an `AgentRole` value in `base_agent.py`;
- the matching `InvocationSource` in `security/gateway.py`;
- a fail-closed entry in `DEFAULT_SOURCE_PROFILES`;
- tests showing that the profile cannot exceed its intended authority.

`BaseAgent.get_invocation_source()` derives the source from the role. Do not use
`AgentRole.GENERAL` for an agent that calls `Executor`; it maps to the unknown
fallback rather than a reviewed specialist profile.

Multiple provider classes may share one gateway role when their authority is
identical. Communication and Email are the existing example. The mesh keeps
them separate by concrete provider identity.

### Capability declaration

```python
from pilot.actions import ActionType
from pilot.agents.base_agent import AgentCapability

def get_capabilities(self) -> list[AgentCapability]:
    return [
        AgentCapability(
            action_type=ActionType.FILE_HASH,
            description="Stream and hash a validated file",
            requires_confirmation=False,
            estimated_duration_ms=1_000,
        )
    ]
```

An `AgentCapability` is routing metadata. It does not independently sandbox
Python code. Real enforcement comes from schema validation, permission tiering,
the Agent Gateway, approval, the shared executor or constrained adapter, and
post-action verification.

### Resource and budget declaration

The mesh reads class-level metadata such as:

```python
from pilot.agents.agent_mesh import AgentBudgetPolicy

mesh_keywords = ("file", "hash", "compare")
mesh_budget = AgentBudgetPolicy(
    max_tokens_per_task=6_000,
    max_actions_per_task=12,
    max_latency_ms_per_action=90_000,
    max_concurrency=1,
)
mesh_filesystem_read = ("user_selected_paths",)
mesh_filesystem_write = ()
mesh_network_domains = ()
mesh_process_names = ()
mesh_credential_names = ()
mesh_clipboard = ()
mesh_devices = ()
```

Use exact, reviewable resources. Do not use wildcard paths/domains or declare
resources the agent does not need.

## Preferred executor-backed pattern

For a narrow built-in domain, follow `_ExecutorDomainAgent` in
`domain_agents.py`: filter to the declared actions, construct a sub-plan, and
send it through the shared executor with the specialist source.

```python
results = await self._executor.execute(
    sub_plan,
    initial_last_output=str((context or {}).get("initial_last_output", "")),
    initial_largest_output=str((context or {}).get("initial_largest_output", "")),
    invocation_source=self.get_invocation_source(),
    scope_override=scope_override,
    user_confirmed=bool((context or {}).get("user_confirmed", False)),
)
```

This preserves the experience ledger, durable action claims, world-model
caution, permission checks, gateway policy, approvals, audit, cancellation, and
verification. Do not call a platform primitive directly merely to avoid these
layers.

Calendar, Email, and SSH demonstrate dedicated adapters. Such an adapter must
enforce a domain-specific allowlist at least as strict as the gateway and must
still emit terminal results and verified outcomes.

## Discovery and startup

`AgentRegistry.discover_agents()` scans `pilot.agents`, registering concrete,
non-abstract `BaseAgent` subclasses defined in their own module. The
`@auto_register` decorator is optional; discovery is the normal built-in path.

The server constructs dependencies, creates discovered providers, registers
every distinct class with the orchestrator and mesh, and starts each provider
once. Constructors must therefore use supported dependency names and avoid
side effects before `start()`.

Dynamic spawning prefers the registry class name:

```json
{ "method": "agent_spawn", "params": { "agent_name": "VisionAgent" } }
```

The legacy `role` parameter supports only the original compatibility roles and
is not the extensibility mechanism.

## Routing and delegation

The mesh selects providers by:

1. exact capability support;
2. persisted callback-observed success and latency;
3. a Bayesian prior for untested providers;
4. narrower scope only when observed quality is tied.

Keywords help preview and explain routing; they cannot make an incapable
provider executable.

Delegation is bounded to depth 3 and fan-out 4, rejects cycles, uses bounded
context references instead of copying entire transcripts, and shares
cancellation. Preserve partial results when a branch fails.

## Adding an action with an agent

An action is not complete when it is added to the enum. Update all relevant
surfaces:

1. `ActionType` and parameter model in `actions.py`;
2. planner parsing/schema and examples;
3. validator and permission/irreversibility classification;
4. executor dispatch or constrained domain adapter;
5. specialist capability and gateway profile;
6. experience, audit, cancellation, and result behavior;
7. unit, negative, replay, and UI status tests.

For browser or desktop UI actions, the provider must consume a fresh grounded
observation, reject missing or ambiguous targets, and verify the resulting
environment state. Keyboard or mouse work against a native application must
carry and re-acquire the intended target window; a successful process spawn or
input API call alone is not a successful task result.

`agent_mesh_status` must continue to report full coverage. Any uncovered action
name is a release blocker.

## Plugins are not ordinary specialists

Reviewed plugin tools may appear in the mesh catalog as guarded external
providers, but the mesh must not import or execute their code. Native tools run
in the constrained child broker; WASM tools run through the WASI broker. Their
manifest grants and destructive approval requirements remain authoritative.
See [Plugin Marketplace](docs/PLUGIN_MARKETPLACE.md).

## Required tests

At minimum, add tests for:

- registry discovery and one-time startup;
- exact capability and resource contracts;
- successful and rejected routing;
- gateway ceiling and deny-list enforcement;
- cancellation and partial-result behavior;
- callback-observed outcome persistence;
- delegation depth, fan-out, and cycle rejection where used;
- action-provider coverage;
- the live RPC/UI contract when user-visible status changes.

Run:

```bash
cd daemon
python -m ruff check pilot tests
python -m ruff format --check pilot tests
python -m pytest

cd ../tauri-app/ui
npx prettier --check .
npm run check
npm run test:unit -- --run
npm run build
```

Also run the relevant Playwright scenarios and verify the restarted daemon in
the browser when the agent changes user-visible routing or authority.

## Pull-request checklist

- [ ] The specialist has a distinct, narrow responsibility.
- [ ] Role, gateway, capability, resource, and budget contracts agree.
- [ ] Every claimed action has a concrete, validated provider.
- [ ] Execution uses the guarded shared path or a stricter adapter.
- [ ] UI actions are grounded, target-bound, and verified after execution.
- [ ] Ledger, audit, cancellation, verification, and terminal responses work.
- [ ] Routing quality uses observed outcomes rather than self-reporting.
- [ ] Negative, replay, and cross-platform cases are covered.
- [ ] `agent_mesh_status` still reports all declared actions covered.
- [ ] README, Architecture, Security, IPC, and marketplace docs were assessed.
