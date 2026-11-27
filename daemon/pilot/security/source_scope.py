"""The gateway profile of the source a request or job came from, carried everywhere its plan runs.

The gateway (``pilot.security.gateway``) checks each step under the profile of whoever runs it: the specialist agent
the orchestrator gives it to, or the caller of the executor. A request's own source (typed, spoken, MCP, a background
job, a voice or gesture workflow) narrows that profile through a ``TaskScopeOverride`` built here, so a step its source
refuses is refused whichever specialist carries it, on every path: the orchestrator, the plain executor, a repair
round, a resumed or later-approved plan. A stored plan keeps its source and that override, and is checked again
under them before it runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pilot.actions import ActionType, PermissionTier
from pilot.security.gateway import (
    InvocationSource,
    SourceProfile,
    TaskScopeOverride,
    action_family,
    profile_for,
    resolve_effective_profile,
)

if TYPE_CHECKING:
    from pilot.config import PilotConfig

# The names a typed request arrives with ("source" of an execute call from the window; it sends none).
TYPED_SOURCE_NAMES = frozenset({"", "interactive", "text", "chat", "typed"})

# The sources whose plans can leave a checkpoint or a durable task: the request pipeline's (typed, spoken, MCP) and
# the voice and gesture workflows' steps.
STORED_PLAN_SOURCES = (
    InvocationSource.INTERACTIVE,
    InvocationSource.VOICE,
    InvocationSource.MCP,
    InvocationSource.GESTURE,
)

# A plan stored without its source runs narrowed by every one of STORED_PLAN_SOURCES (``legacy_scope``), under this
# name: MCP's profile is the one with the longest deny list among them.
LEGACY_SOURCE = InvocationSource.MCP

# Where a background job came from, when that place has a profile of its own: the job runs under the autonomous
# profile narrowed by this one too. Any other job source (a trigger, a schedule, a phone, a recording, a proactive
# suggestion, the window) adds nothing: the autonomous profile is theirs.
JOB_ORIGIN_SOURCES = {
    "voice": InvocationSource.VOICE,
    "gesture": InvocationSource.GESTURE,
    "neural": InvocationSource.NEURAL,
    "mcp": InvocationSource.MCP,
}


def config_of(holder: Any) -> PilotConfig | None:
    """The config ``holder`` (an executor or a gateway) keeps, when it is one with saved source profiles; None makes
    every profile its built-in one."""
    config = getattr(holder, "_config", None)
    profiles = getattr(getattr(config, "gateway", None), "source_profiles", None)
    return config if isinstance(profiles, dict) else None


def as_override(profile: SourceProfile) -> TaskScopeOverride:
    return TaskScopeOverride(
        max_tier=dict(profile.max_tier),
        deny_action_types=list(profile.deny_action_types),
        allow_root=profile.allow_root,
    )


def source_scope(
    config: PilotConfig | None, source: InvocationSource, *narrower: TaskScopeOverride | None
) -> TaskScopeOverride:
    """``source``'s profile, narrowed by each of ``narrower`` (None adds nothing), as an override."""
    profile = profile_for(config, source)
    for override in narrower:
        profile = resolve_effective_profile(profile, override)
    return as_override(profile)


def interaction_source(name: str | None) -> InvocationSource:
    """The gateway source of a request by the name it arrived with: typed, ``voice`` or ``mcp``. Any other name is
    ``UNKNOWN``, which the gateway refuses outright: a new entry must say which source it is."""
    key = str(name or "").strip().lower()
    if key in TYPED_SOURCE_NAMES:
        return InvocationSource.INTERACTIVE
    if key == "voice":
        return InvocationSource.VOICE
    if key == "mcp":
        return InvocationSource.MCP
    return InvocationSource.UNKNOWN


def stored_source(value: str | None) -> InvocationSource | None:
    """The source a stored plan recorded, or None when it recorded none (stored by an older version)."""
    try:
        return InvocationSource(str(value or "").strip()) if value else None
    except ValueError:
        return InvocationSource.UNKNOWN


def legacy_scope(config: PilotConfig | None) -> tuple[InvocationSource, TaskScopeOverride]:
    """The source and scope for a plan stored with no source recorded: every source that could have made it narrows
    it, so it never runs wider than the narrowest of them."""
    return LEGACY_SOURCE, source_scope(
        config, LEGACY_SOURCE, *(as_override(profile_for(config, s)) for s in STORED_PLAN_SOURCES)
    )


def stored_plan_scope(
    config: PilotConfig | None, source: str | None, override: TaskScopeOverride | None
) -> tuple[InvocationSource, TaskScopeOverride]:
    """The source and scope a stored plan runs under again: its own source narrowed by the override it was made under,
    or ``legacy_scope`` when it recorded no source."""
    recorded = stored_source(source)
    if recorded is None:
        legacy, scope = legacy_scope(config)
        return legacy, source_scope(config, legacy, scope, override)
    return recorded, source_scope(config, recorded, override)


def job_scope(
    config: PilotConfig | None,
    invocation_source: InvocationSource,
    job_source: str,
    override: TaskScopeOverride | None,
) -> TaskScopeOverride:
    """The scope every plan of a job runs under: ``invocation_source`` (autonomous for a background job, voice or
    gesture for a workflow), narrowed by the profile of where the job came from (``JOB_ORIGIN_SOURCES``) and by the
    override the job was given."""
    origin = JOB_ORIGIN_SOURCES.get(str(job_source or "").strip().lower())
    narrower = [as_override(profile_for(config, origin))] if origin is not None and origin != invocation_source else []
    return source_scope(config, invocation_source, *narrower, override)


def refusals(
    gateway: Any, actions: list[Any], source: InvocationSource, override: TaskScopeOverride | None
) -> list[tuple[int, str]]:
    """(index, reason) for each of ``actions`` the gateway refuses under ``source`` and ``override``, before anything
    runs or an approval card is shown. Nothing is recorded and the critic is not asked. Uses the gateway's own preview
    when it has one; otherwise the same rule ``AgentGateway.authorize`` applies (deny list, family ceiling, root)."""
    if gateway is None or not actions:
        return []
    preview = getattr(gateway, "preview_denials", None)
    if callable(preview):
        return list(preview(list(actions), [source] * len(actions), override))
    config = getattr(gateway, "_config", None)
    if config is not None and not config.gateway.enabled:
        return []
    profile = resolve_effective_profile(profile_for(config, source), override)
    found: list[tuple[int, str]] = []
    for index, action in enumerate(actions):
        action_type: ActionType = action.action_type
        family = action_family(action_type)
        tier = int(action.permission_tier)
        ceiling = profile.max_tier.get(family.value, int(PermissionTier.READ_ONLY))
        if action_type.value in profile.deny_action_types:
            found.append((index, f"{action_type.value} is denied for source '{source.value}' by gateway policy."))
        elif tier > ceiling:
            found.append(
                (
                    index,
                    f"{action_type.value} (tier={PermissionTier(tier).name}, family={family.value}) exceeds the "
                    f"'{source.value}' source's allowed ceiling ({PermissionTier(ceiling).name}).",
                )
            )
        elif action.permission_tier == PermissionTier.ROOT_CRITICAL and not profile.allow_root:
            found.append((index, f"{action_type.value} requires root, which is denied for source '{source.value}'."))
    return found
