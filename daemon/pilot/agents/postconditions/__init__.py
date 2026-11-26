"""Registry of independent post-condition verifiers, one module per area.

A verifier for an action lives in a module of this package (for example
``mail_calendar.py`` or ``git_repo.py``) and registers itself with
``@register``. ``pilot.agents.verifier`` loads every module of this package
when it is imported, so adding a verifier never edits a shared dict, a shared
``if`` chain or a shared list of modules.

The three outcomes a verifier returns:

- ``verified(detail)``: the post-condition was observed through a path the
  executor did not write. The action counts as independently observed.
- ``mismatch(detail)``: the state was observed and differs from what the
  action should have produced. The action fails verification.
- ``unobserved(detail)``: the verifier could not look (no credentials, a
  platform without the probe, a provider without the folder). The action keeps
  the executor's word, is shown as REPORTED and is not counted as observed.

An exception raised by a verifier fails verification closed ("Verification
error: ..."), exactly as for the verifiers in ``pilot.agents.verifier``. Use it
for an observer that broke, not for a case the verifier knows it cannot see.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

from pilot.actions import ActionResult, ActionType

DEFAULT_TIMEOUT_SECONDS = 20.0


class Outcome(NamedTuple):
    """The verifier's answer, in the shape ``Verifier._verify_single`` returns."""

    passed: bool
    detail: str
    observed: bool


def verified(detail: str) -> Outcome:
    return Outcome(True, detail, True)


def mismatch(detail: str) -> Outcome:
    return Outcome(False, detail, True)


def unobserved(detail: str) -> Outcome:
    return Outcome(True, detail, False)


VerifierFunction = Callable[[ActionResult], Awaitable[Outcome]]


@dataclass(frozen=True)
class Registration:
    action_type: ActionType
    method: str
    check: VerifierFunction
    precheck: bool
    timeout: float
    module: str


_REGISTRY: dict[ActionType, Registration] = {}


def register(
    *action_types: ActionType,
    method: str,
    precheck: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Callable[[VerifierFunction], VerifierFunction]:
    """Register ``check`` as the independent post-condition verifier of ``action_types``.

    ``method`` is the name the capability catalog shows, ending in ``_postcondition``
    (``conditional_`` first when the check can only sometimes observe). ``precheck``
    lets the executor skip an action whose state is already present; set it only
    for a check that is free of side effects and whose action is idempotent (never
    for a send, a post or a push). ``timeout`` bounds the check; when it runs out the
    outcome is ``unobserved``.
    """

    if not action_types:
        raise ValueError("register() needs at least one action type")
    if not method or not method.endswith("_postcondition"):
        raise ValueError(f"Verifier method name must end in _postcondition: {method!r}")
    if timeout <= 0:
        raise ValueError("Verifier timeout must be positive")

    def decorate(check: VerifierFunction) -> VerifierFunction:
        for action_type in action_types:
            existing = _REGISTRY.get(action_type)
            if existing is not None and existing.check is not check:
                raise ValueError(
                    f"{action_type.value} already has a verifier in {existing.module}; "
                    f"a second one in {check.__module__} is refused"
                )
            _REGISTRY[action_type] = Registration(
                action_type=action_type,
                method=method,
                check=check,
                precheck=precheck,
                timeout=timeout,
                module=check.__module__,
            )
        return check

    return decorate


_loaded = False


def load_area_modules() -> Mapping[ActionType, Registration]:
    """Import every module of this package once, so each registers its verifiers.

    An import error is raised, never skipped: a verifier module that does not load
    would otherwise lower the verified count without anyone noticing.
    """

    global _loaded
    if not _loaded:
        for module in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
            if not module.name.startswith("_"):
                importlib.import_module(f"{__name__}.{module.name}")
        _loaded = True
    return registrations()


def registrations() -> Mapping[ActionType, Registration]:
    return MappingProxyType(_REGISTRY)
