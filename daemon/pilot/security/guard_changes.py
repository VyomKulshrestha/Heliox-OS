"""Which setting changes take protection away, so the daemon asks the person before they take effect.

Narrowing a guard is always free. Widening one takes effect only after the person approves it in the daemon's own
approval dialog: turning a protection off, letting more run without asking, or sending the person's requests and
the text Heliox reads to a computer that is not this one. Turning a capture on (Live reading the field being typed
in, Live listening to speech) asks too, once, as a change of setting; using a capability that is on never asks.

Every function here is pure: it compares what is set now with what a request would set and names each widening
in plain words, with what it stops protecting. The handlers in pilot.server call ``widenings_for`` first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
from urllib.parse import urlparse

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

# Where a cloud provider sends requests, for the sentence the person reads.
CLOUD_HOSTS = {
    "openai": "api.openai.com",
    "openrouter": "openrouter.ai",
    "claude": "api.anthropic.com",
    "anthropic": "api.anthropic.com",
    "gemini": "generativelanguage.googleapis.com",
    "meta": "api.llama.com",
}
SUBSCRIPTION_HOSTS = {"codex": "OpenAI, through Codex", "claude": "Anthropic, through Claude Code"}

# Stronger first: a sandbox that isolates more ranks higher. "auto" picks the strongest available.
SANDBOX_RANK = {"firecracker": 4, "docker": 3, "auto": 3, "restricted": 2, "none": 0}


@dataclass(frozen=True)
class Widening:
    key: str  # "security.unrestricted_shell"
    says: str  # what changes, in plain words
    stops: str  # what it stops protecting, or what leaves this computer

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "says": self.says, "stops": self.stops}


def _plain(section: Any) -> dict[str, Any]:
    if is_dataclass(section) and not isinstance(section, type):
        return asdict(section)
    if isinstance(section, dict):
        return dict(section)
    return dict(getattr(section, "__dict__", {}))


def host_of(value: Any) -> str:
    """The host a URL or a bare host name points at; "" for nothing."""
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"//{text}")
    return (parsed.hostname or "").lower()


def is_local(host: str) -> bool:
    return not host or host in LOCAL_HOSTS or host.endswith(".localhost")


def model_destination(model: dict[str, Any]) -> str:
    """Where the model's requests go: "" for this computer, else the host or service, in plain words."""
    provider = str(model.get("provider") or "").lower()
    if provider == "ollama":
        host = host_of(model.get("ollama_base_url"))
        return "" if is_local(host) else host
    if provider == "cloud":
        name = str(model.get("cloud_provider") or "").lower()
        return CLOUD_HOSTS.get(name, name or "a cloud model provider")
    if provider == "subscription":
        name = str(model.get("subscription_provider") or "").lower()
        return SUBSCRIPTION_HOSTS.get(name, name or "a subscription model provider")
    return "" if provider in {"", "local"} else provider


def _bool(values: dict[str, Any], key: str, current: Any) -> bool:
    value = values.get(key, current)
    return value if isinstance(value, bool) else bool(current)


def _number(values: dict[str, Any], key: str, current: Any) -> float:
    value = values.get(key, current)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float(current)


def section_widenings(section: str, current_section: Any, values: dict[str, Any]) -> list[Widening]:
    """The widenings in one update_config request (``values`` for ``section``)."""
    now = _plain(current_section)
    found: list[Widening] = []

    def turned_on(key: str, says: str, stops: str) -> None:
        if not now.get(key) and _bool(values, key, now.get(key)):
            found.append(Widening(f"{section}.{key}", says, stops))

    def turned_off(key: str, says: str, stops: str) -> None:
        if now.get(key) and not _bool(values, key, now.get(key)):
            found.append(Widening(f"{section}.{key}", says, stops))

    def goes_to(key: str, says: str, what: str) -> None:
        if key not in values:
            return
        new, old = host_of(values[key]), host_of(now.get(key))
        if new != old and not is_local(new):
            found.append(Widening(f"{section}.{key}", says.format(host=new), what.format(host=new)))

    if section == "security":
        turned_on(
            "unrestricted_shell",
            "Let Heliox run any shell command",
            "the list of commands Heliox may run and the checks on their arguments",
        )
        turned_on("root_enabled", "Let Heliox act as administrator", "the refusal of actions that need root")
        turned_off("dry_run", "Let Heliox's actions really run", "the dry run, which only shows what would happen")
        turned_off(
            "snapshot_on_destructive",
            "Stop taking a snapshot before destructive actions",
            "the way back after a destructive action",
        )
        if "sandbox_mode" in values:
            old, new = str(now.get("sandbox_mode", "auto")), str(values["sandbox_mode"])
            if SANDBOX_RANK.get(new, 0) < SANDBOX_RANK.get(old, 0):
                found.append(
                    Widening(
                        "security.sandbox_mode",
                        f"Run commands in a weaker sandbox ({new} instead of {old})",
                        "part of the isolation between generated code and your computer",
                    )
                )
        if "sandbox_firecracker_binary" in values and values["sandbox_firecracker_binary"] != now.get(
            "sandbox_firecracker_binary"
        ):
            found.append(
                Widening(
                    "security.sandbox_firecracker_binary",
                    f"Use {values['sandbox_firecracker_binary']} as the sandbox program",
                    "the sandbox Heliox trusts to isolate generated code",
                )
            )
    elif section == "preview":
        turned_off("enabled", "Stop previewing plans before they run", "the preview of each plan")
    elif section == "restrictions":
        labels = {
            "protected_folders": ("Stop protecting {item}", "the protection of that folder"),
            "protected_packages": ("Let Heliox remove the package {item}", "the protection of that package"),
            "blocked_commands": ("Let Heliox run {item}", "the block on that command"),
        }
        for key, (says, stops) in labels.items():
            if key not in values or not isinstance(values[key], list):
                continue
            kept = {str(item) for item in values[key]}
            for item in now.get(key, []) or []:
                if str(item) not in kept:
                    found.append(Widening(f"restrictions.{key}", says.format(item=item), stops))
    elif section == "model":
        merged = {**now, **values}
        new, old = model_destination(merged), model_destination(now)
        if new and new != old:
            found.append(
                Widening(
                    "model.provider",
                    f"Use the language model at {new}",
                    f"Your requests and the text Heliox reads for them will go to {new}, and so will your name, "
                    "your answer style and the parts of your profile that bear on what you ask, unless you turn "
                    "that off (Settings, Your profile and the model).",
                )
            )
    elif section == "proxy":
        for key in ("http", "https"):
            goes_to(key, "Send Heliox's web requests through {host}", "Heliox's web requests will pass through {host}.")
    elif section == "conversation":
        goes_to("search_url", "Search the web with {host}", "What you ask Heliox to look up will go to {host}.")
        old_share = str(now.get("share_profile", "auto") or "auto").casefold()
        new_share = str(values.get("share_profile", old_share) or old_share).casefold()
        if old_share == "off" and new_share != "off":
            found.append(
                Widening(
                    "conversation.share_profile",
                    "Send what Heliox knows about you to a model that is not on this computer",
                    "With a cloud model, your name, your answer style and the parts of your profile that bear on what"
                    " you ask will go with your requests.",
                )
            )
    elif section == "email":
        goes_to("imap_host", "Read mail from {host}", "Heliox will sign in to {host} with your mail password.")
        goes_to("smtp_host", "Send mail through {host}", "Mail Heliox sends will go through {host}.")
    elif section == "calendar":
        goes_to("caldav_url", "Use the calendar at {host}", "Your calendar will be read from and written to {host}.")
    elif section == "timeline":
        turned_on(
            "enabled",
            "Keep the activity timeline",
            "Heliox will record which app and window title are in front, and when, on this computer.",
        )
    elif section == "live":
        # A capture counts when Live is on with it: turning Live on with captures set, or a capture on while Live is.
        captures = {
            "read_focused_text": "Heliox will read the text in the field you are typing in.",
            "listen_to_speech": "Heliox will listen to what you say near the computer, not only to requests.",
        }
        live_was_on, live_on = bool(now.get("enabled")), _bool(values, "enabled", now.get("enabled"))
        for key, words in captures.items():
            was = live_was_on and bool(now.get(key))
            will = live_on and _bool(values, key, now.get(key))
            if will and not was:
                says = "Let Live read what you type" if key == "read_focused_text" else "Let Live listen to your speech"
                found.append(Widening(f"live.{key}", says, words))
    elif section == "gateway" and "risk_gate_enabled" in values:
        turned_off("risk_gate_enabled", "Turn off the risk check", "the world model's check of each plan's risk")
    elif section == "self_healing":
        old_tier = int(now.get("auto_execute_max_tier", 0))
        if _number(values, "auto_execute_max_tier", old_tier) > old_tier:
            found.append(
                Widening(
                    "self_healing.auto_execute_max_tier",
                    f"Let self-healing run tier {int(values['auto_execute_max_tier'])} actions without asking",
                    "your approval of riskier self-healing actions",
                )
            )
        _shorter_wait(found, section, now, values)
    elif section == "narration":
        turned_off(
            "interrupt_on_risk",
            "Stop pausing plans to warn about risk",
            "the narrator's warning before a risky step",
        )
        _shorter_wait(found, section, now, values)
    return found


def _shorter_wait(found: list[Widening], section: str, now: dict[str, Any], values: dict[str, Any]) -> None:
    old = float(now.get("confirm_timeout_seconds", 0) or 0)
    if old and _number(values, "confirm_timeout_seconds", old) < old:
        found.append(
            Widening(
                f"{section}.confirm_timeout_seconds",
                f"Give you {_number(values, 'confirm_timeout_seconds', old):g} s to answer instead of {old:g} s",
                "the time you have to stop it",
            )
        )


def gateway_widenings(profile: str, current_floor: Any, params: dict[str, Any]) -> list[Widening]:
    """A gateway_policy_update for ``profile``: a tier raised, a denied action allowed, root allowed."""
    now = _plain(current_floor)
    found: list[Widening] = []
    raised = params.get("max_tier")
    if isinstance(raised, dict):
        old_tiers = now.get("max_tier", {}) or {}
        for family, tier in raised.items():
            if isinstance(tier, int) and not isinstance(tier, bool) and tier > int(old_tiers.get(family, 0)):
                found.append(
                    Widening(
                        f"gateway.{profile}.{family}",
                        f"Let {profile} requests use {family} actions up to tier {tier}",
                        f"the lower limit on {family} actions for {profile} requests",
                    )
                )
    deny = params.get("deny_action_types")
    if isinstance(deny, list):
        kept = {str(item) for item in deny}
        for action in now.get("deny_action_types", []) or []:
            if str(action) not in kept:
                found.append(
                    Widening(
                        f"gateway.{profile}.deny",
                        f"Let {profile} requests use {action}",
                        f"the block on {action} for {profile} requests",
                    )
                )
    if params.get("allow_root") is True and not now.get("allow_root"):
        found.append(
            Widening(
                f"gateway.{profile}.allow_root",
                f"Let {profile} requests act as administrator",
                f"the refusal of root for {profile} requests",
            )
        )
    return found


def reset_widenings(config: Any, defaults: Any, sections: list[str]) -> list[Widening]:
    """What resetting ``sections`` to ``defaults`` would widen."""
    found: list[Widening] = []
    for name in sections:
        current, default = getattr(config, name, None), getattr(defaults, name, None)
        if current is None or default is None:
            continue
        found.extend(section_widenings(name, current, _plain(default)))
    return found
