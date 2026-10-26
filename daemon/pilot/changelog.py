"""Changelog & Feature Announcements — notifies users of new features."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.changelog")

VERSION = "0.16.0"
PUBLIC_RELEASE_VERSION = "0.16.0"
PUBLISHED_RELEASE_VERSIONS = ("0.16.0", "0.15.0", "0.14.0", "0.13.0", "0.12.0", "0.11.1", "0.9.0")

CHANGELOG = {
    "0.16.0": {
        "title": "To-dos, Watchers, Habits and Quick Answers",
        "date": "2026-10-26",
        "features": [
            {
                "name": "To-Do List",
                "description": (
                    '"Add call the bank to my list", "what\'s on my list" and "mark call the bank as done" are '
                    "handled in code. Open items show in the Today tab, the daily brief and the exported report; "
                    '"mark this email as done" still goes to the planner.'
                ),
                "jarvis_announce": "Tell me what to add to your list, and I'll keep it.",
            },
            {
                "name": "Tell Me When",
                "description": (
                    '"Tell me when Chrome closes", "let me know when my download finishes", "tell me when my battery '
                    'is full" and "tell me when my computer calms down" watch until it happens and say it once. '
                    "Watchers only look, never change anything, and give up after a day."
                ),
                "jarvis_announce": "Ask me to tell you when an app closes or a download finishes.",
            },
            {
                "name": "Repeating and App Reminders",
                "description": (
                    '"Remind me every hour to stand up" and "remind me every weekday at 9 to check email" repeat; '
                    'interval reminders skip the night. "Remind me to submit my timesheet when I open Outlook" waits '
                    "for that app to come to the front."
                ),
                "jarvis_announce": "Reminders can now repeat, or wait until you open an app.",
            },
            {
                "name": "Good Morning, Good Night and Habits",
                "description": (
                    '"Good morning" gives today\'s reminders, your list, streaks to keep and a nearly full drive; '
                    '"good night" sums up the day with focus time and tomorrow\'s first reminder. "Log a workout", '
                    '"how many workouts this week" and "what\'s my workout streak" track habits, and "how long did I '
                    'focus this week" reads your focus history.'
                ),
                "jarvis_announce": "Say good morning or good night, and I'll walk you through your day.",
            },
            {
                "name": "Quick Answers and Helpers",
                "description": (
                    "Sums, percentages, unit conversions, date countdowns and the time in about 60 cities are worked "
                    'out exactly in code. "Generate a password" puts one on the clipboard only; coin flips, dice, '
                    'random picks, a stopwatch and "how long until 5 pm" need no model.'
                ),
                "jarvis_announce": "Ask me sums, conversions, dates or the time anywhere, and I'll answer instantly.",
            },
            {
                "name": "Your Files and This Computer",
                "description": (
                    '"What did I download today", "where is my budget spreadsheet" and "why is my computer slow" are '
                    "answered from folder listings and a quick reading of processor, memory and disks. Heliox only "
                    "reports; closing anything still asks first."
                ),
                "jarvis_announce": "Ask where a file is, or why your computer is slow.",
            },
            {
                "name": "Clipboard History, Snippets and Focus Heads-Ups",
                "description": (
                    "Opt-in clipboard history is kept in memory only and skips passwords and private windows. "
                    '"Save that as my signature" and "copy my signature" reuse text. "During focus, tell me if I open '
                    'YouTube" gives one gentle heads-up per session and blocks nothing.'
                ),
                "jarvis_announce": "I can remember what you copied, save snippets, and nudge you during focus.",
            },
        ],
        "summary": (
            "A to-do list, watchers, repeating and app reminders, good morning and good night, habits, focus history, "
            "quick answers, a password generator, file and download questions, a health check, opt-in clipboard "
            "history, snippets and focus heads-ups"
        ),
    },
    "0.15.0": {
        "title": "Undoable Tasks, Signed Receipts and Your Day",
        "date": "2026-10-24",
        "features": [
            {
                "name": "Per-Task Undo",
                "description": (
                    "Before Heliox writes, moves, copies or deletes a file, replaces the clipboard, sets an "
                    "environment variable or changes the volume or brightness, it keeps what it needs to reverse "
                    "that change. Each finished task shows Undo; the preview lists what will be put back, what "
                    "changed since (left alone) and what can never be undone, and shows a line-by-line diff of "
                    'each file. "Undo the last two tasks" opens each preview in turn. Undo is computed in code, '
                    "never by a model, and every undo needs a click."
                ),
                "jarvis_announce": "You can now undo a task: I keep what I need to put your files back.",
            },
            {
                "name": "Signed Task Receipts",
                "description": (
                    "Every task that ran actions leaves an Ed25519-signed, hash-chained receipt of what ran, what "
                    "you approved and what happened, with contents only as hashes. Verify one in chat or check the "
                    'whole chain in Settings; "what did you do today" and "when did I ask you to" are answered '
                    "from them, and the Today tab exports a Markdown report."
                ),
                "jarvis_announce": "Every task now leaves a signed receipt you can verify.",
            },
            {
                "name": "Your Day: Timeline, Today Tab and Daily Brief",
                "description": (
                    "An opt-in activity timeline records which app and window title were in front (never pixels "
                    'or screen text) so "what was I working on yesterday afternoon" gets an answer. The Today '
                    "tab shows what Heliox ran, where your time went and what can still be undone; an opt-in daily "
                    "brief and usage nudges build on it."
                ),
                "jarvis_announce": "Ask me what you worked on, or open the Today tab for your day at a glance.",
            },
            {
                "name": "Focus, Reminders, Notes and Timers",
                "description": (
                    "Focus sessions, pomodoros and quiet hours hold reminders and the brief while approvals still "
                    "come through. Reminders, timers and quick notes are set and read by voice, kept across "
                    "restarts and answered in code without the planner."
                ),
                "jarvis_announce": "Say focus for 45 minutes, set a timer, or note something, and I'll handle it.",
            },
            {
                "name": "Routines, Voice Shortcuts and Setups",
                "description": (
                    '"Do that every day at 9" turns a finished task into a routine that may use only the actions '
                    'it used; "when I say start work" creates a voice shortcut; "save this as my coding setup" '
                    'remembers your open apps; "do that again" repeats the last task with fresh approvals.'
                ),
                "jarvis_announce": "I can now repeat tasks on a schedule, learn your voice shortcuts and restore your setups.",
            },
            {
                "name": "Phone Task Requests",
                "description": (
                    "A paired Air Handoff phone can send an encrypted, signed request. It waits for Run on the "
                    "computer, or, if allowed, status checks start at once with a fixed read-only list; the answer "
                    "goes back to the phone, which can never approve anything."
                ),
                "jarvis_announce": "You can now send me requests from your paired phone.",
            },
            {
                "name": "Command Palette and Your Data",
                "description": (
                    'Ctrl+K or "what can I say?" lists every built-in command with working examples. Settings -> '
                    "Your data shows everything Heliox keeps on this computer with a Clear button for each, "
                    '"forget my activity from the last hour" removes that stretch, and backups never include '
                    "settings, tokens or keys."
                ),
                "jarvis_announce": "Press Control K to see what you can say, and Settings shows every piece of data I keep.",
            },
        ],
        "summary": (
            "Per-task undo, signed task receipts, an activity timeline with a Today tab and daily brief, focus "
            "sessions, reminders, notes, routines, voice shortcuts, setups, phone task requests, a command "
            "palette and a Your data panel"
        ),
    },
    "0.14.0": {
        "title": "Co-Adaptive Assistance and Governed Desktop Agency",
        "date": "2026-10-16",
        "features": [
            {
                "name": "Assistance Co-Adaptation",
                "description": (
                    "An opt-in, consented selection study models how a person, the interface and Heliox adapt "
                    "to each other and predicts which help suits this person today (a larger target, a confirm "
                    "step, a clarifying question or another input) within rules the person sets. It changes "
                    "presentation only, never commits an ambiguous selection by guessing, and every published "
                    "result so far comes from simulated users."
                ),
                "jarvis_announce": (
                    "I can now run a consented selection study that learns which kind of help suits you, "
                    "within your own rules."
                ),
            },
            {
                "name": "Accessibility-Grounded Desktop Control",
                "description": (
                    "Desktop clicks are grounded in the accessibility tree instead of pixels, buttons can be "
                    "pressed in the background without moving the cursor, browser actions use page snapshots, "
                    "and the person can take the desktop back at any time while Heliox can hand it over when "
                    "it needs help. No virtual machine is involved."
                ),
                "jarvis_announce": (
                    "I now act on real interface elements instead of pixels, and you can take the desktop back "
                    "at any time."
                ),
            },
            {
                "name": "Durable and Scheduled Autonomy",
                "description": (
                    "Autonomous jobs survive daemon restarts and resume only when the person asks, scheduled "
                    "runs are limited to an approved list of actions, unreachable goals are reported instead "
                    "of faked, and each tool call of background work appears as a card in chat."
                ),
                "jarvis_announce": "Background jobs now survive restarts and scheduled runs stay inside an approved list.",
            },
            {
                "name": "Tools from Your MCP Servers",
                "description": (
                    "Heliox can call tools from the person's own MCP servers listed in mcp_servers.json. "
                    "Read-only tools run directly, every other tool asks first, and Settings shows each server "
                    "with a reload button."
                ),
                "jarvis_announce": "I can now use tools from your own MCP servers, asking before anything that changes state.",
            },
            {
                "name": "Voice-First Control While Heliox Works",
                "description": (
                    "Speaking while a task runs does what the person means: corrections, questions and a "
                    "spoken no reach the running task, background approval dialogs stay open while the person "
                    "keeps working, and the approval dialog shows what an action will actually run. The world "
                    "model reviews every autonomous plan."
                ),
                "jarvis_announce": "You can now talk to me while I work, and I show exactly what each approval will run.",
            },
            {
                "name": "Fail-Closed Settings and Approvals",
                "description": (
                    "Settings change safely while work runs and survive unreadable or hand-edited config files; "
                    "an action family a gateway profile does not list is read-only; an approval that is never "
                    "answered expires instead of running; a plan resumed after a restart runs only the ticked "
                    "actions; and a gesture can approve only a dialog the person can see."
                ),
                "jarvis_announce": (
                    "Unanswered approvals now expire instead of running, and I only resume the actions you ticked."
                ),
            },
        ],
        "summary": (
            "Consented assistance co-adaptation, accessibility-grounded desktop control, durable and scheduled "
            "autonomy, MCP tools, voice-first control while working, and fail-closed settings and approvals"
        ),
    },
    "0.13.0": {
        "title": "Verified Autonomy and Runtime Hardening",
        "date": "2026-08-27",
        "features": [
            {
                "name": "Truthful Execution Contracts",
                "description": (
                    "Completed work is revalidated against executor and checkpoint contracts, "
                    "while failed, rejected, skipped, partial, and unavailable paths remain "
                    "visible instead of being reported as successful."
                ),
                "jarvis_announce": (
                    "I now verify completed work more strictly and keep failed or partial outcomes visible."
                ),
            },
            {
                "name": "Transactional Live Settings",
                "description": (
                    "Runtime configuration changes are validated and applied atomically, roll back "
                    "on failure, and can update supported model limits, devices, calendars, camera, "
                    "gesture, voice, and mesh controls without leaving split state."
                ),
                "jarvis_announce": "Live settings now apply atomically and roll back cleanly on failure.",
            },
            {
                "name": "Authenticated Peer Collaboration",
                "description": (
                    "Opt-in LAN peers use authenticated setup, bounded timeouts, constrained delegation, "
                    "and the same permission and startup gates as local collaboration."
                ),
                "jarvis_announce": "Trusted local peers can now collaborate through bounded delegation.",
            },
            {
                "name": "Coordinated Multimodal Control",
                "description": (
                    "Voice, gesture, gaze, cursor, approval, and staged neural inputs arbitrate shared "
                    "resources more reliably, including camera ownership, gesture priority, and "
                    "neural-sidecar diagnostics."
                ),
                "jarvis_announce": "Voice, gesture, gaze, cursor, and staged neural inputs now coordinate more reliably.",
            },
            {
                "name": "Bounded Runtime Resources",
                "description": (
                    "Daemon shutdown now quiesces autonomous jobs, monitors, browser resources, "
                    "subscription CLIs, input services, and collaboration work; idle local TTS and "
                    "model workers release heavy memory outside the long-lived daemon."
                ),
                "jarvis_announce": "Background services and local model workers now release resources more predictably.",
            },
            {
                "name": "Measured Execution Quality",
                "description": (
                    "The executor records step-budget quality, skips actions only when deterministic "
                    "postconditions are already satisfied, preserves fast local status paths, and "
                    "publishes a refreshed reproducible benchmark bundle."
                ),
                "jarvis_announce": "I now measure execution quality and avoid repeating work that is already verified.",
            },
        ],
        "summary": (
            "Truthful outcomes, transactional settings, authenticated collaboration, coordinated "
            "multimodal control, bounded resources, and measured execution quality"
        ),
    },
    "0.12.0": {
        "title": "Governed Intelligence and Handoff",
        "date": "2026-08-16",
        "features": [
            {
                "name": "Existing AI Subscription Providers",
                "description": (
                    "Use an existing Codex or Claude Code login through the provider's official "
                    "CLI, choose an available model, and inspect bounded plan-usage evidence "
                    "without copying OAuth credentials into Heliox."
                ),
                "jarvis_announce": (
                    "You can now use an existing Codex or Claude Code subscription with guarded "
                    "model selection and usage controls."
                ),
            },
            {
                "name": "Local Heliox MCP",
                "description": (
                    "IDE agents can stage Heliox tasks through a local MCP server while the Heliox "
                    "daemon retains identity checks, permission gates, approvals, and execution."
                ),
                "jarvis_announce": "My local MCP now routes IDE requests through the same approval boundary.",
            },
            {
                "name": "Secure Air Handoff",
                "description": (
                    "Cast a deliberately selected Heliox view to a paired mobile browser over an "
                    "ephemeral encrypted session, with explicit start, stop, and gesture controls."
                ),
                "jarvis_announce": "Air Handoff can now share an approved view with a paired phone.",
            },
            {
                "name": "Verified Integrations",
                "description": (
                    "Email, calendar, and SSH settings now reach their executable specialists "
                    "through credential-vault, RPC, policy, and result contracts."
                ),
                "jarvis_announce": "Email, calendar, and SSH integrations now follow one verified path.",
            },
            {
                "name": "Unified Cognitive Runtime",
                "description": (
                    "Companion narration, learned-risk interruption, follow-up suggestions, and "
                    "autonomous execution now use the live production gateway instead of duplicate "
                    "runtime state."
                ),
                "jarvis_announce": "My companion services now share one live cognitive runtime.",
            },
            {
                "name": "Evidence-Driven Reliability",
                "description": (
                    "Local health reviews, neural staged-goal dispatch, model timeouts, microphone "
                    "capture, MCP version reporting, and forensic results now fail more truthfully."
                ),
                "jarvis_announce": "Health, model, microphone, MCP, and forensic failures now report more truthfully.",
            },
        ],
        "summary": (
            "Subscription-backed planning, local MCP, secure Air Handoff, verified integrations, "
            "and one governed cognitive runtime"
        ),
    },
    "0.11.1": {
        "title": "Adaptive Companion Release",
        "date": "2026-08-12",
        "features": [
            {
                "name": "Reliable Windows Release Delivery",
                "description": (
                    "The Windows release pipeline now prefetches and checksum-verifies WiX with "
                    "explicit retries before building MSI and NSIS installers."
                ),
                "jarvis_announce": "Windows release packaging is now deterministic and verified.",
            },
        ],
        "summary": "The complete v0.11 companion feature set with deterministic Windows packaging",
    },
    "0.11.0": {
        "title": "Adaptive Multimodal Companion",
        "date": "2026-08-12",
        "features": [
            {
                "name": "Continuous Companion Loop",
                "description": (
                    "Always-on voice, spoken suggestions, interruption, and autonomous browser "
                    "or application control now share one coordinated execution loop."
                ),
                "jarvis_announce": "My listening, actions, suggestions, and follow-ups now stay coordinated.",
            },
            {
                "name": "Adaptive Intelligence Stack",
                "description": (
                    "A bounded experience ledger, temporal memory, verified online adaptation, "
                    "strategy evolution, and an optional JEPA-style predictor advise future plans."
                ),
                "jarvis_announce": "I can learn from verified outcomes without widening my authority.",
            },
            {
                "name": "Expanded Specialist Mesh",
                "description": (
                    "Twenty-one specialists provide concrete coverage for all 156 declared action "
                    "types across desktop, browser, developer, integration, and research workflows."
                ),
                "jarvis_announce": "My specialist mesh now covers every declared action type.",
            },
            {
                "name": "Reliable Multimodal Control",
                "description": (
                    "Gaze, 3D hand gestures, cursor control, wake-word audio, and workflow bindings "
                    "can operate together with temporal false-positive rejection."
                ),
                "jarvis_announce": "Voice, gaze, and gesture inputs now cooperate more reliably.",
            },
            {
                "name": "Guarded Neural Research Pipeline",
                "description": (
                    "Synthetic BrainFlow and recorded EEGBCI paths now exercise calibrated, signed, "
                    "consent-bounded neural intents without claiming live brain control."
                ),
                "jarvis_announce": "Recorded and synthetic neural research now follows explicit safety controls.",
            },
            {
                "name": "Truthful Execution and Delivery",
                "description": (
                    "Application launch, browser targeting, approvals, cancellations, result reporting, "
                    "first-run Chromium setup, and cross-platform packaging are more deterministic."
                ),
                "jarvis_announce": "I now report completed, blocked, cancelled, and failed work more truthfully.",
            },
        ],
        "summary": "Adaptive learning, coordinated multimodal control, expanded specialists, and guarded neural research",
    },
    "0.10.1": {
        "title": "Reliable Interactive Sessions",
        "date": "2026-07-30",
        "features": [
            {
                "name": "Persistent Private Sessions",
                "description": (
                    "Start isolated chats, reopen prior sessions, and retain adaptive memory "
                    "without leaking one conversation's transcript into another."
                ),
                "jarvis_announce": "Your sessions are now durable, private, and easy to reopen.",
            },
            {
                "name": "Faster Semantic Browser Control",
                "description": (
                    "Common browser tasks use bounded planning, resolve controls by meaning, and "
                    "serialize shared-page actions to avoid slow or conflicting execution."
                ),
                "jarvis_announce": "I can now resolve and use browser controls more quickly and reliably.",
            },
            {
                "name": "Coordinated Companion Services",
                "description": (
                    "Narration, learned-risk interruption, voice, and verified follow-up suggestions "
                    "remain available together while approvals pass through every action gate."
                ),
                "jarvis_announce": "My voice, safety model, and follow-up guidance now stay coordinated.",
            },
            {
                "name": "Safer, Truthful Task Results",
                "description": (
                    "Cancelled work, cloud-provider failures, destructive file changes, and exact "
                    "findings now produce bounded, redacted, recoverable results."
                ),
                "jarvis_announce": "Task outcomes are now clearer, safer, and easier to recover.",
            },
        ],
        "summary": "Persistent sessions, faster browser control, and coordinated companion reliability",
    },
    "0.10.0": {
        "title": "Interactive Companion and Reliability",
        "date": "2026-07-29",
        "features": [
            {
                "name": "Interactive Companion",
                "description": (
                    "Opt-in narration, risk interruption, spoken follow-up, and suggestions now "
                    "share one coordinated conversation loop."
                ),
                "jarvis_announce": "I can now narrate, interrupt, and follow up without voices overlapping.",
            },
            {
                "name": "Learned Risk World Model",
                "description": (
                    "On-device learned predictions now advise the deterministic safety gate and can "
                    "add caution or pause risky actions."
                ),
                "jarvis_announce": "My learned risk model can now warn or pause before an unsafe action.",
            },
            {
                "name": "Unified Camera Intelligence",
                "description": (
                    "Gaze, hand gestures, and cursor control can run together from one camera stream "
                    "without suppressing enabled inputs."
                ),
                "jarvis_announce": "Gaze and gesture controls can now work together from one camera.",
            },
            {
                "name": "Moderated Plugin Marketplace",
                "description": (
                    "Reviewed packages are verified by manifest and SHA-256 before installation, "
                    "and merged catalog entries appear without a desktop release."
                ),
                "jarvis_announce": "Approved marketplace plugins are now verified before installation.",
            },
            {
                "name": "Reliable Approvals and Results",
                "description": (
                    "Approval responses remain live on the active connection, while cancelled, "
                    "blocked, and failed actions now finish with truthful terminal results."
                ),
                "jarvis_announce": "Approvals and failures now stay synchronized with the task.",
            },
            {
                "name": "Durable Voice and Gesture Workflows",
                "description": (
                    "Multi-step workflows can be submitted, paused, resumed, and inspected while "
                    "autonomous healing remains separately permission-gated."
                ),
                "jarvis_announce": "Long-running voice and gesture workflows can now pause and resume safely.",
            },
        ],
        "summary": "Companion intelligence, learned safety, multimodal control, and reliable execution",
    },
    "0.9.0": {
        "title": "JARVIS Autonomy",
        "date": "2026-07-22",
        "features": [
            {
                "name": "Autonomous Background Tasks",
                "description": "Run permission-gated multi-step tasks while continuing to use the app.",
            },
            {
                "name": "Live Execution Narrator",
                "description": "Optionally hear short progress descriptions and risk interruptions.",
            },
            {
                "name": "Local Voice, Gesture, and Gaze Inputs",
                "description": "Use on-device multimodal controls without sending camera frames.",
            },
        ],
        "summary": "Background autonomy and local multimodal control",
    },
}


def get_state_dir() -> Path:
    from pilot.config import STATE_DIR

    return STATE_DIR


def get_last_version() -> str | None:
    state_dir = get_state_dir()
    version_file = state_dir / "last_version.txt"
    if version_file.exists():
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return None


def set_last_version(version: str) -> None:
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    version_file = state_dir / "last_version.txt"
    version_file.write_text(version, encoding="utf-8")


def check_for_updates() -> list[dict[str, Any]]:
    """Check if there are new features since last run."""
    last_version = get_last_version()
    current_version = VERSION

    if last_version is None:
        return get_full_changelog()

    if last_version == current_version:
        return []

    new_features = []
    for ver in CHANGELOG:
        if _compare_versions(ver, last_version) > 0:
            new_features.extend(CHANGELOG[ver]["features"])

    return new_features


def get_full_changelog() -> list[dict[str, Any]]:
    features = []
    for ver in CHANGELOG:
        ver_features = CHANGELOG[ver]["features"]
        # Handle both dict features and string features
        if ver_features and isinstance(ver_features[0], dict):
            for feat in ver_features:
                feat_copy = dict(feat)  # Copy to avoid mutation
                feat_copy["version"] = ver
                feat_copy["date"] = CHANGELOG[ver]["date"]
                features.append(feat_copy)
        else:
            # String feature - convert to dict
            for feat_name in ver_features:
                features.append(
                    {
                        "name": feat_name,
                        "version": ver,
                        "date": CHANGELOG[ver]["date"],
                    }
                )
    return features


def _compare_versions(v1: str, v2: str) -> int:
    """Compare semantic versions. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
    parts1 = [int(x) for x in v1.split(".")]
    parts2 = [int(x) for x in v2.split(".")]
    for p1, p2 in zip(parts1, parts2):
        if p1 > p2:
            return 1
        elif p1 < p2:
            return -1
    return 0


def get_welcome_message() -> dict[str, Any]:
    """Get the welcome message for new users."""
    return {
        "title": "Welcome to Heliox OS",
        "version": VERSION,
        "tagline": "Your biologically-inspired AI assistant",
        "features": get_full_changelog()[:3],
    }


def announce_new_features() -> str:
    """Generate JARVIS announcement for new features."""
    new_features = check_for_updates()

    if not new_features:
        return ""

    lines = [
        "Welcome to Heliox OS version " + VERSION + ".",
    ]

    for feat in new_features[:3]:
        if "jarvis_announce" in feat:
            lines.append(feat["jarvis_announce"])

    lines.append("Say 'What can you do?' to learn more.")

    return " ".join(lines)


def mark_version_seen() -> None:
    """Mark that user has seen current version."""
    set_last_version(VERSION)


def get_cognitive_status() -> dict[str, Any]:
    """Get brief cognitive feature status for HUD."""
    new_features = check_for_updates()
    return {
        "new_features_available": len(new_features) > 0,
        "new_feature_count": len(new_features),
        "version": VERSION,
        "cognitive_engine_available": True,
    }
