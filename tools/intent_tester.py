#!/usr/bin/env python3
"""
intent_tester.py — Offline Intent Tester CLI for Heliox OS

A lightweight developer diagnostic tool that tests how Heliox OS interprets
voice and gesture commands — without needing the daemon or Tauri frontend.

Usage:
    python tools/intent_tester.py "open chrome"
    python tools/intent_tester.py "open chrome" --gesture fist
    python tools/intent_tester.py "delete file" --gesture thumbs_up --confidence 0.9
    python tools/intent_tester.py "open chrome" --json
    python tools/intent_tester.py --batch tools/test_commands.txt

Author: Sanvi Kulkarni (@Sanvi09Kulkarni)
GSSoC '25 Contribution — Issue #<your-issue-number>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# ── Try importing Rich for pretty output ──
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── Add daemon to path so we can import pilot modules ──
DAEMON_PATH = Path(__file__).resolve().parent.parent / "daemon"
sys.path.insert(0, str(DAEMON_PATH))

# ── Import Heliox OS internals ──
try:
    from pilot.multimodal.fusion import (
        MultimodalFusionEngine,
        InputEvent,
        ModalityType,
        GESTURE_MODIFIERS,
        GESTURE_STANDALONE_COMMANDS,
        FusedIntent,
    )
    FUSION_AVAILABLE = True
except ImportError as e:
    FUSION_AVAILABLE = False
    IMPORT_ERROR = str(e)

console = Console() if RICH_AVAILABLE else None

# ── Agent routing rules ──
# Maps keywords in commands to likely agent handlers
AGENT_ROUTING: dict[str, list[str]] = {
    "web agent"     : ["search", "browse", "google", "navigate", "url", "website", "open site"],
    "system agent"  : ["open", "launch", "close", "delete", "file", "folder", "run", "execute", "app"],
    "code agent"    : ["code", "write", "create", "script", "python", "function", "debug", "test"],
    "monitor agent" : ["cpu", "ram", "memory", "disk", "battery", "process", "kill", "performance"],
    "comms agent"   : ["email", "send", "message", "slack", "discord", "notify", "mail"],
}

# ── Risk level rules ──
HIGH_RISK_KEYWORDS   = ["delete", "remove", "kill", "format", "shutdown", "restart", "drop", "wipe"]
MEDIUM_RISK_KEYWORDS = ["close", "cancel", "stop", "modify", "change", "update", "move"]


def determine_agents(command: str) -> list[str]:
    """Determine which agents would handle this command."""
    command_lower = command.lower()
    matched = []
    for agent, keywords in AGENT_ROUTING.items():
        if any(kw in command_lower for kw in keywords):
            matched.append(agent)
    return matched if matched else ["orchestrator (general)"]


def determine_risk(command: str, gesture: str = "") -> tuple[str, str]:
    """Determine risk level and color."""
    command_lower = command.lower()
    if any(kw in command_lower for kw in HIGH_RISK_KEYWORDS):
        return "High ⚠️", "red"
    if any(kw in command_lower for kw in MEDIUM_RISK_KEYWORDS):
        return "Medium 🟡", "yellow"
    return "Low ✅", "green"


def get_gesture_info(gesture_name: str) -> dict:
    """Get gesture metadata."""
    if not gesture_name:
        return {}
    modifier = GESTURE_MODIFIERS.get(gesture_name, "unknown")
    standalone = GESTURE_STANDALONE_COMMANDS.get(gesture_name, "n/a")
    return {
        "modifier"   : modifier.value if hasattr(modifier, "value") else str(modifier),
        "standalone" : standalone,
    }


async def run_fusion(
    voice_text: str,
    gesture_name: str = "",
    voice_confidence: float = 0.95,
    gesture_confidence: float = 0.85,
) -> FusedIntent | None:
    """Run the MultimodalFusionEngine and return a FusedIntent."""
    engine = MultimodalFusionEngine()

    # Create voice event
    voice_event = InputEvent(
        modality=ModalityType.VOICE,
        transcript=voice_text.strip(),
        voice_confidence=voice_confidence,
        is_final=True,
        timestamp=time.time(),
    )

    if gesture_name:
        # Create gesture event slightly before voice (natural timing)
        gesture_event = InputEvent(
            modality=ModalityType.GESTURE,
            gesture_name=gesture_name,
            gesture_confidence=gesture_confidence,
            timestamp=time.time() - 0.5,
        )
        # Feed gesture first, then voice
        await engine.on_gesture_event(gesture_event)

    # Feed voice event — this triggers fusion
    intent = await engine.on_voice_event(voice_event)
    return intent


def print_result_rich(
    voice_text: str,
    gesture_name: str,
    intent: FusedIntent,
    voice_confidence: float,
) -> None:
    """Print beautiful Rich-formatted output."""
    agents   = determine_agents(intent.command)
    risk, risk_color = determine_risk(intent.command, gesture_name)
    gesture_info     = get_gesture_info(gesture_name)

    console.print()
    console.rule("[bold cyan]🧪 HELIOX OS — INTENT TESTER[/bold cyan]")

    # ── Input section ──
    input_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    input_table.add_column("Key",   style="dim", width=16)
    input_table.add_column("Value", style="white")
    input_table.add_row("🎤 Voice",      f'"{voice_text}"')
    input_table.add_row("🤚 Gesture",    gesture_name if gesture_name else "none")
    input_table.add_row("📊 Voice Conf", f"{voice_confidence * 100:.0f}%")
    console.print(Panel(input_table, title="[bold blue]📥 Input[/bold blue]", border_style="blue"))

    # ── Fusion result section ──
    fusion_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    fusion_table.add_column("Key",   style="dim", width=16)
    fusion_table.add_column("Value", style="white")
    fusion_table.add_row("💬 Command",      f"[bold green]{intent.command}[/bold green]")
    fusion_table.add_row("🔀 Fusion Type",  intent.fusion_type)
    fusion_table.add_row("⚡ Confidence",   f"[bold]{intent.confidence * 100:.0f}%[/bold]")

    if gesture_name and gesture_info:
        fusion_table.add_row("✋ Modifier",   gesture_info.get("modifier", "n/a"))

    if intent.metadata.get("time_delta_ms"):
        fusion_table.add_row("⏱️  Time Delta",  f"{intent.metadata['time_delta_ms']:.0f}ms")

    console.print(Panel(fusion_table, title="[bold green]🧠 Fusion Result[/bold green]", border_style="green"))

    # ── Routing section ──
    routing_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    routing_table.add_column("Key",   style="dim", width=16)
    routing_table.add_column("Value", style="white")
    routing_table.add_row("🎯 Agent(s)",   ", ".join(agents))
    routing_table.add_row(f"⚠️  Risk Level", f"[{risk_color}]{risk}[/{risk_color}]")

    console.print(Panel(routing_table, title="[bold yellow]🎯 Agent Routing[/bold yellow]", border_style="yellow"))

    # ── Gesture info (if provided) ──
    if gesture_name and gesture_info:
        gesture_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        gesture_table.add_column("Key",   style="dim", width=16)
        gesture_table.add_column("Value", style="white")
        gesture_table.add_row("✋ Gesture",    gesture_name)
        gesture_table.add_row("🔧 Modifier",   gesture_info.get("modifier", "n/a"))
        gesture_table.add_row("🗣️  Standalone", gesture_info.get("standalone", "n/a"))
        console.print(Panel(gesture_table, title="[bold magenta]✋ Gesture Info[/bold magenta]", border_style="magenta"))

    console.rule("[dim]End of Result[/dim]")
    console.print()


def print_result_plain(
    voice_text: str,
    gesture_name: str,
    intent: FusedIntent,
    voice_confidence: float,
) -> None:
    """Fallback plain text output when Rich is not available."""
    agents = determine_agents(intent.command)
    risk, _ = determine_risk(intent.command, gesture_name)

    print("\n" + "=" * 50)
    print("  HELIOX OS — INTENT TESTER")
    print("=" * 50)
    print(f"\n📥 INPUT")
    print(f"   Voice      : {voice_text}")
    print(f"   Gesture    : {gesture_name or 'none'}")
    print(f"   Confidence : {voice_confidence * 100:.0f}%")
    print(f"\n🧠 FUSION RESULT")
    print(f"   Command    : {intent.command}")
    print(f"   Type       : {intent.fusion_type}")
    print(f"   Confidence : {intent.confidence * 100:.0f}%")
    print(f"\n🎯 ROUTING")
    print(f"   Agents     : {', '.join(agents)}")
    print(f"   Risk       : {risk}")
    print("\n" + "=" * 50 + "\n")


def print_result_json(
    voice_text: str,
    gesture_name: str,
    intent: FusedIntent,
    voice_confidence: float,
) -> None:
    """Print JSON output for CI/CD pipelines."""
    agents = determine_agents(intent.command)
    risk, _ = determine_risk(intent.command, gesture_name)
    output = {
        "input": {
            "voice"            : voice_text,
            "gesture"          : gesture_name or None,
            "voice_confidence" : voice_confidence,
        },
        "fusion": intent.to_dict(),
        "routing": {
            "agents"     : agents,
            "risk_level" : risk,
        },
    }
    print(json.dumps(output, indent=2))


async def test_single(args: argparse.Namespace) -> None:
    """Test a single voice (+ optional gesture) command."""
    if not FUSION_AVAILABLE:
        msg = f"Could not import Heliox OS modules: {IMPORT_ERROR}\n"
        msg += "Make sure you're running from the Heliox-OS root directory."
        if RICH_AVAILABLE:
            console.print(f"[bold red]❌ Import Error:[/bold red] {msg}")
        else:
            print(f"ERROR: {msg}")
        sys.exit(1)

    intent = await run_fusion(
        voice_text=args.command,
        gesture_name=args.gesture or "",
        voice_confidence=args.confidence,
        gesture_confidence=args.gesture_confidence,
    )

    if not intent:
        msg = "Fusion engine returned no intent. Check your input."
        if RICH_AVAILABLE:
            console.print(f"[bold red]❌ {msg}[/bold red]")
        else:
            print(f"ERROR: {msg}")
        sys.exit(1)

    if args.json:
        print_result_json(args.command, args.gesture or "", intent, args.confidence)
    elif RICH_AVAILABLE:
        print_result_rich(args.command, args.gesture or "", intent, args.confidence)
    else:
        print_result_plain(args.command, args.gesture or "", intent, args.confidence)


async def test_batch(args: argparse.Namespace) -> None:
    """Test multiple commands from a text file."""
    batch_file = Path(args.batch)
    if not batch_file.exists():
        print(f"ERROR: Batch file not found: {batch_file}")
        sys.exit(1)

    lines = [l.strip() for l in batch_file.read_text().splitlines() if l.strip() and not l.startswith("#")]

    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]🧪 Batch Testing {len(lines)} commands...[/bold cyan]\n")

    results = []
    for line in lines:
        # Support format: "voice command | gesture_name"
        parts = line.split("|")
        voice  = parts[0].strip()
        gesture = parts[1].strip() if len(parts) > 1 else ""

        intent = await run_fusion(voice_text=voice, gesture_name=gesture)
        if intent:
            agents = determine_agents(intent.command)
            risk, _ = determine_risk(intent.command)
            results.append({
                "voice"      : voice,
                "gesture"    : gesture or "none",
                "command"    : intent.command,
                "type"       : intent.fusion_type,
                "confidence" : f"{intent.confidence * 100:.0f}%",
                "agents"     : ", ".join(agents),
                "risk"       : risk,
            })

    if RICH_AVAILABLE and not args.json:
        table = Table(title="Batch Intent Test Results", box=box.ROUNDED, show_lines=True)
        table.add_column("Voice Input",   style="cyan",   max_width=25)
        table.add_column("Gesture",       style="magenta", max_width=12)
        table.add_column("Fused Command", style="green",  max_width=30)
        table.add_column("Type",          style="blue",   max_width=14)
        table.add_column("Conf",          style="yellow", max_width=6)
        table.add_column("Agents",        style="white",  max_width=20)
        table.add_column("Risk",          max_width=10)

        for r in results:
            risk_color = "red" if "High" in r["risk"] else "yellow" if "Medium" in r["risk"] else "green"
            table.add_row(
                r["voice"], r["gesture"], r["command"],
                r["type"], r["confidence"], r["agents"],
                f"[{risk_color}]{r['risk']}[/{risk_color}]"
            )
        console.print(table)
    elif args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\nVoice: {r['voice']} | Gesture: {r['gesture']}")
            print(f"  → {r['command']} ({r['type']}, {r['confidence']})")
            print(f"  Agents: {r['agents']} | Risk: {r['risk']}")


def list_gestures() -> None:
    """List all supported gestures."""
    if RICH_AVAILABLE:
        table = Table(title="Supported Gestures", box=box.ROUNDED, show_lines=True)
        table.add_column("Gesture Name",     style="cyan")
        table.add_column("Modifier Type",    style="magenta")
        table.add_column("Standalone Command", style="green")

        for gesture, modifier in GESTURE_MODIFIERS.items():
            standalone = GESTURE_STANDALONE_COMMANDS.get(gesture, "—")
            mod_str = modifier.value if hasattr(modifier, "value") else str(modifier)
            table.add_row(gesture, mod_str, standalone)

        console.print(table)
    else:
        print("\nSupported Gestures:")
        print("-" * 50)
        for gesture, modifier in GESTURE_MODIFIERS.items():
            standalone = GESTURE_STANDALONE_COMMANDS.get(gesture, "—")
            mod_str = modifier.value if hasattr(modifier, "value") else str(modifier)
            print(f"  {gesture:<25} {mod_str:<15} {standalone}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="intent_tester",
        description="🧪 Heliox OS — Offline Intent Tester CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/intent_tester.py "open chrome"
  python tools/intent_tester.py "open chrome" --gesture fist
  python tools/intent_tester.py "delete file" --gesture thumbs_up --confidence 0.9
  python tools/intent_tester.py "open chrome" --json
  python tools/intent_tester.py --batch tools/test_commands.txt
  python tools/intent_tester.py --list-gestures
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        help="Voice command to test (e.g. 'open chrome and search python')",
    )
    parser.add_argument(
        "--gesture", "-g",
        default="",
        help="Optional gesture name (e.g. fist, thumbs_up, point_up)",
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.95,
        help="Voice confidence score 0.0–1.0 (default: 0.95)",
    )
    parser.add_argument(
        "--gesture-confidence",
        type=float,
        default=0.85,
        help="Gesture confidence score 0.0–1.0 (default: 0.85)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON (useful for CI/CD pipelines)",
    )
    parser.add_argument(
        "--batch", "-b",
        metavar="FILE",
        help="Test multiple commands from a file (one per line, optionally: 'voice | gesture')",
    )
    parser.add_argument(
        "--list-gestures", "-l",
        action="store_true",
        help="List all supported gesture names and their modifiers",
    )

    args = parser.parse_args()

    # ── Route to correct function ──
    if args.list_gestures:
        list_gestures()
        return

    if args.batch:
        asyncio.run(test_batch(args))
        return

    if not args.command:
        parser.print_help()
        sys.exit(0)

    asyncio.run(test_single(args))


if __name__ == "__main__":
    main()
