#!/usr/bin/env python3
"""Pilot Local CLI — Terminal UI for the Pilot daemon.

Connects to the background websocket server and provides a rich
interactive chat interface with streaming updates.
"""

import asyncio
import contextlib
import json
import os
import sys

import websockets
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()

# Styling
PT_STYLE = PtStyle.from_dict(
    {
        "prompt": "ansiwhite bold",
        "pilot": "ansiyellow bold",
    }
)

BANNER = """[bold cyan]
  ██████╗ ██╗██╗      ██████╗ ████████╗
  ██╔══██╗██║██║     ██╔═══██╗╚══██╔══╝
  ██████╔╝██║██║     ██║   ██║   ██║
  ██╔═══╝ ██║██║     ██║   ██║   ██║
  ██║     ██║███████╗╚██████╔╝   ██║
  ╚═╝     ╚═╝╚══════╝ ╚═════╝    ╚═╝
[/bold cyan][dim]  Autonomous AI Computer Agent v0.3.0[/dim]
"""


class PilotCLI:
    def __init__(self, uri: str = "ws://127.0.0.1:8765"):
        self.uri = uri
        self.ws = None
        self.session = PromptSession(style=PT_STYLE)
        self.request_id = 1
        self.connected = False

    async def connect(self):
        try:
            self.ws = await websockets.connect(self.uri)
            self.connected = True
            console.print("[green]Connected to Pilot daemon.[/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to connect to daemon:[/bold red] {e}")
            console.print("Make sure 'python -m pilot.server' is running in another terminal.")
            sys.exit(1)

    async def send_request(self, method: str, params: dict = None):
        if not self.ws:
            return

        req = {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": self.request_id}
        self.request_id += 1
        await self.ws.send(json.dumps(req))

    def format_plan_preview(self, payload: dict) -> Panel:
        """Format the action plan for user approval."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Type", width=20)
        table.add_column("Target", width=30)
        table.add_column("Risk")

        for action in payload.get("actions", []):
            risk = (
                "[red]High[/red]"
                if action.get("destructive")
                else "[yellow]Elevated[/yellow]"
                if action.get("requires_root")
                else "[green]Low[/green]"
            )
            table.add_row(action.get("action_type", "unknown"), str(action.get("target", ""))[:30], risk)

        text = f"[bold]Explanation:[/bold] {payload.get('explanation', '')}\n\n"

        from rich.console import Group

        group = Group(text, table)
        return Panel(group, title="[bold yellow]Action Plan Generated[/bold yellow]", border_style="yellow")

    def format_multimodal_nodes(self, nodes: list) -> list:
        """Render multimodal elements to the console."""
        rendered = []
        for n in nodes:
            t = n.get("type", "text")
            c = n.get("content", "")

            if t == "text":
                rendered.append(Markdown(c))
            elif t == "code":
                lang = c.get("language", "")
                code = c.get("code", "")
                rendered.append(Markdown(f"```{lang}\n{code}\n```"))
            elif t == "image":
                rendered.append(Text(f"[Image saved to: {c.get('path', 'unknown')}]", style="dim italic"))
            elif t == "table":
                table = Table(show_lines=True)
                for col in c.get("columns", []):
                    table.add_column(str(col))
                for row in c.get("rows", []):
                    table.add_row(*[str(cell) for cell in row])
                rendered.append(table)
            elif t == "file_tree":
                paths = c.get("paths", [])
                rendered.append(Text("\n".join(paths), style="cyan"))
            elif t == "error":
                rendered.append(Panel(c, border_style="red", title="Error"))
            else:
                rendered.append(Text(str(c)))

        return rendered

    async def handle_messages(self, live: Live):
        """Listen for websocket messages and render them."""
        try:
            async for message in self.ws:
                data = json.loads(message)

                # Is it a notification?
                if "method" in data:
                    method = data["method"]
                    params = data.get("params", {})

                    if method == "status":
                        phase = params.get("phase", "thinking")
                        live.update(Spinner("dots", text=f"[cyan]Pilot is {phase}...[/cyan]"))

                    elif method == "plan_preview":
                        live.stop()
                        panel = self.format_plan_preview(params)
                        console.print(panel)
                        live.start()

                    elif method == "confirm_required":
                        live.stop()
                        plan_id = params.get("plan_id")
                        console.print("\n[bold red]⚠️  WARNING: PLAN CONTAINS DESTRUCTIVE/ROOT ACTIONS ⚠️[/bold red]")
                        answer = await self.session.prompt_async(HTML("<ansiyellow>Proceed? [Y/n]: </ansiyellow>"))

                        await self.send_request(
                            "confirm", {"plan_id": plan_id, "approved": answer.lower() in ("y", "yes", "")}
                        )
                        live.start()

                # Is it a response?
                elif "id" in data:
                    result = data.get("result", {})
                    error = data.get("error", None)

                    live.stop()

                    if error:
                        console.print(f"[bold red]Error [{error.get('code')}]:[/bold red] {error.get('message')}")
                    else:
                        status = result.get("status")
                        if status == "success":
                            msg = "[bold green]✓ Execution completed successfully.[/bold green]"
                        elif status == "partial_failure":
                            msg = "[bold yellow]⚠ Execution completed with errors.[/bold yellow]"
                        elif status == "cancelled":
                            msg = "[bold red]⨯ Execution cancelled.[/bold red]"
                            console.print(msg)
                            return
                        else:
                            msg = f"[bold blue]ℹ Status: {status}[/bold blue]"

                        console.print(msg)

                        # Show multimodal results if present
                        for action_result in result.get("results", []):
                            if "nodes" in action_result:
                                # We have rich multimodal nodes!
                                items = self.format_multimodal_nodes(action_result["nodes"])
                                for item in items:
                                    console.print(item)
                            else:
                                # Fallback to plain text
                                output = action_result.get("output", "")
                                if output:
                                    # Output code block
                                    console.print(
                                        Panel(
                                            output,
                                            title=f"Result: {action_result.get('action_type', 'action')}",
                                            border_style="green" if action_result.get("success") else "red",
                                        )
                                    )

                    return  # Exit the handle_messages loop, wait for next prompt

        except websockets.exceptions.ConnectionClosed:
            live.stop()
            console.print("[bold red]Connection to daemon lost.[/bold red]")
            self.connected = False
            sys.exit(1)

    async def run(self):
        console.print(BANNER)
        await self.connect()

        while self.connected:
            try:
                # Get user input
                user_input = await self.session.prompt_async(HTML("<ansiblue>pilot></ansiblue> "))

                if not user_input.strip():
                    continue
                if user_input.strip() in ("exit", "quit"):
                    break
                if user_input.strip() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue

                # Start spinner and send request
                with Live(Spinner("dots", text="[cyan]Thinking...[/cyan]"), refresh_per_second=10) as live:
                    await self.send_request("execute", {"input": user_input})
                    await self.handle_messages(live)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

        if self.ws:
            await self.ws.close()
        console.print("Goodbye!")


if __name__ == "__main__":
    cli = PilotCLI()
    # Windows asyncio bug workaround
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(cli.run())
