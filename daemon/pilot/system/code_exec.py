"""Code Generation & Dynamic Execution — run arbitrary code safely.

Generate and execute Python/PowerShell/Bash scripts on the fly.
Supports sandboxed execution, output capture, and timeout control.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

logger = logging.getLogger("pilot.system.code_exec")


async def execute_python(
    code: str,
    timeout: int = 30,
    capture_output: bool = True,
) -> str:
    """Execute Python code and return the output.

    The code runs in a subprocess for isolation.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE if capture_output else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"ERROR: Script timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            err = stderr.decode("utf-8", errors="replace")
            if err.strip():
                output += f"\n[STDERR]\n{err}"
        if proc.returncode != 0:
            output += f"\n[EXIT CODE: {proc.returncode}]"

        return output.strip() or "(no output)"
    finally:
        os.unlink(script_path)


async def execute_powershell(
    code: str,
    timeout: int = 30,
) -> str:
    """Execute PowerShell code and return the output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ps1", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    try:
        shell = "pwsh" if os.path.exists("C:/Program Files/PowerShell") else "powershell"
        proc = await asyncio.create_subprocess_exec(
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"ERROR: Script timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            err = stderr.decode("utf-8", errors="replace")
            if err.strip():
                output += f"\n[STDERR]\n{err}"

        return output.strip() or "(no output)"
    finally:
        os.unlink(script_path)


async def execute_bash(
    code: str,
    timeout: int = 30,
) -> str:
    """Execute Bash code and return the output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as f:
        f.write("#!/bin/bash\nset -e\n" + code)
        script_path = f.name

    try:
        os.chmod(script_path, 0o755)
        proc = await asyncio.create_subprocess_exec(
            "bash",
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"ERROR: Script timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            err = stderr.decode("utf-8", errors="replace")
            if err.strip():
                output += f"\n[STDERR]\n{err}"

        return output.strip() or "(no output)"
    finally:
        os.unlink(script_path)


async def execute_code(
    code: str,
    language: str = "python",
    timeout: int = 30,
) -> str:
    """Execute code in the specified language.

    language: 'python', 'powershell', 'bash', 'cmd', 'javascript'
    """
    language = language.lower().strip()

    if language in ("python", "py", "python3"):
        return await execute_python(code, timeout)
    elif language in ("powershell", "ps1", "pwsh"):
        return await execute_powershell(code, timeout)
    elif language in ("bash", "sh", "shell"):
        return await execute_bash(code, timeout)
    elif language in ("cmd", "batch", "bat"):
        return await _execute_cmd(code, timeout)
    elif language in ("javascript", "js", "node"):
        return await _execute_node(code, timeout)
    else:
        return f"Unsupported language: {language}. Use: python, powershell, bash, cmd, javascript"


async def _execute_cmd(code: str, timeout: int = 30) -> str:
    """Execute Windows CMD batch script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cmd", delete=False, encoding="utf-8") as f:
        f.write("@echo off\n" + code)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "cmd",
            "/c",
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"ERROR: Script timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += stderr.decode("utf-8", errors="replace")
        return output.strip() or "(no output)"
    finally:
        os.unlink(script_path)


async def _execute_node(code: str, timeout: int = 30) -> str:
    """Execute JavaScript via Node.js."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "node",
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"ERROR: Script timed out after {timeout}s"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += stderr.decode("utf-8", errors="replace")
        return output.strip() or "(no output)"
    finally:
        os.unlink(script_path)


async def generate_and_execute(
    task_description: str,
    language: str = "python",
    timeout: int = 30,
) -> str:
    """Generate code from a task description using the LLM, then execute it.

    This is the most powerful action — the agent writes custom code for any task.
    """
    # Import here to avoid circular dependency
    try:
        from pilot.models.ollama import OllamaClient

        client = OllamaClient()
        if not await client.is_available():
            return "ERROR: Ollama not available for code generation"

        models = await client.list_models()
        if not models:
            return "ERROR: No models available"

        model = models[0]  # Use whatever's available

        prompt = (
            f"Write a {language} script that accomplishes the following task:\n"
            f"{task_description}\n\n"
            f"RULES:\n"
            f"- Output ONLY the code, no explanations\n"
            f"- No markdown code fences\n"
            f"- The script should print its results to stdout\n"
            f"- Handle errors gracefully\n"
            f"- Keep it concise and efficient"
        )

        code = await client.generate(
            model,
            prompt,
            system=f"You are a {language} code generator. Output ONLY executable code.",
            temperature=0.1,
        )

        # Clean up — remove markdown fences if the model added them
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:])  # Remove first line
        if code.endswith("```"):
            code = code[:-3].rstrip()

        result = await execute_code(code, language, timeout)
        return f"[GENERATED CODE]\n{code}\n\n[OUTPUT]\n{result}"

    except Exception as e:
        return f"Code generation failed: {e}"
