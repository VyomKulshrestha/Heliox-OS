"""Planner agent — converts natural language requests into structured ActionPlans.

The Planner has ZERO imports from pilot.system.* — it cannot execute anything.
It only produces Action objects that must pass through validation before reaching the Executor.

Enhanced with 50+ action types for full system control across Windows, Linux, and macOS.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from typing import TYPE_CHECKING

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    ApiRequestParams,
    BrightnessParams,
    BrowserParams,
    ClipboardParams,
    CodeExecParams,
    DBusParams,
    DiskManageParams,
    DownloadParams,
    EmptyParams,
    EnvParams,
    FileIntelParams,
    FileParams,
    GnomeSettingParams,
    KeyboardParams,
    MouseParams,
    NotifyParams,
    OpenApplicationParams,
    OpenUrlParams,
    PackageParams,
    PowerParams,
    ProcessParams,
    RegistryParams,
    ScheduleParams,
    ScreenshotParams,
    ScreenVisionParams,
    ServiceParams,
    ShellCommandParams,
    ShellScriptParams,
    SystemInfoParams,
    TriggerParams,
    VolumeParams,
    WifiParams,
    WindowParams,
)

if TYPE_CHECKING:
    from pilot.memory.store import MemoryStore
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.planner")


def _detect_os() -> str:
    if sys.platform == "win32":
        return "Windows"
    elif sys.platform == "darwin":
        return "macOS"
    return "Linux"


SYSTEM_PROMPT = """\
You are Heliox OS, a hyper-advanced, omnipotent AI operating system that has ABSOLUTE CONTROL over the user's {os} computer.
You are a real code-wiz and system administrator: few are as talented as you at understanding codebases, writing functional and clean code, and iterating until correct.
You receive natural language requests and output ONLY a JSON action plan.
Think step by step about what the user wants, then produce ALL the actions needed to fulfill the request end-to-end.

You are NOT just a chatbot — you are a God-tier autonomous Operating System that natively executes and administrates deep system commands.
When the user asks you to do something, you DO IT by generating the proper action plan without asking for unnecessary permissions unless specifically destructive.
You are fully capable of managing files, navigating the web, writing and running code, tweaking system settings, taking screenshots, and monitoring processes in real-time. Do not hold back your capabilities!

=== APPROACH TO WORK & BEST PRACTICES ===
- Fulfill the user's request using all the tools available to you.
- You are part of an Agentic Loop: You evaluate memory, determine tool calls, execute them, and feed results back into your reasoning loop.
- When encountering difficulties, DO NOT guess or hallucinate. Take time to gather information using search/read/shell commands before concluding a root cause.
- NEVER assume that a given library, CLI tool, or file exists without verifying. Use shell commands or file searches to check the state of the system first.
- When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries, and follow existing patterns.
- If you cannot find some information, use your browser automation tools to find missing context from online sources.
- Multi-step pipelines are your superpower. Chain actions in order. Each action's output is available to the next.
- When running code, ALWAYS add necessary logging and print statements so you can see what happened!

=== AVAILABLE ACTION TYPES ===

# === TIER 1: GAME CHANGERS ===

MOUSE & KEYBOARD CONTROL (Desktop Automation):
- mouse_click, mouse_double_click, mouse_right_click — Params: {{"x": null, "y": null}}
- mouse_move, mouse_drag — Params: {{"x": 100, "y": 100}}
- mouse_scroll — Params: {{"clicks": 10}}
- mouse_position — Params: {{}}
- keyboard_type — Params: {{"text": "hello", "press_enter": false}}
- keyboard_press, keyboard_hotkey — Params: {{"key": "enter"}} / {{"hotkey": "ctrl+c"}}
- keyboard_hold — Params: {{"key": "shift", "duration": 1.0}}

SCREEN UNDERSTANDING (Vision & OCR):
- screen_ocr — Read text from screen. Params: {{"region": null, "language": "eng"}}
- screen_find_text — Find text coordinates. Params: {{"text": "Save"}}
- screen_analyze — Ask Vision LLM about screen. Params: {{"prompt": "What app is open?"}}
- screen_element_map — Map interactive elements. Params: {{}}

BROWSER AUTOMATION (Headless, for reading/interacting):
- browser_navigate, browser_close — Params: {{"url": "https://..."}} / {{}} -- MUST use before browser_extract
- browser_click, browser_click_text — Params: {{"selector": "#submit"}} / {{"text": "Login"}}
- browser_type — Params: {{"selector": "#q", "text": "search query"}}
- browser_select, browser_hover, browser_scroll — Params: {{"selector": "...", "value": "..."}}
- browser_extract, browser_extract_table, browser_extract_links — Params: {{"selector": "h1"}} -- Extracts text into PREV_OUTPUT
- browser_execute_js — Params: {{"script": "return document.title"}}
- browser_screenshot — Params: {{"output_path": "path.png"}}
- browser_new_tab, browser_close_tab, browser_switch_tab — Params: {{"url": "..."}} / {{"index": 0}}
- browser_back, browser_forward, browser_refresh, browser_wait — Params: {{"timeout": 5000}}

REACTIVE TRIGGERS:
- trigger_create — Params: {{"name": "alert", "trigger_type": "cpu_threshold", "condition": {{"threshold":90}}, "action_command": "notify high CPU"}}
- trigger_list, trigger_delete — Params: {{}} / {{"name": "alert"}}
- trigger_start, trigger_stop — Params: {{"name": "alert"}}

# === TIER 2: MASSIVE MULTIPLIERS ===

CODE EXECUTION:
- code_execute — Params: {{"code": "print('hello')", "language": "python"}}
- code_generate_and_run — Params: {{"prompt": "write a python script to rename files"}}

FILE CONTENT INTELLIGENCE & MEMORY:
- file_parse — Read/Parse PDFs, DOCX, XLSX, images. Params: {{"path": "/path/to/file.pdf"}}
- file_search_content — Search inside files. Params: {{"path": "/dir", "query": "secret", "pattern": "*.txt", "is_regex": false}}
- memory_store — Save context or long-term memory for yourself across sessions. Params: {{"key": "user_preferences", "value": "..."}}
- memory_retrieve — Get saved core context. Params: {{"key": "user_preferences"}}

API INTEGRATION, MESSAGING & WEBHOOKS (OpenClaw-style Hub):
- api_request — Params: {{"method": "GET", "url": "https://api.example.com", "headers": {{}}, "body": null}}
- api_github — Params: {{"endpoint": "/user/repos", "method": "GET", "body": null}}
- api_send_email — Params: {{"to": "user@example.com", "subject": "Hi", "body": "Hello"}}
- api_webhook, api_slack, api_discord — Params: {{"url": "...", "payload": {{"text": "Hello"}}}}
- api_whatsapp — Params: {{"phone": "+1234567890", "message": "Your agent update."}}
- api_scrape — Web scraping without browser. Params: {{"url": "https://...", "selector": "h1", "extract": "text"}}

# === STANDARD SYSTEM CONTROLS ===

FILE OPERATIONS:
... (all previously known standard operations)
- file_read, file_write, file_delete, file_move, file_copy, file_list, file_search, file_permissions

SYSTEM ADMINISTRATION / PACKAGE / SERVICE / PROCESS / POWER:
... (all standard commands apply)
- You have full access to shell_command, shell_script, system_info, registry_read, dbus_call, etc.

ENVIRONMENT VARIABLES:
- env_get Params: {{"name": "PATH"}}
- env_set Params: {{"name": "MY_VAR", "value": "hello", "persistent": false}}
- env_list Params: {{"name": "optional-prefix-filter"}}

OPEN URL / APP / NOTIFY:
- open_url Params: {{"url": "https://example.com"}} -- ONLY opens in user's external browser window. CANNOT READ TEXT.
- open_application Params: {{"name": "app-name", "args": []}}
- notify Params: {{"summary": "Title", "body": "message"}}

DOWNLOAD:
- download_file Params: {{"url": "https://example.com/file.zip", "output_path": "/path/to/save", "overwrite": false}}

DISK MANAGEMENT:
- disk_list Params: {{}}
- disk_mount Params: {{"device": "/dev/sda1", "mount_point": "/mnt/data"}}
- disk_unmount Params: {{"mount_point": "/mnt/data"}}

USER INFORMATION:
- user_list Params: {{}}
- user_info Params: {{}}

GNOME SETTINGS (Linux only):
- gnome_setting_read Params: {{"schema": "org.gnome.desktop.interface", "key": "color-scheme"}}
- gnome_setting_write Params: {{"schema": "...", "key": "...", "value": "..."}}

DBUS (Linux only):
- dbus_call Params: {{"bus": "session", "service": "...", "object_path": "/...", "interface": "...", "method": "...", "args": []}}

WINDOWS REGISTRY (Windows only):
- registry_read Params: {{"key_path": "HKCU\\\\Software\\\\...", "value_name": ""}}
- registry_write Params: {{"key_path": "...", "value_name": "...", "value_data": "...", "value_type": "REG_SZ"}}

=== OUTPUT FORMAT ===

Output STRICT JSON (no markdown, no explanation outside JSON):
{{
  "explanation": "Human-readable summary of what will be done",
  "actions": [
    {{
      "action_type": "<type>",
      "target": "<primary resource — file path, URL, app name, etc>",
      "parameters": {{ <type-specific params> }},
      "requires_root": false,
      "destructive": false,
      "reversible": true,
      "rollback_action": null,
      "use_previous_output": false
    }}
  ]
}}

=== RULES ===

1. "target" must ALWAYS be a string (never null). Use the primary resource name.
2. For file operations on {os}: use {path_style} paths.
3. You can use MULTIPLE actions to accomplish complex tasks.
4. Set requires_root=true for operations needing sudo/admin privileges.
5. Set destructive=true for delete/remove operations.
6. Output ONLY valid JSON — no explanation text outside the JSON.
7. If the user's request is ambiguous, pick the most reasonable interpretation.
8. For app-related requests (open Spotify), use open_application.
9. Use shell_script for complex multi-step shell tasks.
10. OUTPUT CHAINING BETWEEN ACTIONS:
    - For file_write: set "use_previous_output": true and "content": "{{PREV_OUTPUT}}" to write the previous action's output to a file.
    - For code_execute (Python): The variable PREV_OUTPUT is automatically available as a Python string containing the previous action's output. Just use it directly: e.g. print(PREV_OUTPUT.count('word')). NEVER assign or define PREV_OUTPUT yourself.
12. For queries about the system (how much RAM, disk space, etc), use the system_info actions.
13. ALWAYS USE THE MOST SPECIFIC TOOL: E.g., for parsing a PDF, use file_parse, DO NOT write a python script. For web scraping, use api_scrape or browser_extract.
14. YOU CAN CLICK AND TYPE: If an app has no CLI/API, use mouse_click, keyboard_type, and screen_ocr to use its GUI manually just like a human would!
15. REASON IN THE APPS: You can write Python code (code_execute) to process data if needed.
16. USE BACKGROUND TRIGGERS: If asked to "watch" or "alert" when something happens, use trigger_create.
17. HOME DIRECTORY: Use "{home}" as the user's home directory.
18. BE PROACTIVE: If the user says "clean my desktop", generate ALL the actions needed.
19. MULTI-STEP PIPELINES: Chain actions in order. Each action's output is available to the next.
20. You are POWERFUL — you can do ANYTHING the user asks. You have TIER 1 desktop automation!
21. When writing Python code inside JSON, escape double quotes and newlines. Prefer single quotes in Python code.
22. For browser_extract: ALWAYS use selector "p" with "multiple": true to extract article text. NEVER use "#content" or "body" — those return the ENTIRE page including navigation and menus. The "p" selector gives clean paragraphs.
23. CRITICAL PIPELINE — "extract from page, save to file, process":
    Step 1: browser_navigate to the URL
    Step 2: browser_extract with selector "p", multiple=true (gets clean paragraphs)
    Step 3: file_write with use_previous_output=true, content="{{PREV_OUTPUT}}"
    Step 4: code_execute with Python. Variables PREV_OUTPUT, PARAGRAPHS, FIRST_3_PARAGRAPHS, FIRST_3_TEXT are auto-available. Use FIRST_3_PARAGRAPHS for "first 3 paragraphs" tasks.
    NEVER add extra steps between browser_extract and file_write.
24. NEVER set destructive=true unless the action deletes data. file_write is NOT destructive.
25. ALWAYS use print() for EVERY result in code_execute. If you don't print, the user sees NOTHING.
26. browser_navigate and open_url DO NOT return page text. You MUST follow them immediately with browser_extract to read the text.
27. EXAMPLE PLAN for "extract first 3 paragraphs from Wikipedia, save to file, count word 'human'":
    {{
      "explanation": "Navigate to Wikipedia, extract paragraphs, save to file, and run python script.",
      "actions": [
        {{"action_type": "browser_navigate", "target": "https://en.wikipedia.org/wiki/Topic", "parameters": {{"url": "https://en.wikipedia.org/wiki/Topic"}}}},
        {{"action_type": "browser_extract", "target": "p", "parameters": {{"selector": "p", "multiple": true}}}},
        {{"action_type": "file_write", "target": "C:\\\\Users\\\\user\\\\Desktop\\\\file.txt", "parameters": {{"path": "C:\\\\Users\\\\user\\\\Desktop\\\\file.txt", "content": "{{PREV_OUTPUT}}"}}, "use_previous_output": true}},
        {{"action_type": "code_execute", "target": "", "parameters": {{"language": "python", "code": "print(f'Human count: {{FIRST_3_TEXT.lower().count(\\'human\\')}}')"}}, "use_previous_output": true}}
      ]
    }}
"""

USER_CONTEXT_TEMPLATE = """\
User preferences and history:
{context}

User request: {request}
"""

RETRY_TEMPLATE = """\
The previous plan FAILED during execution. Here is the error:

{error}

Original request: {request}

IMPORTANT RULES FOR RETRY:
- This is a {os} system. Use {path_style} paths ONLY.
- Home directory is "{home}".
- Desktop is "{home}\\Desktop" (Windows) or "{home}/Desktop" (Linux/Mac).
- NEVER use /mnt/ or /tmp/ or /home/ paths on Windows.
- NEVER use /mnt/prev_output/ — that path does NOT exist.
- DO NOT add mouse_click or keyboard_press actions. They are NOT needed for file/code tasks.
- DO NOT add more than 5 actions. Keep the plan simple and focused.
- If screen_ocr failed, just try screen_ocr again — the OCR engine may need a moment to initialize.
- If browser_extract was missing, add it after browser_navigate.
- ALWAYS use browser_navigate + browser_extract to read web page content. open_url CANNOT read text.
- For Python code: use raw strings r'...' or forward slashes for Windows paths.
- For shell commands: put ONLY the command name in "command", put arguments in "args" list.

Generate a NEW plan that avoids this error. Keep it SIMPLE — fewer actions is better.
"""


class Planner:
    """Converts natural language to structured action plans."""

    def __init__(self, model_router: ModelRouter, memory: MemoryStore) -> None:
        self._model = model_router
        self._memory = memory
        self._system_prompt = SYSTEM_PROMPT.format(
            os=_detect_os(),
            path_style="Windows (C:\\Users\\...)" if sys.platform == "win32" else "Unix (/home/...)",
            home=str(__import__("pathlib").Path.home()),
        )

    async def plan(self, user_input: str, error_context: str = "") -> ActionPlan:
        """Generate an action plan from a natural language request."""
        try:
            context = await self._memory.get_context(user_input)

            if error_context:
                prompt = RETRY_TEMPLATE.format(
                    error=error_context,
                    request=user_input,
                    os=_detect_os(),
                    path_style="Windows (C:\\Users\\...)" if sys.platform == "win32" else "Unix (/home/...)",
                    home=str(__import__("pathlib").Path.home()),
                )
            else:
                prompt = USER_CONTEXT_TEMPLATE.format(
                    context=context or "No prior context.",
                    request=user_input,
                )

            raw_response = await self._model.generate(
                prompt, system=self._system_prompt, json_mode=True, temperature=0.1
            )

            return self._parse_response(raw_response, user_input)

        except Exception as e:
            logger.exception("Planning failed for input: %s", user_input[:100])
            return ActionPlan(error=f"Planning failed: {e}", raw_input=user_input)

    def _parse_response(self, raw: str, user_input: str) -> ActionPlan:
        """Parse LLM JSON output into a validated ActionPlan."""
        clean_raw = raw.strip()
        if clean_raw.startswith("```json"):
            clean_raw = clean_raw.split("```json", 1)[1]
        elif clean_raw.startswith("```"):
            clean_raw = clean_raw.split("```", 1)[1]

        if clean_raw.endswith("```"):
            clean_raw = clean_raw.rsplit("```", 1)[0]

        clean_raw = clean_raw.strip()

        # Debug: log raw LLM response
        logger.debug("Raw LLM response: %s", clean_raw[:500])

        try:
            data = json.loads(clean_raw)
        except json.JSONDecodeError as e:
            return ActionPlan(
                error=f"LLM returned invalid JSON: {e}\\n\\nRaw Response:\\n{clean_raw[:500]}", raw_input=user_input
            )

        if not isinstance(data, dict):
            return ActionPlan(error="LLM response is not a JSON object", raw_input=user_input)

        explanation = data.get("explanation", "")
        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list) or not raw_actions:
            return ActionPlan(error="No actions in plan", explanation=explanation, raw_input=user_input)

        actions: list[Action] = []
        for i, raw_action in enumerate(raw_actions):
            try:
                action = self._parse_action(raw_action)
                actions.append(action)
            except Exception as e:
                logger.warning("Failed to parse action %d: %s", i, e)
                # Skip unparseable actions instead of aborting entire plan
                continue

        if not actions:
            return ActionPlan(
                error="All actions failed to parse",
                explanation=explanation,
                raw_input=user_input,
            )

        # Post-process: fix common LLM structural errors
        actions = self._postprocess_actions(actions)

        return ActionPlan(actions=actions, explanation=explanation, raw_input=user_input)

    # ------------------------------------------------------------------
    # Plan post-processor: fix common LLM structural mistakes
    # ------------------------------------------------------------------

    # Actions that consume previous output (need data in the pipeline)
    _DATA_CONSUMERS = {
        ActionType.FILE_WRITE,
        ActionType.CODE_EXECUTE,
        ActionType.CODE_GENERATE_AND_RUN,
    }

    def _postprocess_actions(self, actions: list[Action]) -> list[Action]:
        """Fix common LLM plan structure errors before execution.

        Fixes applied:
        0. Remove garbage actions (mouse_click spam, keyboard_press spam, fake paths)
        1. open_url → browser_navigate when followed by data-consuming steps
        2. Auto-insert browser_extract after browser_navigate when missing
        3. Fix Windows backslash paths in code_execute Python code
        3b. Remove useless file_read that disrupts data pipeline
        4. screen_analyze → screen_ocr conversion
        5. Linux → Windows path conversion
        6. Shell command splitting
        """
        import re

        fixed = list(actions)

        # --- Fix 0: Remove garbage actions from hallucinated retry plans ---
        # LLM sometimes panics during retries and generates plans with:
        # - Random mouse_click / keyboard_press spam
        # - Actions referencing /mnt/prev_output/ (fake Linux path)
        # - Excessive padding actions that serve no purpose
        _GARBAGE_TYPES = {
            ActionType.MOUSE_CLICK,
            ActionType.MOUSE_MOVE,
            ActionType.MOUSE_SCROLL,
            ActionType.KEYBOARD_PRESS,
            ActionType.KEYBOARD_HOLD,
        }
        # Count how many garbage vs real actions
        real_actions = [a for a in fixed if a.action_type not in _GARBAGE_TYPES]
        garbage_count = len(fixed) - len(real_actions)
        if garbage_count > 2 and real_actions:
            # More than 2 garbage actions = hallucinated retry plan. Strip them.
            logger.info("Post-process: removing %d garbage actions (mouse/keyboard spam)", garbage_count)
            fixed = real_actions

        # Remove actions referencing /mnt/prev_output/ (a hallucinated path)
        fixed = [
            a
            for a in fixed
            if not (
                (hasattr(a.parameters, "path") and "/mnt/prev_output" in (getattr(a.parameters, "path", "") or ""))
                or (
                    hasattr(a.parameters, "output_path")
                    and "/mnt/prev_output" in (getattr(a.parameters, "output_path", "") or "")
                )
                or (a.target and "/mnt/prev_output" in a.target)
            )
        ]

        # --- Fix 1: Convert open_url → browser_navigate when pipeline needs data ---
        for i, action in enumerate(fixed):
            if action.action_type == ActionType.OPEN_URL:
                # Check if any subsequent action wants previous output
                has_consumer = any(
                    a.action_type in self._DATA_CONSUMERS or a.use_previous_output for a in fixed[i + 1 :]
                )
                if has_consumer:
                    url = ""
                    if hasattr(action.parameters, "url"):
                        url = action.parameters.url
                    elif action.target:
                        url = action.target
                    if url and not url.startswith(("http://", "https://", "file://")):
                        url = "https://" + url
                    logger.info("Post-process: converting open_url → browser_navigate for %s", url)
                    fixed[i] = Action(
                        action_type=ActionType.BROWSER_NAVIGATE,
                        target=url,
                        parameters=BrowserParams(url=url),
                        requires_root=False,
                        destructive=False,
                        reversible=True,
                        use_previous_output=False,
                    )

        # --- Fix 2: Auto-insert browser_extract after browser_navigate if missing ---
        i = 0
        while i < len(fixed):
            action = fixed[i]
            if action.action_type == ActionType.BROWSER_NAVIGATE:
                # Check if next action is already browser_extract
                next_is_extract = i + 1 < len(fixed) and fixed[i + 1].action_type == ActionType.BROWSER_EXTRACT
                if not next_is_extract:
                    # Check if any later action needs data
                    has_consumer = any(
                        a.action_type in self._DATA_CONSUMERS or a.use_previous_output for a in fixed[i + 1 :]
                    )
                    if has_consumer:
                        logger.info("Post-process: inserting browser_extract after browser_navigate at index %d", i)
                        extract_action = Action(
                            action_type=ActionType.BROWSER_EXTRACT,
                            target="p",
                            parameters=BrowserParams(selector="p", multiple=True),
                            requires_root=False,
                            destructive=False,
                            reversible=True,
                            use_previous_output=False,
                        )
                        fixed.insert(i + 1, extract_action)
                        i += 1  # skip past the newly inserted action
            i += 1

        # --- Fix 3: Fix Windows paths in Python code ---
        if sys.platform == "win32":
            for action in fixed:
                if action.action_type == ActionType.CODE_EXECUTE and hasattr(action.parameters, "code"):
                    code = action.parameters.code or ""

                    # Replace unescaped Windows paths in single/double quoted strings
                    # e.g. open('C:\Users\user\...') → open(r'C:\Users\user\...')
                    # Pattern: quote, drive letter, colon, backslash (without preceding 'r')
                    def _fix_path_string(m):
                        prefix = m.group(1)  # anything before the quote
                        quote = m.group(2)  # the quote character
                        path = m.group(3)  # the path content
                        # Already a raw string?
                        if prefix.endswith("r") or prefix.endswith("R"):
                            return m.group(0)
                        return prefix + "r" + quote + path + quote

                    code = re.sub(
                        r"""(^|[^rR])(["'])([A-Za-z]:\\[^"']+)\2""",
                        _fix_path_string,
                        code,
                    )
                    action.parameters.code = code

        # --- Fix 3b: Remove useless file_read that disrupts data pipeline ---
        # LLM sometimes inserts file_read on non-existent paths:
        # a) between screen_ocr/browser_extract and code_execute/file_write
        # b) at the START of a plan that has a data-producer later
        # c) at the START of a plan that creates files in that path (file_write/shell_command)
        _DATA_PRODUCERS = {
            ActionType.SCREEN_OCR,
            ActionType.SCREEN_ANALYZE,
            ActionType.BROWSER_EXTRACT,
            ActionType.BROWSER_EXTRACT_TABLE,
            ActionType.BROWSER_EXTRACT_LINKS,
        }
        _FILE_CREATORS = {
            ActionType.FILE_WRITE,
            ActionType.FILE_COPY,
            ActionType.FILE_MOVE,
        }
        has_data_producer = any(a.action_type in _DATA_PRODUCERS for a in fixed)
        i = 0
        while i < len(fixed):
            if fixed[i].action_type == ActionType.FILE_READ:
                # Case a: Between producer and consumer
                is_between = (
                    i > 0
                    and fixed[i - 1].action_type in _DATA_PRODUCERS
                    and i + 1 < len(fixed)
                    and fixed[i + 1].action_type in self._DATA_CONSUMERS
                )
                # Case b: At start of plan, data producer is later
                is_premature = (
                    has_data_producer
                    and not any(fixed[j].action_type in _DATA_PRODUCERS for j in range(i))
                    and any(fixed[j].action_type in _DATA_PRODUCERS for j in range(i + 1, len(fixed)))
                )
                # Case c: file_read on a directory path that will be created by later file_write
                read_path = getattr(fixed[i].parameters, "path", "") or fixed[i].target or ""
                is_dir_check = i == 0 and any(
                    a.action_type in _FILE_CREATORS and (getattr(a.parameters, "path", "") or "").startswith(read_path)
                    for a in fixed[i + 1 :]
                )
                if is_between or is_premature or is_dir_check:
                    logger.info(
                        "Post-process: removing useless file_read at index %d (pattern: %s)",
                        i,
                        "between" if is_between else "premature" if is_premature else "dir-check",
                    )
                    fixed.pop(i)
                    continue
            i += 1

        # --- Fix 4: Auto-convert screen_analyze → screen_ocr when no vision model ---
        # screen_analyze requires a vision LLM (like llava). screen_ocr works with
        # Windows native OCR. If the plan uses screen_analyze, convert it.
        for i, action in enumerate(fixed):
            if action.action_type == ActionType.SCREEN_ANALYZE:
                logger.info("Post-process: converting screen_analyze → screen_ocr at index %d", i)
                fixed[i] = Action(
                    action_type=ActionType.SCREEN_OCR,
                    target=action.target,
                    parameters=ScreenVisionParams(
                        region=getattr(action.parameters, "region", None),
                        language=getattr(action.parameters, "language", "eng"),
                    ),
                    requires_root=False,
                    destructive=False,
                    reversible=True,
                    use_previous_output=action.use_previous_output,
                )

        # --- Fix 5: Convert Linux paths to Windows paths on Windows ---
        if sys.platform == "win32":
            import pathlib

            home = str(pathlib.Path.home())
            for action in fixed:
                # Fix file_write, file_read, etc. paths
                if hasattr(action.parameters, "path"):
                    p = action.parameters.path or ""
                    if p.startswith(("/mnt/", "/tmp/", "/home/")):
                        # Convert /home/user/... → C:\Users\user\...
                        if p.startswith("/home/"):
                            p = p.replace("/home/", home.rsplit("\\", 1)[0] + "\\", 1)
                        elif p.startswith("/mnt/"):
                            p = p.replace("/mnt/", "C:\\", 1)
                        elif p.startswith("/tmp/"):
                            p = p.replace("/tmp/", os.environ.get("TEMP", "C:\\Temp") + "\\", 1)
                        p = p.replace("/", "\\")
                        logger.info("Post-process: fixed Linux path → %s", p)
                        action.parameters.path = p
                        action.target = p
                # Fix screenshot output_path
                if hasattr(action.parameters, "output_path"):
                    op = action.parameters.output_path or ""
                    if op.startswith("/"):
                        op = os.path.join(home, "Pictures", os.path.basename(op))
                        logger.info("Post-process: fixed screenshot path → %s", op)
                        action.parameters.output_path = op
                # Fix destination paths
                if hasattr(action.parameters, "destination"):
                    d = action.parameters.destination or ""
                    if d and d.startswith("/"):
                        if d.startswith("/home/"):
                            d = d.replace("/home/", home.rsplit("\\", 1)[0] + "\\", 1)
                        d = d.replace("/", "\\")
                        action.parameters.destination = d

        # --- Fix 7: Convert slow shell_script with system commands to code_execute ---
        # On Windows, 'systeminfo', 'wmic', 'tasklist', 'netstat' are very slow
        # and often timeout. Convert to code_execute using subprocess for better control.
        if sys.platform == "win32":
            _SLOW_CMDS = {"systeminfo", "wmic", "tasklist", "netstat", "ipconfig", "driverquery"}
            for i, action in enumerate(fixed):
                if action.action_type == ActionType.SHELL_SCRIPT:
                    script = getattr(action.parameters, "script", "") or ""
                    script_lower = script.lower()
                    # Check if script contains slow system commands
                    has_slow = any(cmd in script_lower for cmd in _SLOW_CMDS)
                    if has_slow:
                        # Convert to code_execute with subprocess
                        from pilot.actions import CodeParams

                        python_code = "import subprocess, os\ncommands = {}\n"
                        # Parse lines from the script that look like commands
                        lines = [l.strip() for l in script.split("\n") if l.strip() and not l.strip().startswith("#")]
                        cmd_dict_parts = []
                        for line in lines:
                            cmd_name = line.split()[0] if line.split() else ""
                            if cmd_name.lower() in _SLOW_CMDS or "|" in line:
                                safe_line = line.replace("'", "\\'")
                                cmd_dict_parts.append(f"    '{safe_line}': '{safe_line}'")
                        if cmd_dict_parts:
                            python_code = "import subprocess\nresults = []\n"
                            for line in lines:
                                if line.strip():
                                    safe_line = line.replace("'", "\\'")
                                    python_code += (
                                        f"try:\n"
                                        f"    r = subprocess.run('{safe_line}', shell=True, capture_output=True, text=True, timeout=45)\n"
                                        f"    results.append('=== {safe_line} ===\\n' + r.stdout)\n"
                                        f"except Exception as e:\n"
                                        f"    results.append('=== {safe_line} === FAILED: ' + str(e))\n"
                                    )
                            python_code += "print('\\n'.join(results))\n"

                            fixed[i] = Action(
                                action_type=ActionType.CODE_EXECUTE,
                                target="system_commands",
                                description=action.description or "Run system commands",
                                risk_level=action.risk_level,
                                parameters=CodeParams(
                                    code=python_code,
                                    language="python",
                                ),
                                use_previous_output=False,
                            )
                            logger.info("Post-process: converted slow shell_script to code_execute at index %d", i)

        return fixed

    # Common LLM hallucinations → correct action type values
    ACTION_TYPE_ALIASES: dict[str, str] = {
        "browser_extract_text": "browser_extract",
        "browser_get_text": "browser_extract",
        "browser_read": "browser_extract",
        "browser_read_text": "browser_extract",
        "browser_scrape": "browser_extract",
        "browser_get_content": "browser_extract",
        "browser_open": "browser_navigate",
        "browser_goto": "browser_navigate",
        "browser_go": "browser_navigate",
        "browser_visit": "browser_navigate",
        "browser_load": "browser_navigate",
        "browser_search": "browser_navigate",
        "browser_input": "browser_type",
        "browser_write": "browser_type",
        "browser_run_js": "browser_execute_js",
        "browser_js": "browser_execute_js",
        "browser_get_links": "browser_extract_links",
        "browser_get_tables": "browser_extract_table",
        "code_run": "code_execute",
        "run_code": "code_execute",
        "execute_code": "code_execute",
        "python_execute": "code_execute",
        "run_python": "code_execute",
        "run_script": "shell_script",
        "execute_shell": "shell_command",
        "run_shell": "shell_command",
        "run_command": "shell_command",
        "file_create": "file_write",
        "file_save": "file_write",
        "file_open": "file_read",
        "file_get": "file_read",
        "file_remove": "file_delete",
        "file_rename": "file_move",
        "scrape_url": "api_scrape",
        "web_scrape": "api_scrape",
        "screen_read": "screen_ocr",
        "ocr": "screen_ocr",
        "take_screenshot": "screenshot",
        "capture_screen": "screenshot",
        "mouse_left_click": "mouse_click",
        "type_text": "keyboard_type",
        "press_key": "keyboard_press",
        "hotkey": "keyboard_hotkey",
    }

    def _resolve_action_type(self, raw_type: str) -> ActionType:
        """Resolve an action type string, handling LLM hallucinations."""
        # Strip any 'ActionType.' prefix the LLM might add
        clean = raw_type.strip()
        if clean.startswith("ActionType."):
            clean = clean.split(".", 1)[1].lower()

        # Direct match
        try:
            return ActionType(clean)
        except ValueError:
            pass

        # Check alias table
        if clean in self.ACTION_TYPE_ALIASES:
            resolved = self.ACTION_TYPE_ALIASES[clean]
            logger.info("Resolved hallucinated action type '%s' → '%s'", clean, resolved)
            return ActionType(resolved)

        # Fuzzy match: find closest valid action type
        import difflib

        valid_types = [at.value for at in ActionType]
        matches = difflib.get_close_matches(clean, valid_types, n=1, cutoff=0.6)
        if matches:
            logger.info("Fuzzy-matched action type '%s' → '%s'", clean, matches[0])
            return ActionType(matches[0])

        raise ValueError(f"'{clean}' is not a valid ActionType")

    def _parse_action(self, raw: dict) -> Action:
        action_type = self._resolve_action_type(raw["action_type"])
        target = str(raw.get("target", "") or "")
        params_raw = raw.get("parameters", {}) or {}

        # MIGRATION: LLMs often hallucinate payload keys at the root of the action object
        # instead of inside `parameters`. We must fold them back in.
        standard_keys = {
            "action_type",
            "target",
            "parameters",
            "requires_root",
            "destructive",
            "reversible",
            "rollback_action",
            "use_previous_output",
        }
        for k, v in raw.items():
            if k not in standard_keys and k not in params_raw:
                params_raw[k] = v

        params_raw = self._normalize_params(action_type, params_raw, target)
        parameters = self._parse_parameters(action_type, params_raw)

        rollback = None
        if raw.get("rollback_action"):
            rollback = self._parse_action(raw["rollback_action"])

        return Action(
            action_type=action_type,
            target=target,
            parameters=parameters,
            requires_root=bool(raw.get("requires_root", False)),
            destructive=bool(raw.get("destructive", False)),
            reversible=bool(raw.get("reversible", True)),
            rollback_action=rollback,
            use_previous_output=bool(raw.get("use_previous_output", False)),
        )

    @staticmethod
    def _normalize_params(action_type: ActionType, params: dict, target: str) -> dict:
        """Fill in missing parameters by inferring from target/context."""
        p = dict(params)

        if action_type == ActionType.OPEN_URL:
            if "url" not in p or not p["url"]:
                p["url"] = target or ""
            if p.get("url") and not p["url"].startswith(("http://", "https://")):
                p["url"] = "https://" + p["url"]

        elif action_type == ActionType.OPEN_APPLICATION:
            if "name" not in p or not p["name"]:
                p["name"] = target or ""

        elif action_type == ActionType.NOTIFY:
            if "summary" not in p or not p["summary"]:
                p["summary"] = target or "Pilot notification"

        elif action_type == ActionType.SHELL_COMMAND:
            if "command" not in p and target:
                parts = target.split()
                p["command"] = parts[0]
                if len(parts) > 1:
                    p.setdefault("args", parts[1:])
            # LLM often puts entire command line into 'command' field
            # e.g. {"command": "tree /F C:\\path"} — split it
            if "command" in p and " " in p["command"]:
                parts = p["command"].split()
                p["command"] = parts[0]
                existing_args = p.get("args", [])
                p["args"] = parts[1:] + (existing_args if isinstance(existing_args, list) else [])

        elif action_type == ActionType.SHELL_SCRIPT:
            if "script" not in p and target:
                p["script"] = target

        elif action_type == ActionType.DOWNLOAD_FILE:
            if "url" not in p and target:
                p["url"] = target
            if "output_path" not in p:
                p["output_path"] = str(__import__("pathlib").Path.home() / "Downloads" / "download")

        file_types = {
            ActionType.FILE_READ,
            ActionType.FILE_WRITE,
            ActionType.FILE_DELETE,
            ActionType.FILE_MOVE,
            ActionType.FILE_COPY,
            ActionType.FILE_LIST,
            ActionType.FILE_SEARCH,
            ActionType.FILE_PERMISSIONS,
        }
        if action_type in file_types and ("path" not in p or not p["path"]):
            p["path"] = target

        pkg_types = {
            ActionType.PACKAGE_INSTALL,
            ActionType.PACKAGE_REMOVE,
            ActionType.PACKAGE_SEARCH,
        }
        if action_type in pkg_types and ("name" not in p or not p["name"]):
            p["name"] = target

        svc_types = {
            ActionType.SERVICE_START,
            ActionType.SERVICE_STOP,
            ActionType.SERVICE_RESTART,
            ActionType.SERVICE_ENABLE,
            ActionType.SERVICE_DISABLE,
            ActionType.SERVICE_STATUS,
        }
        if action_type in svc_types and ("name" not in p or not p["name"]):
            p["name"] = target

        # Process management
        if action_type == ActionType.PROCESS_KILL and "name" not in p and "pid" not in p and target:
            try:
                p["pid"] = int(target)
            except ValueError:
                p["name"] = target

        if action_type == ActionType.PROCESS_INFO and "pid" not in p and target:
            with contextlib.suppress(ValueError):
                p["pid"] = int(target)

        # Volume
        if action_type == ActionType.VOLUME_SET and "level" not in p and target:
            with contextlib.suppress(ValueError):
                p["level"] = int(target.replace("%", ""))

        # Brightness
        if action_type == ActionType.BRIGHTNESS_SET and "level" not in p and target:
            with contextlib.suppress(ValueError):
                p["level"] = int(target.replace("%", ""))

        # WiFi
        if action_type == ActionType.WIFI_CONNECT and "ssid" not in p and target:
            p["ssid"] = target

        # Env
        if action_type == ActionType.ENV_GET and "name" not in p and target:
            p["name"] = target

        if action_type == ActionType.ENV_SET and "name" not in p and target:
            if "=" in target:
                parts = target.split("=", 1)
                p["name"] = parts[0]
                p["value"] = parts[1]
            else:
                p["name"] = target

        # Schedule
        if action_type == ActionType.SCHEDULE_DELETE and "name" not in p and target:
            p["name"] = target

        # Window
        window_types = {
            ActionType.WINDOW_FOCUS,
            ActionType.WINDOW_CLOSE,
            ActionType.WINDOW_MINIMIZE,
            ActionType.WINDOW_MAXIMIZE,
        }
        if action_type in window_types and "title" not in p and "process_name" not in p and target:
            p["title"] = target

        # Registry
        if action_type == ActionType.REGISTRY_READ and "key_path" not in p and target:
            p["key_path"] = target

        # Clipboard write
        if action_type == ActionType.CLIPBOARD_WRITE and "content" not in p and target:
            p["content"] = target

        # Browser actions: map target → url/selector/script
        browser_nav_types = {ActionType.BROWSER_NAVIGATE}
        if action_type in browser_nav_types:
            if ("url" not in p or not p["url"]) and target:
                p["url"] = target
            if p.get("url") and not p["url"].startswith(("http://", "https://", "file://")):
                p["url"] = "https://" + p["url"]

        browser_selector_types = {
            ActionType.BROWSER_CLICK,
            ActionType.BROWSER_HOVER,
            ActionType.BROWSER_EXTRACT,
            ActionType.BROWSER_EXTRACT_TABLE,
            ActionType.BROWSER_EXTRACT_LINKS,
            ActionType.BROWSER_SELECT,
            ActionType.BROWSER_WAIT,
        }
        if action_type in browser_selector_types and ("selector" not in p or not p["selector"]) and target:
            p["selector"] = target

        if action_type == ActionType.BROWSER_CLICK_TEXT and ("text" not in p or not p["text"]) and target:
            p["text"] = target

        if action_type == ActionType.BROWSER_TYPE and ("text" not in p or not p["text"]) and target:
            p["text"] = target

        if action_type == ActionType.BROWSER_EXECUTE_JS and ("script" not in p or not p["script"]) and target:
            p["script"] = target

        return p

    @staticmethod
    def _parse_parameters(action_type: ActionType, params: dict):
        """Convert raw params dict into the appropriate Pydantic model."""
        file_types = {
            ActionType.FILE_READ,
            ActionType.FILE_WRITE,
            ActionType.FILE_DELETE,
            ActionType.FILE_MOVE,
            ActionType.FILE_COPY,
            ActionType.FILE_LIST,
            ActionType.FILE_SEARCH,
            ActionType.FILE_PERMISSIONS,
        }
        package_types = {
            ActionType.PACKAGE_INSTALL,
            ActionType.PACKAGE_REMOVE,
            ActionType.PACKAGE_UPDATE,
            ActionType.PACKAGE_SEARCH,
        }
        service_types = {
            ActionType.SERVICE_START,
            ActionType.SERVICE_STOP,
            ActionType.SERVICE_RESTART,
            ActionType.SERVICE_ENABLE,
            ActionType.SERVICE_DISABLE,
            ActionType.SERVICE_STATUS,
        }

        if action_type in file_types:
            return FileParams(**params)
        if action_type in package_types:
            return PackageParams(**params)
        if action_type in service_types:
            return ServiceParams(**params)
        if action_type in (ActionType.GNOME_SETTING_READ, ActionType.GNOME_SETTING_WRITE):
            return GnomeSettingParams(**params)
        if action_type == ActionType.DBUS_CALL:
            return DBusParams(**params)
        if action_type == ActionType.SHELL_COMMAND:
            return ShellCommandParams(**params)
        if action_type == ActionType.SHELL_SCRIPT:
            return ShellScriptParams(**params)
        if action_type == ActionType.OPEN_URL:
            return OpenUrlParams(**params)
        if action_type == ActionType.OPEN_APPLICATION:
            return OpenApplicationParams(**params)
        if action_type == ActionType.NOTIFY:
            return NotifyParams(**params)

        # Process management
        if action_type in (ActionType.PROCESS_LIST, ActionType.PROCESS_KILL, ActionType.PROCESS_INFO):
            return ProcessParams(**params)

        # Clipboard
        if action_type in (ActionType.CLIPBOARD_READ, ActionType.CLIPBOARD_WRITE):
            return ClipboardParams(**params)

        # System info
        if action_type == ActionType.SYSTEM_INFO:
            return SystemInfoParams(**params)
        if action_type in (
            ActionType.DISK_USAGE,
            ActionType.MEMORY_USAGE,
            ActionType.CPU_USAGE,
            ActionType.NETWORK_INFO,
            ActionType.BATTERY_INFO,
            ActionType.DISK_LIST,
            ActionType.USER_LIST,
            ActionType.USER_INFO,
            ActionType.WINDOW_LIST,
            ActionType.VOLUME_GET,
            ActionType.BRIGHTNESS_GET,
            ActionType.WIFI_LIST,
            ActionType.SCHEDULE_LIST,
        ):
            return EmptyParams()

        # Power
        if action_type in (
            ActionType.POWER_SHUTDOWN,
            ActionType.POWER_RESTART,
            ActionType.POWER_SLEEP,
            ActionType.POWER_LOCK,
            ActionType.POWER_LOGOUT,
        ):
            return PowerParams(**params)

        # Schedule
        if action_type in (ActionType.SCHEDULE_CREATE, ActionType.SCHEDULE_DELETE):
            return ScheduleParams(**params)

        # Env
        if action_type in (ActionType.ENV_GET, ActionType.ENV_SET, ActionType.ENV_LIST):
            return EnvParams(**params)

        # Window
        if action_type in (
            ActionType.WINDOW_FOCUS,
            ActionType.WINDOW_CLOSE,
            ActionType.WINDOW_MINIMIZE,
            ActionType.WINDOW_MAXIMIZE,
        ):
            return WindowParams(**params)

        # Volume
        if action_type in (ActionType.VOLUME_SET, ActionType.VOLUME_MUTE):
            return VolumeParams(**params)

        # Brightness
        if action_type == ActionType.BRIGHTNESS_SET:
            return BrightnessParams(**params)

        # Screenshot
        if action_type == ActionType.SCREENSHOT:
            return ScreenshotParams(**params)

        # WiFi
        if action_type in (ActionType.WIFI_CONNECT, ActionType.WIFI_DISCONNECT):
            return WifiParams(**params)

        # Disk
        if action_type in (ActionType.DISK_MOUNT, ActionType.DISK_UNMOUNT):
            return DiskManageParams(**params)

        # Download
        if action_type == ActionType.DOWNLOAD_FILE:
            return DownloadParams(**params)

        # Registry
        if action_type in (ActionType.REGISTRY_READ, ActionType.REGISTRY_WRITE):
            return RegistryParams(**params)

        # Tier 1 & 2 actions
        mouse_types = {
            ActionType.MOUSE_CLICK,
            ActionType.MOUSE_DOUBLE_CLICK,
            ActionType.MOUSE_RIGHT_CLICK,
            ActionType.MOUSE_MOVE,
            ActionType.MOUSE_DRAG,
            ActionType.MOUSE_SCROLL,
            ActionType.MOUSE_POSITION,
        }
        if action_type in mouse_types:
            return MouseParams(**params)

        keyboard_types = {
            ActionType.KEYBOARD_TYPE,
            ActionType.KEYBOARD_PRESS,
            ActionType.KEYBOARD_HOTKEY,
            ActionType.KEYBOARD_HOLD,
        }
        if action_type in keyboard_types:
            return KeyboardParams(**params)

        vision_types = {
            ActionType.SCREEN_OCR,
            ActionType.SCREEN_FIND_TEXT,
            ActionType.SCREEN_ANALYZE,
            ActionType.SCREEN_ELEMENT_MAP,
        }
        if action_type in vision_types:
            return ScreenVisionParams(**params)

        browser_types = {
            ActionType.BROWSER_NAVIGATE,
            ActionType.BROWSER_CLICK,
            ActionType.BROWSER_CLICK_TEXT,
            ActionType.BROWSER_TYPE,
            ActionType.BROWSER_SELECT,
            ActionType.BROWSER_HOVER,
            ActionType.BROWSER_SCROLL,
            ActionType.BROWSER_EXTRACT,
            ActionType.BROWSER_EXTRACT_TABLE,
            ActionType.BROWSER_EXTRACT_LINKS,
            ActionType.BROWSER_EXECUTE_JS,
            ActionType.BROWSER_SCREENSHOT,
            ActionType.BROWSER_FILL_FORM,
            ActionType.BROWSER_NEW_TAB,
            ActionType.BROWSER_CLOSE_TAB,
            ActionType.BROWSER_LIST_TABS,
            ActionType.BROWSER_SWITCH_TAB,
            ActionType.BROWSER_BACK,
            ActionType.BROWSER_FORWARD,
            ActionType.BROWSER_REFRESH,
            ActionType.BROWSER_WAIT,
            ActionType.BROWSER_CLOSE,
            ActionType.BROWSER_PAGE_INFO,
        }
        if action_type in browser_types:
            return BrowserParams(**params)

        trigger_types = {
            ActionType.TRIGGER_CREATE,
            ActionType.TRIGGER_LIST,
            ActionType.TRIGGER_DELETE,
            ActionType.TRIGGER_START,
            ActionType.TRIGGER_STOP,
        }
        if action_type in trigger_types:
            return TriggerParams(**params)

        code_exec_types = {
            ActionType.CODE_EXECUTE,
            ActionType.CODE_GENERATE_AND_RUN,
        }
        if action_type in code_exec_types:
            return CodeExecParams(**params)

        file_intel_types = {
            ActionType.FILE_PARSE,
            ActionType.FILE_SEARCH_CONTENT,
        }
        if action_type in file_intel_types:
            return FileIntelParams(**params)

        api_types = {
            ActionType.API_REQUEST,
            ActionType.API_GITHUB,
            ActionType.API_SEND_EMAIL,
            ActionType.API_WEBHOOK,
            ActionType.API_SLACK,
            ActionType.API_DISCORD,
            ActionType.API_SCRAPE,
        }
        if action_type in api_types:
            return ApiRequestParams(**params)

        raise ValueError(f"Unknown action type: {action_type}")
