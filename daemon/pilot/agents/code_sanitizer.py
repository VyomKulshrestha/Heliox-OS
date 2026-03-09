"""Code sanitizer for LLM-generated Python code.

Fixes common patterns that LLMs generate incorrectly:
- self.method() calls in standalone scripts
- Undefined variables from hallucinated context
- Missing imports
- Syntax issues with string literals
"""

import logging
import re

logger = logging.getLogger("pilot.agents.code_sanitizer")


def sanitize_python_code(code: str) -> str:
    """Fix common LLM code bugs before execution."""
    original = code

    # 1. Remove self.method() calls — LLM sometimes generates code as if inside a class
    # Replace self.browser_extract(...) etc with a comment
    code = re.sub(
        r"self\.\w+\([^)]*\)",
        '""  # removed self.method call',
        code,
    )

    # 2. Fix common undefined variable patterns
    # LLM sometimes uses 'PREV_OUTPUT' without it being defined (preamble adds it)
    # Don't fix this — the preamble handles it

    # 3. Fix unterminated raw strings — r'...\' is invalid because \' escapes the quote
    # Pattern: r'...path\' → r'...path\\' (double the trailing backslash)
    code = re.sub(
        r"""r(["'])(.+?)\\(\1)""",
        lambda m: "r" + m.group(1) + m.group(2) + "\\\\" + m.group(3),
        code,
    )

    # 4. Fix f-string with unescaped braces in regex patterns
    # This is tricky — skip for now

    # 5. Add missing common imports
    needs_os = "os.path" in code or "os.makedirs" in code or "os.listdir" in code
    needs_re = "re.sub(" in code or "re.findall(" in code or "re.search(" in code
    needs_json = "json.loads(" in code or "json.dumps(" in code or "json.load(" in code
    needs_collections = "Counter(" in code and "from collections" not in code

    import_adds = []
    if needs_os and "import os" not in code:
        import_adds.append("import os")
    if needs_re and "import re" not in code:
        import_adds.append("import re")
    if needs_json and "import json" not in code:
        import_adds.append("import json")
    if needs_collections:
        import_adds.append("from collections import Counter")

    if import_adds:
        code = "\n".join(import_adds) + "\n" + code

    # 6. Fix file paths that use single backslashes without r-prefix
    # e.g., open('C:\Users\user\...') → open(r'C:\Users\user\...')
    # Only in open(), with open(), Path() calls
    def _fix_open_path(m):
        prefix = m.group(1)
        quote = m.group(2)
        path = m.group(3)
        # Already raw?
        if prefix.endswith("r") or prefix.endswith("R"):
            return m.group(0)
        return prefix + "r" + quote + path + quote

    code = re.sub(
        r"""(open\s*\(\s*|Path\s*\(\s*)(["'])([A-Za-z]:\\[^"']+)\2""",
        _fix_open_path,
        code,
    )

    # 7. Fix syntax: triple-nested quotes that break
    # e.g., print(f"""...""") inside a string that's already triple-quoted
    # Skip — too complex

    # 8. Syntax check — try to compile and report
    try:
        compile(code, "<pilot_code>", "exec")
    except SyntaxError as e:
        logger.warning("Code has syntax error after sanitization: %s (line %d)", e.msg, e.lineno or 0)
        # Try to fix common syntax errors
        if "unterminated string literal" in str(e.msg):
            # Try replacing problematic regex lines with simpler versions
            lines = code.split("\n")
            if e.lineno and e.lineno <= len(lines):
                bad_line = lines[e.lineno - 1]
                # If line has unmatched quotes in a regex, comment it out
                if "re.sub" in bad_line or "re.findall" in bad_line:
                    lines[e.lineno - 1] = "# SANITIZED: " + bad_line
                    code = "\n".join(lines)
                    logger.info("Commented out problematic regex line %d", e.lineno)

    if code != original:
        logger.info("Code sanitizer made changes to LLM-generated code")

    return code
