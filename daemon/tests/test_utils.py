"""Tests for core utility functions.

Covers:
- sanitizer.py: regex validators, path validation, URL normalization
- code_sanitizer.py: string transformations, regex extractors, import injection
"""

import pytest
from unittest.mock import MagicMock


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.security.unrestricted_shell = False
    return cfg


@pytest.fixture
def sanitizer(config):
    from pilot.security.sanitizer import Sanitizer
    s = Sanitizer(config)
    s._is_windows = False  # Force posix mode for consistent cross-platform tests
    return sanitizer_posix(config)


def sanitizer_posix(config):
    from pilot.security.sanitizer import Sanitizer
    s = Sanitizer(config)
    s._is_windows = False
    return s


# ─── Regex Patterns (module-level, pure) ─────────────────────────────────────

class TestRegexPatterns:
    def test_shell_metachar_semicolon(self):
        from pilot.security.sanitizer import SHELL_METACHARACTERS
        assert SHELL_METACHARACTERS.search("rm -rf /; echo done")

    def test_shell_metachar_pipe(self):
        from pilot.security.sanitizer import SHELL_METACHARACTERS
        assert SHELL_METACHARACTERS.search("cat file | grep foo")

    def test_shell_metachar_backtick(self):
        from pilot.security.sanitizer import SHELL_METACHARACTERS
        assert SHELL_METACHARACTERS.search("`whoami`")

    def test_shell_metachar_clean(self):
        from pilot.security.sanitizer import SHELL_METACHARACTERS
        assert not SHELL_METACHARACTERS.search("/home/user/file.txt")

    def test_path_traversal_unix(self):
        from pilot.security.sanitizer import PATH_TRAVERSAL
        assert PATH_TRAVERSAL.search("/home/user/../etc/passwd")

    def test_path_traversal_windows_style(self):
        from pilot.security.sanitizer import PATH_TRAVERSAL
        assert PATH_TRAVERSAL.search("C:\\Users\\..\\Windows")

    def test_path_traversal_clean(self):
        from pilot.security.sanitizer import PATH_TRAVERSAL
        assert not PATH_TRAVERSAL.search("/home/user/documents")

    def test_valid_package_name_simple(self):
        from pilot.security.sanitizer import VALID_PACKAGE_NAME
        assert VALID_PACKAGE_NAME.match("requests")

    def test_valid_package_name_with_extras(self):
        from pilot.security.sanitizer import VALID_PACKAGE_NAME
        assert VALID_PACKAGE_NAME.match("my-package.v2_3")

    def test_invalid_package_name_starts_with_dot(self):
        from pilot.security.sanitizer import VALID_PACKAGE_NAME
        assert not VALID_PACKAGE_NAME.match(".hidden")

    def test_invalid_package_name_empty(self):
        from pilot.security.sanitizer import VALID_PACKAGE_NAME
        assert not VALID_PACKAGE_NAME.match("")

    def test_valid_service_name(self):
        from pilot.security.sanitizer import VALID_SERVICE_NAME
        assert VALID_SERVICE_NAME.match("nginx")

    def test_valid_service_name_with_at(self):
        from pilot.security.sanitizer import VALID_SERVICE_NAME
        assert VALID_SERVICE_NAME.match("getty@tty1")

    def test_invalid_service_name_space(self):
        from pilot.security.sanitizer import VALID_SERVICE_NAME
        assert not VALID_SERVICE_NAME.match("my service")

    def test_valid_gsettings_schema(self):
        from pilot.security.sanitizer import VALID_GSETTINGS_SCHEMA
        assert VALID_GSETTINGS_SCHEMA.match("org.gnome.desktop.interface")

    def test_invalid_gsettings_schema_no_org(self):
        from pilot.security.sanitizer import VALID_GSETTINGS_SCHEMA
        assert not VALID_GSETTINGS_SCHEMA.match("gnome.desktop")

    def test_valid_gsettings_key(self):
        from pilot.security.sanitizer import VALID_GSETTINGS_KEY
        assert VALID_GSETTINGS_KEY.match("font-name")

    def test_invalid_gsettings_key_uppercase(self):
        from pilot.security.sanitizer import VALID_GSETTINGS_KEY
        assert not VALID_GSETTINGS_KEY.match("FontName")

    def test_invalid_gsettings_key_starts_with_digit(self):
        from pilot.security.sanitizer import VALID_GSETTINGS_KEY
        assert not VALID_GSETTINGS_KEY.match("1key")

    def test_valid_dbus_name(self):
        from pilot.security.sanitizer import VALID_DBUS_NAME
        assert VALID_DBUS_NAME.match("org.freedesktop.NetworkManager")

    def test_invalid_dbus_name_starts_with_digit(self):
        from pilot.security.sanitizer import VALID_DBUS_NAME
        assert not VALID_DBUS_NAME.match("1invalid")

    def test_valid_dbus_path(self):
        from pilot.security.sanitizer import VALID_DBUS_PATH
        assert VALID_DBUS_PATH.match("/org/freedesktop/NetworkManager")

    def test_invalid_dbus_path_no_leading_slash(self):
        from pilot.security.sanitizer import VALID_DBUS_PATH
        assert not VALID_DBUS_PATH.match("org/freedesktop")

    def test_valid_url_https(self):
        from pilot.security.sanitizer import VALID_URL
        assert VALID_URL.match("https://example.com/path?q=1")

    def test_valid_url_http(self):
        from pilot.security.sanitizer import VALID_URL
        assert VALID_URL.match("http://localhost:8080")

    def test_invalid_url_no_scheme(self):
        from pilot.security.sanitizer import VALID_URL
        assert not VALID_URL.match("example.com")

    def test_invalid_url_ftp(self):
        from pilot.security.sanitizer import VALID_URL
        assert not VALID_URL.match("ftp://files.example.com")


# ─── Sanitizer.validate_path ──────────────────────────────────────────────────

class TestValidatePath:
    def test_empty_path_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="Empty path"):
            s.validate_path("", 0)

    def test_path_traversal_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="Path traversal"):
            s.validate_path("/home/user/../etc/passwd", 0)

    def test_metachar_in_path_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="shell metacharacters"):
            s.validate_path("/home/user;rm -rf /", 0)

    def test_relative_path_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="absolute"):
            s.validate_path("relative/path/file.txt", 0)

    def test_valid_absolute_path(self, config):
        s = sanitizer_posix(config)
        s.validate_path("/home/user/documents/file.txt", 0)  # no raise

    def test_dotdot_in_parts_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError):
            s.validate_path("/home/user/../../etc", 0)


# ─── Sanitizer.validate_url ───────────────────────────────────────────────────

class TestValidateUrl:
    def test_empty_url_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="Empty URL"):
            s.validate_url("", 0)

    def test_valid_https_url(self, config):
        s = sanitizer_posix(config)
        s.validate_url("https://example.com", 0)  # no raise

    def test_valid_http_url(self, config):
        s = sanitizer_posix(config)
        s.validate_url("http://localhost:3000/api", 0)  # no raise

    def test_url_without_scheme_gets_normalized(self, config):
        # URL without scheme gets https:// prepended, then validated
        s = sanitizer_posix(config)
        s.validate_url("example.com", 0)  # no raise — normalized to https://example.com

    def test_invalid_url_with_spaces_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="Invalid URL"):
            s.validate_url("https://exa mple.com", 0)


# ─── Sanitizer.validate_shell_command ────────────────────────────────────────

class TestValidateShellCommand:
    def test_safe_command_allowed(self, config):
        s = sanitizer_posix(config)
        s.validate_shell_command("ls", ["-la"], 0)  # no raise

    def test_unsafe_command_blocked(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="not in the safe whitelist"):
            s.validate_shell_command("nc", ["-lvp", "4444"], 0)

    def test_metachar_in_arg_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="metacharacters"):
            s.validate_shell_command("ls", ["/tmp; rm -rf /"], 0)

    def test_unrestricted_mode_allows_any_command(self, config):
        config.security.unrestricted_shell = True
        s = sanitizer_posix(config)
        s.validate_shell_command("nc", ["-lvp", "4444"], 0)  # no raise


# ─── Sanitizer.validate_package_name ─────────────────────────────────────────

class TestValidatePackageName:
    def test_valid_package(self, config):
        s = sanitizer_posix(config)
        s.validate_package_name("numpy", 0)  # no raise

    def test_invalid_package_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="Invalid package name"):
            s.validate_package_name(".bad-pkg", 0)

    def test_empty_package_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError):
            s.validate_package_name("", 0)


# ─── Sanitizer.validate_service_name ─────────────────────────────────────────

class TestValidateServiceName:
    def test_valid_service(self, config):
        s = sanitizer_posix(config)
        s.validate_service_name("ssh", 0)  # no raise

    def test_invalid_service_raises(self, config):
        from pilot.security.sanitizer import SanitizationError
        s = sanitizer_posix(config)
        with pytest.raises(SanitizationError, match="Invalid service name"):
            s.validate_service_name("my service!", 0)


# ─── SanitizationError ───────────────────────────────────────────────────────

class TestSanitizationError:
    def test_error_message_format(self):
        from pilot.security.sanitizer import SanitizationError
        err = SanitizationError(3, "Something went wrong")
        assert "Action [3]" in str(err)
        assert "Something went wrong" in str(err)
        assert err.action_index == 3

    def test_error_index_zero(self):
        from pilot.security.sanitizer import SanitizationError
        err = SanitizationError(0, "msg")
        assert err.action_index == 0


# ─── code_sanitizer.sanitize_python_code ─────────────────────────────────────

class TestSanitizePythonCode:
    def test_empty_string(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        assert sanitize_python_code("") == ""

    def test_removes_self_method_call(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = 'result = self.browser_extract(url, "div")'
        out = sanitize_python_code(code)
        assert "self.browser_extract" not in out
        assert "removed self.method call" in out

    def test_removes_multiple_self_calls(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "self.foo()\nself.bar(x, y)\nprint('done')"
        out = sanitize_python_code(code)
        assert "self.foo" not in out
        assert "self.bar" not in out

    def test_fixes_raw_string_trailing_backslash(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        # r'C:\path\' is invalid — trailing backslash escapes the quote
        code = r"""path = r'C:\Users\test\'"""
        out = sanitize_python_code(code)
        # Should have doubled trailing backslash
        assert "\\\\" in out

    def test_injects_import_os(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "x = os.path.join('/tmp', 'file')"
        out = sanitize_python_code(code)
        assert "import os" in out

    def test_injects_import_re(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "m = re.search(r'\\d+', text)"
        out = sanitize_python_code(code)
        assert "import re" in out

    def test_injects_import_json(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "data = json.loads(response)"
        out = sanitize_python_code(code)
        assert "import json" in out

    def test_injects_counter_import(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "c = Counter(words)"
        out = sanitize_python_code(code)
        assert "from collections import Counter" in out

    def test_no_duplicate_import_os(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "import os\nx = os.path.join('/a', 'b')"
        out = sanitize_python_code(code)
        assert out.count("import os") == 1

    def test_no_duplicate_import_re(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "import re\nm = re.search(r'\\d+', s)"
        out = sanitize_python_code(code)
        assert out.count("import re") == 1

    def test_fixes_windows_path_in_open(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = 'f = open("C:\\Users\\user\\file.txt")'
        out = sanitize_python_code(code)
        assert 'r"C:\\' in out or "r'" in out  # got r-prefix

    def test_valid_code_unchanged(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "x = 1 + 2\nprint(x)\n"
        out = sanitize_python_code(code)
        assert out == code

    def test_syntax_error_unterminated_string_commented(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        # Inject a line that causes unterminated string + has re.sub
        # After self-removal it may trigger the syntax fix path
        bad = 'result = re.sub(r"["\n, text)\nprint("ok")'
        out = sanitize_python_code(bad)
        # Should not raise — just return something (possibly commented)
        assert isinstance(out, str)

    def test_no_import_when_not_needed(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        code = "print('hello world')"
        out = sanitize_python_code(code)
        assert "import os" not in out
        assert "import re" not in out
        assert "import json" not in out

    def test_returns_string_always(self):
        from pilot.agents.code_sanitizer import sanitize_python_code
        for inp in ["", "   ", "\n\n", "x=1"]:
            assert isinstance(sanitize_python_code(inp), str)