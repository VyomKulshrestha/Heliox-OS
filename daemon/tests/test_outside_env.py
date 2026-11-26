"""The daemon leaves an AppImage's start-up settings out of its own environment (pilot.system.outside_env), by the
rule the desktop app uses (tauri-app/src-tauri/src/outside_env.rs), so a program it starts sees the desktop's
settings. Tables of before and after environments; nothing here reads or changes this process's environment."""

from __future__ import annotations

import pytest

from pilot.system.outside_env import clean_this_process, outside, without_the_hooks_share

APP = "/tmp/.mount_HelioxAb12"


def appimage(**extra: str) -> dict[str, str]:
    return {"APPIMAGE": "/home/sam/Apps/Heliox-OS.AppImage", "APPDIR": APP, **extra}


def test_the_launchers_settings_as_ci_read_them_from_the_daemon():
    before = appimage(
        ARGV0="/home/sam/Apps/Heliox-OS.AppImage",
        APPIMAGE_EXTRACT_AND_RUN="1",
        GTK_DATA_PREFIX=APP,
        GDK_BACKEND="x11",
        GTK_THEME="Adwaita:light",
        XDG_DATA_DIRS=f"{APP}/usr/share/:{APP}/usr/share:/usr/share:",
        PYTHONHOME=f"{APP}/usr/",
        PYTHONPATH=f"{APP}/usr/share/pyshared/:",
        PYTHONDONTWRITEBYTECODE="1",
        PERLLIB=f"{APP}/usr/share/perl5/:{APP}/usr/lib/perl5/:",
        PATH=f"{APP}/usr/bin/:/usr/local/bin:/usr/bin:/bin",
        DISPLAY=":99",
        HOME="/home/sam",
    )
    assert outside(before) == {
        "APPIMAGE_EXTRACT_AND_RUN": "1",
        "DISPLAY": ":99",
        "HOME": "/home/sam",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }


GTK = {"GTK_DATA_PREFIX": APP}
ROWS = [  # (case, the environment besides APPIMAGE and APPDIR, the name, what a program sees)
    ("the hook's x11", {**GTK, "GDK_BACKEND": "x11"}, "GDK_BACKEND", None),
    ("a person's wayland", {**GTK, "GDK_BACKEND": "wayland"}, "GDK_BACKEND", "wayland"),
    ("x11 with no GTK hook is the person's", {"GDK_BACKEND": "x11"}, "GDK_BACKEND", "x11"),
    ("recorded before the start-up", {**GTK, "GDK_BACKEND": "x11", "GDK_BACKEND_ORIG": "wayland"}, "GDK_BACKEND",
     "wayland"),
    ("recorded the other way", {**GTK, "GDK_BACKEND": "x11", "APPIMAGE_ORIGINAL_GDK_BACKEND": "wayland"},
     "GDK_BACKEND", "wayland"),
    ("the hook's dark theme", {**GTK, "GTK_THEME": "Adwaita:dark"}, "GTK_THEME", None),
    ("an old hook's theme", {**GTK, "GTK_THEME": "Adwaita"}, "GTK_THEME", None),
    ("the theme asked of the AppImage", {**GTK, "APPIMAGE_GTK_THEME": "Arc", "GTK_THEME": "Arc"}, "GTK_THEME", None),
    ("a person's theme with the hook", {**GTK, "GTK_THEME": "Arc-Dark"}, "GTK_THEME", "Arc-Dark"),
    ("a person's theme, no hook", {"GTK_THEME": "Adwaita:dark"}, "GTK_THEME", "Adwaita:dark"),
    ("AppRun.c's bytecode switch", {"PYTHONHOME": f"{APP}/usr/", "PYTHONDONTWRITEBYTECODE": "1"},
     "PYTHONDONTWRITEBYTECODE", None),
    ("a person's bytecode switch", {"PYTHONDONTWRITEBYTECODE": "1"}, "PYTHONDONTWRITEBYTECODE", "1"),
    ("the GStreamer hook's reuse switch",
     {"GST_PLUGIN_SCANNER": f"{APP}/usr/lib/gst-plugin-scanner", "GST_REGISTRY_REUSE_PLUGIN_SCANNER": "no"},
     "GST_REGISTRY_REUSE_PLUGIN_SCANNER", None),
    ("the person's data folders come back",
     {**GTK, "XDG_DATA_DIRS": f"{APP}/usr/share/:{APP}/usr/share:/usr/share:/usr/local/share:/usr/share"},
     "XDG_DATA_DIRS", "/usr/local/share:/usr/share"),
    ("no data folders before: unset, the desktop's default",
     {**GTK, "XDG_DATA_DIRS": f"{APP}/usr/share:/usr/share:"}, "XDG_DATA_DIRS", None),
    ("the runtime's ARGV0", {"ARGV0": "Heliox.AppImage"}, "ARGV0", None),
    ("the runtime's OWD", {"OWD": "/home/sam"}, "OWD", None),
    ("a person's own variable", {"EDITOR": "vim"}, "EDITOR", "vim"),
]  # fmt: skip


@pytest.mark.parametrize(("case", "extra", "name", "sees"), ROWS, ids=[row[0] for row in ROWS])
def test_before_and_after(case, extra, name, sees):
    assert outside(appimage(**extra)).get(name) == sees


def test_without_an_appimage_nothing_changes():
    plain = {
        "GDK_BACKEND": "x11",
        "GTK_THEME": "Adwaita:dark",
        "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_DATA_DIRS": "/usr/share:/usr/share",
        "GTK_DATA_PREFIX": "/usr",
        "APPIMAGE": "",
    }
    assert outside(plain) == plain


def test_the_hooks_share_is_only_the_entry_right_after_its_own():
    assert without_the_hooks_share(f"{APP}/usr/share:/usr/share:", APP) == f"{APP}/usr/share:"
    assert without_the_hooks_share("/usr/share:/opt/share", APP) == "/usr/share:/opt/share"
    assert without_the_hooks_share(f"{APP}//usr/share/:/usr/share/:/usr/share", APP) == f"{APP}//usr/share/:/usr/share"


def test_the_daemon_cleans_its_environment_once_on_linux_only_and_logs_names_not_values(caplog):
    env = appimage(GTK_DATA_PREFIX=APP, GDK_BACKEND="x11", HOME="/home/sam")
    windows = dict(env)
    assert clean_this_process(windows, platform="win32") == [] and windows == env
    with caplog.at_level("INFO", logger="pilot.system.outside_env"):
        changed = clean_this_process(env, platform="linux")
    assert changed == ["APPDIR", "APPIMAGE", "GDK_BACKEND", "GTK_DATA_PREFIX"]
    assert env == {"HOME": "/home/sam"}
    assert APP not in caplog.text and "x11" not in caplog.text
    assert clean_this_process(env, platform="linux") == []  # a clean environment is left alone


def test_a_daemon_the_app_started_cleanly_is_left_alone():
    clean = {"HOME": "/home/sam", "PATH": "/usr/bin", "GDK_BACKEND": "wayland", "XDG_DATA_DIRS": "/usr/share"}
    assert clean_this_process(dict(clean), platform="linux") == []
