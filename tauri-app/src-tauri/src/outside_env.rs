//! The environment for a program the app starts outside itself, when the app runs from an AppImage.
//!
//! An AppImage's AppRun points the app at the libraries, data and modules inside the AppImage (its folder is
//! $APPDIR while it runs): LD_LIBRARY_PATH, XDG_DATA_DIRS, GTK_*, GDK_*, GIO_*, and the like. Every program the app
//! started inherited that: the first-run setup script and its downloader, the daemon's Python, the system's opener.
//! A system program loading the AppImage's libraries can fail before it runs a line; in CI on f6a1f8e the system's
//! python3 died at start-up inside the AppImage's setup ("Fatal Python error ... <no Python frame>"), so a person
//! with the AppImage could not set Heliox up.
//!
//! `outside` is the one rule, pure and tested: untouched when the app is not running from an AppImage (Windows,
//! macOS, the .deb and the .rpm, by construction); otherwise PYTHONHOME and PYTHONPATH dropped, every entry under
//! $APPDIR removed from the path-list variables, a variable left empty unset, and the person's own value put back
//! where the AppImage's start-up saved one. `apply` gives it to a `Command`, and `open_browser` and `open` stand in
//! for the opener crate, which starts the system's opener with the app's own environment and takes no other.

use std::collections::BTreeMap;
use std::process::Command;

/// Python's own search settings: no system Python wants the AppImage's.
const DROPPED: [&str; 2] = ["PYTHONHOME", "PYTHONPATH"];

/// Path lists (or single paths) the AppImage's start-up points into itself.
const PATH_VARIABLES: [&str; 5] = [
    "LD_LIBRARY_PATH",
    "PATH",
    "XDG_DATA_DIRS",
    "GI_TYPELIB_PATH",
    "GSETTINGS_SCHEMA_DIR",
];

/// Families of variables the toolkit start-up hooks set, each a path or a path list.
const PATH_PREFIXES: [&str; 5] = ["GIO_", "GTK_", "GDK_", "GST_", "QT_"];

/// Where an AppImage's start-up may have kept the person's own value of NAME, in the order they are tried. No
/// AppRun Tauri ships is known to keep one; the rule restores it where one does.
fn saved_originals(name: &str) -> [String; 2] {
    [format!("APPIMAGE_ORIGINAL_{name}"), format!("{name}_ORIG")]
}

fn is_path_variable(name: &str) -> bool {
    PATH_VARIABLES.contains(&name) || PATH_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
}

/// The value without its entries under `appdir`, or None when nothing is left.
fn without_appdir(value: &str, appdir: &str) -> Option<String> {
    let inside = |entry: &str| entry == appdir || entry.starts_with(&format!("{appdir}/"));
    let kept: Vec<&str> = value.split(':').filter(|entry| !inside(entry)).collect();
    if kept.iter().all(|entry| entry.is_empty()) {
        None
    } else {
        Some(kept.join(":"))
    }
}

/// Whether this environment is an AppImage's: the runtime sets APPIMAGE (the file) and APPDIR (where it is mounted).
pub fn from_appimage(env: &BTreeMap<String, String>) -> bool {
    env.get("APPIMAGE").is_some_and(|value| !value.is_empty())
        || env.get("APPDIR").is_some_and(|value| !value.is_empty())
}

/// The environment to give a program started outside the AppImage (see the module's comment).
pub fn outside(env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    if !from_appimage(env) {
        return env.clone();
    }
    let appdir = env
        .get("APPDIR")
        .map(|dir| dir.trim_end_matches('/').to_string())
        .filter(|dir| !dir.is_empty());
    let mut result = BTreeMap::new();
    for (name, value) in env {
        if DROPPED.contains(&name.as_str()) || name == "APPDIR" {
            continue;
        }
        if let Some(original) = saved_originals(name)
            .iter()
            .find_map(|saved| env.get(saved))
        {
            if !original.is_empty() {
                result.insert(name.clone(), original.clone());
            }
            continue;
        }
        match (&appdir, is_path_variable(name)) {
            (Some(appdir), true) => {
                if let Some(kept) = without_appdir(value, appdir) {
                    result.insert(name.clone(), kept);
                }
            }
            _ => {
                result.insert(name.clone(), value.clone());
            }
        }
    }
    result
}

fn current() -> BTreeMap<String, String> {
    std::env::vars().collect()
}

/// Give `command` the environment for a program outside the AppImage. Call it right after `Command::new`, before
/// any `.env(...)` of the caller's own: it replaces the whole environment when the app runs from an AppImage, and
/// does nothing otherwise.
pub fn apply(command: &mut Command) -> &mut Command {
    let env = current();
    if from_appimage(&env) {
        command.env_clear().envs(outside(&env));
    }
    command
}

/// Open a web or mail address in the system's browser, as the opener crate does: $BROWSER when it is set, else the
/// desktop's opener. From an AppImage, it is started with the environment for programs outside it.
pub fn open_browser(address: &str) -> Result<(), String> {
    let env = current();
    if !from_appimage(&env) {
        return opener::open_browser(address).map_err(|error| error.to_string());
    }
    let outside = outside(&env);
    let browser = outside.get("BROWSER").and_then(|value| {
        value
            .split(':')
            .find(|program| !program.is_empty())
            .map(str::to_string)
    });
    spawn_outside(browser.as_deref().unwrap_or("xdg-open"), address, &outside)
}

/// Open a folder or file with the desktop's opener (the log folder). From an AppImage, as `open_browser`.
pub fn open(target: &std::path::Path) -> Result<(), String> {
    let env = current();
    if !from_appimage(&env) {
        return opener::open(target).map_err(|error| error.to_string());
    }
    spawn_outside("xdg-open", &target.to_string_lossy(), &outside(&env))
}

fn spawn_outside(
    program: &str,
    argument: &str,
    env: &BTreeMap<String, String>,
) -> Result<(), String> {
    Command::new(program)
        .arg(argument)
        .env_clear()
        .envs(env)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("{program}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    const APP: &str = "/tmp/.mount_HelioxAb12";

    fn appimage(extra: &[(&str, &str)]) -> BTreeMap<String, String> {
        let mut found = env(&[
            ("APPIMAGE", "/home/sam/Apps/Heliox-OS.AppImage"),
            ("APPDIR", APP),
        ]);
        found.extend(env(extra));
        found
    }

    #[test]
    fn untouched_when_not_running_from_an_appimage() {
        let plain = env(&[
            ("PATH", "/usr/bin:/bin"),
            ("PYTHONPATH", "/home/sam/lib"),
            ("LD_LIBRARY_PATH", "/opt/lib"),
            ("GTK_PATH", "/usr/lib/gtk"),
        ]);
        assert_eq!(outside(&plain), plain);
    }

    #[test]
    fn python_search_settings_are_dropped() {
        let result = outside(&appimage(&[
            ("PYTHONHOME", "/tmp/x/usr"),
            ("PYTHONPATH", "/a:/b"),
        ]));
        assert!(!result.contains_key("PYTHONHOME") && !result.contains_key("PYTHONPATH"));
    }

    #[test]
    fn appimage_entries_go_wherever_they_are_in_a_list() {
        let first = format!("{APP}/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/lib");
        let middle = format!("/usr/local/bin:{APP}/usr/bin:/usr/bin");
        let last = format!("/usr/local/share:/usr/share:{APP}/usr/share");
        let result = outside(&appimage(&[
            ("LD_LIBRARY_PATH", first.as_str()),
            ("PATH", middle.as_str()),
            ("XDG_DATA_DIRS", last.as_str()),
        ]));
        assert_eq!(
            result["LD_LIBRARY_PATH"],
            "/usr/lib/x86_64-linux-gnu:/opt/lib"
        );
        assert_eq!(result["PATH"], "/usr/local/bin:/usr/bin");
        assert_eq!(result["XDG_DATA_DIRS"], "/usr/local/share:/usr/share");
    }

    #[test]
    fn a_variable_left_empty_is_unset_not_empty() {
        let pixbuf = format!("{APP}/usr/lib/gdk-pixbuf-2.0/loaders.cache");
        let gtk = format!("{APP}/usr/lib/gtk-3.0:{APP}/usr/lib");
        let result = outside(&appimage(&[
            ("GDK_PIXBUF_MODULE_FILE", pixbuf.as_str()),
            ("GTK_PATH", gtk.as_str()),
            ("GIO_MODULE_DIR", APP),
        ]));
        for name in ["GDK_PIXBUF_MODULE_FILE", "GTK_PATH", "GIO_MODULE_DIR"] {
            assert!(!result.contains_key(name), "{name} should be unset");
        }
    }

    #[test]
    fn the_persons_own_value_comes_back_where_one_was_kept() {
        let ld = format!("{APP}/usr/lib");
        let data = format!("{APP}/usr/share:/usr/share");
        let result = outside(&appimage(&[
            ("LD_LIBRARY_PATH", ld.as_str()),
            ("APPIMAGE_ORIGINAL_LD_LIBRARY_PATH", "/home/sam/lib"),
            ("XDG_DATA_DIRS", data.as_str()),
            ("XDG_DATA_DIRS_ORIG", "/usr/local/share:/usr/share"),
        ]));
        assert_eq!(result["LD_LIBRARY_PATH"], "/home/sam/lib");
        assert_eq!(result["XDG_DATA_DIRS"], "/usr/local/share:/usr/share");
    }

    #[test]
    fn nothing_left_points_into_the_appimage_and_the_rest_is_kept() {
        let path = format!("{APP}/usr/bin:/usr/bin");
        let result = outside(&appimage(&[
            ("PATH", path.as_str()),
            ("HOME", "/home/sam"),
            ("DISPLAY", ":0"),
            ("QT_PLUGIN_PATH", "/usr/lib/qt/plugins"),
        ]));
        assert!(
            result.values().all(|value| !value.contains(APP)),
            "{result:?}"
        );
        assert_eq!(result["HOME"], "/home/sam");
        assert_eq!(result["DISPLAY"], ":0");
        assert_eq!(result["QT_PLUGIN_PATH"], "/usr/lib/qt/plugins");
        assert_eq!(result["APPIMAGE"], "/home/sam/Apps/Heliox-OS.AppImage"); // the file, outside the mount
        assert!(!result.contains_key("APPDIR"));
    }

    #[test]
    fn an_entry_that_only_starts_like_the_appdir_is_kept() {
        let path = format!("{APP}-other/bin:{APP}/usr/bin");
        let result = outside(&appimage(&[("PATH", path.as_str())]));
        assert_eq!(result["PATH"], format!("{APP}-other/bin"));
    }

    #[test]
    fn every_place_the_app_starts_a_program_uses_the_rule() {
        // Each production file that builds a Command for a program outside the app names outside_env::apply right
        // after it; the opener goes through outside_env too. Tests and Windows-only commands are not counted.
        for (file, source) in [
            ("commands.rs", include_str!("commands.rs")),
            ("main.rs", include_str!("main.rs")),
            ("not_answering.rs", include_str!("not_answering.rs")),
            ("setup.rs", include_str!("setup.rs")),
            ("uninstall.rs", include_str!("uninstall.rs")),
        ] {
            let end = ["\nmod tests {", "\nmod startup_tests {"]
                .iter()
                .filter_map(|marker| source.find(marker))
                .min()
                .unwrap_or(source.len());
            let production = &source[..end];
            let commands = production.matches("Command::new(").count();
            let applied = production.matches("outside_env::apply(").count();
            assert!(
                applied >= commands,
                "{file}: {commands} Command::new, {applied} outside_env::apply"
            );
            assert!(
                !production.contains("opener::"),
                "{file} calls the opener crate directly"
            );
        }
        let navigation = include_str!("navigation.rs");
        let end = navigation.find("\nmod tests {").unwrap_or(navigation.len());
        assert!(
            !navigation[..end].contains("opener::"),
            "navigation.rs calls the opener crate directly"
        );
    }
}
