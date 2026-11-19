//! The environment for a program the app starts outside itself, when the app runs from an AppImage.
//!
//! An AppImage's start-up points the app at the libraries, data and modules inside the AppImage (its folder is
//! $APPDIR while it runs). The AppRun Tauri bundles (AppImageKit's AppRun.c, "apprun-old") prepends folders under
//! $APPDIR to PATH, LD_LIBRARY_PATH, PYTHONPATH, XDG_DATA_DIRS, PERLLIB, GSETTINGS_SCHEMA_DIR, QT_PLUGIN_PATH and
//! GST_PLUGIN_SYSTEM_PATH(_1_0), keeping the person's value after its own entries, and sets PYTHONHOME to
//! $APPDIR/usr/; the GTK and GStreamer hooks of linuxdeploy then set GTK_*, GDK_*, GIO_*, GI_TYPELIB_PATH and more,
//! most of them to paths under $APPDIR with no trace of a value the person had. Every program the app started
//! inherited that: the first-run setup script and its downloader, the daemon's Python, the system's opener. A system
//! program loading the AppImage's libraries can fail before it runs a line; in CI on f6a1f8e the system's python3
//! died at start-up inside the AppImage's setup ("Fatal Python error ... <no Python frame>").
//!
//! Until 20 November 2026 the rule cleaned a fixed list of names, and PERLLIB, which the AppRun also sets, reached
//! the daemon (Linux package job, every run from 13 to 19 November). The set of names depends on the AppRun and the
//! hooks a build happens to bundle, so the rule now goes by what a value is:
//!
//! - untouched when the app is not running from an AppImage (Windows, macOS, the .deb and the .rpm);
//! - otherwise a variable whose value points into the AppImage's folder (the whole value, one entry of a ':' list,
//!   or a mention after a separator) gets the value the start-up recorded for it before it ran
//!   (APPIMAGE_ORIGINAL_<NAME> or <NAME>_ORIG), when one was recorded; else its entries under the folder are removed;
//!   a variable with nothing left, or still mentioning the folder, is unset;
//! - a variable that does not point into the folder is kept as it is, whatever its name: a person's own value stays;
//! - PYTHONHOME and PYTHONPATH are dropped whatever they hold, so the daemon's Python searches only its environment.
//!
//! The folder is $APPDIR; when the runtime did not set it (an AppImage extracted to run), it is the folder the app's
//! own program runs from, the part before /usr/bin/. `apply` gives the result to a `Command`; `open_browser` and
//! `open` stand in for the opener crate, which starts the system's opener with the app's own environment.

use std::collections::BTreeMap;
use std::process::Command;

/// Python's own search settings: the daemon's Python and the system's python3 look only in their own folders.
const DROPPED: [&str; 2] = ["PYTHONHOME", "PYTHONPATH"];

/// Where an AppImage's start-up may have recorded the value NAME had before it ran, in the order they are tried.
/// The AppRun Tauri bundles records none; other AppRun scripts do, and the rule uses a record where there is one.
fn recorded_originals(name: &str) -> [String; 2] {
    [format!("APPIMAGE_ORIGINAL_{name}"), format!("{name}_ORIG")]
}

/// Whether one path is the folder or under it ("/tmp/.mount_x//usr" counts; "/tmp/.mount_x-other" does not).
fn inside(entry: &str, folder: &str) -> bool {
    entry
        .strip_prefix(folder)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Whether the value mentions the folder anywhere as a path of its own: at the start or after a character that
/// cannot be part of a path name (':', '=', ';', ',', a space, a quote), and followed by its end, '/' or ':'.
fn mentions(value: &str, folder: &str) -> bool {
    value.match_indices(folder).any(|(at, _)| {
        let before = value[..at].chars().next_back();
        let after = value[at + folder.len()..].chars().next();
        let starts = before.is_none_or(|c| !(c.is_alphanumeric() || "/._-~+@".contains(c)));
        let ends = after.is_none_or(|c| c == '/' || c == ':');
        starts && ends
    })
}

/// The value as a program outside the AppImage should see it, or None when the variable should be unset.
fn cleaned(value: &str, folder: &str) -> Option<String> {
    if !mentions(value, folder) {
        return Some(value.to_string());
    }
    let kept: Vec<&str> = value
        .split(':')
        .filter(|entry| !inside(entry, folder))
        .collect();
    let joined = kept.join(":");
    if kept.iter().all(|entry| entry.is_empty()) || mentions(&joined, folder) {
        None
    } else {
        Some(joined)
    }
}

/// Whether this environment is an AppImage's: the runtime sets APPIMAGE (the file) and APPDIR (where it is mounted).
pub fn from_appimage(env: &BTreeMap<String, String>) -> bool {
    env.get("APPIMAGE").is_some_and(|value| !value.is_empty())
        || env.get("APPDIR").is_some_and(|value| !value.is_empty())
}

/// The AppImage's folder: $APPDIR, else the part of the app's program path before /usr/bin/.
fn image_folder(env: &BTreeMap<String, String>, program: Option<&str>) -> Option<String> {
    let trimmed = |dir: &str| {
        let dir = dir.trim_end_matches('/');
        (!dir.is_empty()).then(|| dir.to_string())
    };
    env.get("APPDIR").and_then(|dir| trimmed(dir)).or_else(|| {
        program
            .and_then(|program| program.split_once("/usr/bin/"))
            .and_then(|(head, _)| trimmed(head))
    })
}

/// The environment to give a program started outside the AppImage (see the module's comment). `program` is the
/// path of the app's own program, used only when APPDIR is not set.
pub fn outside_with(
    env: &BTreeMap<String, String>,
    program: Option<&str>,
) -> BTreeMap<String, String> {
    if !from_appimage(env) {
        return env.clone();
    }
    let folder = image_folder(env, program);
    let mut result = BTreeMap::new();
    for (name, value) in env {
        if DROPPED.contains(&name.as_str()) {
            continue;
        }
        let Some(folder) = folder.as_deref() else {
            result.insert(name.clone(), value.clone());
            continue;
        };
        if !mentions(value, folder) {
            result.insert(name.clone(), value.clone());
            continue;
        }
        let recorded = recorded_originals(name)
            .iter()
            .find_map(|record| env.get(record));
        if let Some(kept) = cleaned(recorded.unwrap_or(value), folder) {
            if !kept.is_empty() {
                result.insert(name.clone(), kept);
            }
        }
    }
    result
}

/// `outside_with` for the app as it runs now.
pub fn outside(env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let program = std::env::current_exe().ok();
    outside_with(env, program.as_deref().and_then(|path| path.to_str()))
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

    fn outside(env: &BTreeMap<String, String>) -> BTreeMap<String, String> {
        outside_with(env, None)
    }

    #[test]
    fn untouched_when_not_running_from_an_appimage() {
        let plain = env(&[
            ("PATH", "/usr/bin:/bin"),
            ("PYTHONPATH", "/home/sam/lib"),
            ("LD_LIBRARY_PATH", "/opt/lib"),
            ("GTK_PATH", "/usr/lib/gtk"),
            ("PERLLIB", "/tmp/.mount_HelioxAb12/usr/share/perl5/"),
        ]);
        assert_eq!(
            outside_with(&plain, Some("/opt/Heliox/usr/bin/heliox-os")),
            plain
        );
    }

    #[test]
    fn python_search_settings_are_dropped() {
        let result = outside(&appimage(&[
            ("PYTHONHOME", "/tmp/x/usr"),
            ("PYTHONPATH", "/a:/b"),
        ]));
        assert!(!result.contains_key("PYTHONHOME") && !result.contains_key("PYTHONPATH"));
    }

    /// The environment the Linux package job read from the app on cfd60bc, with the person's own values the AppRun
    /// kept after its entries.
    #[test]
    fn the_appruns_environment_as_ci_read_it() {
        let lib = format!("{APP}/usr/lib/:{APP}/usr/lib/x86_64-linux-gnu/:{APP}/lib64/:");
        let path =
            format!("{APP}/usr/bin/:{APP}/usr/sbin/:{APP}/bin/:/usr/local/bin:/usr/bin:/bin");
        let perllib = format!("{APP}/usr/share/perl5/:{APP}/usr/lib/perl5/:");
        let gst = format!("{APP}/usr/lib/gstreamer-1.0:");
        let schemas =
            format!("{APP}/usr/share/glib-2.0/schemas/:{APP}//usr/share/glib-2.0/schemas");
        let pixbuf = format!("{APP}//usr/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache");
        let exe = format!("{APP}//usr");
        let result = outside(&appimage(&[
            ("LD_LIBRARY_PATH", lib.as_str()),
            ("PATH", path.as_str()),
            ("PERLLIB", perllib.as_str()),
            ("GST_PLUGIN_SYSTEM_PATH_1_0", gst.as_str()),
            ("GSETTINGS_SCHEMA_DIR", schemas.as_str()),
            ("GDK_PIXBUF_MODULE_FILE", pixbuf.as_str()),
            ("GTK_DATA_PREFIX", APP),
            ("GTK_EXE_PREFIX", exe.as_str()),
            ("HOME", "/home/sam"),
        ]));
        assert_eq!(result["PATH"], "/usr/local/bin:/usr/bin:/bin");
        assert_eq!(result["HOME"], "/home/sam");
        assert_eq!(result["APPIMAGE"], "/home/sam/Apps/Heliox-OS.AppImage"); // the file, outside the folder
        for name in [
            "APPDIR",
            "LD_LIBRARY_PATH",
            "PERLLIB",
            "GST_PLUGIN_SYSTEM_PATH_1_0",
            "GSETTINGS_SCHEMA_DIR",
            "GDK_PIXBUF_MODULE_FILE",
            "GTK_DATA_PREFIX",
            "GTK_EXE_PREFIX",
        ] {
            assert!(!result.contains_key(name), "{name} should be unset");
        }
        assert!(
            result.values().all(|value| !value.contains(APP)),
            "{result:?}"
        );
    }

    #[test]
    fn a_path_list_keeps_the_persons_entries_wherever_the_appimages_are() {
        let first = format!("{APP}/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/lib");
        let middle = format!("/usr/local/bin:{APP}/usr/bin:/usr/bin");
        let last = format!("/home/sam/perl5/lib:{APP}/usr/share/perl5/");
        let result = outside(&appimage(&[
            ("LD_LIBRARY_PATH", first.as_str()),
            ("PATH", middle.as_str()),
            ("PERLLIB", last.as_str()),
        ]));
        assert_eq!(
            result["LD_LIBRARY_PATH"],
            "/usr/lib/x86_64-linux-gnu:/opt/lib"
        );
        assert_eq!(result["PATH"], "/usr/local/bin:/usr/bin");
        assert_eq!(result["PERLLIB"], "/home/sam/perl5/lib");
    }

    #[test]
    fn a_variable_no_list_names_is_cleaned_by_its_value() {
        let made_up = format!("{APP}/usr/lib/made-up:/usr/lib/made-up");
        let single = format!("{APP}/usr/share/thing.conf");
        let result = outside(&appimage(&[
            ("SOME_FUTURE_HOOK_PATH", made_up.as_str()),
            ("ANOTHER_HOOKS_FILE", single.as_str()),
        ]));
        assert_eq!(result["SOME_FUTURE_HOOK_PATH"], "/usr/lib/made-up");
        assert!(!result.contains_key("ANOTHER_HOOKS_FILE"));
    }

    #[test]
    fn nothing_left_means_unset_not_empty() {
        let only_ours = format!("{APP}/usr/lib/gtk-3.0:{APP}/usr/lib:");
        let result = outside(&appimage(&[
            ("GTK_PATH", only_ours.as_str()),
            ("GIO_MODULE_DIR", APP),
        ]));
        assert!(!result.contains_key("GTK_PATH"));
        assert!(!result.contains_key("GIO_MODULE_DIR"));
    }

    #[test]
    fn a_value_recorded_before_the_start_up_comes_back() {
        let ld = format!("{APP}/usr/lib:/home/sam/lib");
        let data = format!("{APP}/usr/share:/usr/share");
        let theme = format!("{APP}/usr/share/themes");
        let result = outside(&appimage(&[
            ("LD_LIBRARY_PATH", ld.as_str()),
            (
                "APPIMAGE_ORIGINAL_LD_LIBRARY_PATH",
                "/home/sam/lib:/opt/lib",
            ),
            ("XDG_DATA_DIRS", data.as_str()),
            ("XDG_DATA_DIRS_ORIG", "/usr/local/share:/usr/share"),
            ("GTK_THEME_DIR", theme.as_str()),
            ("GTK_THEME_DIR_ORIG", ""), // recorded as not set before
        ]));
        assert_eq!(result["LD_LIBRARY_PATH"], "/home/sam/lib:/opt/lib");
        assert_eq!(result["XDG_DATA_DIRS"], "/usr/local/share:/usr/share");
        assert!(!result.contains_key("GTK_THEME_DIR"));
    }

    #[test]
    fn a_record_is_used_only_for_a_value_that_points_into_the_appimage() {
        let result = outside(&appimage(&[("EDITOR", "vim"), ("EDITOR_ORIG", "nano")]));
        assert_eq!(result["EDITOR"], "vim");
    }

    #[test]
    fn a_persons_value_naming_other_folders_is_kept() {
        let result = outside(&appimage(&[
            ("PERL5LIB", "/home/sam/perl5/lib/perl5"),
            ("QT_PLUGIN_PATH", "/usr/lib/qt/plugins"),
            ("MY_TOOL_HOME", "/home/sam/tmp/.mount_HelioxAb12/tool"), // contains APPDIR's text, elsewhere
            ("NEIGHBOUR", "/tmp/.mount_HelioxAb12-other/bin"),
            ("DISPLAY", ":0"),
        ]));
        assert_eq!(result["PERL5LIB"], "/home/sam/perl5/lib/perl5");
        assert_eq!(result["QT_PLUGIN_PATH"], "/usr/lib/qt/plugins");
        assert_eq!(
            result["MY_TOOL_HOME"],
            "/home/sam/tmp/.mount_HelioxAb12/tool"
        );
        assert_eq!(result["NEIGHBOUR"], "/tmp/.mount_HelioxAb12-other/bin");
        assert_eq!(result["DISPLAY"], ":0");
    }

    #[test]
    fn a_mention_that_is_not_a_list_entry_unsets_the_variable() {
        let option = format!("--plugins={APP}/usr/lib/x");
        let result = outside(&appimage(&[("SOME_TOOL_OPTIONS", option.as_str())]));
        assert!(!result.contains_key("SOME_TOOL_OPTIONS"));
    }

    #[test]
    fn without_appdir_the_folder_is_where_the_apps_program_runs_from() {
        let mut found = env(&[
            ("APPIMAGE", "/home/sam/Apps/Heliox-OS.AppImage"),
            ("PERLLIB", "/tmp/appimage_extracted_ab/usr/share/perl5/:"),
            ("PATH", "/tmp/appimage_extracted_ab/usr/bin/:/usr/bin"),
        ]);
        found.insert("HOME".into(), "/home/sam".into());
        let result = outside_with(&found, Some("/tmp/appimage_extracted_ab/usr/bin/heliox-os"));
        assert!(!result.contains_key("PERLLIB"));
        assert_eq!(result["PATH"], "/usr/bin");
        assert_eq!(result["HOME"], "/home/sam");
    }

    #[test]
    fn the_folder_check_needs_a_whole_path_component() {
        assert!(inside(&format!("{APP}//usr"), APP));
        assert!(inside(APP, APP));
        assert!(!inside(&format!("{APP}-other/bin"), APP));
        assert!(mentions(&format!("x:{APP}/lib"), APP));
        assert!(!mentions(&format!("/home{APP}/lib"), APP));
        assert!(!mentions(&format!("{APP}x/lib"), APP));
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
