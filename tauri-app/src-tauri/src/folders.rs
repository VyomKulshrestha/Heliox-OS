//! Every folder Heliox keeps, found by the daemon's own rules (daemon/pilot/config.py), and the only file in the app
//! that builds one by hand. Everything else asks here: the log panel, the token, setup, removal, the shortcut, the
//! single-instance lock.
//!
//! - config, data, state: "heliox-os" under $XDG_CONFIG_HOME, $XDG_DATA_HOME, $XDG_STATE_HOME when the variable,
//!   with surrounding blanks removed, is an absolute path; otherwise under ~/.config, ~/.local/share, ~/.local/state.
//!   On every system (pilot.config `_xdg` and `_heliox_dir`).
//! - runtime (token files, port lock): pilot.config `_default_runtime_dir`, see `runtime_for`.
//! - cache: ~/.cache/heliox (pilot.config CACHE_DIR), whatever the variables say.
//! - setup: ~/.config/heliox-os (pilot.config SETUP_DIR): the environment, tools, setup files, the app's own
//!   shortcut and lock, whatever $XDG_CONFIG_HOME says.
//!
//! daemon/tests/heliox_folders_cases.json holds the cases this file and pilot.config are both tested against. A
//! source test (`no_other_file_builds_a_heliox_folder`) fails when another file in src/ puts "heliox-os" in a path.

use std::path::{Path, PathBuf};

const NAME: &str = "heliox-os";

/// Heliox's folders on one system, for one person. Removal reads config, data and cache through the functions below
/// (uninstall.rs builds them from its own SystemDirs); here they are for the shared table's test.
#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub struct Folders {
    pub config: PathBuf,
    pub data: PathBuf,
    pub state: PathBuf,
    pub runtime: PathBuf,
    pub cache: PathBuf,
    pub setup: PathBuf,
}

impl Folders {
    /// This process's folders: its environment, home folder ("." when none is known, as the app always did), uid.
    pub fn here() -> Self {
        Self::resolve(
            this_system(),
            &|name| std::env::var(name).ok(),
            &dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")),
            current_uid(),
            &folder_writable,
        )
    }

    /// Pure: `system` is "linux", "macos" or "windows"; `var` reads an environment variable; `writable` says whether
    /// an existing folder may be written (only /run/user/<uid> is asked).
    pub fn resolve(
        system: &str,
        var: &dyn Fn(&str) -> Option<String>,
        home: &Path,
        uid: u32,
        writable: &dyn Fn(&Path) -> bool,
    ) -> Self {
        let xdg = |name: &str| xdg_base(var(name).as_deref());
        Folders {
            config: config_dir(xdg("XDG_CONFIG_HOME").as_deref(), home),
            data: data_dir(xdg("XDG_DATA_HOME").as_deref(), home),
            state: state_dir(xdg("XDG_STATE_HOME").as_deref(), home),
            runtime: runtime_for(system, var, home, uid, writable),
            cache: cache_dir(home),
            setup: setup_dir(home),
        }
    }

    /// The daemon's log (pilot.config LOG_FILE), what the log panel shows and "Clear logs" empties.
    pub fn log_file(&self) -> PathBuf {
        self.state.join("pilot.log")
    }

    /// The Python environment setup builds.
    pub fn env(&self) -> PathBuf {
        self.setup.join("env")
    }

    /// Left by the app when it had to end a daemon that did not answer its stop (supervisor::stop_on_purpose), for
    /// the next start's run record (pilot.system.run_record APP_NOTE_FILE), beside run_state.json.
    pub fn app_note_file(&self) -> PathBuf {
        self.state.join("ended_by_app")
    }

    /// The shortcut the person chose, kept by the app beside setup's files.
    pub fn shortcut_file(&self) -> PathBuf {
        self.setup.join("shortcut.txt")
    }
}

/// An XDG base folder as pilot.config `_xdg` reads it: blanks removed; unset, empty or relative means "not set".
pub fn xdg_base(value: Option<&str>) -> Option<PathBuf> {
    let value = value?.trim();
    let path = PathBuf::from(value);
    (!value.is_empty() && path.is_absolute()).then_some(path)
}

/// Settings and keys: "heliox-os" under $XDG_CONFIG_HOME (already read by `xdg_base`) or ~/.config.
pub fn config_dir(xdg: Option<&Path>, home: &Path) -> PathBuf {
    under(xdg, home, &[".config"])
}

/// Memory, timeline, models: "heliox-os" under $XDG_DATA_HOME or ~/.local/share.
pub fn data_dir(xdg: Option<&Path>, home: &Path) -> PathBuf {
    under(xdg, home, &[".local", "share"])
}

/// Logs and the run record: "heliox-os" under $XDG_STATE_HOME or ~/.local/state.
pub fn state_dir(xdg: Option<&Path>, home: &Path) -> PathBuf {
    under(xdg, home, &[".local", "state"])
}

fn under(xdg: Option<&Path>, home: &Path, fallback: &[&str]) -> PathBuf {
    let base = match xdg {
        Some(base) => base.to_path_buf(),
        None => fallback
            .iter()
            .fold(home.to_path_buf(), |path, part| path.join(part)),
    };
    base.join(NAME)
}

pub fn cache_dir(home: &Path) -> PathBuf {
    home.join(".cache").join("heliox")
}

pub fn setup_dir(home: &Path) -> PathBuf {
    home.join(".config").join(NAME)
}

/// %LOCALAPPDATA% as pilot.config reads it: an empty value counts as unset.
pub fn local_app_data(var: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    var("LOCALAPPDATA")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The Heliox folder that holds the runtime folder when $XDG_RUNTIME_DIR is not set, on Windows
/// (%LOCALAPPDATA%\heliox-os) and macOS (~/Library/Caches/heliox-os); None on Linux, where the runtime folder
/// itself is the Heliox folder. Removal takes it whole.
pub fn runtime_home(system: &str, local_app_data: Option<&Path>, home: &Path) -> Option<PathBuf> {
    match system {
        "windows" => Some(
            local_app_data
                .map(Path::to_path_buf)
                .unwrap_or_else(|| home.join("AppData").join("Local"))
                .join(NAME),
        ),
        "macos" => Some(home.join("Library").join("Caches").join(NAME)),
        _ => None,
    }
}

/// The daemon's runtime folder, where it writes its token files (auth_token, mcp_auth_token, neural_auth_token)
/// and its port lock, by pilot.config `_default_runtime_dir`, case by case:
///   $XDG_RUNTIME_DIR/heliox-os when that variable is set and not blank, on every system;
///   macOS: ~/Library/Caches/heliox-os/runtime;
///   Windows: %LOCALAPPDATA%\heliox-os\runtime, or ~\AppData\Local\heliox-os\runtime without it;
///   Linux and other Unix: /run/user/<uid>/heliox-os when that folder exists and is writable, else
///   $TMPDIR/heliox-os-runtime-<uid> ("/tmp" when TMPDIR is unset), with the real uid.
pub fn runtime_for(
    system: &str,
    var: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    uid: u32,
    writable: &dyn Fn(&Path) -> bool,
) -> PathBuf {
    if let Some(xdg) = var("XDG_RUNTIME_DIR").filter(|value| !value.trim().is_empty()) {
        return PathBuf::from(xdg.trim()).join(NAME);
    }
    if let Some(folder) = runtime_home(system, local_app_data(var).as_deref(), home) {
        return folder.join("runtime");
    }
    let run_user = PathBuf::from(format!("/run/user/{uid}"));
    if writable(&run_user) {
        return run_user.join(NAME);
    }
    let tmp = var("TMPDIR").unwrap_or_else(|| "/tmp".to_string());
    PathBuf::from(tmp).join(format!("{NAME}-runtime-{uid}"))
}

/// Named as the daemon names its runtime folder on Linux: "heliox-os", or "heliox-os-runtime-<uid>" under $TMPDIR.
#[cfg_attr(not(unix), allow(dead_code))] // as uninstall.rs runtime_folder_is_ours, its one caller
pub fn is_runtime_folder_name(name: &str, uid: u32) -> bool {
    name == NAME || name == format!("{NAME}-runtime-{uid}")
}

/// Where versions before 13 November 2026 looked for the token, read after the daemon's own folder so a daemon
/// started by an older setup is still found.
pub fn older_token_places(local_app_data: Option<&Path>, home: &Path) -> Vec<PathBuf> {
    let local = local_app_data
        .map(Path::to_path_buf)
        .unwrap_or_else(|| home.join("AppData").join("Local"));
    vec![
        local.join(NAME).join("runtime").join("auth_token"),
        local.join("pilot").join("runtime").join("auth_token"),
        local.join(NAME).join("auth_token"),
        local.join("pilot").join("auth_token"),
        PathBuf::from("/run/user/1000")
            .join(NAME)
            .join("auth_token"),
        PathBuf::from("/run/user/1000/pilot/auth_token"),
    ]
}

pub fn this_system() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(windows) {
        "windows"
    } else {
        "linux"
    }
}

#[cfg(unix)]
pub fn current_uid() -> u32 {
    // SAFETY: getuid has no preconditions and cannot fail.
    unsafe { libc::getuid() }
}

#[cfg(not(unix))]
pub fn current_uid() -> u32 {
    1000 // pilot.config's value where there is no uid; the Windows rule does not use it
}

/// An existing folder this process may write into (pilot.config: is_dir() and os.access(W_OK)).
#[cfg(unix)]
fn folder_writable(folder: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    if !folder.is_dir() {
        return false;
    }
    let Ok(path) = std::ffi::CString::new(folder.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: a valid NUL-terminated path; access only reads it.
    unsafe { libc::access(path.as_ptr(), libc::W_OK) == 0 }
}

#[cfg(not(unix))]
fn folder_writable(folder: &Path) -> bool {
    folder.is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "{root}" in the table is "C:" on Windows and nothing elsewhere, so a "set" XDG value is an absolute path on
    /// the machine running the test, as pilot.config's absolute check needs.
    fn fill(text: &str) -> String {
        text.replace("{root}", if cfg!(windows) { "C:" } else { "" })
    }

    fn slashes(path: &Path) -> String {
        path.to_string_lossy().replace('\\', "/")
    }

    /// The table pilot.config is tested against too (daemon/tests/test_heliox_folders_cases.py): the app finds
    /// every folder where the daemon puts it, on every system, with the XDG variables set, blank, relative or unset.
    #[test]
    fn every_folder_is_where_the_daemon_puts_it_on_every_system() {
        let table: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../daemon/tests/heliox_folders_cases.json"
        )))
        .unwrap();
        let cases = table["cases"].as_array().unwrap();
        assert!(cases.len() >= 30);
        for case in cases {
            let env: std::collections::HashMap<String, String> = case["env"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| (name.clone(), fill(value.as_str().unwrap())))
                .collect();
            let uid = case["uid"].as_u64().unwrap() as u32;
            let run_user_writable = case["run_user_writable"].as_bool().unwrap();
            let run_user = format!("/run/user/{uid}");
            let found = Folders::resolve(
                case["system"].as_str().unwrap(),
                &|name| env.get(name).cloned(),
                Path::new(case["home"].as_str().unwrap()),
                uid,
                &|folder| run_user_writable && slashes(folder) == run_user,
            );
            let expected = case["expected"].as_object().unwrap();
            assert!(!expected.is_empty(), "{case}");
            for (key, want) in expected {
                let got = match key.as_str() {
                    "config" => &found.config,
                    "data" => &found.data,
                    "state" => &found.state,
                    "runtime" => &found.runtime,
                    "cache" => &found.cache,
                    "setup" => &found.setup,
                    other => panic!("unknown folder {other} in {case}"),
                };
                assert_eq!(
                    slashes(got),
                    fill(want.as_str().unwrap()),
                    "{key} in {case}"
                );
            }
        }
    }

    #[test]
    fn the_daemon_reads_the_note_the_app_leaves_where_the_app_leaves_it() {
        let run_record = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../daemon/pilot/system/run_record.py"
        ));
        let folders = Folders::resolve("linux", &|_| None, Path::new("/home/sam"), 1000, &|_| true);
        let name = folders.app_note_file();
        let name = name.file_name().unwrap().to_string_lossy();
        assert!(run_record.contains(&format!("APP_NOTE_FILE = \"{name}\"")));
        assert!(run_record.contains(&format!("\"{}\"", crate::supervisor::NO_ANSWER_AT_STOP)));
        assert_eq!(
            folders.app_note_file().parent(),
            Some(folders.state.as_path())
        );
    }

    #[test]
    fn the_log_file_follows_xdg_state_home() {
        let home = Path::new("/home/sam");
        let state = if cfg!(windows) {
            "C:/var/state"
        } else {
            "/var/state"
        };
        let folders = Folders::resolve(
            "linux",
            &|name| (name == "XDG_STATE_HOME").then(|| state.to_string()),
            home,
            1000,
            &|_| true,
        );
        assert_eq!(
            slashes(&folders.log_file()),
            format!("{state}/heliox-os/pilot.log")
        );
        assert_eq!(
            slashes(&folders.shortcut_file()),
            "/home/sam/.config/heliox-os/shortcut.txt"
        );
        assert_eq!(slashes(&folders.env()), "/home/sam/.config/heliox-os/env");
    }

    /// A folder built by hand elsewhere is how the log panel came to read ~/.local/state while the daemon wrote under
    /// $XDG_STATE_HOME. Outside this file and outside the test modules, "heliox-os" may appear in comments and in text
    /// meant for people (a string with a space in it), never in a string that could be a path piece.
    #[test]
    fn no_other_file_builds_a_heliox_folder() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut found = Vec::new();
        let mut checked = 0;
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !name.ends_with(".rs") || name == "folders.rs" {
                continue;
            }
            checked += 1;
            let text = std::fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            // Each file's tests are one module at its end ("#[cfg(test)]" then "mod tests {"); a "#[cfg(test)]" on a
            // single item, such as main.rs's "mod real_supervision;", does not end the code.
            let end = (1..lines.len())
                .find(|&i| {
                    lines[i - 1].trim() == "#[cfg(test)]"
                        && lines[i].starts_with("mod ")
                        && lines[i].trim_end().ends_with('{')
                })
                .unwrap_or(lines.len());
            for (number, line) in lines[..end].iter().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for literal in string_literals(line) {
                    if literal.contains(NAME) && !literal.contains(' ') {
                        found.push(format!("{name}:{}: {}", number + 1, line.trim()));
                    }
                }
            }
        }
        assert!(
            checked >= 8,
            "only {checked} files read in {}",
            src.display()
        );
        assert!(
            found.is_empty(),
            "build these through folders.rs:\n{}",
            found.join("\n")
        );
    }

    #[test]
    fn the_source_check_sees_path_pieces_and_not_sentences() {
        let line =
            r#"let a = home.join("heliox-os"); let b = "Quit, or select heliox-os in the list";"#;
        assert_eq!(
            string_literals(line),
            vec!["heliox-os", "Quit, or select heliox-os in the list"]
        );
        assert_eq!(string_literals(r#"x("a\"b", 'c')"#), vec![r#"a\"b"#]);
    }

    /// The double-quoted string literals on one line (a literal that runs past the line ends at the line's end).
    fn string_literals(line: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\'' && i + 2 < bytes.len() && bytes[i + 2] == b'\'' {
                i += 3; // a char literal such as '"'
                continue;
            }
            if bytes[i] == b'"' {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += if bytes[j] == b'\\' { 2 } else { 1 };
                }
                out.push(&line[start..j.min(bytes.len())]);
                i = j + 1;
                continue;
            }
            i += 1;
        }
        out
    }
}
