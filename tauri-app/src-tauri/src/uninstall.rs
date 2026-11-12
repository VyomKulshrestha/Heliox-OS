//! "Remove Heliox's files": everything Heliox put on this computer, listed with sizes before anything is deleted.
//!
//! The list is fixed by this module from the home folder and the system's app-data folders, never taken from the
//! UI, and deletion never follows a link out of those folders: a symbolic link (or Windows junction) inside them is
//! removed as a link, and what it points to is left alone. Every location is a folder whose name is Heliox's alone,
//! or, where a folder's name could belong to another program, the exact files Heliox wrote in it (the folder goes
//! only if that leaves it empty). With no known home folder nothing is listed or removed, and every path must be
//! absolute and at least two folders below the root as a second lock.
//!
//! The person chooses between removing the app's files and keeping their data (settings, memory, timeline,
//! receipts, undo copies, persona, quarantined files, voice calibration, and the keys in the system credential
//! store), or removing everything, in which case the keys Heliox saved in the credential store are removed first by
//! the daemon's own code (`pilot.system.credential_cleanup`, which knows the services and accounts), and whatever
//! remains is listed by name. Things Heliox cannot or should not remove itself are listed with what the person can
//! do. See docs/FIRST_RUN.md.

use serde::Serialize;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CREDENTIAL_SERVICES: &str = include_str!("../setup/credential_services.json");
const CREDENTIAL_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Rebuilt by setup or downloaded again on demand: environment, private Python, tools, logs, caches, models.
    App,
    /// The person's own: settings, memory, timeline, receipts, undo copies, persona, quarantine.
    Data,
}

/// What is removed at a location.
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// A folder whose name is Heliox's alone: removed whole.
    Folder,
    /// A folder another program could also use: only these files, then the folder if it is left empty.
    Files(Vec<&'static str>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub path: PathBuf,
    pub label: String,
    pub kind: Kind,
    pub target: Target,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Location {
    pub path: PathBuf,
    pub label: String,
    pub kind: Kind,
    pub bytes: u64,
}

/// Where the system keeps per-app data, for the app-data folders of the desktop app itself.
pub struct SystemDirs {
    pub home: PathBuf,
    /// Windows: %LOCALAPPDATA%. Unused elsewhere.
    pub local_app_data: Option<PathBuf>,
    /// "windows", "macos" or "linux".
    pub os: &'static str,
}

const APP_ID: &str = "com.helioxos.app";

/// Folders inside ~/.config/heliox-os that setup builds and can build again; the rest of that folder is data.
pub const CONFIG_APP_CHILDREN: [&str; 6] =
    ["env", "python", "tools", "uv-cache", "setup", "env.broken"];

fn place(path: PathBuf, label: &str, kind: Kind, target: Target) -> Place {
    Place {
        path,
        label: label.into(),
        kind,
        target,
    }
}

/// Every place Heliox writes, with its kind and what is removed there. Pure: computed from the system folders,
/// tested on every system. Beside each: why the name is Heliox's alone, or why only named files are removed.
pub fn locations(dirs: &SystemDirs) -> Vec<Place> {
    use Kind::{App, Data};
    use Target::{Files, Folder};
    let home = &dirs.home;
    // "heliox-os" under .config, .local/share and .local/state: Heliox's name, used by the daemon and the app.
    let config = home.join(".config").join("heliox-os");
    let mut out: Vec<Place> = CONFIG_APP_CHILDREN
        .iter()
        .map(|child| place(config.join(child), &format!("Setup: {child}"), App, Folder))
        .collect();
    out.push(place(
        config.clone(),
        "Settings and keys (config.toml, restrictions, shortcut)",
        Data,
        Folder,
    ));
    let data = home.join(".local").join("share").join("heliox-os");
    out.push(place(
        data.join("models"),
        "Downloaded speech models",
        App,
        Folder,
    ));
    out.push(place(
        data,
        "Memory, timeline, receipts and undo copies",
        Data,
        Folder,
    ));
    out.push(place(
        home.join(".local").join("state").join("heliox-os"),
        "Logs",
        App,
        Folder,
    ));
    // "heliox" under .cache and ".heliox": Heliox's name (cognitive caches, persona, plugins, quarantine).
    out.push(place(
        home.join(".cache").join("heliox"),
        "Persona, voice calibration and other learned data",
        Data,
        Folder,
    ));
    out.push(place(
        home.join(".heliox"),
        "Plugins, screenshots, persona file and quarantined files",
        Data,
        Folder,
    ));
    // ".pilot" is a name another program can use: only the two files the daemon writes there
    // (agents/self_heal.py and system/context.py), and the folder only if that leaves it empty.
    out.push(place(
        home.join(".pilot"),
        "Usage patterns and failure memory",
        Data,
        Files(vec!["usage_patterns.json", "failure_memory.json"]),
    ));
    match dirs.os {
        "windows" => {
            let local = dirs
                .local_app_data
                .clone()
                .unwrap_or_else(|| home.join("AppData").join("Local"));
            // "heliox-os" (the daemon's runtime folder) and "com.helioxos.app" (the app's identifier): Heliox's.
            out.push(place(local.join("heliox-os"), "Runtime files", App, Folder));
            out.push(place(
                local.join(APP_ID),
                "The app's web view data",
                App,
                Folder,
            ));
        }
        "macos" => {
            let library = home.join("Library");
            out.push(place(
                library.join("Caches").join("heliox-os"),
                "Runtime files",
                App,
                Folder,
            ));
            for (folder, label) in [
                ("Application Support", "The app's data"),
                ("Caches", "The app's caches"),
                ("WebKit", "The app's web view data"),
            ] {
                out.push(place(library.join(folder).join(APP_ID), label, App, Folder));
            }
        }
        _ => {
            out.push(place(
                home.join(".local").join("share").join(APP_ID),
                "The app's web view data",
                App,
                Folder,
            ));
            out.push(place(
                home.join(".cache").join(APP_ID),
                "The app's caches",
                App,
                Folder,
            ));
        }
    }
    out
}

/// The second lock: an absolute path at least two folders below the root (never "/", "C:\\" or a bare home name).
pub fn safe_location(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .filter(|c| matches!(c, Component::Normal(_)))
            .count()
            >= 2
}

/// What a removal deletes: with `keep_data`, only app places; otherwise everything.
pub fn chosen(all: &[Place], keep_data: bool) -> Vec<Place> {
    all.iter()
        .filter(|p| !keep_data || p.kind == Kind::App)
        .cloned()
        .collect()
}

/// Bytes under `path`, not following links (a link counts as itself).
pub fn size_of(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .map(|entries| entries.flatten().map(|entry| size_of(&entry.path())).sum())
        .unwrap_or(0)
}

fn target_size(place: &Place) -> u64 {
    match &place.target {
        Target::Folder => size_of(&place.path),
        Target::Files(files) => files.iter().map(|f| size_of(&place.path.join(f))).sum(),
    }
}

/// Remove one path without following links out of it: a link is removed as a link.
pub fn remove(path: &Path) -> std::io::Result<()> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if meta.file_type().is_symlink() || !meta.is_dir() {
        // On Windows a directory symlink or junction is removed with remove_dir, a file link with remove_file.
        return std::fs::remove_file(path).or_else(|_| std::fs::remove_dir(path));
    }
    // remove_dir_all does not follow symbolic links or junctions inside the folder (Rust 1.58 and later).
    std::fs::remove_dir_all(path)
}

fn remove_place(place: &Place) -> std::io::Result<()> {
    match &place.target {
        Target::Folder => remove(&place.path),
        Target::Files(files) => {
            for file in files {
                remove(&place.path.join(file))?;
            }
            // Only an empty folder: another program's files there stay, and so does the folder.
            if std::fs::read_dir(&place.path)
                .map(|mut entries| entries.next().is_none())
                .unwrap_or(false)
            {
                std::fs::remove_dir(&place.path)?;
            }
            Ok(())
        }
    }
}

/// The system folders, or None when the home folder is not known (then nothing is listed or removed).
pub fn system_dirs() -> Option<SystemDirs> {
    dirs_from_home(dirs::home_dir())
}

/// No home folder, or one that is not an absolute path: None, and nothing is listed or removed. Pure.
pub fn dirs_from_home(home: Option<PathBuf>) -> Option<SystemDirs> {
    let home = home.filter(|h| h.is_absolute())?;
    Some(SystemDirs {
        home,
        local_app_data: std::env::var_os("LOCALAPPDATA").map(PathBuf::from),
        os: if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "linux"
        },
    })
}

/// What is there now, with sizes; the settings and data folders leave out what is listed on its own.
pub fn plan(dirs: &SystemDirs) -> Vec<Location> {
    let all = locations(dirs);
    let config = dirs.home.join(".config").join("heliox-os");
    let models = dirs
        .home
        .join(".local")
        .join("share")
        .join("heliox-os")
        .join("models");
    all.iter()
        .filter(|p| safe_location(&p.path) && std::fs::symlink_metadata(&p.path).is_ok())
        .filter(|p| target_size(p) > 0 || matches!(p.target, Target::Folder))
        .map(|p| {
            let mut bytes = target_size(p);
            if p.path == config {
                let children: u64 = CONFIG_APP_CHILDREN
                    .iter()
                    .map(|c| size_of(&config.join(c)))
                    .sum();
                bytes = bytes.saturating_sub(children);
            }
            if Some(p.path.as_path()) == models.parent() {
                bytes = bytes.saturating_sub(size_of(&models));
            }
            Location {
                path: p.path.clone(),
                label: p.label.clone(),
                kind: p.kind,
                bytes,
            }
        })
        .collect()
}

/// What Heliox does not remove itself, and what the person can do.
pub fn manual_steps(os: &str) -> Vec<String> {
    let mut steps = vec![
        "Uninstall the Heliox app itself the way your system removes apps.".to_string(),
        "Shared caches are left alone because other programs use them: pip's download cache and the Hugging Face \
         cache (~/.cache/huggingface), which holds the local voice and Whisper models."
            .to_string(),
        "Heliox installs no login item, service or scheduled task, so there is nothing to remove there.".to_string(),
    ];
    match os {
        "macos" => steps.push(
            "Privacy permissions (Microphone, Accessibility, Screen Recording, Automation) stay listed in System \
             Settings > Privacy & Security until you remove Heliox-OS there, or run: tccutil reset All com.helioxos.app"
                .to_string(),
        ),
        "linux" => steps.push(
            "On Wayland, the global shortcut is stored by your desktop: remove \"Show or hide Heliox\" in its \
             keyboard shortcut settings."
                .to_string(),
        ),
        _ => {}
    }
    steps
}

// ── the credential store ──

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Credentials {
    /// "kept" (the person kept their data), "removed" (the daemon's cleanup ran), "listed" (it could not run).
    pub outcome: String,
    pub removed: Vec<String>,
    pub remaining: Vec<String>,
    pub how: String,
}

pub fn credential_services() -> Vec<String> {
    serde_json::from_str::<serde_json::Value>(CREDENTIAL_SERVICES)
        .ok()
        .and_then(|data| data["services"].as_array().cloned())
        .map(|list| {
            list.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn credential_how(os: &str) -> String {
    match os {
        "windows" => "Open Credential Manager > Windows Credentials and remove the entries for the services named.",
        "macos" => "Open Keychain Access, search the login keychain for each service named and delete it.",
        _ => "Open Passwords and Keys (Seahorse) or your desktop's wallet, search for each service named and delete it.",
    }
    .to_string()
}

/// The report when the daemon's cleanup could not run: every Heliox service, by name, to remove by hand.
pub fn credentials_listed(os: &str) -> Credentials {
    Credentials {
        outcome: "listed".into(),
        removed: Vec::new(),
        remaining: credential_services()
            .into_iter()
            .map(|s| format!("{s} (every entry)"))
            .collect(),
        how: credential_how(os),
    }
}

/// Read the cleanup's JSON line; anything unreadable means it did not run.
pub fn parse_credentials(output: &str, os: &str) -> Credentials {
    let Some(data) = output
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<serde_json::Value>(line.trim()).ok())
    else {
        return credentials_listed(os);
    };
    let list = |key: &str| -> Vec<String> {
        data[key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    Credentials {
        outcome: "removed".into(),
        removed: list("removed"),
        remaining: list("remaining"),
        how: data["how"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| credential_how(os)),
    }
}

/// Run the daemon's own cleanup with the environment's Python, bounded in time.
fn remove_credentials(home: &Path, os: &str) -> Credentials {
    let env = home.join(".config").join("heliox-os").join("env");
    let python = if os == "windows" {
        env.join("Scripts").join("python.exe")
    } else {
        env.join("bin").join("python")
    };
    let mut command = Command::new(python);
    command
        .args(["-m", "pilot.system.credential_cleanup", "remove"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let Ok(mut child) = command.spawn() else {
        return credentials_listed(os);
    };
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < CREDENTIAL_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(100))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return credentials_listed(os);
            }
        }
    }
    let mut output = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        use std::io::Read;
        let _ = stdout.read_to_string(&mut output);
    }
    parse_credentials(&output, os)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RemovalReport {
    pub removed: Vec<String>,
    pub failed: Vec<String>,
    pub freed_bytes: u64,
    pub kept_data: bool,
    pub credentials: Credentials,
    pub manual_steps: Vec<String>,
}

/// Why a removal failed, in the person's terms.
pub fn failure_note(path: &Path, os: &str, error: &str) -> String {
    let held = os == "windows" && path.ends_with(APP_ID);
    if held {
        format!(
            "{}: the running app holds this folder open; it goes when you uninstall the app.",
            path.display()
        )
    } else {
        format!("{}: {error}", path.display())
    }
}

/// Delete the chosen places. The paths come only from `locations`, never from the caller. With "everything", the
/// keys in the credential store go first, through the daemon's code, while its environment still exists.
pub fn remove_files(dirs: &SystemDirs, keep_data: bool, credentials: Credentials) -> RemovalReport {
    let mut report = RemovalReport {
        kept_data: keep_data,
        credentials,
        manual_steps: manual_steps(dirs.os),
        ..RemovalReport::default()
    };
    for place in chosen(&locations(dirs), keep_data) {
        if !safe_location(&place.path) {
            report.failed.push(format!(
                "{}: refused, not a safe location",
                place.path.display()
            ));
            continue;
        }
        if std::fs::symlink_metadata(&place.path).is_err() {
            continue;
        }
        let bytes = target_size(&place);
        match remove_place(&place) {
            Ok(()) => {
                report.freed_bytes += bytes;
                report.removed.push(place.path.display().to_string());
            }
            Err(error) => {
                report
                    .failed
                    .push(failure_note(&place.path, dirs.os, &error.to_string()))
            }
        }
    }
    report
}

#[derive(Clone, Debug, Serialize)]
pub struct FilesPlan {
    pub home_known: bool,
    pub message: String,
    pub locations: Vec<Location>,
    pub manual_steps: Vec<String>,
    pub credential_services: Vec<String>,
}

const NO_HOME: &str =
    "Heliox cannot tell which folder is your home folder, so it lists and removes nothing.";

#[tauri::command]
pub fn heliox_files_plan() -> FilesPlan {
    match system_dirs() {
        Some(dirs) => FilesPlan {
            home_known: true,
            message: String::new(),
            locations: plan(&dirs),
            manual_steps: manual_steps(dirs.os),
            credential_services: credential_services(),
        },
        None => FilesPlan {
            home_known: false,
            message: NO_HOME.into(),
            locations: Vec::new(),
            manual_steps: Vec::new(),
            credential_services: Vec::new(),
        },
    }
}

/// Remove Heliox's files. `confirm` must be "remove", which the screen sends only after the person ticked that
/// they understand. The daemon is stopped first, with the supervisor paused for good: there is nothing left to
/// restart.
#[tauri::command]
pub fn heliox_files_remove(
    app: tauri::AppHandle,
    keep_data: bool,
    confirm: String,
) -> Result<RemovalReport, String> {
    use tauri::Manager;
    if confirm != "remove" {
        return Err("Nothing was removed: the removal was not confirmed.".into());
    }
    let Some(dirs) = system_dirs() else {
        return Err(NO_HOME.into());
    };
    crate::setup::DAEMON_PAUSED.store(true, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut guard) = app.state::<crate::DaemonProcess>().0.lock() {
        if let Some(mut child) = guard.take() {
            crate::supervisor::kill_tree(&mut child);
            let _ = child.wait();
        }
    }
    Ok(remove_heliox(&dirs, keep_data))
}

/// The removal itself, once the daemon is stopped: with "everything", the saved keys through the daemon's own
/// code while its environment still exists, then the files.
pub fn remove_heliox(dirs: &SystemDirs, keep_data: bool) -> RemovalReport {
    let credentials = if keep_data {
        Credentials {
            outcome: "kept".into(),
            remaining: credential_services(),
            how: credential_how(dirs.os),
            ..Credentials::default()
        }
    } else {
        remove_credentials(&dirs.home, dirs.os)
    };
    remove_files(dirs, keep_data, credentials)
}

/// Quit the app after a removal: nothing is left for it to run.
#[tauri::command]
pub fn heliox_quit(app: tauri::AppHandle) {
    app.exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs_for(home: &Path, os: &'static str) -> SystemDirs {
        SystemDirs {
            home: home.to_path_buf(),
            local_app_data: Some(home.join("AppData").join("Local")),
            os,
        }
    }

    #[test]
    fn every_location_is_inside_the_home_folder_and_classified() {
        for os in ["windows", "macos", "linux"] {
            let home = std::env::temp_dir().join("home-sam");
            let all = locations(&dirs_for(&home, os));
            assert!(all.iter().all(|p| p.path.starts_with(&home)), "{os}");
            assert!(all.iter().all(|p| safe_location(&p.path)), "{os}");
            let data: Vec<_> = all.iter().filter(|p| p.kind == Kind::Data).collect();
            assert!(data.iter().any(|p| p.path.ends_with(".heliox")), "{os}");
            let pilot = all.iter().find(|p| p.path.ends_with(".pilot")).unwrap();
            assert_eq!(
                pilot.target,
                Target::Files(vec!["usage_patterns.json", "failure_memory.json"])
            );
        }
        let mac = locations(&dirs_for(&std::env::temp_dir().join("u"), "macos"));
        assert!(mac.iter().any(|p| p
            .path
            .ends_with(Path::new("Library").join("WebKit").join(APP_ID))));
    }

    #[test]
    fn only_absolute_paths_two_folders_deep_are_safe() {
        assert!(!safe_location(Path::new(".config/heliox-os")));
        assert!(!safe_location(Path::new("./.heliox")));
        assert!(!safe_location(
            std::env::temp_dir().ancestors().last().unwrap()
        ));
        let root = std::env::temp_dir()
            .ancestors()
            .last()
            .unwrap()
            .to_path_buf();
        assert!(!safe_location(&root.join("home")));
        assert!(safe_location(&root.join("home").join("sam")));
    }

    #[test]
    fn without_a_known_home_folder_nothing_is_listed_or_removed() {
        assert!(dirs_from_home(None).is_none());
        assert!(dirs_from_home(Some(PathBuf::from("."))).is_none());
        assert!(dirs_from_home(Some(PathBuf::from("relative/home"))).is_none());
        assert!(dirs_from_home(Some(std::env::temp_dir())).is_some());
    }

    #[test]
    fn keeping_data_removes_only_app_files() {
        let all = locations(&dirs_for(&std::env::temp_dir().join("h"), "linux"));
        let keep = chosen(&all, true);
        assert!(keep.iter().any(|p| p.path.ends_with("env")));
        assert!(keep.iter().any(|p| p.path.ends_with("models")));
        assert!(!keep
            .iter()
            .any(|p| p.path.ends_with(".heliox") || p.path.ends_with(".pilot")));
        assert_eq!(chosen(&all, false).len(), all.len());
    }

    #[test]
    fn removal_keeps_data_when_asked_never_follows_a_link_and_spares_another_programs_pilot_files()
    {
        let root = std::env::temp_dir().join(format!("heliox-uninstall-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let home = root.join("home");
        let outside = root.join("outside");
        let config = home.join(".config").join("heliox-os");
        std::fs::create_dir_all(config.join("env").join("lib")).unwrap();
        std::fs::write(
            config.join("env").join("lib").join("big.bin"),
            vec![0u8; 4096],
        )
        .unwrap();
        std::fs::write(config.join("config.toml"), "theme = 'dark'\n").unwrap();
        std::fs::create_dir_all(home.join(".heliox").join("quarantine")).unwrap();
        std::fs::write(
            home.join(".heliox").join("quarantine").join("kept.txt"),
            "mine",
        )
        .unwrap();
        let pilot = home.join(".pilot");
        std::fs::create_dir_all(&pilot).unwrap();
        std::fs::write(pilot.join("usage_patterns.json"), "{}").unwrap();
        std::fs::write(pilot.join("failure_memory.json"), "{}").unwrap();
        std::fs::write(pilot.join("other-program.conf"), "not Heliox's").unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("precious.txt"), "not Heliox's").unwrap();
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(&outside, config.join("env").join("link"));
        #[cfg(windows)]
        let _ = std::os::windows::fs::symlink_dir(&outside, config.join("env").join("link"));

        let dirs = dirs_for(&home, "linux");
        let listed = plan(&dirs);
        assert!(
            listed
                .iter()
                .find(|l| l.path.ends_with("env"))
                .unwrap()
                .bytes
                >= 4096
        );
        assert!(listed.iter().find(|l| l.path == config).unwrap().bytes < 4096);

        let report = remove_files(&dirs, true, Credentials::default());
        assert!(report.failed.is_empty(), "{:?}", report.failed);
        assert!(!config.join("env").exists());
        assert!(config.join("config.toml").exists(), "data kept");
        assert!(home
            .join(".heliox")
            .join("quarantine")
            .join("kept.txt")
            .exists());
        assert!(
            outside.join("precious.txt").exists(),
            "nothing outside is touched"
        );

        let report = remove_files(&dirs, false, Credentials::default());
        assert!(report.failed.is_empty(), "{:?}", report.failed);
        assert!(!config.exists() && !home.join(".heliox").exists());
        assert!(
            !pilot.join("usage_patterns.json").exists()
                && !pilot.join("failure_memory.json").exists()
        );
        assert!(
            pilot.join("other-program.conf").exists(),
            "another program's file in .pilot stays"
        );
        assert!(outside.join("precious.txt").exists());

        std::fs::remove_file(pilot.join("other-program.conf")).unwrap();
        std::fs::write(pilot.join("usage_patterns.json"), "{}").unwrap();
        remove_files(&dirs, false, Credentials::default());
        assert!(!pilot.exists(), "an empty .pilot goes");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_credential_report_comes_from_the_daemon_or_lists_every_service() {
        let out = "warning\n{\"removed\": [\"pilot-ai-command-center / anthropic\"], \"remaining\": [], \"how\": \"Open Keychain Access\"}\n";
        let parsed = parse_credentials(out, "macos");
        assert_eq!(parsed.outcome, "removed");
        assert_eq!(parsed.removed, vec!["pilot-ai-command-center / anthropic"]);
        let listed = parse_credentials("Traceback (most recent call last)", "windows");
        assert_eq!(listed.outcome, "listed");
        assert!(listed
            .remaining
            .iter()
            .any(|s| s.starts_with("pilot-ai-command-center")));
        assert!(listed.how.contains("Credential Manager"));
        assert_eq!(credential_services().len(), 4);
    }

    #[test]
    fn a_failure_on_the_web_view_folder_on_windows_says_why() {
        let note = failure_note(
            &std::env::temp_dir().join("Local").join(APP_ID),
            "windows",
            "Access is denied.",
        );
        assert!(note.contains("goes when you uninstall the app"));
        assert!(failure_note(Path::new("/x/y"), "linux", "busy").ends_with(": busy"));
    }

    /// A real removal from this computer's home folder, after a real first run: CI only (the job sets
    /// HELIOX_REAL_REMOVAL to "keep" or "everything", and GITHUB_ACTIONS is "true" there), never on a person's
    /// machine. Prints the report as one line for the job's log.
    #[test]
    #[ignore = "removes Heliox's files from this computer's home folder; run by CI after a real first run"]
    fn real_removal_from_a_set_up_home() {
        let Ok(mode) = std::env::var("HELIOX_REAL_REMOVAL") else {
            return;
        };
        if std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true") {
            return;
        }
        let keep_data = match mode.as_str() {
            "keep" => true,
            "everything" => false,
            other => panic!("HELIOX_REAL_REMOVAL must be keep or everything, not {other:?}"),
        };
        let dirs = system_dirs().expect("the runner has a home folder");
        let env = dirs.home.join(".config").join("heliox-os").join("env");
        let had_env = env.exists();
        let before = plan(&dirs);
        println!(
            "REMOVAL_PLAN {}",
            serde_json::to_string(&before).unwrap_or_default()
        );
        let report = remove_heliox(&dirs, keep_data);
        println!(
            "REMOVAL_REPORT {}",
            serde_json::to_string(&report).unwrap_or_default()
        );
        assert!(report.failed.is_empty(), "{:?}", report.failed);
        assert!(!env.exists(), "the environment is removed in both modes");
        if keep_data {
            assert_eq!(report.credentials.outcome, "kept");
            for location in before.iter().filter(|l| l.kind == Kind::Data) {
                assert!(location.path.exists(), "kept: {}", location.path.display());
            }
        } else {
            if had_env {
                assert_eq!(
                    report.credentials.outcome, "removed",
                    "the daemon's cleanup ran"
                );
                assert!(
                    report.credentials.remaining.is_empty(),
                    "{:?}",
                    report.credentials.remaining
                );
            } else {
                // The app's files went earlier with "keep my data": nothing can run the daemon's cleanup now, so
                // every service is named for the person to remove by hand.
                assert_eq!(report.credentials.outcome, "listed");
                assert_eq!(
                    report.credentials.remaining.len(),
                    credential_services().len()
                );
            }
            for location in &before {
                assert!(!location.path.exists(), "left: {}", location.path.display());
            }
        }
    }

    #[test]
    fn manual_steps_name_what_heliox_leaves_to_the_person() {
        assert!(manual_steps("macos")
            .iter()
            .any(|s| s.contains("tccutil reset All com.helioxos.app")));
        assert!(manual_steps("linux").iter().any(|s| s.contains("Wayland")));
    }
}
