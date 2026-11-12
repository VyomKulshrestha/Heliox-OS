//! First-run setup: the person chooses what to install, and every failure is shown in the app.
//!
//! The steps live in `setup/first-run.sh` (macOS, Linux) and `setup/first-run.ps1` (Windows), embedded here and
//! run with the Heliox data folder; CI runs the same scripts from an empty home folder. This module only starts
//! them, turns their `@@` lines into a state the UI shows (`setup-progress` events and `setup_status`), and starts
//! the daemon when the core is in. Nothing is installed until the person presses the button on the first screen,
//! and a failed optional group never blocks the app. See docs/FIRST_RUN.md.
//!
//! It also updates an environment an older version of the app installed (`setup/update.py`, run with the
//! environment's base Python): offered on the same screen, planned before anything changes, applied all or
//! nothing, with the daemon stopped during the update and the supervisor paused (`DAEMON_PAUSED`).

use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

/// Set while setup changes the daemon's environment (update, repair): the supervisor must not restart the daemon.
pub static DAEMON_PAUSED: AtomicBool = AtomicBool::new(false);

const SCRIPT_SH: &str = include_str!("../setup/first-run.sh");
const SCRIPT_PS1: &str = include_str!("../setup/first-run.ps1");
const PYTHON_VERSIONS: &str = include_str!("../setup/python-versions.json");
const UPDATE_PY: &str = include_str!("../setup/update.py");

/// (min, max, private): the Python versions an existing install may be, and the one Heliox downloads.
pub fn python_versions() -> (String, String, String) {
    let data: serde_json::Value = serde_json::from_str(PYTHON_VERSIONS).unwrap_or_default();
    let get = |key: &str, fallback: &str| data[key].as_str().unwrap_or(fallback).to_string();
    (
        get("min", "3.11"),
        get("max", "3.13"),
        get("private", "3.12"),
    )
}

/// Optional groups the person can choose, and the daemon extras each installs. The UI cannot ask for anything
/// outside this list.
pub fn group_extras(id: &str) -> Option<&'static str> {
    match id {
        "voice" => Some("voice"),
        "screen" => Some("vision"),
        "browser" => Some("browser"),
        "local" => Some("llamacpp"),
        "platform" => {
            if cfg!(target_os = "macos") {
                Some("macos")
            } else if cfg!(target_os = "linux") {
                Some("linux")
            } else {
                None
            }
        }
        "more" => Some("full,automation,documents,supervision,network,ssh,wasm"),
        _ => None,
    }
}

/// What pip installs: the released daemon matching this app, or a local checkout when `HELIOX_DAEMON_SOURCE` is set
/// (development and CI).
pub fn requirement(extras: Option<&str>, version: &str, source: Option<&str>) -> String {
    let extras = extras.map(|e| format!("[{e}]")).unwrap_or_default();
    match source {
        Some(path) => format!("{path}{extras}"),
        None => format!("pilot-daemon{extras}=={version}"),
    }
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Step {
    pub id: String,
    pub status: String, // "running", "done", "failed", "skipped"
    pub message: String,
    pub hint: String,
    pub detail: String, // the last line of output, shown as progress
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct PackageChange {
    pub name: String,
    pub from: String,
    pub to: String,
    pub bytes: u64,
}

/// What an update will do, from `update.py plan`; nothing changes until the person applies it.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct UpdatePlan {
    pub installed: String,
    pub target: String,
    pub groups: Vec<String>,
    pub planned: bool,
    pub changes: Vec<PackageChange>,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct PythonPlan {
    pub kind: String, // "system", "download", "none", "" before the probe
    pub version: String,
    pub path: String,
    pub download_bytes: u64,
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SetupState {
    /// "installed" (nothing to do), "needs_choice" (first screen), "running", "done", "failed",
    /// "needs_update" (an older daemon is installed), "planning" and "updating"
    pub phase: String,
    pub python: PythonPlan,
    pub steps: Vec<Step>,
    pub log_tail: Vec<String>,
    pub update: UpdatePlan,
}

impl SetupState {
    fn step(&mut self, id: &str) -> &mut Step {
        if let Some(index) = self.steps.iter().position(|s| s.id == id) {
            return &mut self.steps[index];
        }
        self.steps.push(Step {
            id: id.to_string(),
            ..Step::default()
        });
        self.steps.last_mut().expect("just pushed")
    }

    /// Apply one line of the script's output.
    pub fn apply(&mut self, line: &str) {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("@@python ") {
            let mut parts = rest.splitn(3, ' ');
            let kind = parts.next().unwrap_or("").to_string();
            let second = parts.next().unwrap_or("").to_string();
            let third = parts.next().unwrap_or("").to_string();
            self.python = if kind == "system" {
                PythonPlan {
                    kind,
                    version: second,
                    path: third,
                    ..PythonPlan::default()
                }
            } else {
                PythonPlan {
                    kind,
                    download_bytes: second.parse().unwrap_or(0),
                    source: third,
                    ..PythonPlan::default()
                }
            };
        } else if let Some(rest) = line.strip_prefix("@@step ") {
            let mut parts = rest.splitn(3, ' ');
            let id = parts.next().unwrap_or("").to_string();
            let status = parts.next().unwrap_or("").to_string();
            let message = parts.next().unwrap_or("").to_string();
            let step = self.step(&id);
            step.status = status;
            step.message = message;
            step.detail.clear();
        } else if let Some(rest) = line.strip_prefix("@@hint ") {
            let (id, hint) = rest.split_once(' ').unwrap_or((rest, ""));
            self.step(id).hint = hint.to_string();
        } else if let Some(rest) = line.strip_prefix("@@update ") {
            let words: Vec<&str> = rest.split(' ').collect();
            match words.as_slice() {
                ["installed", installed, "target", target, "groups", groups] => {
                    self.update.installed = installed.to_string();
                    self.update.target = target.to_string();
                    self.update.groups = groups
                        .split(',')
                        .filter(|g| !g.is_empty() && *g != "-")
                        .map(str::to_string)
                        .collect();
                }
                ["changes", _, "bytes", bytes] => {
                    self.update.total_bytes = bytes.parse().unwrap_or(0);
                    self.update.planned = true;
                }
                _ => {}
            }
        } else if let Some(rest) = line.strip_prefix("@@change ") {
            let words: Vec<&str> = rest.split(' ').collect();
            if let [name, from, to, bytes] = words.as_slice() {
                self.update.changes.push(PackageChange {
                    name: name.to_string(),
                    from: if *from == "-" {
                        String::new()
                    } else {
                        from.to_string()
                    },
                    to: to.to_string(),
                    bytes: bytes.parse().unwrap_or(0),
                });
            }
        } else if line.starts_with("@@") {
            // Measurements for CI (@@size, @@disk) and anything newer: not shown to the person.
        } else if !line.trim().is_empty() {
            if let Some(step) = self.steps.iter_mut().rev().find(|s| s.status == "running") {
                step.detail = line.chars().take(200).collect();
            }
            self.log_tail.push(line.to_string());
            if self.log_tail.len() > 40 {
                self.log_tail.remove(0);
            }
        }
    }

    /// The phase after the script exited with `code`: the core decides; optional groups never fail setup.
    pub fn finish(&mut self, code: Option<i32>) {
        let core_done = self
            .steps
            .iter()
            .any(|s| s.id == "core" && s.status == "done");
        for step in self.steps.iter_mut().filter(|s| s.status == "running") {
            step.status = "failed".into();
            if step.message.is_empty() {
                step.message = "Stopped before it finished.".into();
            }
        }
        self.phase = if core_done { "done" } else { "failed" }.into();
        if !core_done && !self.steps.iter().any(|s| s.status == "failed") {
            let step = self.step("core");
            step.status = "failed".into();
            step.message = format!("Setup stopped (exit code {}).", code.unwrap_or(-1));
            step.hint = "Press Retry. The details below show the last lines of output.".into();
        }
    }
}

pub struct Setup(pub Mutex<SetupState>);

fn data_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("heliox-os")
}

fn script_command(mode: &str) -> std::io::Result<Command> {
    let dir = data_dir().join("setup");
    std::fs::create_dir_all(&dir)?;
    let mut command = if cfg!(target_os = "windows") {
        let path = dir.join("first-run.ps1");
        std::fs::write(&path, SCRIPT_PS1)?;
        let mut command = Command::new("powershell");
        command.args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ]);
        command.arg(path).arg(mode);
        command
    } else {
        let path = dir.join("first-run.sh");
        std::fs::write(&path, SCRIPT_SH.replace("\r\n", "\n"))?;
        let mut command = Command::new("/bin/sh");
        command.arg(path).arg(mode);
        command
    };
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    command
        .env("HELIOX_SETUP_DIR", data_dir())
        .env("HELIOX_PYTHON_MIN", python_versions().0)
        .env("HELIOX_PYTHON_MAX", python_versions().1)
        .env("HELIOX_PYTHON_PRIVATE", python_versions().2)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    Ok(command)
}

/// The state at startup: nothing to do when the daemon's environment exists or a daemon already answers;
/// otherwise the first screen. What setting up Python will take is filled in by `probe_in_background`.
pub fn initial_state(installed: bool) -> SetupState {
    SetupState {
        phase: if installed {
            "installed"
        } else {
            "needs_choice"
        }
        .into(),
        ..SetupState::default()
    }
}

/// Find out, without installing anything, whether this machine has a suitable Python or needs a private one.
pub fn probe_in_background(app: AppHandle) {
    let needed = app
        .state::<Setup>()
        .0
        .lock()
        .map(|state| state.phase == "needs_choice")
        .unwrap_or(false);
    if !needed {
        let app = app.clone();
        std::thread::spawn(move || check_for_update(&app));
        return;
    }
    std::thread::spawn(move || {
        let result = script_command("probe").and_then(|mut command| command.output());
        let setup = app.state::<Setup>();
        let Ok(mut state) = setup.0.lock() else {
            return;
        };
        match result {
            Ok(out) => {
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    state.apply(line);
                }
            }
            Err(error) => {
                let step = state.step("python");
                step.status = "failed".into();
                step.message = format!("Setup could not start: {error}");
                step.hint =
                    "Reinstall Heliox; the setup script could not be written to your home folder."
                        .into();
                state.phase = "failed".into();
            }
        }
        emit(&app, &state);
    });
}

/// Start the daemon the setup just installed and hand it to the app, as if it had been there at launch.
fn start_daemon_after_setup(app: &AppHandle) {
    if std::net::TcpStream::connect((crate::DAEMON_HOST, crate::DAEMON_PORT)).is_ok() {
        // A group added later from Settings: the daemon is already running and picks the new parts up on restart.
        return;
    }
    let child = crate::launch_daemon();
    if child.is_none() {
        if let Ok(mut state) = app.state::<Setup>().0.lock() {
            let step = state.step("start");
            step.status = "failed".into();
            step.message = "Heliox was installed, but its background service did not start.".into();
            step.hint = "Quit and reopen Heliox. If it happens again, the log is in ~/.local/state/heliox-os.".into();
            state.phase = "failed".into();
        }
        return;
    }
    if let Ok(mut guard) = app.state::<crate::DaemonProcess>().0.lock() {
        *guard = child;
    }
    // At launch there was no daemon to watch, so the supervisor did not start; it watches this one from now on.
    crate::supervise_daemon(app.clone());
}

fn emit(app: &AppHandle, state: &SetupState) {
    let _ = app.emit("setup-progress", state.clone());
}

/// Run the setup with the chosen groups in a background thread; the daemon is started when the core is in.
pub fn run(app: AppHandle, groups: Vec<String>) -> Result<(), String> {
    {
        let setup = app.state::<Setup>();
        let mut state = setup
            .0
            .lock()
            .map_err(|_| "Setup state is unavailable.".to_string())?;
        if state.phase == "running" {
            return Err("Setup is already running.".into());
        }
        state.phase = "running".into();
        state.steps.retain(|s| s.status == "done" && s.id != "core");
        state.log_tail.clear();
    }
    let version = env!("CARGO_PKG_VERSION");
    let source = std::env::var("HELIOX_DAEMON_SOURCE").ok();
    let wanted: Vec<String> = groups
        .iter()
        .filter_map(|id| {
            group_extras(id).map(|extras| {
                format!(
                    "{id}={}",
                    requirement(Some(extras), version, source.as_deref())
                )
            })
        })
        .collect();
    let spawned = script_command("run").and_then(|mut command| {
        command
            .env(
                "HELIOX_CORE_REQ",
                requirement(None, version, source.as_deref()),
            )
            .env("HELIOX_GROUPS", wanted.join(";"))
            .spawn()
    });
    let child = match spawned {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Setup could not start: {error}");
            if let Ok(mut state) = app.state::<Setup>().0.lock() {
                let step = state.step("python");
                step.status = "failed".into();
                step.message = message.clone();
                step.hint = "Press Retry. If it happens again, reinstall Heliox.".into();
                state.phase = "failed".into();
                emit(&app, &state);
            }
            return Err(message);
        }
    };
    std::thread::spawn(move || {
        let code = stream_lines(&app, child);
        let core_done = {
            let setup = app.state::<Setup>();
            let mut state = match setup.0.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            state.finish(code);
            state.phase == "done"
        };
        if core_done {
            record_groups(&app);
            start_daemon_after_setup(&app);
        }
        if let Ok(state) = app.state::<Setup>().0.lock() {
            emit(&app, &state);
        }
    });
    Ok(())
}

/// Feed a child's stdout and stderr into the state line by line, emitting progress; returns its exit code.
fn stream_lines(app: &AppHandle, mut child: std::process::Child) -> Option<i32> {
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (sender, receiver) = std::sync::mpsc::channel::<String>();
    if let Some(err) = stderr {
        let sender = sender.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let _ = sender.send(line);
            }
        });
    }
    if let Some(out) = stdout {
        let sender = sender.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let _ = sender.send(line);
            }
        });
    }
    drop(sender);
    for line in receiver {
        let setup = app.state::<Setup>();
        if let Ok(mut state) = setup.0.lock() {
            state.apply(&line);
            emit(app, &state);
        };
    }
    child.wait().ok().and_then(|status| status.code())
}

/// The groups file: which optional groups this environment has, so an update or a repair keeps them.
fn groups_file() -> PathBuf {
    data_dir().join("setup.json")
}

fn recorded_groups() -> Vec<String> {
    std::fs::read_to_string(groups_file())
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|data| data["groups"].as_array().cloned())
        .map(|list| {
            list.iter()
                .filter_map(|g| g.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The groups recorded so far plus those this run installed (its steps that are done and are groups).
pub fn merged_groups(recorded: &[String], steps: &[Step]) -> Vec<String> {
    let mut groups: Vec<String> = recorded.to_vec();
    for step in steps {
        if step.status == "done" && group_extras(&step.id).is_some() && !groups.contains(&step.id) {
            groups.push(step.id.clone());
        }
    }
    groups
}

fn record_groups(app: &AppHandle) {
    let steps = app
        .state::<Setup>()
        .0
        .lock()
        .map(|state| state.steps.clone())
        .unwrap_or_default();
    let data = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "groups": merged_groups(&recorded_groups(), &steps),
    });
    let _ = std::fs::write(groups_file(), format!("{data:#}\n"));
}

/// The version of the daemon in the environment, from its `.dist-info` folder name; None when it is not there.
pub fn daemon_version<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<String> {
    names.into_iter().find_map(|name| {
        name.strip_prefix("pilot_daemon-")
            .and_then(|rest| rest.strip_suffix(".dist-info"))
            .map(str::to_string)
    })
}

fn version_parts(text: &str) -> Vec<u64> {
    text.split('.')
        .map(|part| {
            part.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0)
        })
        .collect()
}

/// Whether the installed daemon is older than this app (a newer one is never downgraded).
pub fn is_older(installed: &str, app: &str) -> bool {
    version_parts(installed) < version_parts(app)
}

fn site_packages(env: &std::path::Path) -> Vec<PathBuf> {
    let mut found = vec![env.join("Lib").join("site-packages")];
    if let Ok(entries) = std::fs::read_dir(env.join("lib")) {
        for entry in entries.flatten() {
            found.push(entry.path().join("site-packages"));
        }
    }
    found
}

fn installed_daemon_version() -> Option<String> {
    for dir in site_packages(&data_dir().join("env")) {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let names: Vec<String> = entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            if let Some(version) = daemon_version(names.iter().map(String::as_str)) {
                return Some(version);
            }
        }
    }
    None
}

/// The Python the environment was made from, from `pyvenv.cfg`'s `home` line.
pub fn base_python(pyvenv_cfg: &str, windows: bool) -> Option<PathBuf> {
    let home = pyvenv_cfg.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == "home").then(|| PathBuf::from(value.trim()))
    })?;
    if windows {
        return Some(home.join("python.exe"));
    }
    ["python3", "python"]
        .iter()
        .map(|name| home.join(name))
        .find(|path| path.exists())
        .or_else(|| Some(home.join("python3")))
}

fn extras_argument() -> String {
    ["platform", "voice", "screen", "browser", "local", "more"]
        .iter()
        .map(|id| format!("{id}={}", group_extras(id).unwrap_or("")))
        .collect::<Vec<_>>()
        .join(";")
}

fn update_command(mode: &str) -> std::io::Result<Command> {
    let dir = data_dir().join("setup");
    std::fs::create_dir_all(&dir)?;
    let script = dir.join("update.py");
    std::fs::write(&script, UPDATE_PY)?;
    let env = data_dir().join("env");
    let cfg = std::fs::read_to_string(env.join("pyvenv.cfg"))?;
    let python = base_python(&cfg, cfg!(target_os = "windows")).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "the environment does not name its Python",
        )
    })?;
    let mut command = Command::new(python);
    command
        .arg(script)
        .arg(mode)
        .arg("--env")
        .arg(env)
        .arg("--version")
        .arg(env!("CARGO_PKG_VERSION"))
        .arg("--groups-file")
        .arg(groups_file())
        .arg("--extras")
        .arg(extras_argument());
    if let Ok(source) = std::env::var("HELIOX_DAEMON_SOURCE") {
        command.arg("--source").arg(source);
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    Ok(command)
}

/// At start, when Heliox is installed: an environment from an older version of the app is offered an update.
fn check_for_update(app: &AppHandle) {
    let Some(installed) = installed_daemon_version() else {
        return;
    };
    if !is_older(&installed, env!("CARGO_PKG_VERSION")) {
        return;
    }
    if let Ok(mut state) = app.state::<Setup>().0.lock() {
        state.phase = "needs_update".into();
        state.update = UpdatePlan {
            installed,
            target: env!("CARGO_PKG_VERSION").into(),
            ..UpdatePlan::default()
        };
        emit(app, &state);
    }
}

fn fail_update(app: &AppHandle, message: String) -> Result<(), String> {
    if let Ok(mut state) = app.state::<Setup>().0.lock() {
        let step = state.step("update");
        step.status = "failed".into();
        step.message = message.clone();
        step.hint = "Nothing was changed. Press Retry.".into();
        state.phase = "failed".into();
        emit(app, &state);
    }
    Err(message)
}

/// Work out what the update changes (packages and download size). Changes nothing.
pub fn update_plan(app: AppHandle) -> Result<(), String> {
    {
        let setup = app.state::<Setup>();
        let mut state = setup
            .0
            .lock()
            .map_err(|_| "Setup state is unavailable.".to_string())?;
        state.phase = "planning".into();
        state.update.changes.clear();
        state.update.planned = false;
        state.steps.clear();
        state.log_tail.clear();
    }
    let child = match update_command("plan").and_then(|mut command| command.spawn()) {
        Ok(child) => child,
        Err(error) => return fail_update(&app, format!("The update could not start: {error}")),
    };
    std::thread::spawn(move || {
        let code = stream_lines(&app, child);
        if let Ok(mut state) = app.state::<Setup>().0.lock() {
            state.phase = if code == Some(0) && state.update.planned {
                "needs_update"
            } else {
                "failed"
            }
            .into();
            emit(&app, &state);
        }
    });
    Ok(())
}

/// Apply the update: stop the daemon (the supervisor is paused), update all or nothing, start it again.
pub fn update_apply(app: AppHandle) -> Result<(), String> {
    {
        let setup = app.state::<Setup>();
        let mut state = setup
            .0
            .lock()
            .map_err(|_| "Setup state is unavailable.".to_string())?;
        if state.phase == "updating" {
            return Err("The update is already running.".into());
        }
        state.phase = "updating".into();
        state.steps.clear();
        state.log_tail.clear();
    }
    DAEMON_PAUSED.store(true, Ordering::SeqCst);
    if let Ok(mut guard) = app.state::<crate::DaemonProcess>().0.lock() {
        if let Some(mut child) = guard.take() {
            crate::supervisor::kill_tree(&mut child);
            let _ = child.wait();
        }
    }
    let child = match update_command("apply").and_then(|mut command| command.spawn()) {
        Ok(child) => child,
        Err(error) => {
            DAEMON_PAUSED.store(false, Ordering::SeqCst);
            start_daemon_after_setup(&app);
            return fail_update(&app, format!("The update could not start: {error}"));
        }
    };
    std::thread::spawn(move || {
        let code = stream_lines(&app, child);
        if let Ok(mut state) = app.state::<Setup>().0.lock() {
            state.phase = if code == Some(0) { "done" } else { "failed" }.into();
        }
        DAEMON_PAUSED.store(false, Ordering::SeqCst);
        // Updated or put back as it was: either way the daemon starts again from the environment.
        start_daemon_after_setup(&app);
        if let Ok(state) = app.state::<Setup>().0.lock() {
            emit(&app, &state);
        }
    });
    Ok(())
}

#[tauri::command]
pub fn setup_update_plan(app: AppHandle) -> Result<(), String> {
    update_plan(app)
}

#[tauri::command]
pub fn setup_update_apply(app: AppHandle) -> Result<(), String> {
    update_apply(app)
}

/// "Not now": keep using the installed version for this run; the offer comes back at the next start.
#[tauri::command]
pub fn setup_update_later(app: AppHandle) -> SetupState {
    let setup = app.state::<Setup>();
    let Ok(mut state) = setup.0.lock() else {
        return SetupState::default();
    };
    if state.phase == "needs_update" || state.phase == "failed" {
        state.phase = "installed".into();
    }
    state.clone()
}

#[tauri::command]
pub fn setup_status(app: AppHandle) -> SetupState {
    app.state::<Setup>()
        .0
        .lock()
        .map(|state| state.clone())
        .unwrap_or_default()
}

#[tauri::command]
pub fn setup_run(app: AppHandle, groups: Vec<String>) -> Result<(), String> {
    run(app, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_lines_become_steps_hints_and_progress() {
        let mut state = SetupState::default();
        state.apply("@@python download 16682639 https://github.com/astral-sh/uv/releases/download/0.13.5/uv.tar.gz");
        assert_eq!(state.python.kind, "download");
        assert_eq!(state.python.download_bytes, 16682639);
        state.apply("@@step core running Installing Heliox's core");
        state.apply("Collecting numpy==2.2.6");
        state.apply("@@size core 112640");
        assert_eq!(state.steps[0].detail, "Collecting numpy==2.2.6");
        state.apply("@@step core done Core installed");
        state.apply("@@step voice failed Could not be installed.");
        state.apply("@@hint voice Press Retry; the rest of Heliox works without it.");
        state.finish(Some(2));
        assert_eq!(state.phase, "done"); // an optional group failing does not fail setup
        assert_eq!(
            state.steps[1].hint,
            "Press Retry; the rest of Heliox works without it."
        );
    }

    #[test]
    fn a_core_that_did_not_finish_fails_setup_with_a_reason() {
        let mut state = SetupState::default();
        state.apply("@@python system 3.12.3 /usr/bin/python3.12");
        assert_eq!(state.python.path, "/usr/bin/python3.12");
        state.apply("@@step core running Installing Heliox's core");
        state.finish(Some(1));
        assert_eq!(state.phase, "failed");
        assert_eq!(state.steps[0].status, "failed");
        let mut silent = SetupState::default();
        silent.finish(Some(9));
        assert!(silent.steps[0].message.contains("exit code 9"));
    }

    #[test]
    fn an_older_daemon_is_found_and_offered_an_update_never_a_downgrade() {
        let names = [
            "numpy-2.2.6.dist-info",
            "pilot_daemon-0.15.0.dist-info",
            "pilot",
        ];
        assert_eq!(daemon_version(names), Some("0.15.0".to_string()));
        assert_eq!(daemon_version(["numpy-2.2.6.dist-info"]), None);
        assert!(is_older("0.15.0", "0.16.0"));
        assert!(is_older("0.9.1", "0.16.0"));
        assert!(!is_older("0.16.0", "0.16.0"));
        assert!(!is_older("0.17.0", "0.16.0"));
    }

    #[test]
    fn the_base_python_comes_from_pyvenv_cfg() {
        let cfg = "home = C:\\Python312\ninclude-system-site-packages = false\n";
        assert_eq!(
            base_python(cfg, true),
            Some(PathBuf::from("C:\\Python312").join("python.exe"))
        );
        assert_eq!(base_python("version = 3.12\n", true), None);
    }

    #[test]
    fn update_lines_become_the_plan_and_groups_are_kept() {
        let mut state = SetupState::default();
        state.apply("@@update installed 0.15.0 target 0.16.0 groups platform,voice");
        state.apply("@@change pilot-daemon 0.15.0 0.16.0 812345");
        state.apply("@@change mcp - 2.1.0 0");
        state.apply("@@update changes 2 bytes 812345");
        assert_eq!(state.update.installed, "0.15.0");
        assert_eq!(state.update.groups, vec!["platform", "voice"]);
        assert!(state.update.planned);
        assert_eq!(state.update.changes.len(), 2);
        assert_eq!(state.update.changes[1].from, "");
        assert_eq!(state.update.total_bytes, 812345);
        let steps = vec![
            Step {
                id: "core".into(),
                status: "done".into(),
                ..Step::default()
            },
            Step {
                id: "voice".into(),
                status: "done".into(),
                ..Step::default()
            },
            Step {
                id: "browser".into(),
                status: "failed".into(),
                ..Step::default()
            },
        ];
        assert_eq!(
            merged_groups(&["platform".to_string()], &steps),
            vec!["platform", "voice"]
        );
    }

    #[test]
    fn the_python_range_comes_from_the_shared_file() {
        let (min, max, private) = python_versions();
        let data: serde_json::Value = serde_json::from_str(PYTHON_VERSIONS).unwrap();
        assert_eq!(min, data["min"].as_str().unwrap());
        assert_eq!(max, data["max"].as_str().unwrap());
        assert_eq!(private, data["private"].as_str().unwrap());
    }

    #[test]
    fn only_known_groups_install_and_requirements_pin_the_version() {
        assert_eq!(group_extras("voice"), Some("voice"));
        assert_eq!(group_extras("rm -rf"), None);
        assert_eq!(requirement(None, "0.16.0", None), "pilot-daemon==0.16.0");
        assert_eq!(
            requirement(Some("voice"), "0.16.0", None),
            "pilot-daemon[voice]==0.16.0"
        );
        assert_eq!(
            requirement(Some("vision"), "0.16.0", Some("/src/daemon")),
            "/src/daemon[vision]"
        );
    }
}
