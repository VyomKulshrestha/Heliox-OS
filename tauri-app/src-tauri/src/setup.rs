//! First-run setup: the person chooses what to install, and every failure is shown in the app.
//!
//! The steps live in `setup/first-run.sh` (macOS, Linux) and `setup/first-run.ps1` (Windows), embedded here and
//! run with the Heliox data folder; CI runs the same scripts from an empty home folder. This module only starts
//! them, turns their `@@` lines into a state the UI shows (`setup-progress` events and `setup_status`), and starts
//! the daemon when the core is in. Nothing is installed until the person presses the button on the first screen,
//! and a failed optional group never blocks the app. See docs/FIRST_RUN.md.

use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

const SCRIPT_SH: &str = include_str!("../setup/first-run.sh");
const SCRIPT_PS1: &str = include_str!("../setup/first-run.ps1");
const PYTHON_VERSIONS: &str = include_str!("../setup/python-versions.json");

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
pub struct PythonPlan {
    pub kind: String, // "system", "download", "none", "" before the probe
    pub version: String,
    pub path: String,
    pub download_bytes: u64,
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SetupState {
    /// "installed" (nothing to do), "needs_choice" (first screen), "running", "done", "failed"
    pub phase: String,
    pub python: PythonPlan,
    pub steps: Vec<Step>,
    pub log_tail: Vec<String>,
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
    let mut child = match spawned {
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
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    std::thread::spawn(move || {
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
                emit(&app, &state);
            };
        }
        let code = child.wait().ok().and_then(|status| status.code());
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
            start_daemon_after_setup(&app);
        }
        if let Ok(state) = app.state::<Setup>().0.lock() {
            emit(&app, &state);
        }
    });
    Ok(())
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
