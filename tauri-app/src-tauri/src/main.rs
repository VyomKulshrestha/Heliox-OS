// Heliox OS — AI System Control Agent
// Tauri v2 application entry point
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod commands;
mod file_access;
mod hotkey;
mod setup;
mod supervisor;
mod tray;
mod uninstall;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use sysinfo::Disks;
use sysinfo::System;
use tauri::{Emitter, Manager};
/// Global handle to the Python daemon process so we can kill it on exit.
struct DaemonProcess(Mutex<Option<Child>>);
/// Set when the app is exiting, so the supervisor does not start the daemon the app is stopping.
static APP_EXITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// One supervisor thread at a time; cleared when it ends, so a later call (after first-run setup) can start one.
static SUPERVISING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Clears SUPERVISING when the supervisor thread ends, however it ends.
struct SupervisingGuard;

impl Drop for SupervisingGuard {
    fn drop(&mut self) {
        SUPERVISING.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

const DAEMON_HOST: &str = "127.0.0.1";
const DAEMON_PORT: u16 = 8785;
fn get_app_data_dir() -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".config").join("heliox-os")
}
pub(crate) fn pilot_log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local")
        .join("state")
        .join("heliox-os")
        .join("pilot.log")
}
pub(crate) fn get_venv_python() -> std::path::PathBuf {
    let venv_dir = get_app_data_dir().join("env");
    #[cfg(target_os = "windows")]
    {
        venv_dir.join("Scripts").join("python.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        venv_dir.join("bin").join("python3")
    }
}
/// Try to launch the daemon using a specific python path.
fn try_spawn_with(python: &std::path::Path) -> Option<Child> {
    let mut cmd = Command::new(python);
    cmd.args(["-m", "pilot.server"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    match cmd.spawn() {
        Ok(child) => Some(child),
        Err(e) => {
            eprintln!(
                "[Heliox OS] Failed to spawn daemon with {:?}: {}",
                python, e
            );
            None
        }
    }
}

/// Allow the child to fail fast, but do not wait for model initialization.
/// The Svelte client already exposes a reconnecting "Connecting" state, and
/// first startup can legitimately take well over eight seconds.
fn daemon_child_started(child: &mut Child, source: &str) -> bool {
    std::thread::sleep(Duration::from_millis(500));
    match child.try_wait() {
        Ok(Some(status)) => {
            eprintln!("[Heliox OS] Daemon exited during {source} startup with status: {status}");
            false
        }
        Ok(None) => true,
        Err(error) => {
            eprintln!("[Heliox OS] Failed to inspect daemon after {source} startup: {error}");
            false
        }
    }
}

#[tauri::command]
fn get_system_stats() -> serde_json::Value {
    let mut system = System::new_all();
    system.refresh_cpu_all();
    // CPU
    let cpu = system.global_cpu_usage();
    let cpu_name = system
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_default();
    let total_ram_gb = system.total_memory() / 1024 / 1024 / 1024;
    let disks_info = Disks::new_with_refreshed_list();
    let mut disk_size = 0;
    for disk in &disks_info {
        disk_size += disk.total_space() / 1024 / 1024 / 1024;
    }
    // RAM
    let total_memory = system.total_memory();
    let used_memory = system.used_memory();
    let ram = (used_memory as f64 / total_memory as f64) * 100.0;
    // DISKS
    let disks = Disks::new_with_refreshed_list();
    let mut total_disk = 0;
    let mut used_disk = 0;
    for disk in &disks {
        total_disk += disk.total_space();
        used_disk += disk.total_space() - disk.available_space();
    }
    let disk = (used_disk as f64 / total_disk as f64) * 100.0;
    serde_json::json!({
    "cpu": cpu,
    "ram": ram,
    "disk": disk,
    "network_up": serde_json::Value::Null,
    "network_down": serde_json::Value::Null,
    "cpu_name": cpu_name,
    "cpu_usage": cpu,
    "total_ram": total_ram_gb,
    "disk_size": disk_size
    })
}

#[tauri::command]
fn system_info() -> serde_json::Value {
    let mut system = System::new_all();
    system.refresh_all();
    let total_memory = system.total_memory();
    let used_memory = system.used_memory();
    let memory_percent = if total_memory == 0 {
        0.0
    } else {
        (used_memory as f64 / total_memory as f64) * 100.0
    };
    let disks = Disks::new_with_refreshed_list();
    let disk_total: u64 = disks.iter().map(|disk| disk.total_space()).sum();
    let disk_used: u64 = disks
        .iter()
        .map(|disk| disk.total_space().saturating_sub(disk.available_space()))
        .sum();
    let disk_percent = if disk_total == 0 {
        0.0
    } else {
        (disk_used as f64 / disk_total as f64) * 100.0
    };

    serde_json::json!({
        "status": "ok",
        "cpu_percent": system.global_cpu_usage(),
        "memory_percent": memory_percent,
        "memory_used": used_memory,
        "memory_total": total_memory,
        "disk_percent": disk_percent,
        "disk_used": disk_used,
        "disk_total": disk_total,
        "hostname": System::host_name().unwrap_or_else(|| "HELIOX".to_string()),
        "uptime_seconds": System::uptime(),
    })
}

#[tauri::command]
fn get_terminal_logs() -> Result<Vec<String>, String> {
    let log_path = pilot_log_path();
    if !log_path.exists() {
        return Ok(Vec::new());
    }
    let contents = std::fs::read_to_string(&log_path)
        .map_err(|error| format!("Could not read {}: {error}", log_path.display()))?;
    Ok(contents
        .lines()
        .rev()
        .take(100)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(str::to_string)
        .collect())
}

#[tauri::command]
fn get_log_count() -> usize {
    let log_path = pilot_log_path();
    std::fs::read_to_string(log_path)
        .map(|contents| contents.lines().count())
        .unwrap_or(0)
}

#[tauri::command]
fn get_rss_feed() -> Vec<serde_json::Value> {
    let mut feed = vec![serde_json::json!({
        "title": format!(
            "Heliox OS local build v{}",
            env!("CARGO_PKG_VERSION")
        ),
        "url": "https://github.com/VyomKulshrestha/Heliox-OS/releases",
        "source": "Current Build"
    })];
    if let Ok(output) = std::process::Command::new("git")
        .args([
            "tag",
            "-l",
            "--sort=-creatordate",
            "--format=%(refname:short)|%(creatordate:short)|%(subject)",
        ])
        .output()
    {
        if let Ok(text) = String::from_utf8(output.stdout) {
            for line in text.lines().take(4) {
                let parts: Vec<&str> = line.split('|').collect();
                if !parts.is_empty() && !parts[0].is_empty() {
                    feed.push(serde_json::json!({
                        "title": format!("Release {}: {}", parts[0], if parts.len() > 2 && !parts[2].is_empty() { parts[2] } else { "Official Heliox OS Distribution" }),
                        "url": format!("https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/{}", parts[0]),
                        "source": format!("Release Tag ({})", if parts.len() > 1 && !parts[1].is_empty() { parts[1] } else { "Published" })
                    }));
                }
            }
        }
    }
    feed
}
#[tauri::command]
fn get_agent_activity() -> Result<Vec<serde_json::Value>, String> {
    Err("Agent activity is unavailable because no native agent supervisor is configured".into())
}
#[tauri::command]
fn get_temperature_stats() -> Result<serde_json::Value, String> {
    Err("Hardware temperature sensors are unavailable in this build".into())
}
fn spawn_daemon() -> Option<Child> {
    if TcpStream::connect((DAEMON_HOST, DAEMON_PORT)).is_ok() {
        println!(
            "[Heliox OS] Reusing daemon already listening on ws://{}:{}",
            DAEMON_HOST, DAEMON_PORT
        );
        BORROW_AT_START.store(true, std::sync::atomic::Ordering::SeqCst);
        return None;
    }

    if let Some(child) = launch_daemon() {
        return Some(child);
    }

    // Strategy 3: no environment yet. Nothing is installed here: the app shows the first-run screen (setup.rs),
    // where the person chooses what to install.
    if get_venv_python().exists() {
        eprintln!("[Heliox OS] Warning: venv exists but daemon failed to start.");
    } else {
        println!("[Heliox OS] No daemon found. Waiting for first-run setup in the app.");
    }

    None
}

/// Start the daemon from the app's venv, else from the system Python (strategies 1 and 2).
fn launch_daemon() -> Option<Child> {
    let data_dir = get_app_data_dir();
    let _ = std::fs::create_dir_all(&data_dir);

    let venv_python = get_venv_python();

    // Strategy 1: isolated venv python
    if venv_python.exists() {
        if let Some(mut child) = try_spawn_with(&venv_python) {
            println!("[Heliox OS] AI daemon spawned from venv");

            if daemon_child_started(&mut child, "venv") {
                println!("[Heliox OS] Daemon is initializing; the UI will reconnect when ready");
                return Some(child);
            }
            if exited_as_another_copy(&mut child) {
                println!("[Heliox OS] Another copy of the daemon is serving; using it");
                BORROW_AT_START.store(true, std::sync::atomic::Ordering::SeqCst);
                return None;
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    // Strategy 2: system python
    #[cfg(target_os = "windows")]
    let sys_python = PathBuf::from("python");
    #[cfg(not(target_os = "windows"))]
    let sys_python = PathBuf::from("python3");

    if let Some(mut child) = try_spawn_with(&sys_python) {
        println!("[Heliox OS] AI daemon spawned from system Python");

        if daemon_child_started(&mut child, "system Python") {
            println!("[Heliox OS] Daemon is initializing; the UI will reconnect when ready");
            return Some(child);
        }
        if exited_as_another_copy(&mut child) {
            println!("[Heliox OS] Another copy of the daemon is serving; using it");
            BORROW_AT_START.store(true, std::sync::atomic::Ordering::SeqCst);
            return None;
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    None
}

/// Set when a daemon this app started found another copy serving the port (exit code 3), or when the app found a
/// daemon already listening at launch: the supervisor then watches that copy instead of owning one.
static BORROW_AT_START: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True when the child has exited with the daemon's "another copy is serving" code. Strategy 2 (system Python) is
/// not tried then: it would find the same copy.
fn exited_as_another_copy(child: &mut Child) -> bool {
    match child.try_wait() {
        Ok(Some(status)) => {
            supervisor::exit_observation(status.code()) == supervisor::Observation::AnotherCopy
        }
        _ => false,
    }
}

/// Is anything answering on the daemon's port? In borrowed mode that is the only question: the other copy's own
/// supervisor (or the person who ran it) deals with its start-up and hangs.
fn daemon_port_answers() -> bool {
    use std::net::{SocketAddr, ToSocketAddrs};
    let addresses: Vec<SocketAddr> = (DAEMON_HOST, DAEMON_PORT)
        .to_socket_addrs()
        .map(|found| found.collect())
        .unwrap_or_default();
    addresses
        .iter()
        .any(|address| TcpStream::connect_timeout(address, Duration::from_secs(1)).is_ok())
}

/// Watch the daemon this app started; restart it when it exits or stops answering (see supervisor.rs).
/// A daemon the app did not start (`None` in DaemonProcess) is left alone. Safe to call again later, for example
/// after first-run setup starts the daemon: a second call while a watcher runs does nothing.
pub(crate) fn supervise_daemon(app: tauri::AppHandle) {
    use supervisor::{Observation, Step, Watch, CHECK_EVERY};

    let owned = app
        .state::<DaemonProcess>()
        .0
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(false);
    let borrowed = !owned && BORROW_AT_START.load(std::sync::atomic::Ordering::SeqCst);
    if !(owned || borrowed) || SUPERVISING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        let _guard = SupervisingGuard;
        let mut watch = Watch::default();
        watch.started();
        if borrowed {
            watch.borrow();
            BORROW_AT_START.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let paused = || setup::DAEMON_PAUSED.load(std::sync::atomic::Ordering::SeqCst);
        let exiting = || APP_EXITING.load(std::sync::atomic::Ordering::SeqCst);
        let mut was_paused = false;
        loop {
            std::thread::sleep(CHECK_EVERY);
            if APP_EXITING.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            if paused() {
                // Setup is changing the daemon's environment (update, repair); it starts the daemon again itself.
                was_paused = true;
                continue;
            }
            let state = app.state::<DaemonProcess>();
            if was_paused {
                // Own what setup started, or quietly borrow what answers; decided once, when the pause ends.
                was_paused = false;
                let owns = state.0.lock().map(|guard| guard.is_some()).unwrap_or(false);
                watch.resume_after_pause(owns);
                continue;
            }
            let exited = if watch.borrowed {
                None // another copy's daemon: only its port is watched
            } else {
                match state.0.lock() {
                    Ok(mut guard) => match guard.as_mut() {
                        Some(child) => match child.try_wait() {
                            Ok(None) => None,
                            Ok(Some(status)) => Some(supervisor::exit_observation(status.code())),
                            Err(_) => Some(Observation::Exited),
                        },
                        // A restart that failed to launch: keep trying, unless it found another copy serving.
                        None if BORROW_AT_START
                            .swap(false, std::sync::atomic::Ordering::SeqCst) =>
                        {
                            Some(Observation::AnotherCopy)
                        }
                        None => Some(Observation::Exited),
                    },
                    Err(_) => return,
                }
            };
            let seen = if let Some(exit) = exited {
                exit
            } else if watch.borrowed {
                if daemon_port_answers() {
                    Observation::Answered
                } else {
                    Observation::Silent
                }
            } else if tauri::async_runtime::block_on(commands::daemon_answers()) {
                Observation::Answered
            } else {
                Observation::Silent
            };
            let step = watch.observe(seen, CHECK_EVERY);
            if let Some(line) = supervisor::notice(step) {
                eprintln!("[Heliox OS] {line}");
                let _ = app.emit("daemon-supervisor", serde_json::json!({ "message": line }));
            }
            match step {
                Step::Wait => {}
                Step::KillHung => {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.as_mut() {
                            supervisor::kill_tree(child);
                        }
                    }
                }
                Step::Restart { after } => {
                    let stop = || paused() || exiting();
                    supervisor::wait_unless(
                        after,
                        Duration::from_millis(250),
                        stop,
                        std::thread::sleep,
                    );
                    if exiting() {
                        return;
                    }
                    if !supervisor::may_launch(paused(), exiting()) {
                        continue; // an update or repair began during the back-off: setup starts the daemon
                    }
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(mut old) = guard.take() {
                            // Only a live process is killed: an exited one's id may already belong to another.
                            if matches!(old.try_wait(), Ok(None)) {
                                supervisor::kill_tree(&mut old);
                            } else {
                                let _ = old.wait();
                            }
                        }
                        if TcpStream::connect((DAEMON_HOST, DAEMON_PORT)).is_ok() {
                            // Someone else started a daemon meanwhile (for example from a terminal): leave it, and
                            // watch it as a borrowed one.
                            println!("[Heliox OS] Another daemon is listening; not starting one");
                            watch.borrow();
                            continue;
                        }
                        *guard = launch_daemon();
                    }
                    watch.started();
                }
                Step::Borrow => {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(mut old) = guard.take() {
                            let _ = old.wait(); // it has exited: reap it
                        }
                    }
                }
                Step::TakeOver => {
                    if exiting() {
                        return;
                    }
                    if !supervisor::may_launch(paused(), exiting()) {
                        continue;
                    }
                    if let Ok(mut guard) = state.0.lock() {
                        *guard = launch_daemon();
                    }
                    watch.started();
                }
            }
        }
    });
}

fn stop_daemon(state: &DaemonProcess) {
    APP_EXITING.store(true, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            match child.try_wait() {
                Ok(Some(_)) => {
                    println!("[Heliox OS] Python daemon already exited");
                }
                Ok(None) => {
                    supervisor::kill_tree(&mut child);
                    println!("[Heliox OS] Python daemon stopped");
                }
                Err(e) => {
                    eprintln!("[Heliox OS] Failed to inspect daemon before stop: {}", e);
                    supervisor::kill_tree(&mut child);
                }
            }
        }
    }
}
mod instance;
mod not_answering;

/// Whether Heliox counts as installed at launch, without waiting on the daemon: an environment, or (only when there
/// is none) a daemon already answering, asked lazily.
fn installed_at_launch(venv_exists: bool, port_answers: impl FnOnce() -> bool) -> bool {
    venv_exists || port_answers()
}

fn port_answers(limit: Duration) -> bool {
    use std::net::SocketAddr;
    let address = SocketAddr::from(([127, 0, 0, 1], DAEMON_PORT));
    TcpStream::connect_timeout(&address, limit).is_ok()
}

/// After the window is up: find a daemon already listening or start one, then watch it (supervise_daemon). The window
/// meanwhile shows "Connecting...", and a request typed in it is kept until the daemon answers.
fn start_daemon_after_window(app: tauri::AppHandle) {
    std::thread::spawn(move || {
        let child = spawn_daemon();
        if let Ok(mut guard) = app.state::<DaemonProcess>().0.lock() {
            *guard = child;
        }
        supervise_daemon(app);
    });
}

fn main() {
    // One copy of the app per person: a second launch brings the first window forward and exits, before it starts
    // a daemon, a tray or a supervisor of its own (instance.rs).
    let first_copy = match instance::claim(&get_app_data_dir()) {
        instance::Launch::First(first) => Some(first),
        instance::Launch::Second => {
            if instance::ask_first_to_show(&get_app_data_dir()) {
                println!("[Heliox OS] Heliox is already open; brought its window forward");
            } else {
                // The first copy holds the lock but does not answer (hung, or with no window): say so in a window
                // of this launch's own, with how to close it; never end the other copy from here.
                not_answering::tell_the_person();
            }
            return;
        }
        instance::Launch::Unknown(reason) => {
            eprintln!(
                "[Heliox OS] Could not check for another open copy ({reason}); opening anyway"
            );
            None
        }
    };
    // The window first. The daemon is found or started after it, in the background (start_daemon_after_window):
    // spawning it here held the window back by the half second daemon_child_started waits, a full second when the
    // venv's daemon died and the system Python was tried too.
    let installed = installed_at_launch(get_venv_python().exists(), || {
        // Only without an environment (a daemon run from a checkout): refused at once when nothing listens.
        port_answers(Duration::from_millis(150))
    });
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .manage(DaemonProcess(Mutex::new(None)))
        .manage(setup::Setup(Mutex::new(setup::initial_state(installed))))
        .manage(commands::NeuralProcess::new())
        .manage(file_access::AllowedPaths::new())
        .manage(commands::GestureCursor::init())
        .setup(move |app| {
            let window = app.get_webview_window("main").unwrap();

            // Show the window when the user starts the app, rather than hiding it
            window.show().unwrap();
            window.set_focus().unwrap();
            if let Some(first) = first_copy {
                let handle = app.handle().clone();
                let listening = first.listen(move || {
                    if let Some(window) = handle.get_webview_window("main") {
                        let _ = window.unminimize();
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                });
                if let Err(error) = listening {
                    eprintln!("[Heliox OS] A second launch will not find this window: {error}");
                }
                app.manage(first); // the lock is held for as long as the app runs
            }
            tray::setup_tray(app)?;
            hotkey::register_hotkey(app)?;
            setup::probe_in_background(app.handle().clone());
            start_daemon_after_window(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                println!("[Heliox OS] Main window close requested");
                let _ = window;
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::toggle_window,
            commands::get_daemon_status,
            commands::send_to_daemon,
            commands::confirm_action,
            commands::open_terminal,
            commands::clear_logs,
            commands::restart_agents,
            commands::system_scan,
            commands::get_uptime,
            commands::get_dashboard_status,
            get_system_stats,
            system_info,
            get_log_count,
            get_temperature_stats,
            get_terminal_logs,
            get_rss_feed,
            get_agent_activity,
            commands::open_logs_folder,
            commands::apply_git_conflict_resolution,
            commands::get_hotkey,
            setup::setup_status,
            setup::setup_run,
            setup::setup_update_plan,
            setup::setup_update_apply,
            setup::setup_update_later,
            setup::setup_repair,
            uninstall::heliox_files_plan,
            uninstall::heliox_files_remove,
            uninstall::heliox_quit,
            commands::get_hotkey_status,
            commands::set_hotkey,
            commands::get_auth_token,
            commands::extract_file_text,
            commands::move_gesture_cursor,
            commands::click_gesture_cursor,
            commands::get_neural_sidecar_status,
            commands::start_neural_sidecar,
            commands::stop_neural_sidecar,
            commands::export_neural_recording,
            commands::run_neural_benchmark,
            file_access::register_allowed_path,
            file_access::revoke_allowed_path,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Heliox OS")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app_handle.state::<DaemonProcess>();
                stop_daemon(&state);
                let neural = app_handle.state::<commands::NeuralProcess>();
                commands::stop_neural_process(neural.inner());
            }
        });
}

#[cfg(test)]
mod startup_tests {
    use super::*;

    #[cfg(target_os = "windows")]
    fn spawn_short_lived_child() -> Child {
        Command::new("cmd")
            .args(["/C", "exit", "7"])
            .spawn()
            .expect("spawn short-lived child")
    }

    #[cfg(not(target_os = "windows"))]
    fn spawn_short_lived_child() -> Child {
        Command::new("sh")
            .args(["-c", "exit 7"])
            .spawn()
            .expect("spawn short-lived child")
    }

    #[cfg(target_os = "windows")]
    fn spawn_long_lived_child() -> Child {
        Command::new("cmd")
            .args(["/C", "ping -n 4 127.0.0.1 >NUL"])
            .spawn()
            .expect("spawn long-lived child")
    }

    #[cfg(not(target_os = "windows"))]
    fn spawn_long_lived_child() -> Child {
        Command::new("sh")
            .args(["-c", "sleep 2"])
            .spawn()
            .expect("spawn long-lived child")
    }

    #[test]
    fn installed_at_launch_never_waits_on_the_port_when_there_is_an_environment() {
        let mut asked = false;
        assert!(installed_at_launch(true, || {
            asked = true;
            false
        }));
        assert!(!asked);
        assert!(installed_at_launch(false, || true));
        assert!(!installed_at_launch(false, || false));
    }

    #[test]
    fn startup_accepts_a_child_that_is_still_initializing() {
        let mut child = spawn_long_lived_child();

        assert!(daemon_child_started(&mut child, "test"));

        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn startup_rejects_a_child_that_exits_immediately() {
        let mut child = spawn_short_lived_child();

        assert!(!daemon_child_started(&mut child, "test"));

        let _ = child.wait();
    }

    #[test]
    fn offline_system_info_uses_real_host_units() {
        let info = system_info();
        let stats = get_system_stats();

        assert_eq!(info["status"], "ok");
        assert!(info["memory_total"].as_u64().unwrap_or(0) > 1024 * 1024);
        assert!(info["disk_total"].as_u64().unwrap_or(0) > 1024 * 1024);
        assert!(!info["hostname"].as_str().unwrap_or_default().is_empty());
        assert!(stats["total_ram"]
            .as_u64()
            .is_some_and(|value| value < 1024));
    }

    #[test]
    fn native_release_feed_uses_package_version() {
        let feed = get_rss_feed();
        let title = feed[0]["title"].as_str().unwrap_or_default();

        assert!(title.contains(env!("CARGO_PKG_VERSION")));
    }
}
