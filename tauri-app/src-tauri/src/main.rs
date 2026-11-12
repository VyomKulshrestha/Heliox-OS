// Heliox OS — AI System Control Agent
// Tauri v2 application entry point
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod commands;
mod file_access;
mod hotkey;
mod supervisor;
mod tray;
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

/// Run the first-time venv + pip install in a background thread (non-blocking).
fn daemon_requirement() -> String {
    format!("pilot-daemon[all]=={}", env!("CARGO_PKG_VERSION"))
}

fn playwright_browser_install_args() -> [&'static str; 4] {
    ["-m", "playwright", "install", "chromium"]
}

fn setup_venv_in_background() {
    std::thread::spawn(|| {
        let data_dir = get_app_data_dir();
        let _ = std::fs::create_dir_all(&data_dir);
        let venv_dir = data_dir.join("env");
        println!(
            "[Heliox OS] First run detected — setting up virtual environment in background..."
        );
        // 1. Create venv
        #[cfg(target_os = "windows")]
        let sys_python = "python";
        #[cfg(not(target_os = "windows"))]
        let sys_python = "python3";
        let mut venv_cmd = Command::new(sys_python);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            venv_cmd.creation_flags(0x08000000);
        }
        let ok = venv_cmd
            .args(["-m", "venv", venv_dir.to_str().unwrap()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("[Heliox OS] Background setup: failed to create venv. Is Python installed?");
            return;
        }
        // 2. pip install pilot-daemon
        #[cfg(target_os = "windows")]
        let pip_exe = venv_dir.join("Scripts").join("pip.exe");
        #[cfg(not(target_os = "windows"))]
        let pip_exe = venv_dir.join("bin").join("pip");
        #[cfg(target_os = "windows")]
        let venv_python = venv_dir.join("Scripts").join("python.exe");
        #[cfg(not(target_os = "windows"))]
        let venv_python = venv_dir.join("bin").join("python");

        let mut pip_cmd = Command::new(&pip_exe);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            pip_cmd.creation_flags(0x08000000);
        }

        // [all] pulls in every optional feature this project ships (voice,
        // vision, browser automation, gesture cursor, self-healing,
        // supervision, gateway/risk-gate, documents, cognitive, ssh,
        // network, wasm) so real users get every shipped feature out of
        // the box, not a partial install requiring separate manual pip
        // steps per feature. Every one of these still defaults to off/
        // config-gated behavior where the feature is privacy- or
        // safety-sensitive (e.g. supervision's global keyboard/mouse hook
        // stays behind its own explicit one-time "I understand" consent
        // checkbox in Settings regardless of whether pynput is installed)
        // -- installing the package is not the same as enabling the
        // feature.
        // Keep the desktop and daemon on the same release. An unpinned install
        // can silently pair an older desktop with a newer, incompatible RPC
        // surface after a future PyPI publication.
        let daemon_requirement = daemon_requirement();
        let ok = pip_cmd
            .args(["install", daemon_requirement.as_str()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if !ok {
            eprintln!("[Heliox OS] Background setup: pip install failed.");
            return;
        }

        // Installing the Playwright Python package does not install a browser
        // executable. Provision Chromium during the same background setup so
        // browser actions work without a separate developer command.
        let mut browser_cmd = Command::new(&venv_python);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            browser_cmd.creation_flags(0x08000000);
        }
        let browser_ok = browser_cmd
            .args(playwright_browser_install_args())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !browser_ok {
            eprintln!("[Heliox OS] Background setup: Chromium installation failed.");
            return;
        }

        println!("[Heliox OS] Background setup complete — restart the app to activate AI backend.");
    });
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
        return None;
    }

    if let Some(child) = launch_daemon() {
        return Some(child);
    }

    // Strategy 3: background install if venv doesn't exist
    if !get_venv_python().exists() {
        println!("[Heliox OS] No daemon found. Starting background installation...");
        setup_venv_in_background();
    } else {
        eprintln!("[Heliox OS] Warning: venv exists but daemon failed to start.");
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
        let _ = child.kill();
        let _ = child.wait();
    }

    None
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
    if !owned || SUPERVISING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        let _guard = SupervisingGuard;
        let mut watch = Watch::default();
        watch.started();
        loop {
            std::thread::sleep(CHECK_EVERY);
            if APP_EXITING.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let state = app.state::<DaemonProcess>();
            let exited = match state.0.lock() {
                Ok(mut guard) => match guard.as_mut() {
                    Some(child) => !matches!(child.try_wait(), Ok(None)),
                    None => true, // a restart that failed to launch: keep trying
                },
                Err(_) => return,
            };
            let seen = if exited {
                Observation::Exited
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
                    std::thread::sleep(after);
                    if APP_EXITING.load(std::sync::atomic::Ordering::SeqCst) {
                        return;
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
                            // Someone else started a daemon meanwhile (for example from a terminal): leave it.
                            println!("[Heliox OS] Another daemon is listening; not starting one");
                            return;
                        }
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
fn main() {
    // Spawn the Python daemon before building the Tauri app
    let daemon_child = spawn_daemon();
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .manage(DaemonProcess(Mutex::new(daemon_child)))
        .manage(commands::NeuralProcess::new())
        .manage(file_access::AllowedPaths::new())
        .manage(commands::GestureCursor::init())
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();

            // Show the window when the user starts the app, rather than hiding it
            window.show().unwrap();
            window.set_focus().unwrap();
            tray::setup_tray(app)?;
            hotkey::register_hotkey(app)?;
            supervise_daemon(app.handle().clone());
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
            commands::take_screenshot,
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

    #[test]
    fn desktop_installs_matching_daemon_version() {
        assert_eq!(
            daemon_requirement(),
            format!("pilot-daemon[all]=={}", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn desktop_first_run_installs_playwright_chromium() {
        assert_eq!(
            playwright_browser_install_args(),
            ["-m", "playwright", "install", "chromium"]
        );
    }
}
