// Heliox OS — AI System Control Agent
// Tauri v2 application entry point

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod hotkey;
mod tray;

use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::Manager;

/// Global handle to the Python daemon process so we can kill it on exit.
struct DaemonProcess(Mutex<Option<Child>>);

const DAEMON_HOST: &str = "127.0.0.1";
const DAEMON_PORT: u16 = 8785;

fn get_app_data_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".heliox-os")
}

fn get_venv_python() -> PathBuf {
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
fn try_spawn_with(python: &Path) -> Option<Child> {
    let mut cmd = Command::new(python);
    cmd.args(["-m", "pilot.server"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());

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

/// Wait until the daemon accepts TCP connections on the configured host/port.
fn wait_for_daemon(host: &str, port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;

    while Instant::now() < deadline {
        if TcpStream::connect((host, port)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    false
}

/// Run the first-time venv + pip install in a background thread (non-blocking).
fn setup_venv_in_background() {
    std::thread::spawn(|| {
        let data_dir = get_app_data_dir();
        let _ = std::fs::create_dir_all(&data_dir);
        let venv_dir = data_dir.join("env");

        println!("[Heliox OS] First run detected — setting up virtual environment in background...");

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

        #[cfg(target_os = "windows")]
        let pip_exe = venv_dir.join("Scripts").join("pip.exe");
        #[cfg(not(target_os = "windows"))]
        let pip_exe = venv_dir.join("bin").join("pip");

        let mut pip_cmd = Command::new(&pip_exe);

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            pip_cmd.creation_flags(0x08000000);
        }

        let ok = pip_cmd
            .args(["install", "pilot-daemon"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if ok {
            println!("[Heliox OS] Background setup complete — restart the app to activate AI backend.");
        } else {
            eprintln!("[Heliox OS] Background setup: pip install failed.");
        }
    });
}

fn spawn_daemon() -> Option<Child> {
    let data_dir = get_app_data_dir();
    let _ = std::fs::create_dir_all(&data_dir);

    let venv_python = get_venv_python();

    // Strategy 1: isolated venv python
    if venv_python.exists() {
        if let Some(mut child) = try_spawn_with(&venv_python) {
            println!("[Heliox OS] AI daemon spawned from venv");

            if wait_for_daemon(DAEMON_HOST, DAEMON_PORT, Duration::from_secs(8)) {
                println!(
                    "[Heliox OS] AI daemon is ready on ws://{}:{}",
                    DAEMON_HOST, DAEMON_PORT
                );
                return Some(child);
            }

            match child.try_wait() {
                Ok(Some(status)) => {
                    eprintln!(
                        "[Heliox OS] Daemon exited early after venv spawn with status: {}",
                        status
                    );
                }
                Ok(None) => {
                    eprintln!(
                        "[Heliox OS] Daemon spawned from venv but did not become ready in time"
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[Heliox OS] Failed to inspect daemon process after venv spawn: {}",
                        e
                    );
                }
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

        if wait_for_daemon(DAEMON_HOST, DAEMON_PORT, Duration::from_secs(8)) {
            println!(
                "[Heliox OS] AI daemon is ready on ws://{}:{}",
                DAEMON_HOST, DAEMON_PORT
            );
            return Some(child);
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!(
                    "[Heliox OS] Daemon exited early after system Python spawn with status: {}",
                    status
                );
            }
            Ok(None) => {
                eprintln!(
                    "[Heliox OS] Daemon spawned from system Python but did not become ready in time"
                );
            }
            Err(e) => {
                eprintln!(
                    "[Heliox OS] Failed to inspect daemon process after system Python spawn: {}",
                    e
                );
            }
        }

        let _ = child.kill();
        let _ = child.wait();
    }

    // Strategy 3: background install if venv doesn't exist
    if !venv_python.exists() {
        println!("[Heliox OS] No daemon found. Starting background installation...");
        setup_venv_in_background();
    } else {
        eprintln!("[Heliox OS] Warning: venv exists but daemon failed to start.");
    }

    None
}

fn stop_daemon(state: &DaemonProcess) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            match child.try_wait() {
                Ok(Some(_)) => {
                    println!("[Heliox OS] Python daemon already exited");
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    println!("[Heliox OS] Python daemon stopped");
                }
                Err(e) => {
                    eprintln!("[Heliox OS] Failed to inspect daemon before stop: {}", e);
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }
}

fn main() {
    let daemon_child = spawn_daemon();

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_shell::init())
        .manage(DaemonProcess(Mutex::new(daemon_child)))
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            window.show().unwrap();
            window.set_focus().unwrap();

            tray::setup_tray(app)?;
            hotkey::register_hotkey(app)?;
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
        ])
        .build(tauri::generate_context!())
        .expect("error while building Heliox OS")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app_handle.state::<DaemonProcess>();
                stop_daemon(&state);
            }
        });
}