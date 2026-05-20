// Heliox OS — AI System Control Agent
// Tauri v2 application entry point

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod hotkey;
mod tray;

use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

/// Global handle to the Python daemon process so we can kill it on exit.
struct DaemonProcess(Mutex<Option<Child>>);

fn get_app_data_dir() -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".heliox-os")
}

fn get_venv_python() -> std::path::PathBuf {
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

    cmd.spawn().ok()
}

/// Run the first-time venv + pip install in a background thread (non-blocking).
fn setup_venv_in_background() {
    std::thread::spawn(|| {
        let data_dir = get_app_data_dir();
        let _ = std::fs::create_dir_all(&data_dir);
        let venv_dir = data_dir.join("env");

        println!("[Heliox OS] First run detected — setting up virtual environment in background...");

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

    // === Strategy 1: Try the isolated venv python (production installs) ===
    let venv_python = get_venv_python();
    if venv_python.exists() {
        if let Some(child) = try_spawn_with(&venv_python) {
            println!("[Heliox OS] AI daemon spawned from venv");
            return Some(child);
        }
    }

    // === Strategy 2: Try system python directly (local dev with `pip install -e daemon/`) ===
    #[cfg(target_os = "windows")]
    let sys_python = std::path::PathBuf::from("python");
    #[cfg(not(target_os = "windows"))]
    let sys_python = std::path::PathBuf::from("python3");

    if let Some(child) = try_spawn_with(&sys_python) {
        println!("[Heliox OS] AI daemon spawned from system Python");
        return Some(child);
    }

    // === Strategy 3: Nothing worked — kick off background setup, don't block UI ===
    if !venv_python.exists() {
        println!("[Heliox OS] No daemon found. Starting background installation...");
        setup_venv_in_background();
    } else {
        eprintln!("[Heliox OS] Warning: venv exists but daemon failed to start.");
    }

    None
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
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            
            // Show the window when the user starts the app, rather than hiding it
            window.show().unwrap();
            window.set_focus().unwrap();

            tray::setup_tray(app)?;
            hotkey::register_hotkey(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Kill the daemon when the app window is destroyed
                if let Some(state) = window.try_state::<DaemonProcess>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(ref mut child) = *guard {
                            let _ = child.kill();
                            println!("[Heliox OS] Python daemon stopped");
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::toggle_window,
            commands::get_daemon_status,
            commands::send_to_daemon,
            commands::confirm_action,
            commands::open_logs_folder,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Heliox OS");
}
