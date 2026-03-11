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

fn spawn_daemon() -> Option<Child> {
    let mut possible_dirs = vec![std::path::PathBuf::from("daemon")];

    // Safely build paths relative to the executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            possible_dirs.push(exe_dir.join("daemon"));
            if let Some(p1) = exe_dir.parent() {
                possible_dirs.push(p1.join("daemon"));
                if let Some(p2) = p1.parent() {
                    possible_dirs.push(p2.join("daemon"));
                }
            }
        }
    }

    if let Some(home) = dirs::home_dir() {
        possible_dirs.push(home.join(".heliox-os").join("daemon"));
    }

    let daemon_dir = possible_dirs.into_iter().find(|d| d.join("pilot").exists());

    let mut cmd = Command::new("python");
    cmd.args(["-m", "pilot.server"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    if let Some(dir) = daemon_dir {
        cmd.current_dir(&dir);
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let child = cmd.spawn().ok();

    if child.is_some() {
        println!("[Heliox OS] Python daemon spawned successfully");
    } else {
        eprintln!("[Heliox OS] Warning: Could not spawn Python daemon. Is Python installed?");
    }

    child
}

fn main() {
    // Spawn the Python daemon before building the Tauri app
    let daemon_child = spawn_daemon();

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running Heliox OS");
}
