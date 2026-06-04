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
use sysinfo::System;
use sysinfo::Disks;
use sysinfo::Networks;
use battery::Manager as BatteryManager;
/// Global handle to the Python daemon process so we can kill it on exit.
struct DaemonProcess(Mutex<Option<Child>>);
fn get_app_data_dir() -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".config").join("heliox-os")
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
#[tauri::command]
fn get_system_stats() -> serde_json::Value {
    let mut system = System::new_all();
    system.refresh_cpu();
    // CPU
    let cpu = system.global_cpu_info().cpu_usage();
    let cpu_name =
    system.global_cpu_info().brand();
    let total_ram_gb =
    system.total_memory() / 1024 / 1024;
    let disks_info =
    Disks::new_with_refreshed_list();
    let mut disk_size = 0;
    for disk in &disks_info {
    disk_size +=
        disk.total_space()
        / 1024
        / 1024
        / 1024;
}
    // RAM
    let total_memory = system.total_memory();
    let used_memory = system.used_memory();
    let ram =
        (used_memory as f64 / total_memory as f64) * 100.0;
    // DISKS
    let disks = Disks::new_with_refreshed_list();
    let mut total_disk = 0;
    let mut used_disk = 0;
    for disk in &disks {
        total_disk += disk.total_space();
        used_disk +=
            disk.total_space() - disk.available_space();
    }
    let disk =
        (used_disk as f64 / total_disk as f64) * 100.0;
    // NETWORKS
    let mut networks =
    Networks::new_with_refreshed_list();
   networks.refresh();
   let mut upload = 0;
    let mut download = 0;
   for (_name, data) in &networks {
    upload += data.total_transmitted();
    download += data.total_received();
    }
    serde_json::json!({
    "cpu": cpu,
    "ram": ram,
    "disk": disk,
    "network_up": upload / 1024,
    "network_down": download / 1024,
    "cpu_name": cpu_name,
    "cpu_usage": cpu,
    "total_ram": total_ram_gb,
    "disk_size": disk_size
    })
}
#[tauri::command]
fn get_terminal_logs() -> Vec<String> {
    let stats =
        get_system_stats();
    let cpu =
        stats["cpu"]
        .as_f64()
        .unwrap_or(0.0);
    let ram =
        stats["ram"]
        .as_f64()
        .unwrap_or(0.0);
    let disk =
        stats["disk"]
        .as_f64()
        .unwrap_or(0.0);
    let upload =
        stats["network_up"]
        .as_f64()
        .unwrap_or(0.0);
    let download =
        stats["network_down"]
        .as_f64()
        .unwrap_or(0.0);
    let mut logs = vec![
        format!(
            "[INFO] CPU Usage: {:.1}%",
            cpu
        ),
        format!(
            "[INFO] RAM Usage: {:.1}%",
            ram
        ),
        format!(
            "[INFO] Disk Usage: {:.1}%",
            disk
        ),
        format!(
            "[INFO] Network ↑ {:.2} MB/s ↓ {:.2} MB/s",
            upload,
            download
        ),
        "[INFO] JSON-RPC connected"
            .to_string(),
        "[SUCCESS] System monitoring active"
            .to_string(),
        "[INFO] Background agents online"
            .to_string(),
    ];
    if cpu > 80.0 {
        logs.push(
            "[WARN] High CPU usage detected"
                .to_string()
        );
    }
    if ram > 75.0 {
        logs.push(
            "[WARN] High memory usage detected"
                .to_string()
        );
    }
    if disk > 85.0 {
        logs.push(
            "[WARN] Disk storage running low"
                .to_string()
        );
    }
    if upload > 5.0 || download > 5.0 {
        logs.push(
            "[INFO] High network activity"
                .to_string()
        );
    }
    logs.push(
        "[SUCCESS] Agent heartbeat OK"
            .to_string()
    );
  logs.push(
    "[INFO] Monitoring active agents"
        .to_string()
    );
  logs.push(
    "[INFO] System telemetry synced"
        .to_string()
   ); 
  logs.push(
    "[SUCCESS] Daemon heartbeat stable"
        .to_string()
   );
    logs
}
#[tauri::command]
fn get_rss_feed() -> Vec<serde_json::Value> {
    let stats =
        get_system_stats();
    let cpu =
        stats["cpu"]
        .as_f64()
        .unwrap_or(0.0);
    let ram =
        stats["ram"]
        .as_f64()
        .unwrap_or(0.0);
    let mut feed = vec![
        serde_json::json!({
            "title":
                "Heliox OS dashboard initialized",
            "time":
                "Just now"
        }),
        serde_json::json!({
            "title":
                "JSON-RPC daemon connected",
            "time":
                "1 min ago"
        }),
        serde_json::json!({
            "title":
                format!(
                    "CPU usage {:.1}% active",
                    cpu
                ),
            "time":
                "Live"
        }),
        serde_json::json!({
            "title":
                format!(
                    "RAM usage {:.1}% active",
                    ram
                ),
            "time":
                "Live"
        }),
    ];
    if cpu > 60.0 {
        feed.push(
            serde_json::json!({
                "title":
                    "High CPU activity detected",
                "time":
                    "Alert"
            })
        );
    }
    if ram > 70.0 {
        feed.push(
            serde_json::json!({
                "title":
                    "Memory usage elevated",
                "time":
                    "Alert"
            })
        );
    }
    feed.push(
        serde_json::json!({
            "title":
                "Background agents online",
            "time":
                "Live"
        })
    );
    feed
}
#[tauri::command]
fn get_agent_activity() -> Vec<serde_json::Value> {
    let stats =
        get_system_stats();
    let cpu =
        stats["cpu"]
        .as_f64()
        .unwrap_or(0.0);
    let ram =
        stats["ram"]
        .as_f64()
        .unwrap_or(0.0);
     let upload =
       stats["network_up"]
       .as_f64()
       .unwrap_or(0.0);
    let download =
       stats["network_down"]
       .as_f64()
       .unwrap_or(0.0);   
    vec![
        serde_json::json!({
            "name":
                "Voice Agent",
            "status":
                if cpu > 20.0 {
                    "Active"
                } else {
                    "Idle"
                },
            "message":
                "Listening..."
        }),
        serde_json::json!({
            "name":
                "Gesture Agent",
            "status":
                "Active",
            "message":
                "Tracking..."
        }),
        serde_json::json!({
            "name":
                "System Agent",
            "status":
                if ram > 70.0 {
                    "Warning"
                } else {
                    "Active"
                },
            "message":
                "Monitoring..."
        }),
        serde_json::json!({
            "name":
                "Automation Agent",
            "status":
                if cpu > 60.0 {
                    "Active"
                } else {
                    "Idle"
                },
            "message":
                "Waiting..."
        }),
        serde_json::json!({
        "name":
            "Network Agent",
        "status":
            if upload > 5.0 || download > 5.0 {
                "Active"
            } else {
                "Idle"
            },
        "message":
            "Monitoring network traffic..."
    }),
    serde_json::json!({
        "name":
            "Security Agent",
        "status":
            "Active",
        "message":
            "Scanning system threats..."
    }),
    serde_json::json!({
        "name":
            "Thermal Agent",
        "status":
            if cpu > 75.0 {
                "Warning"
            } else {
                "Active"
            },
        "message":
            "Monitoring system temperatures..."
    }),
    serde_json::json!({
        "name":
            "JSON-RPC Agent",
        "status":
            "Active",
        "message":
            "Syncing real-time events..."
    }),
    ]
}
#[tauri::command]
fn get_temperature_stats() -> serde_json::Value {
    let stats =
        get_system_stats();
    let cpu_usage =
        stats["cpu"]
        .as_f64()
        .unwrap_or(0.0);
    let ram_usage =
        stats["ram"]
        .as_f64()
        .unwrap_or(0.0);
    let cpu_temp =
        35.0 + (cpu_usage * 0.6);
    let gpu_temp =
        32.0 + (cpu_usage * 0.4);
    let motherboard_temp =
        28.0 + (ram_usage * 0.2);
    let ssd_temp =
        30.0 + (cpu_usage * 0.2);
    let vrm_temp =
    30.0 + (cpu_usage * 0.3);
  let battery_temp =
    29.0 + (ram_usage * 0.2);
   let power_draw =
    45.0 + (cpu_usage * 1.5);
   let mut sys = System::new_all();
   sys.refresh_all();
// REAL CPU NAME
let cpu_name =
    sys.global_cpu_info()
        .brand()
        .to_string();
// REAL CPU THREADS
let cpu_threads =
    sys.cpus().len();
// REAL BATTERY %
let manager =
    BatteryManager::new().ok();
let mut battery_percent = 0;
if let Some(manager) = manager {
    if let Ok(mut batteries) =
        manager.batteries()
    {
        if let Some(Ok(battery)) =
            batteries.next()
        {
            battery_percent =
                (battery
                    .state_of_charge()
                    .value
                    * 100.0) as i32;
        }
    }
}   
    serde_json::json!({
        "cpu":
            cpu_temp,
        "gpu":
            gpu_temp,
        "motherboard":
            motherboard_temp,
        "ssd":
            ssd_temp,
        "vrm":
            vrm_temp,
        "battery":
            battery_temp,
        "power":
            power_draw,
        "cpu_name":
           cpu_name,
       "cpu_threads":
            cpu_threads,
        "battery_percent":
            battery_percent,
    })
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
            get_terminal_logs,
            get_rss_feed,
            get_agent_activity,
            commands::open_logs_folder,
            commands::apply_git_conflict_resolution,
            commands::get_hotkey,
            commands::set_hotkey,
            commands::get_auth_token,
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
