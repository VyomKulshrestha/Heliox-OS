use enigo::{Button, Coordinate, Direction, Enigo, Mouse, Settings};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use sysinfo::System;
use tauri::{AppHandle, Manager};
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

type DaemonSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// The longest a connection and its sign-in may take. A daemon that is suspended or stuck still has its port
/// accepted by the system, so without a limit the websocket handshake waited for ever (the real supervisor check
/// hung on Windows on 2f6f461 and 5816eb0 waiting on such a daemon).
const CONNECT_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

async fn connect_authenticated_to(url: &str, auth_token: &str) -> Result<DaemonSocket, String> {
    tokio::time::timeout(CONNECT_LIMIT, connect_and_sign_in(url, auth_token))
        .await
        .map_err(|_| {
            format!(
                "The daemon did not let the app sign in within {} s",
                CONNECT_LIMIT.as_secs()
            )
        })?
}

async fn connect_and_sign_in(url: &str, auth_token: &str) -> Result<DaemonSocket, String> {
    if auth_token.is_empty() {
        return Err("Daemon authentication token is unavailable".to_string());
    }

    let (mut ws, _) = connect_async(url)
        .await
        .map_err(|e| format!("Daemon connection failed: {e}"))?;
    let auth_request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "auth",
        "params": {"token": auth_token},
        "id": "native-auth"
    });
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        auth_request.to_string().into(),
    ))
    .await
    .map_err(|e| format!("Daemon authentication send failed: {e}"))?;

    let response = ws
        .next()
        .await
        .ok_or_else(|| "Daemon closed before authentication completed".to_string())?
        .map_err(|e| format!("Daemon authentication failed: {e}"))?;
    let parsed: serde_json::Value =
        serde_json::from_str(response.to_text().map_err(|e| e.to_string())?)
            .map_err(|e| format!("Invalid daemon authentication response: {e}"))?;
    let authenticated = parsed
        .get("result")
        .and_then(|result| result.get("status"))
        .and_then(|status| status.as_str())
        == Some("authenticated");
    if !authenticated {
        let message = parsed
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(|message| message.as_str())
            .unwrap_or("Daemon rejected authentication");
        return Err(message.to_string());
    }

    Ok(ws)
}

async fn connect_authenticated() -> Result<DaemonSocket, String> {
    connect_authenticated_to("ws://127.0.0.1:8785", &get_auth_token()).await
}

/// Managed state wrapping a single lazily-shared `Enigo` instance for the
/// gesture-cursor bridge. Constructing `Enigo` can involve a real connection
/// to the platform's input subsystem (X11/Wayland on Linux, etc.), so it's
/// created once at startup and reused, not recreated per cursor-move call —
/// that matters for a ~30fps continuous stream. `Option` because
/// `Enigo::new()` can fail (e.g. no display session available); when it
/// does, cursor-control commands fail gracefully with an error instead of
/// panicking the whole app.
pub struct GestureCursor(pub Mutex<Option<Enigo>>);

/// Desktop-owned neurod child. It is deliberately separate from the daemon
/// process so a decoder crash/disconnect disarms neural control without taking
/// down Heliox itself.
pub struct NeuralProcess(Mutex<Option<Child>>);

impl NeuralProcess {
    pub fn new() -> Self {
        Self(Mutex::new(None))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralLaunchOptions {
    source: String,
    artifact_path: Option<String>,
    playback_path: Option<String>,
    board_id: Option<i32>,
    serial_port: Option<String>,
    lsl_name: Option<String>,
    synthetic_frequency: Option<f64>,
    record_raw: Option<bool>,
    recording_file: Option<String>,
    recording_purpose: Option<String>,
    retention_days: Option<u16>,
    allow_bids_export: Option<bool>,
}

#[derive(Serialize)]
pub struct NeuralSidecarStatus {
    running: bool,
    pid: Option<u32>,
}

fn checked_data_file(
    value: Option<&str>,
    label: &str,
    required: bool,
) -> Result<Option<PathBuf>, String> {
    let Some(raw) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return if required {
            Err(format!("{label} is required for this neural source"))
        } else {
            Ok(None)
        };
    };
    let path = PathBuf::from(raw)
        .canonicalize()
        .map_err(|_| format!("{label} does not exist or cannot be read"))?;
    if !path.is_file() {
        return Err(format!("{label} must point to a file"));
    }
    Ok(Some(path))
}

fn checked_new_file(value: Option<&str>, label: &str) -> Result<Option<PathBuf>, String> {
    let Some(raw) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let requested = PathBuf::from(raw);
    if requested.exists() {
        return Err(format!(
            "{label} already exists; Heliox never overwrites neural data"
        ));
    }
    let name = requested
        .file_name()
        .ok_or_else(|| format!("{label} needs a file name"))?;
    let parent = requested
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .canonicalize()
        .map_err(|_| format!("{label} parent directory does not exist"))?;
    Ok(Some(parent.join(name)))
}

fn checked_new_neural_recording(value: Option<&str>) -> Result<Option<PathBuf>, String> {
    let path = checked_new_file(value, "Recording destination")?;
    if let Some(path) = &path {
        let is_neeg = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("neeg"));
        if !is_neeg {
            return Err("Neural recording destination must use the .neeg extension".to_string());
        }
    }
    Ok(path)
}

fn neural_sidecar_args(options: &NeuralLaunchOptions) -> Result<Vec<String>, String> {
    if !matches!(
        options.source.as_str(),
        "synthetic" | "playback" | "brainflow" | "lsl"
    ) {
        return Err("Unsupported neural source".to_string());
    }
    let artifact = checked_data_file(
        options.artifact_path.as_deref(),
        "Calibration artifact",
        options.source != "synthetic",
    )?;
    let playback = checked_data_file(
        options.playback_path.as_deref(),
        "Playback recording",
        options.source == "playback",
    )?;
    let frequency = options.synthetic_frequency.unwrap_or(12.0);
    if !frequency.is_finite() || !(6.0..=40.0).contains(&frequency) {
        return Err("Synthetic frequency must be between 6 and 40 Hz".to_string());
    }
    let board_id = options.board_id.unwrap_or(0);
    if !(-1000..=1000).contains(&board_id) {
        return Err("BrainFlow board id is out of range".to_string());
    }
    let serial = options.serial_port.as_deref().unwrap_or("").trim();
    if serial.len() > 256 || serial.chars().any(char::is_control) {
        return Err("Serial port is invalid".to_string());
    }
    let lsl_name = options.lsl_name.as_deref().unwrap_or("HelioxEEG").trim();
    if lsl_name.is_empty() || lsl_name.len() > 128 || lsl_name.chars().any(char::is_control) {
        return Err("LSL stream name is invalid".to_string());
    }

    let mut args = vec![
        "-m".to_string(),
        "pilot.neural.rpc_client".to_string(),
        "--source".to_string(),
        options.source.clone(),
        "--synthetic-frequency".to_string(),
        frequency.to_string(),
        "--board-id".to_string(),
        board_id.to_string(),
        "--lsl-name".to_string(),
        lsl_name.to_string(),
    ];
    if let Some(path) = artifact {
        args.extend([
            "--artifact".to_string(),
            path.to_string_lossy().into_owned(),
        ]);
    }
    if let Some(path) = playback {
        args.extend([
            "--playback".to_string(),
            path.to_string_lossy().into_owned(),
        ]);
    }
    if !serial.is_empty() {
        args.extend(["--serial-port".to_string(), serial.to_string()]);
    }
    if options.record_raw.unwrap_or(false) {
        let purpose = options.recording_purpose.as_deref().unwrap_or("").trim();
        if !(3..=256).contains(&purpose.len()) || purpose.chars().any(char::is_control) {
            return Err("A 3-256 character recording purpose is required".to_string());
        }
        let retention = options.retention_days.unwrap_or(7);
        if !(1..=365).contains(&retention) {
            return Err("Neural recording retention must be 1-365 days".to_string());
        }
        let recording = checked_new_neural_recording(options.recording_file.as_deref())?;
        args.extend([
            "--record-raw".to_string(),
            "--recording-purpose".to_string(),
            purpose.to_string(),
            "--retention-days".to_string(),
            retention.to_string(),
        ]);
        if let Some(path) = recording {
            args.extend([
                "--recording-file".to_string(),
                path.to_string_lossy().into_owned(),
            ]);
        }
        if options.allow_bids_export.unwrap_or(false) {
            args.push("--allow-bids-export".to_string());
        }
    }
    Ok(args)
}

fn neural_python() -> PathBuf {
    let configured = crate::get_venv_python();
    if configured.exists() {
        return configured;
    }
    #[cfg(target_os = "windows")]
    return PathBuf::from("python");
    #[cfg(not(target_os = "windows"))]
    return PathBuf::from("python3");
}

fn hidden_python_command() -> Command {
    let mut command = Command::new(neural_python());
    crate::outside_env::apply(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

fn reap_neural_process(guard: &mut Option<Child>) -> Result<(), String> {
    let finished = match guard.as_mut() {
        Some(child) => child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some(),
        None => false,
    };
    if finished {
        *guard = None;
    }
    Ok(())
}

#[tauri::command]
pub fn get_neural_sidecar_status(
    state: tauri::State<NeuralProcess>,
) -> Result<NeuralSidecarStatus, String> {
    let mut guard = state.0.lock().map_err(|error| error.to_string())?;
    reap_neural_process(&mut guard)?;
    Ok(NeuralSidecarStatus {
        running: guard.is_some(),
        pid: guard.as_ref().map(Child::id),
    })
}

#[tauri::command]
pub fn start_neural_sidecar(
    state: tauri::State<NeuralProcess>,
    options: NeuralLaunchOptions,
) -> Result<NeuralSidecarStatus, String> {
    let args = neural_sidecar_args(&options)?;
    let mut guard = state.0.lock().map_err(|error| error.to_string())?;
    reap_neural_process(&mut guard)?;
    if guard.is_some() {
        return Err("The neural sidecar is already running".to_string());
    }
    let mut command = hidden_python_command();
    command.args(args);
    let child = command
        .spawn()
        .map_err(|error| format!("Could not start the neural sidecar: {error}"))?;
    let pid = child.id();
    *guard = Some(child);
    Ok(NeuralSidecarStatus {
        running: true,
        pid: Some(pid),
    })
}

#[tauri::command]
pub fn export_neural_recording(recording: String, destination: String) -> Result<(), String> {
    let recording = checked_data_file(Some(&recording), "Encrypted neural recording", true)?
        .ok_or_else(|| "Encrypted neural recording is required".to_string())?;
    let destination = PathBuf::from(destination.trim());
    if destination.as_os_str().is_empty() {
        return Err("BIDS export destination is required".to_string());
    }
    if destination.exists() {
        return Err("BIDS export destination must not already exist".to_string());
    }
    let parent = destination
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .canonicalize()
        .map_err(|_| "BIDS export parent directory does not exist".to_string())?;
    let name = destination
        .file_name()
        .ok_or_else(|| "BIDS export destination needs a directory name".to_string())?;
    let destination = parent.join(name);
    let status = hidden_python_command()
        .args([
            "-m",
            "pilot.neural.recording",
            &recording.to_string_lossy(),
            &destination.to_string_lossy(),
        ])
        .status()
        .map_err(|error| format!("Could not start neural export: {error}"))?;
    if !status.success() {
        return Err(
            "Neural export failed; verify consent, keyring access, and the recording file"
                .to_string(),
        );
    }
    Ok(())
}

fn neural_benchmark_args(
    benchmark: &str,
    subject: Option<u16>,
    runs: Option<Vec<u8>>,
) -> Result<Vec<String>, String> {
    let mut args = vec!["-m".to_string(), "pilot.neural.benchmark".to_string()];
    match benchmark {
        "brainflow-synthetic" => {
            args.extend([
                benchmark.to_string(),
                "--seconds".to_string(),
                "2".to_string(),
            ]);
        }
        "eegbci" => {
            let subject = subject.unwrap_or(1);
            if !(1..=109).contains(&subject) {
                return Err("EEGBCI subject must be between 1 and 109".to_string());
            }
            let runs = runs.unwrap_or_else(|| vec![6, 10, 14]);
            if runs.len() < 2
                || runs.len() > 6
                || runs
                    .iter()
                    .any(|run| !matches!(run, 4 | 6 | 8 | 10 | 12 | 14))
            {
                return Err("Use 2-6 registered EEGBCI motor-imagery runs".to_string());
            }
            args.extend([
                benchmark.to_string(),
                "--subject".to_string(),
                subject.to_string(),
                "--runs".to_string(),
            ]);
            args.extend(runs.into_iter().map(|run| run.to_string()));
        }
        _ => return Err("Unsupported neural benchmark".to_string()),
    }
    Ok(args)
}

#[tauri::command]
pub async fn run_neural_benchmark(
    benchmark: String,
    subject: Option<u16>,
    runs: Option<Vec<u8>>,
) -> Result<serde_json::Value, String> {
    let args = neural_benchmark_args(&benchmark, subject, runs)?;
    tokio::task::spawn_blocking(move || {
        let output = hidden_python_command()
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|error| format!("Could not start neural benchmark: {error}"))?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr);
            let summary = detail.lines().last().unwrap_or("benchmark failed");
            return Err(format!("Neural benchmark failed: {summary}"));
        }
        serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("Neural benchmark returned invalid JSON: {error}"))
    })
    .await
    .map_err(|error| format!("Neural benchmark worker failed: {error}"))?
}

#[tauri::command]
pub fn stop_neural_sidecar(
    state: tauri::State<NeuralProcess>,
) -> Result<NeuralSidecarStatus, String> {
    stop_neural_process(&state);
    Ok(NeuralSidecarStatus {
        running: false,
        pid: None,
    })
}

pub fn stop_neural_process(state: &NeuralProcess) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl GestureCursor {
    pub fn init() -> Self {
        match Enigo::new(&Settings::default()) {
            Ok(enigo) => Self(Mutex::new(Some(enigo))),
            Err(e) => {
                eprintln!("[Heliox OS] Gesture cursor control unavailable: {}", e);
                Self(Mutex::new(None))
            }
        }
    }
}

/// Move the OS mouse cursor to an absolute screen position, driven by the
/// gesture-cursor bridge (continuous, ~30fps stream from GestureControl.svelte
/// while cursor mode is active). Bypasses the WebSocket/daemon path entirely
/// for latency — see the "Architecture decision" section in the gesture
/// cursor bridge design notes (GESTURES.md).
#[tauri::command]
pub fn move_gesture_cursor(
    state: tauri::State<GestureCursor>,
    x: i32,
    y: i32,
) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let enigo = guard
        .as_mut()
        .ok_or("Gesture cursor control is not available on this platform/session")?;
    enigo
        .move_mouse(x, y, Coordinate::Abs)
        .map_err(|e| e.to_string())
}

/// Perform a left mouse click at the cursor's current position — used for
/// the pinch-to-click gesture while cursor mode is active.
#[tauri::command]
pub fn click_gesture_cursor(state: tauri::State<GestureCursor>) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    let enigo = guard
        .as_mut()
        .ok_or("Gesture cursor control is not available on this platform/session")?;
    enigo
        .button(Button::Left, Direction::Click)
        .map_err(|e| e.to_string())
}

// 1. Command to show/hide main application window
#[tauri::command]
pub async fn toggle_window(app: AppHandle) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or("Main window not found")?;

    if window.is_visible().unwrap_or(false) {
        window.hide().map_err(|e| e.to_string())?;
    } else {
        window.show().map_err(|e| e.to_string())?;
        window.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}

// Internal worker to parse handshake ping data
/// One authenticated ping for the daemon supervisor: true when the daemon answered within the time limit.
pub(crate) async fn daemon_answers() -> bool {
    let attempt = async {
        let mut ws = connect_authenticated().await.ok()?;
        let request = serde_json::json!({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "supervisor"});
        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            request.to_string().into(),
        ))
        .await
        .ok()?;
        let reply = ws.next().await?.ok()?;
        Some(reply.is_text())
    };
    matches!(
        tokio::time::timeout(crate::supervisor::PING_TIMEOUT, attempt).await,
        Ok(Some(true))
    )
}

/// Longest wait for the reply to a one-off request (a git resolution writes one file).
const REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Read until the reply whose id is `id`. The daemon also sends this connection notifications it sends to every
/// window (late_parts, storage, Live), and those can arrive before the reply; they are skipped.
async fn read_reply<S>(ws: &mut S, id: &serde_json::Value) -> Result<serde_json::Value, String>
where
    S: futures_util::Stream<
            Item = Result<
                tokio_tungstenite::tungstenite::Message,
                tokio_tungstenite::tungstenite::Error,
            >,
        > + Unpin,
{
    use tokio_tungstenite::tungstenite::Message;
    loop {
        let message = ws
            .next()
            .await
            .ok_or_else(|| "Daemon closed before replying".to_string())?
            .map_err(|e| format!("Daemon connection failed: {e}"))?;
        let text = match message {
            Message::Text(text) => text,
            Message::Close(_) => return Err("Daemon closed before replying".to_string()),
            _ => continue, // ping, pong and binary frames are not replies
        };
        let parsed: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("Invalid daemon reply: {e}"))?;
        if parsed.get("id") == Some(id) {
            return Ok(parsed);
        }
    }
}

/// Send one request on an authenticated socket and return its reply, within `limit`.
async fn request_reply(
    ws: &mut DaemonSocket,
    request: &serde_json::Value,
    limit: std::time::Duration,
) -> Result<serde_json::Value, String> {
    let id = request
        .get("id")
        .cloned()
        .ok_or_else(|| "Request has no id".to_string())?;
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        request.to_string().into(),
    ))
    .await
    .map_err(|e| format!("Send failed: {e}"))?;
    tokio::time::timeout(limit, read_reply(ws, &id))
        .await
        .map_err(|_| REPLY_TIMED_OUT.to_string())?
}

const REPLY_TIMED_OUT: &str = "The daemon did not reply in time";

/// What a stop request got back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopAnswer {
    /// The daemon said it is stopping.
    Stopping,
    /// Nothing listened, there was no token yet, or the daemon refused: it was not reached.
    NotReached,
    /// It took the connection but did not answer in time: stuck or suspended.
    NoAnswer,
}

/// Ask the daemon this app started to stop on purpose (daemon_stop), before the app ends it: the daemon then marks
/// its run as ended cleanly (pilot.system.run_record), so quitting, updating or repairing is not reported as an
/// unclean end next time. At most 3 s.
pub(crate) async fn ask_daemon_to_stop(reason: &str) -> StopAnswer {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "daemon_stop",
        "params": {"reason": reason},
        "id": "native-stop"
    });
    let attempt = async {
        let Ok(mut ws) = connect_authenticated().await else {
            return StopAnswer::NotReached;
        };
        match request_reply(&mut ws, &request, std::time::Duration::from_secs(2)).await {
            Ok(reply)
                if reply.get("result").and_then(|r| r.get("status"))
                    == Some(&serde_json::json!("ok")) =>
            {
                StopAnswer::Stopping
            }
            Err(error) if error == REPLY_TIMED_OUT => StopAnswer::NoAnswer,
            _ => StopAnswer::NotReached,
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(3), attempt)
        .await
        .unwrap_or(StopAnswer::NoAnswer)
}

/// One authenticated request to the daemon and its result, within `limit` (the real supervisor check reads
/// startup_status with it).
#[cfg(test)]
pub(crate) async fn daemon_result(
    method: &str,
    limit: std::time::Duration,
) -> Result<serde_json::Value, String> {
    let request =
        serde_json::json!({"jsonrpc": "2.0", "method": method, "params": {}, "id": "native-check"});
    let mut ws = connect_authenticated().await?;
    let reply = request_reply(&mut ws, &request, limit).await?;
    result_or_error(reply)
}

#[tauri::command]

pub fn open_terminal() -> Result<String, String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut command = Command::new("cmd");
    crate::outside_env::apply(&mut command);
    command
        .args([
            "/C",
            &format!("start powershell -NoProfile -NoExit -Command \"cd '{}'; echo '=== Heliox OS System Terminal Active ==='\"", cwd.display())
        ])
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok("Terminal Opened Successfully".into())
}
#[tauri::command]
pub fn clear_logs() -> Result<String, String> {
    clear_log_file(&crate::pilot_log_path())
}

fn clear_log_file(log_path: &std::path::Path) -> Result<String, String> {
    if !log_path.exists() {
        return Ok("No daemon log file exists yet".into());
    }
    std::fs::write(log_path, "")
        .map_err(|error| format!("Could not clear {}: {error}", log_path.display()))?;
    Ok("Daemon log cleared".into())
}
#[tauri::command]
pub fn restart_agents() -> Result<String, String> {
    Err("No managed agent supervisor is configured; no agents were restarted".into())
}
#[tauri::command]
pub fn system_scan() -> serde_json::Value {
    let mut sys = System::new_all();
    sys.refresh_all();
    let total_mem = sys.total_memory() / (1024 * 1024);
    let used_mem = sys.used_memory() / (1024 * 1024);
    serde_json::json!({
        "scan_scope": "Resource telemetry only; no malware or threat scan was performed",
        "host_os": format!("{} ({})", System::name().unwrap_or_else(|| "Windows".into()), System::os_version().unwrap_or_else(|| "10/11".into())),
        "cpu_processor": sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default(),
        "active_threads": sys.cpus().len(),
        "memory_utilization": format!("{} MB / {} MB ({:.0}%)", used_mem, total_mem, (used_mem as f32 / total_mem as f32) * 100.0),
        "system_uptime": format!("{}h {}m", System::uptime() / 3600, (System::uptime() % 3600) / 60)
    })
}
#[tauri::command]
pub fn get_uptime() -> String {
    let mut sys = System::new_all();
    sys.refresh_all();
    let uptime = System::uptime();
    let days = uptime / 86400;
    let hours = (uptime % 86400) / 3600;
    let mins = (uptime % 3600) / 60;
    format!("{}d {}h {}m", days, hours, mins)
}
#[tauri::command]
pub fn get_dashboard_status() -> serde_json::Value {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();
    serde_json::json!({
        "connected": std::net::TcpStream::connect(("127.0.0.1", 8785)).is_ok(),
        "agents": serde_json::Value::Null,
        "cpu": format!(
            "{:.0}%",
            sys.global_cpu_usage()
        ),
        "memory": format!(
            "{:.0}%",
            (sys.used_memory() as f32
            / sys.total_memory() as f32) * 100.0
        ),
        "network_up": "Unavailable",
        "network_down": "Unavailable"
    })
}

#[tauri::command]
pub fn open_logs_folder() -> Result<(), String> {
    let log_path = crate::pilot_log_path();
    let log_dir = log_path
        .parent()
        .ok_or_else(|| "Could not resolve daemon log directory".to_string())?;

    if !log_dir.exists() {
        std::fs::create_dir_all(log_dir).map_err(|e| e.to_string())?;
    }

    crate::outside_env::open(log_dir)?;
    Ok(())
}

#[tauri::command]
pub async fn apply_git_conflict_resolution(
    _window: tauri::Window,
    path: String,
    full_block: String,
    resolved_code: String,
) -> Result<serde_json::Value, String> {
    git_resolution(&path, &full_block, &resolved_code).await
}

/// The daemon's result for one conflict resolution ({"status": "success" | "error", "message"}; the window checks
/// the status), or the error that kept the request from being answered within REPLY_TIMEOUT.
pub(crate) async fn git_resolution(
    path: &str,
    full_block: &str,
    resolved_code: &str,
) -> Result<serde_json::Value, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "apply_git_resolution",
        "params": {
            "path": path,
            "full_block": full_block,
            "resolved_code": resolved_code
        },
        "id": "native-git-resolution"
    });

    let mut ws = connect_authenticated().await?;
    let reply = request_reply(&mut ws, &request, REPLY_TIMEOUT).await?;
    result_or_error(reply)
}

fn result_or_error(reply: serde_json::Value) -> Result<serde_json::Value, String> {
    if let Some(result) = reply.get("result") {
        return Ok(result.clone());
    }
    if let Some(error) = reply.get("error") {
        return Err(error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Daemon error")
            .to_string());
    }
    Ok(reply)
}

// 5. Get the currently active global shortcut
#[tauri::command]
pub fn get_hotkey(app: AppHandle) -> String {
    crate::hotkey::load_saved_shortcut(&app)
}

// 5b. The active shortcut and, when it could not be registered at startup, why
#[tauri::command]
pub fn get_hotkey_status(app: AppHandle) -> serde_json::Value {
    serde_json::json!({
        "shortcut": crate::hotkey::load_saved_shortcut(&app),
        "default": crate::hotkey::default_shortcut(),
        "error": crate::hotkey::registration_error(),
    })
}

// 6. Update the global shortcut from the frontend settings panel
#[tauri::command]
pub fn set_hotkey(app: AppHandle, shortcut: String) -> Result<(), String> {
    crate::hotkey::update_shortcut(&app, &shortcut)
}

/// The token the window and the app's own requests authenticate with: the daemon's runtime folder first
/// (folders::Folders::runtime, by the daemon's own rule), then the places older versions used. Before 13 November
/// 2026 only those older places were read, so on macOS, and on Linux outside uid 1000's /run/user folder, the window
/// never found the token.
///
/// Returns an empty string if the file does not exist yet (daemon still starting up).
#[tauri::command]
pub fn get_auth_token() -> String {
    let folders = crate::folders::Folders::here();
    let home = dirs::home_dir().unwrap_or_default();
    let local = crate::folders::local_app_data(&|name| std::env::var(name).ok());
    let mut places = vec![folders.runtime.join("auth_token")];
    places.extend(crate::folders::older_token_places(local.as_deref(), &home));
    for path in places {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let trimmed = content.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    String::new()
}

#[tauri::command]
pub async fn extract_file_text(app: AppHandle, path: String) -> Result<String, String> {
    // 1. Canonicalize ΓÇö resolves "..", symlinks, etc.
    let canonical = std::fs::canonicalize(&path).map_err(|e| format!("Invalid path: {}", e))?;

    // 2. Check the user-consent allowlist
    // A grant is used once (file_access): the person dropped this file, and it is read now.
    let allowed = app.state::<crate::file_access::AllowedPaths>();
    if !allowed.take(&canonical) {
        return Err("Access denied: file was not dropped on Heliox's window".into());
    }

    // 3. Enforce a 50 MB size cap to prevent memory exhaustion
    const MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;
    let metadata = std::fs::metadata(&canonical).map_err(|e| format!("Cannot stat file: {}", e))?;
    if metadata.len() > MAX_FILE_BYTES {
        return Err("File too large (>50 MB)".into());
    }

    // 4. Offload blocking I/O to a background thread
    let path_str = canonical.to_string_lossy().to_string();
    tokio::task::spawn_blocking(move || extract_text_from_path(&path_str))
        .await
        .map_err(|e| format!("Task join error: {}", e))?
}

fn extract_text_from_path(path: &str) -> Result<String, String> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if extension == "pdf" {
        match pdf_extract::extract_text(path) {
            Ok(text) => Ok(text),
            Err(e) => Err(format!("Failed to parse PDF: {}", e)),
        }
    } else if extension == "docx" {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

        let mut document_xml = archive
            .by_name("word/document.xml")
            .map_err(|e| e.to_string())?;
        let mut xml_content = String::new();
        std::io::Read::read_to_string(&mut document_xml, &mut xml_content)
            .map_err(|e| e.to_string())?;

        let mut text = String::new();
        let mut in_tag = false;
        let mut tag_name = String::new();
        for c in xml_content.chars() {
            if c == '<' {
                in_tag = true;
                tag_name.clear();
            } else if c == '>' {
                in_tag = false;
                if tag_name.starts_with("w:p") {
                    text.push('\n');
                }
            } else if in_tag {
                tag_name.push(c);
            } else {
                text.push(c);
            }
        }
        Ok(text)
    } else {
        match std::fs::read_to_string(path) {
            Ok(text) => Ok(text),
            Err(e) => Err(format!("Failed to read file: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    /// A daemon that takes the connection and never completes the handshake (suspended, stuck): the app gives up
    /// after CONNECT_LIMIT instead of waiting for ever.
    #[tokio::test]
    async fn a_daemon_that_never_answers_the_handshake_is_given_up_on() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(60)).await; // holds the connection, says nothing
        });
        let started = std::time::Instant::now();

        let outcome = connect_authenticated_to(&format!("ws://{address}"), "test-token").await;

        assert!(outcome.unwrap_err().contains("did not let the app sign in"));
        let took = started.elapsed();
        assert!(took >= CONNECT_LIMIT && took < CONNECT_LIMIT + std::time::Duration::from_secs(5));
        server.abort();
    }

    #[tokio::test]
    async fn daemon_bridge_authenticates_before_returning_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let first = socket.next().await.unwrap().unwrap();
            let request: serde_json::Value =
                serde_json::from_str(first.to_text().unwrap()).unwrap();

            assert_eq!(request["method"], "auth");
            assert_eq!(request["params"]["token"], "test-token");
            socket
                .send(tokio_tungstenite::tungstenite::Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "result": {"status": "authenticated"},
                        "id": "native-auth"
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
        });

        let url = format!("ws://{address}");
        let socket = connect_authenticated_to(&url, "test-token").await;

        assert!(socket.is_ok());
        server.await.unwrap();
    }

    fn text(value: serde_json::Value) -> tokio_tungstenite::tungstenite::Message {
        tokio_tungstenite::tungstenite::Message::Text(value.to_string().into())
    }

    #[tokio::test]
    async fn one_off_requests_skip_notifications_and_return_their_own_reply() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let _auth = socket.next().await.unwrap().unwrap();
            socket
                .send(text(serde_json::json!({"jsonrpc": "2.0", "result": {"status": "authenticated"}, "id": "native-auth"})))
                .await
                .unwrap();
            let request: serde_json::Value =
                serde_json::from_str(socket.next().await.unwrap().unwrap().to_text().unwrap())
                    .unwrap();
            // What the daemon sends every window by itself, before the reply: these must not be taken for it.
            socket
                .send(text(serde_json::json!({"jsonrpc": "2.0", "method": "late_parts", "params": {"ready": ["neural"]}})))
                .await
                .unwrap();
            socket
                .send(text(serde_json::json!({"jsonrpc": "2.0", "result": {"version": "wrong"}, "id": "someone-else"})))
                .await
                .unwrap();
            socket
                .send(tokio_tungstenite::tungstenite::Message::Ping(
                    vec![1].into(),
                ))
                .await
                .unwrap();
            socket
                .send(text(serde_json::json!({"jsonrpc": "2.0", "result": {"version": "1.2.3"}, "id": request["id"]})))
                .await
                .unwrap();
            let _ = socket.next().await; // keep the socket open until the client is done
        });

        let mut ws = connect_authenticated_to(&format!("ws://{address}"), "test-token")
            .await
            .unwrap();
        let request = serde_json::json!({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "native-ping"});
        let reply = request_reply(&mut ws, &request, std::time::Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(reply["result"]["version"], "1.2.3");
        drop(ws);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn a_reply_that_never_comes_is_an_error_not_a_wrong_answer() {
        let messages = vec![
            Ok(text(
                serde_json::json!({"jsonrpc": "2.0", "method": "storage", "params": {}}),
            )),
            Ok(text(
                serde_json::json!({"jsonrpc": "2.0", "result": {"ok": true}, "id": 7}),
            )),
        ];
        let mut stream = futures_util::stream::iter(messages);

        let outcome = read_reply(&mut stream, &serde_json::json!("native-git-resolution")).await;

        assert_eq!(outcome.unwrap_err(), "Daemon closed before replying");
    }

    #[test]
    fn a_git_resolution_error_reply_is_an_error() {
        let reply = serde_json::json!({"jsonrpc": "2.0", "error": {"code": -32010, "message": "Disk full"}, "id": "native-git-resolution"});

        assert_eq!(result_or_error(reply).unwrap_err(), "Disk full");
    }

    #[tokio::test]
    async fn daemon_bridge_rejects_missing_auth_token_without_connecting() {
        let result = connect_authenticated_to("ws://127.0.0.1:1", "").await;

        assert_eq!(
            result.unwrap_err(),
            "Daemon authentication token is unavailable"
        );
    }

    #[test]
    fn synthetic_neural_launch_needs_no_external_artifact() {
        let args = neural_sidecar_args(&NeuralLaunchOptions {
            source: "synthetic".to_string(),
            artifact_path: None,
            playback_path: None,
            board_id: None,
            serial_port: None,
            lsl_name: None,
            synthetic_frequency: Some(12.0),
            record_raw: None,
            recording_file: None,
            recording_purpose: None,
            retention_days: None,
            allow_bids_export: None,
        })
        .unwrap();

        assert!(args
            .windows(2)
            .any(|pair| pair == ["--source", "synthetic"]));
        assert!(!args.iter().any(|arg| arg == "--artifact"));
    }

    #[test]
    fn live_neural_launch_requires_a_calibration_artifact() {
        let error = neural_sidecar_args(&NeuralLaunchOptions {
            source: "brainflow".to_string(),
            artifact_path: None,
            playback_path: None,
            board_id: Some(0),
            serial_port: None,
            lsl_name: None,
            synthetic_frequency: None,
            record_raw: None,
            recording_file: None,
            recording_purpose: None,
            retention_days: None,
            allow_bids_export: None,
        })
        .unwrap_err();

        assert!(error.contains("Calibration artifact is required"));
    }

    #[test]
    fn raw_neural_recording_requires_and_carries_explicit_bounded_consent() {
        let args = neural_sidecar_args(&NeuralLaunchOptions {
            source: "synthetic".to_string(),
            artifact_path: None,
            playback_path: None,
            board_id: None,
            serial_port: None,
            lsl_name: None,
            synthetic_frequency: Some(12.0),
            record_raw: Some(true),
            recording_file: None,
            recording_purpose: Some("local accessibility calibration".to_string()),
            retention_days: Some(14),
            allow_bids_export: Some(true),
        })
        .unwrap();

        assert!(args.iter().any(|arg| arg == "--record-raw"));
        assert!(args.iter().any(|arg| arg == "--allow-bids-export"));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--retention-days", "14"]));
    }

    #[test]
    fn neural_benchmarks_accept_only_fixed_no_hardware_workflows() {
        let synthetic = neural_benchmark_args("brainflow-synthetic", None, None).unwrap();
        assert!(synthetic.iter().any(|value| value == "brainflow-synthetic"));

        let recorded = neural_benchmark_args("eegbci", Some(2), Some(vec![6, 10])).unwrap();
        assert!(recorded.windows(2).any(|pair| pair == ["--subject", "2"]));
        assert!(recorded
            .windows(3)
            .any(|values| values == ["--runs", "6", "10"]));

        assert!(neural_benchmark_args("shell", None, None).is_err());
        assert!(neural_benchmark_args("eegbci", Some(0), None).is_err());
        assert!(neural_benchmark_args("eegbci", Some(1), Some(vec![1, 2])).is_err());
    }

    #[test]
    fn scan_discloses_its_limited_scope() {
        let scan = system_scan();
        assert_eq!(
            scan["scan_scope"],
            "Resource telemetry only; no malware or threat scan was performed"
        );
        assert!(scan.get("status").is_none());
    }

    #[test]
    fn restart_fails_when_no_supervisor_exists() {
        assert!(restart_agents().is_err());
    }

    #[test]
    fn clear_log_file_clears_the_requested_file() {
        let path = std::env::temp_dir().join(format!(
            "heliox-log-test-{}-{}.log",
            std::process::id(),
            System::uptime()
        ));
        std::fs::write(&path, "recorded evidence\n").unwrap();
        assert_eq!(clear_log_file(&path).unwrap(), "Daemon log cleared");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
        std::fs::remove_file(path).unwrap();
    }
}
