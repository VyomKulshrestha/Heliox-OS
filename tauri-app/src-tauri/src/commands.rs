use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Window, Emitter};

#[derive(Serialize, Deserialize, Clone)]
pub struct DaemonStatus {
    pub connected: bool,
    pub version: String,
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

// 2. Command to check daemon connection and ping status
#[tauri::command]
pub async fn get_daemon_status(window: tauri::Window) -> Result<DaemonStatus, String> {
    // Pass window down to the ping checker
    let status = match try_ping_daemon(window).await {
        Ok(version) => DaemonStatus {
            connected: true,
            version,
        },
        Err(_) => DaemonStatus {
            connected: false,
            version: String::new(),
        },
    };
    Ok(status)
}

// 3. Command triggered by UI input prompts
#[tauri::command]
pub async fn send_to_daemon(window: tauri::Window, method: String, params: serde_json::Value) -> Result<(), String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    // Pass window directly to send streaming chunks
    send_rpc(window, request).await
}

// 4. Command to confirm user specific milestones or plans
#[tauri::command]
pub async fn confirm_action(window: tauri::Window, plan_id: String, confirmed: bool) -> Result<(), String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "confirm",
        "params": {
            "plan_id": plan_id,
            "confirmed": confirmed
        },
        "id": 1
    });

    send_rpc(window, request).await
}

// Internal worker to parse handshake ping data
async fn try_ping_daemon(window: tauri::Window) -> Result<String, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "ping",
        "params": {},
        "id": 1
    });

    // Temporary mock response parsing since ping won't stream chunks
    let url = "ws://127.0.0.1:8785";
    use tokio_tungstenite::connect_async;
    use futures_util::{SinkExt, StreamExt};

    let (mut ws, _) = connect_async(url).await.map_err(|e| e.to_string())?;
    let msg = serde_json::to_string(&request).map_err(|e| e.to_string())?;
    ws.send(tokio_tungstenite::tungstenite::Message::Text(msg)).await.map_err(|e| e.to_string())?;

    if let Some(Ok(response)) = ws.next().await {
        let text = response.to_text().map_err(|e| e.to_string())?;
        let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let version = parsed
            .get("result")
            .and_then(|r| r.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        return Ok(version);
    }
    Err("Ping failed".to_string())
}

// Main streaming loop broadcasting raw frames back to Svelte context
async fn send_rpc(window: tauri::Window, request: serde_json::Value) -> Result<(), String> {
    use tokio_tungstenite::connect_async;
    use futures_util::{SinkExt, StreamExt};

    let url = "ws://127.0.0.1:8785";
    let (mut ws, _) = connect_async(url)
        .await
        .map_err(|e| format!("Conn failed: {}", e))?;

    let msg = serde_json::to_string(&request).map_err(|e| e.to_string())?;
    ws.send(tokio_tungstenite::tungstenite::Message::Text(msg))
        .await
        .map_err(|e| format!("Send failed: {}", e))?;

    // Actively loop over streaming messages instead of breaking instantly
    while let Some(Ok(response)) = ws.next().await {
        let text = response.to_text().map_err(|e| e.to_string())?;
        let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        
        window.emit("llm-chunk", &parsed).map_err(|e| e.to_string())?;
    }

    window.emit("llm-complete", "DONE").map_err(|e| e.to_string())?;
    Ok(())
}
#[tauri::command]
pub fn open_logs_folder(app: tauri::AppHandle) -> Result<(), String> {
    let log_dir = app
        .path()
        .app_log_dir()
        .map_err(|e| e.to_string())?;

    if !log_dir.exists() {
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    }

    opener::open(&log_dir).map_err(|e| e.to_string())?;
    Ok(())
}
#[tauri::command]
pub async fn apply_git_conflict_resolution(
    _window: tauri::Window,
    path: String,
    full_block: String,
    resolved_code: String,
) -> Result<serde_json::Value, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "apply_git_resolution",
        "params": {
            "path": path,
            "full_block": full_block,
            "resolved_code": resolved_code
        },
        "id": 1
    });

    let url = "ws://127.0.0.1:8785";
    use tokio_tungstenite::connect_async;
    use futures_util::{SinkExt, StreamExt};

    let (mut ws, _) = connect_async(url).await.map_err(|e| e.to_string())?;
    let msg = serde_json::to_string(&request).map_err(|e| e.to_string())?;
    ws.send(tokio_tungstenite::tungstenite::Message::Text(msg)).await.map_err(|e| e.to_string())?;

    if let Some(Ok(response)) = ws.next().await {
        let text = response.to_text().map_err(|e| e.to_string())?;
        let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if let Some(result) = parsed.get("result") {
            return Ok(result.clone());
        }
        if let Some(error) = parsed.get("error") {
            return Err(error.get("message").and_then(|m| m.as_str()).unwrap_or("Daemon error").to_string());
        }
        return Ok(parsed);
    }
    Err("Failed to receive response from daemon".to_string())
}

// 5. Get the currently active global shortcut
#[tauri::command]
pub fn get_hotkey(app: AppHandle) -> String {
    crate::hotkey::load_saved_shortcut(&app)
}

// 6. Update the global shortcut from the frontend settings panel
#[tauri::command]
pub fn set_hotkey(app: AppHandle, shortcut: String) -> Result<(), String> {
    crate::hotkey::update_shortcut(&app, &shortcut)
}