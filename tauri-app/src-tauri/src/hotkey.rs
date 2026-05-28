use tauri::{App, AppHandle, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use std::fs;

fn get_shortcut_path(_app: &AppHandle) -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".config").join("heliox-os").join("shortcut.txt")
}

pub fn load_saved_shortcut(app: &AppHandle) -> String {
    let path = get_shortcut_path(app);
    let saved = fs::read_to_string(path).unwrap_or_default();
    let trimmed = saved.trim();
    if trimmed.is_empty() {
        // Use platform-appropriate default shortcut
        #[cfg(target_os = "macos")]
        return "Cmd+Space".to_string();
        #[cfg(not(target_os = "macos"))]
        return "Ctrl+Space".to_string();
    }
    trimmed.to_string()
}

fn save_shortcut(app: &AppHandle, shortcut: &str) {
    let path = get_shortcut_path(app);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, shortcut);
}

pub fn register_hotkey(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let app_handle = app.handle().clone();
    let shortcut = load_saved_shortcut(&app_handle);
    do_register(&app_handle, &shortcut)?;
    Ok(())
}

pub fn do_register(app: &AppHandle, shortcut: &str) -> Result<(), Box<dyn std::error::Error>> {
    let _ = app.global_shortcut().unregister_all();

    let app_handle = app.clone();

    app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
        
        if event.state != ShortcutState::Pressed {
            return;
        }

        if let Some(window) = app_handle.get_webview_window("main") {
            let is_visible = window.is_visible().unwrap_or(false);
            let is_minimized = window.is_minimized().unwrap_or(false);
            let is_focused = window.is_focused().unwrap_or(false);
            if is_visible && !is_minimized && is_focused {
                // Window is fully visible and focused , hide it completely
                let _ = window.hide();
            } else {
                // Window is hidden or minimized , bring it up properly
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
    })?;

    Ok(())
}

pub fn update_shortcut(app: &AppHandle, new_shortcut: &str) -> Result<(), String> {
    do_register(app, new_shortcut).map_err(|e| e.to_string())?;
    save_shortcut(app, new_shortcut);
    Ok(())
}