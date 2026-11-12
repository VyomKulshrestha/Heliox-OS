use std::fs;
use std::sync::Mutex;
use tauri::{App, AppHandle, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

/// Why the active shortcut could not be registered, so the app can say so instead of printing to stderr.
static REGISTRATION_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// The shortcut used when the person has not chosen one: free on a stock install of each system.
/// macOS: Cmd+Space opens Spotlight and Ctrl+Space switches input sources, so Option+Space ("Alt+Space").
/// Windows and Linux: Ctrl+Space (GNOME switches input sources with Super+Space; KDE's KRunner uses Alt+Space).
pub fn default_shortcut() -> &'static str {
    if cfg!(target_os = "macos") {
        "Alt+Space"
    } else {
        "Ctrl+Space"
    }
}

/// Shown in Settings on Wayland, where the shortcut comes from the desktop's portal.
pub const WAYLAND_NOTE: &str = "Wayland: your desktop sets this shortcut. Heliox asks for it through the desktop when it starts; change the keys in your desktop's keyboard settings.";

/// A Wayland session on Linux (the variables a Wayland compositor sets for its clients).
pub fn wayland_session() -> bool {
    cfg!(target_os = "linux")
        && (std::env::var_os("WAYLAND_DISPLAY").is_some()
            || std::env::var("XDG_SESSION_TYPE")
                .map(|kind| kind.eq_ignore_ascii_case("wayland"))
                .unwrap_or(false))
}

pub fn registration_error() -> Option<String> {
    REGISTRATION_ERROR
        .lock()
        .ok()
        .and_then(|error| error.clone())
}

fn set_registration_error(error: Option<String>) {
    if let Ok(mut slot) = REGISTRATION_ERROR.lock() {
        *slot = error;
    }
}

fn get_shortcut_path(_app: &AppHandle) -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".config").join("heliox-os").join("shortcut.txt")
}

pub fn load_saved_shortcut(app: &AppHandle) -> String {
    let path = get_shortcut_path(app);
    let saved = fs::read_to_string(path).unwrap_or_default();
    let trimmed = saved.trim();
    if trimmed.is_empty() {
        return default_shortcut().to_string();
    }
    trimmed.to_string()
}

fn save_shortcut(app: &AppHandle, shortcut: &str) -> Result<(), String> {
    let path = get_shortcut_path(app);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Could not create shortcut settings: {e}"))?;
    }
    fs::write(path, shortcut).map_err(|e| format!("Could not save shortcut: {e}"))
}

pub fn register_hotkey(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let app_handle = app.handle().clone();
    let shortcut = load_saved_shortcut(&app_handle);
    if wayland_session() {
        // The plugin grabs keys through X11, which a Wayland session does not allow; the daemon binds the shortcut
        // through the desktop's GlobalShortcuts portal instead (daemon/pilot/system/wayland_shortcut.py).
        set_registration_error(Some(WAYLAND_NOTE.to_string()));
        return Ok(());
    }
    match do_register(&app_handle, &shortcut) {
        Ok(()) => set_registration_error(None),
        Err(e) => {
            eprintln!(
                "[Heliox OS] Warning: Failed to register hotkey {}: {}",
                shortcut, e
            );
            set_registration_error(Some(e.to_string()));
        }
    }
    Ok(())
}

pub fn do_register(app: &AppHandle, shortcut: &str) -> Result<(), Box<dyn std::error::Error>> {
    let _ = app.global_shortcut().unregister_all();

    let app_handle = app.clone();

    app.global_shortcut()
        .on_shortcut(shortcut, move |_app, _shortcut, event| {
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
    let trimmed = new_shortcut.trim();
    if trimmed.is_empty() {
        return Err("Shortcut cannot be empty".to_string());
    }

    let previous = load_saved_shortcut(app);
    if let Err(error) = do_register(app, trimmed) {
        let _ = do_register(app, &previous);
        return Err(format!("Could not register '{trimmed}': {error}"));
    }

    if let Err(error) = save_shortcut(app, trimmed) {
        let _ = do_register(app, &previous);
        return Err(error);
    }

    set_registration_error(None);
    Ok(())
}
