use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;

/// Thread-safe allowlist of file paths the user has explicitly selected
/// (e.g. via drag-and-drop). Only files in this set may be read by
/// `extract_file_text`, preventing Local File Disclosure attacks from
/// a compromised webview.
pub struct AllowedPaths(Mutex<HashSet<PathBuf>>);

impl AllowedPaths {
    pub fn new() -> Self {
        Self(Mutex::new(HashSet::new()))
    }

    /// Check whether a canonicalized path is in the allowlist.
    pub fn contains(&self, path: &PathBuf) -> bool {
        self.0.lock().map(|set| set.contains(path)).unwrap_or(false)
    }

    /// Add a canonicalized path to the allowlist.
    pub fn insert(&self, path: PathBuf) {
        if let Ok(mut set) = self.0.lock() {
            set.insert(path);
        }
    }

    /// Remove a path from the allowlist.
    pub fn remove(&self, path: &PathBuf) {
        if let Ok(mut set) = self.0.lock() {
            set.remove(path);
        }
    }
}

// -- Tauri commands --

/// Register a user-selected file path so `extract_file_text` may read it.
/// Called from the frontend immediately after a drag-and-drop event provides
/// the path, *before* requesting text extraction.
#[tauri::command]
pub fn register_allowed_path(app: tauri::AppHandle, path: String) -> Result<(), String> {
    use tauri::Manager;

    let canonical =
        std::fs::canonicalize(&path).map_err(|e| format!("Cannot resolve path: {}", e))?;

    let state = app.state::<AllowedPaths>();
    state.insert(canonical);
    Ok(())
}

/// Remove a file path from the allowlist (e.g. when the user removes an
/// attachment chip).
#[tauri::command]
pub fn revoke_allowed_path(app: tauri::AppHandle, path: String) -> Result<(), String> {
    use tauri::Manager;

    if let Ok(canonical) = std::fs::canonicalize(&path) {
        let state = app.state::<AllowedPaths>();
        state.remove(&canonical);
    }
    Ok(())
}
