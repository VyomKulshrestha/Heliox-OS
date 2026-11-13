//! Which files the window may have read, and who decides it.
//!
//! `extract_file_text` reads a file only when the person put it in front of Heliox: dropped on the window. The
//! grant comes from the native drop event in Rust (`grant_dropped`, from the window's DragDrop event), never from a
//! command the page can call: before, the page called `register_allowed_path` itself, so script running in the
//! window could grant itself any file. Each grant is used once: `extract_file_text` takes it when it reads the file.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Files the person dropped on the window and that have not been read yet.
pub struct AllowedPaths(Mutex<HashSet<PathBuf>>);

impl AllowedPaths {
    pub fn new() -> Self {
        Self(Mutex::new(HashSet::new()))
    }

    /// Record a dropped file, by its canonical path. Paths that cannot be resolved are skipped.
    pub fn grant(&self, path: &Path) -> Option<PathBuf> {
        let canonical = std::fs::canonicalize(path).ok()?;
        if let Ok(mut set) = self.0.lock() {
            set.insert(canonical.clone());
        }
        Some(canonical)
    }

    /// Use a grant: true (and the grant is gone) when the canonical path was dropped and not read since.
    pub fn take(&self, path: &PathBuf) -> bool {
        self.0
            .lock()
            .map(|mut set| set.remove(path))
            .unwrap_or(false)
    }
}

/// The window's native drop event: grant each dropped file, and the paths that were granted (for the page to ask
/// for, through the "heliox://files-dropped" event emitted after the grant, so a read never races the grant).
pub fn grant_dropped(allowed: &AllowedPaths, paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|path| {
            allowed
                .grant(path)
                .map(|_| path.to_string_lossy().to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dropped_file_can_be_read_once_and_nothing_else_can() {
        let dir = std::env::temp_dir().join(format!("heliox-drop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dropped = dir.join("notes.txt");
        let other = dir.join("secret.txt");
        std::fs::write(&dropped, "a").unwrap();
        std::fs::write(&other, "b").unwrap();
        let allowed = AllowedPaths::new();
        assert_eq!(
            grant_dropped(&allowed, std::slice::from_ref(&dropped)).len(),
            1
        );
        let canonical = std::fs::canonicalize(&dropped).unwrap();
        assert!(!allowed.take(&std::fs::canonicalize(&other).unwrap()));
        assert!(allowed.take(&canonical));
        assert!(!allowed.take(&canonical), "a grant is used once");
        assert!(grant_dropped(&allowed, &[dir.join("missing.txt")]).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
