//! The app's window shows the app and nothing else.
//!
//! A link in text the window renders (a model's reply, a page's text, a file's contents, a plugin's description) must
//! never take the window to another page: there, a page Heliox does not control would look like Heliox. Every
//! navigation is checked here (`on_navigation`, a plugin hook that runs for each one). Only the app's own pages load;
//! a web or mail address opens in the system's browser or mail app instead, and anything else (javascript:, data:,
//! file:, blob:) goes nowhere. The page's own click handler does the same first (ui/src/lib/utils/links.ts); this is
//! the floor under it. Commands are only reachable from the app's own origin: the capabilities file names no remote
//! URL, so a page from anywhere else could not call them even if it loaded.

use tauri::Url;

/// Whether ``url`` is one of the app's own pages: the bundled app (tauri://localhost on macOS and Linux,
/// http(s)://tauri.localhost on Windows), the blank page a window starts on, and in a debug build the dev server.
pub fn is_app_page(url: &Url) -> bool {
    match url.scheme() {
        "tauri" => url.host_str() == Some("localhost"),
        "about" => url.as_str() == "about:blank",
        "http" | "https" => {
            url.host_str() == Some("tauri.localhost")
                || (cfg!(debug_assertions)
                    && matches!(url.host_str(), Some("localhost") | Some("127.0.0.1"))
                    && url.port() == Some(1420))
        }
        _ => false,
    }
}

/// Whether ``url`` may be handed to the system to open: a web page or a mail address, nothing else.
pub fn opens_outside(url: &Url) -> bool {
    matches!(url.scheme(), "http" | "https" | "mailto") && !is_app_page(url)
}

/// The navigation hook: true lets the window load ``url``. Anything else is refused, and a web or mail address is
/// opened by the system instead.
pub fn allow(url: &Url) -> bool {
    if is_app_page(url) {
        return true;
    }
    if opens_outside(url) {
        if let Err(error) = opener::open_browser(url.as_str()) {
            eprintln!("[Heliox OS] Could not open {url} in the browser: {error}");
        }
    } else {
        eprintln!(
            "[Heliox OS] The window was not allowed to go to a {} address",
            url.scheme()
        );
    }
    false
}

/// For the page's link handler: open a web or mail address in the system's browser or mail app. Anything else is
/// refused, so the page cannot use this to open a file or run a program.
#[tauri::command]
pub fn open_external_link(url: String) -> Result<(), String> {
    let parsed = Url::parse(&url).map_err(|_| "Not a link Heliox opens".to_string())?;
    if !opens_outside(&parsed) {
        return Err("Heliox opens only web and mail links outside the app".into());
    }
    opener::open_browser(parsed.as_str()).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap()
    }

    #[test]
    fn only_the_apps_own_pages_load_in_the_window() {
        for page in [
            "tauri://localhost/",
            "tauri://localhost/index.html",
            "http://tauri.localhost/",
            "about:blank",
        ] {
            assert!(is_app_page(&url(page)), "{page}");
        }
        for page in [
            "https://example.com/",
            "http://tauri.localhost.example.com/",
            "tauri://evil/",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
            "blob:https://example.com/1",
            "about:srcdoc",
        ] {
            assert!(!is_app_page(&url(page)), "{page}");
        }
    }

    #[test]
    fn only_web_and_mail_addresses_open_outside() {
        assert!(opens_outside(&url("https://example.com/a?b=c&d")));
        assert!(opens_outside(&url("mailto:sam@example.com")));
        for page in [
            "javascript:alert(1)",
            "file:///C:/Windows/System32/calc.exe",
            "data:text/html,x",
            "tauri://localhost/",
        ] {
            assert!(!opens_outside(&url(page)), "{page}");
        }
        assert!(open_external_link("file:///etc/passwd".into()).is_err());
        assert!(open_external_link("javascript:alert(1)".into()).is_err());
        assert!(open_external_link("not a url".into()).is_err());
    }
}
