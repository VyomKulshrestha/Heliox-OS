//! The app's window shows the app and nothing else.
//!
//! A link in text the window renders (a model's reply, a page's text, a file's contents, a plugin's description) must
//! never take the window to another page: there, a page Heliox does not control would look like Heliox. Every
//! navigation is checked here (`on_navigation`, a plugin hook that runs for each one). Only the app's own pages load;
//! every other navigation is refused and opens nothing. A web or mail address opens in the system's browser or mail
//! app only when the person clicked a link: the page's one click guard (ui/src/lib/utils/links.ts) stops the click
//! and calls `open_external_link`, which checks the scheme again. This hook is the floor under that guard. Commands are only reachable from the app's own origin: the capabilities file names no remote
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

/// The navigation hook: true lets the window load ``url``; anything else is refused, and nothing is opened for it.
/// A navigation the page did not ask for by a person's click (a script setting `location`, a refresh tag, a form
/// that submits itself, an anchor followed by script) must not become the person's browser opening an address they
/// never chose, with whatever its query carries. A clicked link reaches the browser only through
/// `open_external_link`, which the page's one click guard calls for a click the person made (ui/src/lib/utils/links.ts).
/// Only the scheme and host are logged: the rest of a refused address can carry data.
pub fn allow(url: &Url) -> bool {
    if is_app_page(url) {
        return true;
    }
    eprintln!(
        "[Heliox OS] The window was not allowed to go to {}:{}; nothing was opened",
        url.scheme(),
        url.host_str()
            .map(|host| format!("//{host}"))
            .unwrap_or_default()
    );
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
    crate::outside_env::open_browser(parsed.as_str())
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
        assert!(open_external_link("tauri://localhost/settings".into()).is_err());
    }

    #[test]
    fn a_refused_navigation_opens_nothing() {
        for page in [
            "https://example.com/?token=secret",
            "mailto:sam@example.com",
            "javascript:alert(1)",
            "file:///etc/passwd",
        ] {
            assert!(!allow(&url(page)), "{page}");
        }
        assert!(allow(&url("tauri://localhost/")));
        // The hook itself never hands an address to the system: only open_external_link does, for a person's click.
        let source = include_str!("navigation.rs");
        let hook = &source[source.find("pub fn allow").unwrap()
            ..source.find("/// For the page's link handler").unwrap()];
        assert!(
            !hook.contains("opener::"),
            "the navigation hook must not open anything"
        );
    }
}
