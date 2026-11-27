//! The window's page when the web view's content process ends under it (a crash, a memory limit, a kill).
//!
//! Without this, on Linux the window went blank and stayed blank until the person restarted Heliox: wry connects
//! nothing to WebKitGTK's `web-process-terminated`. WebView2 on Windows has the same gap (wry adds no `ProcessFailed`
//! handler: the window shows the web view's own error page). On macOS Tauri already reloads the page by itself, but
//! without any limit, so a page that ends its process on every load would reload for ever.
//!
//! The same rule everywhere the toolkit gives a hook: one line on stderr, then the page is reloaded. The page
//! reconnects to the daemon as it does at start; the daemon, its tasks and an approval it is waiting on are not
//! touched (a plan approval comes back through `resume_task`, the same path as a dropped connection). At most
//! `MOST_RELOADS` reloads within `WITHIN`: one more end in that time and the app stops reloading and shows a native
//! message that the window could not be shown, with a button that restarts Heliox (the daemon keeps running and the
//! new copy uses it).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Reloads allowed within `WITHIN`. A single crash recovers at once; a page that ends its process on every load (a
/// broken graphics driver, say) stops after a few seconds instead of flickering for ever.
pub(crate) const MOST_RELOADS: usize = 3;
pub(crate) const WITHIN: Duration = Duration::from_secs(120);

pub(crate) const TITLE: &str = "Heliox";
pub(crate) const GAVE_UP: &str = "Heliox's window stopped working three times in two minutes, so it was not \
     loaded again. Heliox is still running in the background, and nothing you asked it to do was stopped. Restart \
     Heliox to open its window again.";
pub(crate) const RESTART: &str = "Restart Heliox";
pub(crate) const CLOSE: &str = "Close";

/// What to do after the page's process ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Next {
    /// Reload; `count` is this reload's number within `WITHIN`.
    Reload { count: usize },
    /// Too many in `WITHIN`: tell the person and offer a restart.
    GiveUp,
}

/// The reloads of the last `WITHIN`.
#[derive(Default)]
pub(crate) struct Reloads {
    recent: VecDeque<Instant>,
}

impl Reloads {
    pub(crate) fn after_end(&mut self, now: Instant) -> Next {
        while self
            .recent
            .front()
            .is_some_and(|then| now.saturating_duration_since(*then) >= WITHIN)
        {
            self.recent.pop_front();
        }
        if self.recent.len() >= MOST_RELOADS {
            return Next::GiveUp;
        }
        self.recent.push_back(now);
        Next::Reload {
            count: self.recent.len(),
        }
    }
}

/// The line for stderr when the page's process ended and the page is reloaded.
pub(crate) fn reload_line(why: &str, count: usize) -> String {
    format!(
        "[Heliox OS] The window's web process ended ({why}); reloading the page ({count} of {MOST_RELOADS} allowed in {} s)",
        WITHIN.as_secs()
    )
}

pub(crate) fn gave_up_line(why: &str) -> String {
    format!(
        "[Heliox OS] The window's web process ended ({why}) again after {MOST_RELOADS} reloads in {} s; not reloading, offering a restart",
        WITHIN.as_secs()
    )
}

/// Restart Heliox from the main thread: there the app ends without its exit event, so the daemon it started keeps
/// running (and the new copy finds and uses it); from another thread the exit event would stop the daemon. The
/// single-instance lock is let go first, or the new copy would find this one still holding it and hand over to it.
pub(crate) fn restart(app: &tauri::AppHandle) {
    let handle = app.clone();
    let scheduled = app.run_on_main_thread(move || {
        use tauri::Manager;
        if let Some(first) = handle.try_state::<crate::instance::FirstCopy>() {
            first.release();
        }
        crate::commands::stop_neural_process(
            handle.state::<crate::commands::NeuralProcess>().inner(),
        );
        eprintln!("[Heliox OS] Restarting Heliox to open its window again");
        handle.restart();
    });
    if let Err(error) = scheduled {
        eprintln!("[Heliox OS] Could not restart Heliox: {error}");
    }
}

/// Linux: WebKitGTK's `web-process-terminated` on the main window's web view.
#[cfg(target_os = "linux")]
pub(crate) fn watch(window: &tauri::WebviewWindow) {
    use tauri::Manager;
    let app = window.app_handle().clone();
    let watching = window.with_webview(move |platform| {
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;
        use webkit2gtk::WebViewExt;

        let view = platform.inner();
        let reloads = RefCell::new(Reloads::default());
        let reloading = Rc::new(Cell::new(false));
        let loaded = reloading.clone();
        view.connect_load_changed(move |_, event| {
            if event == webkit2gtk::LoadEvent::Finished && loaded.replace(false) {
                eprintln!("[Heliox OS] The window's page loaded again after its web process ended");
            }
        });
        view.connect_web_process_terminated(move |view, reason| {
            let why = format!("{reason:?}");
            let next = reloads.borrow_mut().after_end(Instant::now());
            match next {
                Next::Reload { count } => {
                    eprintln!("{}", reload_line(&why, count));
                    reloading.set(true);
                    view.reload();
                }
                Next::GiveUp => {
                    eprintln!("{}", gave_up_line(&why));
                    offer_restart_gtk(&app);
                }
            }
        });
    });
    if let Err(error) = watching {
        eprintln!("[Heliox OS] The window's web process is not watched: {error}");
    }
}

/// A GTK message on the app's own main loop, with Restart and Close.
#[cfg(target_os = "linux")]
fn offer_restart_gtk(app: &tauri::AppHandle) {
    use gtk::prelude::*;
    let dialog = gtk::MessageDialog::new(
        None::<&gtk::Window>,
        gtk::DialogFlags::MODAL,
        gtk::MessageType::Error,
        gtk::ButtonsType::None,
        GAVE_UP,
    );
    dialog.set_title(TITLE);
    dialog.add_button(CLOSE, gtk::ResponseType::Close);
    dialog.add_button(RESTART, gtk::ResponseType::Accept);
    dialog.set_default_response(gtk::ResponseType::Accept);
    let app = app.clone();
    dialog.connect_response(move |dialog, response| {
        dialog.close();
        if response == gtk::ResponseType::Accept {
            restart(&app);
        }
    });
    dialog.show_all();
}

/// Windows: WebView2's `ProcessFailed` on the main window's web view. Only the page's own process ending is
/// reloaded; the whole web view ending (its browser process) cannot be reloaded and offers the restart at once. A
/// GPU or utility process ending is recovered by WebView2 itself, and an unresponsive page is not an ended one.
#[cfg(target_os = "windows")]
pub(crate) fn watch(window: &tauri::WebviewWindow) {
    use tauri::Manager;
    let app = window.app_handle().clone();
    let watching = window.with_webview(move |platform| {
        use webview2_com::Microsoft::Web::WebView2::Win32::{
            COREWEBVIEW2_PROCESS_FAILED_KIND,
            COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED,
            COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED,
        };
        use webview2_com::ProcessFailedEventHandler;

        let core = match unsafe { platform.controller().CoreWebView2() } {
            Ok(core) => core,
            Err(error) => {
                eprintln!("[Heliox OS] The window's web process is not watched: {error}");
                return;
            }
        };
        let mut reloads = Reloads::default();
        let handler = ProcessFailedEventHandler::create(Box::new(move |sender, args| {
            let Some(args) = args else { return Ok(()) };
            let mut kind = COREWEBVIEW2_PROCESS_FAILED_KIND::default();
            unsafe { args.ProcessFailedKind(&mut kind)? };
            if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_BROWSER_PROCESS_EXITED {
                eprintln!("{}", gave_up_line("its browser process exited"));
                offer_restart_with_a_tool(&app);
                return Ok(());
            }
            if kind != COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED {
                return Ok(());
            }
            match reloads.after_end(Instant::now()) {
                Next::Reload { count } => {
                    eprintln!("{}", reload_line("its render process exited", count));
                    if let Some(sender) = sender {
                        unsafe { sender.Reload()? };
                    }
                }
                Next::GiveUp => {
                    eprintln!("{}", gave_up_line("its render process exited"));
                    offer_restart_with_a_tool(&app);
                }
            }
            Ok(())
        }));
        let mut token = 0i64;
        if let Err(error) = unsafe { core.add_ProcessFailed(&handler, &mut token) } {
            eprintln!("[Heliox OS] The window's web process is not watched: {error}");
        }
    });
    if let Err(error) = watching {
        eprintln!("[Heliox OS] The window's web process is not watched: {error}");
    }
}

/// macOS: Tauri reloads by itself unless the app sets this hook; set, it keeps the reload and adds the limit.
#[cfg(target_os = "macos")]
pub(crate) fn on_content_process_terminate(webview: &tauri::Webview) {
    use std::sync::Mutex;
    use tauri::Manager;
    static RELOADS: Mutex<Option<Reloads>> = Mutex::new(None);
    let next = match RELOADS.lock() {
        Ok(mut guard) => guard
            .get_or_insert_with(Reloads::default)
            .after_end(Instant::now()),
        Err(_) => Next::Reload { count: 1 },
    };
    match next {
        Next::Reload { count } => {
            eprintln!("{}", reload_line("its content process ended", count));
            if let Err(error) = webview.reload() {
                eprintln!("[Heliox OS] Could not reload the window's page: {error}");
            }
        }
        Next::GiveUp => {
            eprintln!("{}", gave_up_line("its content process ended"));
            offer_restart_with_a_tool(webview.app_handle());
        }
    }
}

/// Windows and macOS: the system's own dialog tool (as not_answering.rs does), on a thread of its own so the
/// window's thread is not held; the restart, when chosen, goes back to the main thread.
#[cfg(any(target_os = "windows", target_os = "macos"))]
fn offer_restart_with_a_tool(app: &tauri::AppHandle) {
    let app = app.clone();
    std::thread::spawn(move || {
        let Some(dialog) = restart_dialog(if cfg!(target_os = "windows") {
            "windows"
        } else {
            "macos"
        }) else {
            return;
        };
        let mut command = std::process::Command::new(dialog.program);
        crate::outside_env::apply(&mut command);
        command.args(&dialog.args);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        match command.output() {
            Ok(output) if chose_restart(&String::from_utf8_lossy(&output.stdout)) => restart(&app),
            Ok(_) => {}
            Err(error) => {
                eprintln!("[Heliox OS] Could not show the message about the window: {error}")
            }
        }
    });
}

/// The dialog with Close and Restart for a system; its standard output names the button pressed.
#[cfg_attr(target_os = "linux", allow(dead_code))] // Linux shows a GTK dialog in the app's own process
pub(crate) fn restart_dialog(os: &str) -> Option<crate::not_answering::Dialog> {
    match os {
        "windows" => {
            let quoted = GAVE_UP.replace('\'', "''");
            // A form with the two buttons by name (a message box would only say OK and Cancel).
            let script = format!(
                "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; \
                 $f = New-Object Windows.Forms.Form; $f.Text = '{TITLE}'; $f.Width = 480; $f.Height = 210; \
                 $f.StartPosition = 'CenterScreen'; $f.TopMost = $true; $f.FormBorderStyle = 'FixedDialog'; \
                 $f.MaximizeBox = $false; $f.MinimizeBox = $false; \
                 $l = New-Object Windows.Forms.Label; $l.Text = '{quoted}'; $l.Left = 16; $l.Top = 14; \
                 $l.Width = 440; $l.Height = 100; $f.Controls.Add($l); \
                 $r = New-Object Windows.Forms.Button; $r.Text = '{RESTART}'; $r.Left = 236; $r.Top = 122; \
                 $r.Width = 120; $r.DialogResult = 'OK'; $f.AcceptButton = $r; $f.Controls.Add($r); \
                 $c = New-Object Windows.Forms.Button; $c.Text = '{CLOSE}'; $c.Left = 364; $c.Top = 122; \
                 $c.Width = 90; $c.DialogResult = 'Cancel'; $f.CancelButton = $c; $f.Controls.Add($c); \
                 if ($f.ShowDialog() -eq 'OK') {{ '{RESTART}' }} else {{ '{CLOSE}' }}"
            );
            Some(crate::not_answering::Dialog {
                program: "powershell",
                args: vec![
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-WindowStyle".into(),
                    "Hidden".into(),
                    "-Command".into(),
                    script,
                ],
            })
        }
        "macos" => {
            let quoted = GAVE_UP.replace('\\', "\\\\").replace('"', "\\\"");
            Some(crate::not_answering::Dialog {
                program: "osascript",
                args: vec![
                    "-e".into(),
                    format!(
                        "display alert \"{TITLE}\" message \"{quoted}\" buttons {{\"{CLOSE}\", \"{RESTART}\"}} default button \"{RESTART}\""
                    ),
                ],
            })
        }
        _ => None,
    }
}

/// Whether the dialog's output says Restart was pressed (PowerShell prints the word; osascript prints
/// "button returned:Restart Heliox").
#[cfg_attr(target_os = "linux", allow(dead_code))]
pub(crate) fn chose_restart(output: &str) -> bool {
    output.lines().any(|line| line.trim().ends_with(RESTART))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_reloads_in_two_minutes_then_a_restart_is_offered() {
        let start = Instant::now();
        let mut reloads = Reloads::default();
        assert_eq!(reloads.after_end(start), Next::Reload { count: 1 });
        assert_eq!(
            reloads.after_end(start + Duration::from_secs(1)),
            Next::Reload { count: 2 }
        );
        assert_eq!(
            reloads.after_end(start + Duration::from_secs(2)),
            Next::Reload { count: 3 }
        );
        assert_eq!(
            reloads.after_end(start + Duration::from_secs(3)),
            Next::GiveUp
        );
        // Giving up does not count as a reload, so it stays given up within the window.
        assert_eq!(
            reloads.after_end(start + Duration::from_secs(4)),
            Next::GiveUp
        );
    }

    #[test]
    fn ends_far_apart_always_reload() {
        let start = Instant::now();
        let mut reloads = Reloads::default();
        for step in 0..10u64 {
            let next = reloads.after_end(start + WITHIN * step as u32);
            assert_eq!(next, Next::Reload { count: 1 }, "end number {step}");
        }
    }

    #[test]
    fn an_old_reload_leaves_the_window_and_frees_a_place() {
        let start = Instant::now();
        let mut reloads = Reloads::default();
        reloads.after_end(start);
        reloads.after_end(start + Duration::from_secs(60));
        reloads.after_end(start + Duration::from_secs(61));
        assert_eq!(
            reloads.after_end(start + WITHIN + Duration::from_secs(1)),
            Next::Reload { count: 3 }
        );
    }

    #[test]
    fn the_lines_and_the_message_say_what_happened() {
        assert!(reload_line("Crashed", 1).contains("reloading the page (1 of 3 allowed in 120 s)"));
        assert!(gave_up_line("Crashed").contains("not reloading, offering a restart"));
        assert!(GAVE_UP.contains("still running in the background"));
        assert!(GAVE_UP.ends_with("Restart Heliox to open its window again."));
    }

    #[test]
    fn the_dialogs_offer_close_and_restart_and_their_answers_are_read() {
        for os in ["windows", "macos"] {
            let dialog = restart_dialog(os).expect("a dialog");
            let all = dialog.args.join(" ");
            assert!(all.contains(RESTART) && all.contains(CLOSE), "{os}: {all}");
        }
        assert!(restart_dialog("linux").is_none()); // GTK in the app's own process there
        assert!(chose_restart("button returned:Restart Heliox\n"));
        assert!(chose_restart("Restart Heliox\r\n"));
        assert!(!chose_restart("button returned:Close\n"));
        assert!(!chose_restart("Close\r\n"));
        assert!(!chose_restart(""));
    }
}
