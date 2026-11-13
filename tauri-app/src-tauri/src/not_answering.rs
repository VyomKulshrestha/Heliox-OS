//! A second launch whose first copy holds the lock but does not answer: tell the person in a window of its own.
//!
//! Without this, double-clicking Heliox while a hung copy (with no window) holds the lock did nothing at all, every
//! time. The second launch now shows one small dialog that says Heliox is already running but not answering, how to
//! close it on this system, and has a Close button. It never ends the other copy itself.
//!
//! No new crate: each system's own dialog tool, all present by default, run with arguments (never through a shell).
//! - Windows: PowerShell with Windows Forms (a label and a "Close" button).
//! - macOS: `osascript` `display alert` with a "Close" button.
//! - Linux: the first of `zenity`, `kdialog`, `xmessage` that exists; if none does, `notify-send`.
//!
//! If no tool could be started, the sentence still goes to stderr.

use std::process::Command;

pub(crate) const TITLE: &str = "Heliox";

/// The words for this system.
pub(crate) fn message(os: &str) -> String {
    let how = match os {
        "windows" => {
            "Close it with Quit on the Heliox icon in the notification area of the taskbar, or in Task Manager \
             (Ctrl+Shift+Esc): select Heliox-OS and choose End task."
        }
        "macos" => {
            "Close it with Quit on the Heliox icon in the menu bar, or in Activity Monitor: select Heliox-OS and \
             choose Quit (or press Option+Command+Esc and choose Force Quit)."
        }
        _ => {
            "Close it with Quit on the Heliox icon in the system tray, or in your system monitor: select heliox-os \
             and end it."
        }
    };
    format!(
        "Heliox is already running but is not answering, so it cannot open a window. {how} Then open Heliox again."
    )
}

/// One way to show the dialog: a program and its arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dialog {
    pub(crate) program: &'static str,
    pub(crate) args: Vec<String>,
}

/// The dialogs to try on this system, in order.
pub(crate) fn dialogs(os: &str, text: &str) -> Vec<Dialog> {
    match os {
        "windows" => {
            // Single-quoted PowerShell strings: a quote inside is written twice.
            let quoted = text.replace('\'', "''");
            let script = format!(
                "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; \
                 $f = New-Object Windows.Forms.Form; $f.Text = '{TITLE}'; $f.Width = 460; $f.Height = 210; \
                 $f.StartPosition = 'CenterScreen'; $f.TopMost = $true; $f.FormBorderStyle = 'FixedDialog'; \
                 $f.MaximizeBox = $false; $f.MinimizeBox = $false; \
                 $l = New-Object Windows.Forms.Label; $l.Text = '{quoted}'; $l.Left = 16; $l.Top = 14; \
                 $l.Width = 420; $l.Height = 100; $f.Controls.Add($l); \
                 $b = New-Object Windows.Forms.Button; $b.Text = 'Close'; $b.Left = 344; $b.Top = 122; \
                 $b.Width = 90; $b.DialogResult = 'OK'; $f.AcceptButton = $b; $f.CancelButton = $b; \
                 $f.Controls.Add($b); [void]$f.ShowDialog()"
            );
            vec![Dialog {
                program: "powershell",
                args: vec![
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-WindowStyle".into(),
                    "Hidden".into(),
                    "-Command".into(),
                    script,
                ],
            }]
        }
        "macos" => {
            // AppleScript string: backslashes and double quotes escaped.
            let quoted = text.replace('\\', "\\\\").replace('"', "\\\"");
            vec![Dialog {
                program: "osascript",
                args: vec![
                    "-e".into(),
                    format!(
                        "display alert \"{TITLE} is not answering\" message \"{quoted}\" buttons {{\"Close\"}} default button \"Close\""
                    ),
                ],
            }]
        }
        _ => vec![
            Dialog {
                program: "zenity",
                args: vec![
                    "--warning".into(),
                    format!("--title={TITLE}"),
                    "--ok-label=Close".into(),
                    "--no-markup".into(),
                    format!("--text={text}"),
                ],
            },
            Dialog {
                program: "kdialog",
                args: vec![
                    "--title".into(),
                    TITLE.into(),
                    "--ok-label".into(),
                    "Close".into(),
                    "--sorry".into(),
                    text.into(),
                ],
            },
            Dialog {
                program: "xmessage",
                args: vec![
                    "-center".into(),
                    "-buttons".into(),
                    "Close:0".into(),
                    text.into(),
                ],
            },
            Dialog {
                program: "notify-send",
                args: vec![
                    "--urgency=critical".into(),
                    format!("{TITLE} is not answering"),
                    text.into(),
                ],
            },
        ],
    }
}

fn this_system() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

/// Show the dialog and wait until the person closes it. Returns which program showed it, if any.
pub(crate) fn tell_the_person() -> Option<&'static str> {
    let text = message(this_system());
    eprintln!("[Heliox OS] {text}");
    for dialog in dialogs(this_system(), &text) {
        let mut command = Command::new(dialog.program);
        crate::outside_env::apply(&mut command);
        command.args(&dialog.args);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW: no console flashes up behind the dialog
        }
        // A tool that is missing fails to start, and one with no display to use fails: try the next one.
        if command.status().is_ok_and(|status| status.success()) {
            return Some(dialog.program);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_words_say_what_happened_and_how_to_close_it_on_each_system() {
        for (os, place) in [
            ("windows", "Task Manager"),
            ("macos", "Activity Monitor"),
            ("linux", "system monitor"),
        ] {
            let text = message(os);
            assert!(text.starts_with("Heliox is already running but is not answering"));
            assert!(text.contains(place), "{os}: {text}");
            assert!(text.contains("Quit on the Heliox icon"));
            assert!(text.ends_with("Then open Heliox again."));
        }
    }

    #[test]
    fn every_system_has_a_dialog_with_a_close_button_and_nothing_that_ends_the_other_copy() {
        for os in ["windows", "macos", "linux"] {
            let text = message(os);
            let tried = dialogs(os, &text);
            assert!(!tried.is_empty());
            let first = tried[0].args.join(" ");
            assert!(first.contains("Close"), "{os}: {first}");
            for dialog in &tried {
                let all = dialog.args.join(" ").to_lowercase();
                assert!(
                    !all.contains("stop-process")
                        && !all.contains("taskkill")
                        && !all.contains("pkill"),
                    "{os} must never end the other copy"
                );
            }
        }
        let linux: Vec<&str> = dialogs("linux", "x").iter().map(|d| d.program).collect();
        assert_eq!(linux, ["zenity", "kdialog", "xmessage", "notify-send"]);
    }

    /// Builds the real form with the real words and disposes of it instead of showing it: the script parses and
    /// Windows Forms loads, with no window on the screen.
    #[cfg(target_os = "windows")]
    #[test]
    fn the_windows_dialog_script_runs() {
        let dialog = dialogs("windows", &message("windows")).remove(0);
        let mut args = dialog.args.clone();
        let script = args
            .pop()
            .unwrap()
            .replace("[void]$f.ShowDialog()", "$f.Dispose()");
        assert!(script.ends_with("$f.Dispose()"));
        args.push(script);
        let status = Command::new(dialog.program)
            .args(&args)
            .status()
            .expect("PowerShell");
        assert!(status.success());
    }

    #[test]
    fn quotes_in_the_words_cannot_break_out_of_the_script() {
        let windows = dialogs("windows", "it's \"here\"");
        assert!(windows[0].args.last().unwrap().contains("'it''s \"here\"'"));
        let mac = dialogs("macos", "a \"b\" c\\d");
        assert!(mac[0].args[1].contains("a \\\"b\\\" c\\\\d"));
    }
}
