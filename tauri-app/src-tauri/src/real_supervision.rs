//! The supervisor restarting a real daemon after a crash and after a hang, the app stopping it on purpose (quitting,
//! an update, a repair, removal) when it answers and when it cannot, and the log panel reading the daemon's own log,
//! on a CI runner.
//!
//!     cargo test real_supervisor -- --ignored --nocapture
//!
//! Needs HELIOX_REAL_SUPERVISOR_PYTHON (a Python with the daemon installed) and a complete fake home in the
//! environment (HOME, USERPROFILE, LOCALAPPDATA, no XDG_CONFIG/DATA_HOME, XDG_STATE_HOME outside the default
//! ~/.local/state so the log panel is checked where it went wrong, XDG_RUNTIME_DIR inside it on Linux): the CI step
//! sets them. The supervisor loop is the app's own (supervisor::tick); only the launcher, the
//! notice sink and the time limits are the check's. The daemon runs under a small Python wrapper, the process the
//! supervisor owns, which on request kills the daemon (a crash) or suspends it (a hang) and exits with the daemon's
//! code, as a crashed interpreter would.
//!
//! Prints CHECK lines and fails at the end if one failed. Whatever happens, everything it started is ended.

use crate::supervisor::{self, Host, Limits, Tick, Watch};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const WRAPPER: &str = r#"
import pathlib, subprocess, sys, time
import psutil
control = pathlib.Path(sys.argv[1])
daemon = subprocess.Popen([sys.executable, "-m", "pilot.server"])
with open(control / "daemons.txt", "a", encoding="utf-8") as pids:
    pids.write(f"{daemon.pid}\n")
order = control / "order"


def tree():
    """The daemon and everything under it: on Windows a virtual environment's python.exe is a launcher, and the
    interpreter that serves is its child, so freezing the launcher alone froze nothing that answers."""
    try:
        root = psutil.Process(daemon.pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


seen = set()
while daemon.poll() is None:
    new = [process.pid for process in tree()[1:] if process.pid not in seen]
    if new:
        seen.update(new)
        with open(control / "children.txt", "a", encoding="utf-8") as pids:  # for the clean-up, whatever happens
            pids.write("".join(f"{pid}\n" for pid in new))
    if order.exists():
        what = order.read_text(encoding="utf-8").strip()
        order.unlink()
        if what == "crash":
            daemon.kill()
        elif what == "freeze":
            for process in tree():
                process.suspend()
    time.sleep(0.1)
with open(control / "exits.txt", "a", encoding="utf-8") as codes:  # how the daemon itself ended
    codes.write(f"{daemon.returncode}\n")
sys.exit(daemon.returncode or 70)
"#;

struct CheckHost {
    python: PathBuf,
    control: PathBuf,
    daemon: Mutex<Option<Child>>,
    notices: Mutex<Vec<(Instant, String)>>,
}

impl Host for CheckHost {
    fn exiting(&self) -> bool {
        false
    }
    fn paused(&self) -> bool {
        false
    }
    fn daemon(&self) -> &Mutex<Option<Child>> {
        &self.daemon
    }
    fn take_borrow_at_start(&self) -> bool {
        false
    }
    fn port_answers(&self) -> bool {
        false
    }
    fn daemon_answers(&self) -> bool {
        tauri::async_runtime::block_on(crate::commands::daemon_answers())
    }
    fn port_taken(&self) -> bool {
        false
    }
    fn launch(&self) -> Option<Child> {
        let mut command = Command::new(&self.python);
        command
            .args(["-c", WRAPPER])
            .arg(&self.control)
            .stdin(std::process::Stdio::null());
        if let Some(seen) = supervisor::take_previous_exit() {
            command.env(supervisor::PREVIOUS_EXIT_VARIABLE, seen);
        }
        command.spawn().ok()
    }
    fn notice(&self, line: &str) {
        println!("INFO notice: {line}");
        if let Ok(mut notices) = self.notices.lock() {
            notices.push((Instant::now(), line.to_string()));
        }
    }
    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

fn startup_status() -> Option<serde_json::Value> {
    tauri::async_runtime::block_on(crate::commands::daemon_result(
        "startup_status",
        Duration::from_secs(5),
    ))
    .ok()
}

/// Wait for a daemon that says it is ready; its startup_status, or None after `limit`.
fn wait_ready(limit: Duration) -> Option<serde_json::Value> {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if let Some(status) = startup_status() {
            if status.get("state").and_then(|s| s.as_str()) == Some("ready") {
                return Some(status);
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    None
}

/// Tick the supervisor until a notice containing `wanted` has been given and a new daemon has been launched after
/// it, or `limit` passes. Returns the time from `since` to the relaunch.
fn tick_until_relaunch(
    host: &CheckHost,
    watch: &mut Watch,
    wanted: &str,
    since: Instant,
    limit: Duration,
) -> Option<Duration> {
    let mut paused = false;
    let deadline = Instant::now() + limit;
    let launches_before = launches(&host.control);
    let mut last = Instant::now();
    while Instant::now() < deadline {
        host.sleep(watch.limits.check_every);
        let elapsed = watch.limits.counted(last.elapsed()); // as supervisor::run counts a check
        last = Instant::now();
        if supervisor::tick(host, watch, &mut paused, elapsed) == Tick::Stop {
            return None;
        }
        let said = host
            .notices
            .lock()
            .map(|notices| notices.iter().any(|(_, line)| line.contains(wanted)))
            .unwrap_or(false);
        if said && launches(&host.control) > launches_before {
            return Some(since.elapsed());
        }
    }
    None
}

fn launches(control: &Path) -> usize {
    std::fs::read_to_string(control.join("daemons.txt"))
        .map(|text| text.lines().count())
        .unwrap_or(0)
}

/// The exit code of the last daemon that ended while its wrapper still ran ("0": it stopped by itself, cleanly).
fn last_daemon_exit(control: &Path) -> String {
    std::fs::read_to_string(control.join("exits.txt"))
        .unwrap_or_default()
        .lines()
        .last()
        .unwrap_or("")
        .to_string()
}

/// Every daemon process any wrapper started that is still running, as "pid name".
fn still_running(control: &Path) -> Vec<String> {
    let pids = std::fs::read_to_string(control.join("daemons.txt")).unwrap_or_default()
        + &std::fs::read_to_string(control.join("children.txt")).unwrap_or_default();
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    pids.lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter_map(|pid| {
            let process = system.process(sysinfo::Pid::from_u32(pid))?;
            let gone = matches!(
                process.status(),
                sysinfo::ProcessStatus::Zombie | sysinfo::ProcessStatus::Dead
            );
            (!gone).then(|| format!("{pid} {}", process.name().to_string_lossy()))
        })
        .collect()
}

/// Stop the daemon the check owns as the app does on purpose; how long it took and how the wrapper ended.
fn stop_now(host: &CheckHost, reason: &str) -> (Duration, Option<std::process::ExitStatus>) {
    let mut child = host
        .daemon
        .lock()
        .unwrap()
        .take()
        .expect("a daemon to stop");
    let started = Instant::now();
    supervisor::stop_on_purpose(&mut child, reason);
    let took = started.elapsed();
    (took, child.try_wait().ok().flatten())
}

/// Wait up to `limit` for every daemon process to be gone.
fn all_gone(control: &Path, limit: Duration) -> Vec<String> {
    let deadline = Instant::now() + limit;
    loop {
        let left = still_running(control);
        if left.is_empty() || Instant::now() >= deadline {
            return left;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn order(control: &Path, what: &str) {
    let _ = std::fs::write(control.join("order"), what);
}

fn ended(status: &Option<serde_json::Value>) -> String {
    status
        .as_ref()
        .and_then(|s| s.get("last_run"))
        .and_then(|run| run.get("ended"))
        .and_then(|e| e.as_str())
        .unwrap_or("")
        .to_string()
}

fn notice_of(status: &Option<serde_json::Value>) -> String {
    status
        .as_ref()
        .and_then(|s| s.get("last_run"))
        .and_then(|run| run.get("notice"))
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string()
}

/// End the wrapper and every daemon any wrapper started, whatever state they are in.
fn end_everything(host: &CheckHost) {
    if let Ok(mut guard) = host.daemon.lock() {
        if let Some(mut child) = guard.take() {
            supervisor::kill_tree(&mut child);
            let _ = child.wait();
        }
    }
    let pids = std::fs::read_to_string(host.control.join("daemons.txt")).unwrap_or_default()
        + &std::fs::read_to_string(host.control.join("children.txt")).unwrap_or_default();
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    for pid in pids
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
    {
        if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
            process.kill();
        }
    }
}

#[test]
#[ignore = "starts and kills real daemons: run on a CI runner with a fake home (HELIOX_REAL_SUPERVISOR_PYTHON)"]
fn real_supervisor_restarts_a_crashed_and_a_hung_daemon() {
    let Some(python) = std::env::var_os("HELIOX_REAL_SUPERVISOR_PYTHON").map(PathBuf::from) else {
        println!("NOT RUN - HELIOX_REAL_SUPERVISOR_PYTHON is not set");
        return;
    };
    let control = std::env::var_os("RUNNER_TEMP")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("heliox-supervisor-check-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&control);
    std::fs::create_dir_all(&control).expect("a control folder");
    let host = CheckHost {
        python,
        control: control.clone(),
        daemon: Mutex::new(None),
        notices: Mutex::new(Vec::new()),
    };
    let mut watch = Watch::default();
    watch.limits = Limits {
        check_every: Duration::from_secs(1),
        startup_grace: Duration::from_secs(20),
        hang_after: Duration::from_secs(8),
        stable_after: Duration::from_secs(3600), // two crashes in a row: the back-off must double
        // a check counts for at most the pause, the ping's own limit and a second, as in the app
        longest_counted_check: Duration::from_secs(1)
            + supervisor::PING_TIMEOUT
            + Duration::from_secs(1),
        ..Limits::default()
    };
    let _ = supervisor::take_previous_exit();

    let mut results: Vec<bool> = Vec::new();
    let mut check = |name: &str, ok: bool, detail: String| {
        println!(
            "CHECK supervisor/{name}: {} - {detail}",
            if ok { "PASS" } else { "FAIL" }
        );
        results.push(ok);
    };

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        *host.daemon.lock().unwrap() = host.launch();
        watch.started();
        let first = wait_ready(Duration::from_secs(180));
        check(
            "first-start",
            first.is_some(),
            format!("ready: {}", first.is_some()),
        );

        // The log panel, "Clear logs" and "Open logs folder" use pilot_log_path: it must be the file this daemon
        // writes, under $XDG_STATE_HOME when that is set (until 13 November 2026 it was always ~/.local/state).
        let log = crate::pilot_log_path();
        let lines = std::fs::read_to_string(&log)
            .map(|text| text.lines().count())
            .unwrap_or(0);
        let expected = crate::folders::Folders::here().state.join("pilot.log");
        let state_home = std::env::var("XDG_STATE_HOME").unwrap_or_default();
        check(
            "log-panel-reads-the-daemons-log",
            lines > 0
                && log == expected
                && (state_home.is_empty() || log.starts_with(state_home.trim())),
            format!(
                "{} has {lines} lines; XDG_STATE_HOME {}",
                log.display(),
                if state_home.is_empty() {
                    "unset".to_string()
                } else {
                    format!("\"{state_home}\"")
                }
            ),
        );

        // A crash: the daemon ends; the wrapper exits with its code, as a crashed interpreter would.
        let crashed_at = Instant::now();
        order(&control, "crash");
        let back = tick_until_relaunch(
            &host,
            &mut watch,
            "Starting it again in 2 s.",
            crashed_at,
            Duration::from_secs(30),
        );
        check(
            "crash-restarted-in-the-back-off",
            back.is_some_and(|took| took <= Duration::from_secs(8)),
            format!(
                "notice \"...Starting it again in 2 s.\", started again {back:?} after the crash"
            ),
        );
        let after = wait_ready(Duration::from_secs(180));
        let said = ended(&after);
        check(
            "crash-reported-with-its-code",
            said.starts_with("stopped with exit code")
                && !notice_of(&after).to_lowercase().contains("crash"),
            format!(
                "last_run ended \"{said}\", notice \"{}\"",
                notice_of(&after)
            ),
        );

        // A second crash in a row: the wait doubles.
        let crashed_at = Instant::now();
        order(&control, "crash");
        let back = tick_until_relaunch(
            &host,
            &mut watch,
            "Starting it again in 4 s.",
            crashed_at,
            Duration::from_secs(30),
        );
        check(
            "second-crash-doubles-the-wait",
            back.is_some_and(|took| {
                took >= Duration::from_secs(4) && took <= Duration::from_secs(10)
            }),
            format!(
                "notice \"...Starting it again in 4 s.\", started again {back:?} after the crash"
            ),
        );
        let _ = wait_ready(Duration::from_secs(180));

        // A hang: the daemon is suspended; after the grace and the hang rule it is ended and started again. The
        // grace is ticked through first, while the daemon answers.
        let mut paused = false;
        let grace_until = Instant::now() + watch.limits.startup_grace + Duration::from_secs(2);
        let mut last = Instant::now();
        while Instant::now() < grace_until {
            host.sleep(watch.limits.check_every);
            let elapsed = watch.limits.counted(last.elapsed());
            last = Instant::now();
            supervisor::tick(&host, &mut watch, &mut paused, elapsed);
        }
        let frozen_at = Instant::now();
        order(&control, "freeze");
        let back = tick_until_relaunch(
            &host,
            &mut watch,
            "stopped answering, so it is being restarted.",
            frozen_at,
            Duration::from_secs(240),
        );
        let told_after = host.notices.lock().ok().and_then(|notices| {
            notices
                .iter()
                .find(|(at, line)| *at >= frozen_at && line.contains("stopped answering"))
                .map(|(at, _)| at.duration_since(frozen_at))
        });
        // The rule in wall time: told within hang_after plus at most one check (the ping waits out its own limit).
        let bound =
            watch.limits.hang_after + watch.limits.longest_counted_check + Duration::from_secs(3);
        check(
            "hang-told-within-the-rule",
            told_after.is_some_and(|took| took <= bound),
            format!("told {told_after:?} after the freeze; the rule allows {bound:?}"),
        );
        check(
            "hang-ended-and-restarted",
            back.is_some(),
            format!("notice \"...stopped answering, so it is being restarted.\", started again {back:?} after the freeze"),
        );
        let after = wait_ready(Duration::from_secs(180));
        let said = ended(&after);
        check(
            "hang-reported",
            said == "stopped answering",
            format!(
                "last_run ended \"{said}\", notice \"{}\"",
                notice_of(&after)
            ),
        );

        // Stopped on purpose while it answers (quitting, an update, a repair, removal all call stop_on_purpose):
        // the daemon is asked, ends by itself within STOP_ON_PURPOSE_WAIT, and the next start reports nothing.
        let (took, wrapper) = stop_now(&host, "the check");
        let code = last_daemon_exit(&control);
        let left = all_gone(&control, Duration::from_secs(5));
        check(
            "stop-on-purpose-ends-cleanly",
            code == "0" && took < supervisor::STOP_ON_PURPOSE_WAIT && left.is_empty(),
            format!(
                "the daemon ended by itself with code {code} after {took:?} (the fallback comes at {:?}); wrapper \
                 {wrapper:?}; still running: {left:?}",
                supervisor::STOP_ON_PURPOSE_WAIT
            ),
        );
        *host.daemon.lock().unwrap() = host.launch();
        let after = wait_ready(Duration::from_secs(180));
        let last_run = after.as_ref().and_then(|s| s.get("last_run")).cloned();
        check(
            "stop-on-purpose-next-start-reports-nothing",
            after.is_some() && matches!(last_run, None | Some(serde_json::Value::Null)),
            format!("ready: {}, last_run {last_run:?}", after.is_some()),
        );

        // Stopped on purpose while it cannot answer (suspended, with everything under it): the request gets no
        // answer within its own limit (3 s), so the app ends the daemon at once and does not wait the 5 s.
        order(&control, "freeze");
        std::thread::sleep(Duration::from_millis(1500)); // the wrapper looks for orders every 0.1 s
        let (took, wrapper) = stop_now(&host, "the check");
        let left = all_gone(&control, Duration::from_secs(5));
        let ask_limit = Duration::from_secs(3);
        check(
            "stop-on-purpose-ends-a-daemon-that-cannot-answer",
            took <= ask_limit + Duration::from_secs(2) && left.is_empty(),
            format!(
                "ended after {took:?} (asking waits at most {ask_limit:?}); wrapper {wrapper:?}; still running: \
                 {left:?}"
            ),
        );
        *host.daemon.lock().unwrap() = host.launch();
        let after = wait_ready(Duration::from_secs(180));
        let last_run = after.as_ref().and_then(|s| s.get("last_run")).cloned();
        // Not a clean end: the daemon never marked its run as ended, so the next start says so.
        check(
            "stop-on-purpose-of-a-daemon-that-cannot-answer-is-reported",
            after.is_some() && last_run.as_ref().is_some_and(|run| !run.is_null()),
            format!(
                "ready: {}, last_run ended \"{}\", notice \"{}\"",
                after.is_some(),
                ended(&after),
                notice_of(&after)
            ),
        );
    }));
    end_everything(&host);
    let _ = std::fs::remove_dir_all(&control);
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
    assert!(
        !results.is_empty() && results.iter().all(|ok| *ok),
        "a CHECK failed"
    );
}
