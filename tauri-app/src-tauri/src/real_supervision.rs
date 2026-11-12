//! The supervisor restarting a real daemon after a crash and after a hang, on a CI runner.
//!
//!     cargo test real_supervisor -- --ignored --nocapture
//!
//! Needs HELIOX_REAL_SUPERVISOR_PYTHON (a Python with the daemon installed) and a complete fake home in the
//! environment (HOME, USERPROFILE, LOCALAPPDATA, no XDG_CONFIG/DATA/STATE_HOME, XDG_RUNTIME_DIR inside it on
//! Linux): the CI step sets them. The supervisor loop is the app's own (supervisor::tick); only the launcher, the
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
while daemon.poll() is None:
    if order.exists():
        what = order.read_text(encoding="utf-8").strip()
        order.unlink()
        if what == "crash":
            daemon.kill()
        elif what == "freeze":
            psutil.Process(daemon.pid).suspend()
    time.sleep(0.1)
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
    let pids = std::fs::read_to_string(host.control.join("daemons.txt")).unwrap_or_default();
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
