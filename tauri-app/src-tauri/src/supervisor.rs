//! Keeps the daemon the app started alive.
//!
//! The app starts the Python daemon at launch (`spawn_daemon` in main.rs). Before this, nothing watched it: if it
//! crashed or hung, the window showed "Reconnecting..." until the person restarted the app. Now a thread checks the
//! daemon the app owns every few seconds. A daemon that exited is started again after a back-off (2 s, doubling to
//! 60 s, reset after five quiet minutes); one that stops answering its RPC ping for HANG_AFTER (a minute of wall
//! time, measured between checks) after its start-up grace is killed and then started again. One check counts for
//! at most LONGEST_COUNTED_CHECK, so a sleep and wake, which makes one gap between checks look long, is never a hang
//! by itself. Each step is announced on the `daemon-supervisor` event in one plain line, and the window reconnects on
//! its own as it always did.
//!
//! A daemon the app did not start (for example one run from a terminal during development, found already listening
//! at launch) is never touched: `DaemonProcess` holds `None` and the supervisor only watches its port ("borrowed").
//!
//! A daemon that exits with code 3 found another copy serving the port (`AlreadyRunningError` in the daemon). That is
//! not a crash: there is no restart and no alarm, one calm line says the window is using the copy that is running,
//! and the supervisor borrows that copy too. If a borrowed copy's port stays silent for TAKE_OVER_AFTER, it has gone
//! and this app starts its own daemon.

use std::path::Path;
use std::process::{Child, Command};
use std::time::Duration;

/// What the supervisor saw end the daemon it restarts, passed to the new one in this variable for its run record
/// (pilot.system.run_record): an exit code, "signal" (ended by a signal) or "hung" (it stopped answering and was
/// ended). Set once per restart and taken by the next launch.
pub(crate) const PREVIOUS_EXIT_VARIABLE: &str = "HELIOX_PREVIOUS_EXIT";
static PREVIOUS_EXIT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
/// How long a daemon asked to stop on purpose gets to end before it is ended (it marks its run clean at once).
pub(crate) const STOP_ON_PURPOSE_WAIT: Duration = Duration::from_secs(5);

pub(crate) fn note_previous_exit(code: Option<i32>) {
    if code == Some(ANOTHER_COPY_EXIT_CODE) {
        return; // not an end of the daemon this app ran
    }
    let seen = code.map_or_else(|| "signal".to_string(), |code| code.to_string());
    if let Ok(mut slot) = PREVIOUS_EXIT.lock() {
        if slot.as_deref() != Some("hung") {
            *slot = Some(seen); // the exit of a daemon ended for hanging is that end, already noted
        }
    }
}

pub(crate) fn note_hung() {
    if let Ok(mut slot) = PREVIOUS_EXIT.lock() {
        *slot = Some("hung".to_string());
    }
}

pub(crate) fn take_previous_exit() -> Option<String> {
    PREVIOUS_EXIT.lock().ok().and_then(|mut slot| slot.take())
}

/// End the daemon this app started on purpose (quitting, an update, a repair, removal): ask it to stop first, so
/// it marks its run as ended cleanly, give it STOP_ON_PURPOSE_WAIT, then end whatever is left. One that took the
/// request and did not answer is ended at once, and the next start says it had stopped answering: the app leaves
/// that in the state folder (folders::Folders::app_note_file), because after quitting no app is left to pass it.
pub(crate) fn stop_on_purpose(child: &mut Child, reason: &str) {
    let answer = tauri::async_runtime::block_on(crate::commands::ask_daemon_to_stop(reason));
    if answer == crate::commands::StopAnswer::Stopping {
        let waited = std::time::Instant::now();
        while waited.elapsed() < STOP_ON_PURPOSE_WAIT {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    if let Some(note) = note_for(answer) {
        leave_note(&crate::folders::Folders::here().app_note_file(), note);
    }
    kill_tree(child);
}

/// What the app leaves for the next start when it had to end a daemon that did not answer its stop; the run record
/// says "stopped answering" for it (pilot.system.run_record).
pub(crate) const NO_ANSWER_AT_STOP: &str = "hung-at-stop";

/// The note a stop leaves: only for a daemon that took the request and did not answer. One that was not reached
/// (still starting, already gone) is not said to have stopped answering.
fn note_for(answer: crate::commands::StopAnswer) -> Option<&'static str> {
    (answer == crate::commands::StopAnswer::NoAnswer).then_some(NO_ANSWER_AT_STOP)
}

fn leave_note(file: &Path, note: &str) {
    if let Some(folder) = file.parent() {
        let _ = std::fs::create_dir_all(folder);
    }
    let _ = std::fs::write(file, note);
}

/// Start the daemon in a process group of its own on macOS and Linux, so kill_tree can end it and everything under
/// it with one signal, including children that left its tree. Nothing to do on Windows (kill_tree walks the tree).
pub(crate) fn own_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(not(unix))]
    let _ = command;
}

pub(crate) const CHECK_EVERY: Duration = Duration::from_secs(5);
/// First start-up can load speech models for well over a minute on a slow or busy machine.
pub(crate) const STARTUP_GRACE: Duration = Duration::from_secs(300);
/// The daemon answers ping from its event loop; a minute without an answer means that loop is stuck.
pub(crate) const HANG_AFTER: Duration = Duration::from_secs(60);
/// After this long without a restart, the back-off starts again from 2 s.
pub(crate) const STABLE_AFTER: Duration = Duration::from_secs(300);
pub(crate) const PING_TIMEOUT: Duration = Duration::from_secs(10);
/// The most one check counts for: the pause between checks, a ping that waited out its whole limit, and a second.
/// A longer gap (the computer slept, the process was stopped) counts as this much, never as a minute of silence.
pub(crate) const LONGEST_COUNTED_CHECK: Duration =
    Duration::from_secs(CHECK_EVERY.as_secs() + PING_TIMEOUT.as_secs() + 1);
/// The daemon's exit code when another copy already serves its port (pilot.server main, AlreadyRunningError).
pub(crate) const ANOTHER_COPY_EXIT_CODE: i32 = 3;
/// While borrowing another copy's daemon: how long its port may stay silent before this app starts its own. A minute,
/// so a developer restarting a daemon from a source checkout (import, data-folder check, tokens, listen) gets the port
/// back before the app takes it.
pub(crate) const TAKE_OVER_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Observation {
    Exited,
    /// Exited with ANOTHER_COPY_EXIT_CODE: another copy serves the port.
    AnotherCopy,
    Answered,
    Silent,
}

/// What an exit status means: code 3 is "another copy is serving", anything else is a crash.
pub(crate) fn exit_observation(code: Option<i32>) -> Observation {
    if code == Some(ANOTHER_COPY_EXIT_CODE) {
        Observation::AnotherCopy
    } else {
        Observation::Exited
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    Wait,
    Restart {
        after: Duration,
    },
    KillHung,
    /// Another copy serves the port: drop the exited child, watch that copy, say so once.
    Borrow,
    /// The borrowed copy's port went silent: start this app's own daemon now.
    TakeOver,
}

/// The supervisor's time limits: the constants above, or shorter ones a real check sets so it does not wait out
/// minutes on a CI runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    pub(crate) check_every: Duration,
    pub(crate) startup_grace: Duration,
    pub(crate) hang_after: Duration,
    pub(crate) stable_after: Duration,
    pub(crate) take_over_after: Duration,
    pub(crate) longest_counted_check: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            check_every: CHECK_EVERY,
            startup_grace: STARTUP_GRACE,
            hang_after: HANG_AFTER,
            stable_after: STABLE_AFTER,
            take_over_after: TAKE_OVER_AFTER,
            longest_counted_check: LONGEST_COUNTED_CHECK,
        }
    }
}

impl Limits {
    /// What a gap of `measured` between two checks counts for: its wall time, up to longest_counted_check.
    pub(crate) fn counted(&self, measured: Duration) -> Duration {
        measured.min(self.longest_counted_check)
    }
}

/// Pure bookkeeping for one owned daemon: what each observation means. Tested without processes or sockets.
#[derive(Debug, Default)]
pub(crate) struct Watch {
    pub(crate) restarts_in_row: u32,
    /// Watching a daemon another copy started; observations are "its port answers" or not.
    pub(crate) borrowed: bool,
    pub(crate) limits: Limits,
    since_start: Duration,
    since_answer: Duration,
}

impl Watch {
    pub(crate) fn started(&mut self) {
        self.borrowed = false;
        self.since_start = Duration::ZERO;
        self.since_answer = Duration::ZERO;
    }

    /// Setup paused the supervisor (an update or a repair replaced the environment) and has now ended the pause.
    /// Setup starts the daemon again itself: if it did, this app owns that one; if not, whatever answers on the port
    /// is borrowed, quietly. Before, every paused tick reset the watch to "owned", so an app that had been borrowing
    /// announced "stopped, starting it again" the moment the pause ended.
    pub(crate) fn resume_after_pause(&mut self, owns_a_daemon: bool) {
        if owns_a_daemon {
            self.started();
        } else {
            self.borrow();
        }
    }

    /// From now on, watch another copy's daemon instead of one this app owns.
    pub(crate) fn borrow(&mut self) {
        self.borrowed = true;
        self.restarts_in_row = 0;
        self.since_answer = Duration::ZERO;
    }

    pub(crate) fn observe(&mut self, seen: Observation, elapsed: Duration) -> Step {
        self.since_start += elapsed;
        if self.borrowed {
            return match seen {
                Observation::Answered => {
                    self.since_answer = Duration::ZERO;
                    Step::Wait
                }
                _ => {
                    self.since_answer += elapsed;
                    if self.since_answer >= self.limits.take_over_after {
                        Step::TakeOver
                    } else {
                        Step::Wait
                    }
                }
            };
        }
        match seen {
            Observation::AnotherCopy => {
                self.borrow();
                Step::Borrow
            }
            Observation::Exited => {
                let after = restart_delay(self.restarts_in_row);
                self.restarts_in_row += 1;
                Step::Restart { after }
            }
            Observation::Answered => {
                self.since_answer = Duration::ZERO;
                if self.since_start >= self.limits.stable_after {
                    self.restarts_in_row = 0;
                }
                Step::Wait
            }
            Observation::Silent => {
                self.since_answer += elapsed;
                if self.since_start >= self.limits.startup_grace
                    && self.since_answer >= self.limits.hang_after
                {
                    Step::KillHung
                } else {
                    Step::Wait
                }
            }
        }
    }
}

/// What the supervisor loop needs from the app, so a real check drives the same loop with its own launcher, notice
/// sink and limits (main.rs's AppHost is the app's).
pub(crate) trait Host {
    fn exiting(&self) -> bool;
    fn paused(&self) -> bool;
    /// The daemon this app started, if any.
    fn daemon(&self) -> &std::sync::Mutex<Option<Child>>;
    /// A launch found another copy serving (exit code 3), or one was listening at launch; taken once.
    fn take_borrow_at_start(&self) -> bool;
    /// Borrowed mode: does anything answer on the daemon's port?
    fn port_answers(&self) -> bool;
    /// Owned mode: does the daemon answer an authenticated ping in time?
    fn daemon_answers(&self) -> bool;
    /// Right before a restart: is someone else already listening on the port?
    fn port_taken(&self) -> bool;
    fn launch(&self) -> Option<Child>;
    /// The one line a person sees (the window's daemon-supervisor event).
    fn notice(&self, line: &str);
    fn sleep(&self, duration: Duration);
}

/// Whether the loop goes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tick {
    Continue,
    Stop,
}

/// The supervisor until the app exits: one tick every `watch.limits.check_every`, each counting the wall time since
/// the one before (up to longest_counted_check).
pub(crate) fn run(host: &impl Host, watch: &mut Watch) {
    let mut was_paused = false;
    let mut last = std::time::Instant::now();
    loop {
        host.sleep(watch.limits.check_every);
        let now = std::time::Instant::now();
        let elapsed = watch.limits.counted(now - last);
        last = now;
        if tick(host, watch, &mut was_paused, elapsed) == Tick::Stop {
            return;
        }
    }
}

/// One look at the daemon and what follows from it: a notice, a kill, a restart after the back-off, a borrow.
/// `elapsed`: the time this look counts for since the last one.
pub(crate) fn tick(
    host: &impl Host,
    watch: &mut Watch,
    was_paused: &mut bool,
    elapsed: Duration,
) -> Tick {
    if host.exiting() {
        return Tick::Stop;
    }
    if host.paused() {
        // Setup is changing the daemon's environment (update, repair); it starts the daemon again itself.
        *was_paused = true;
        return Tick::Continue;
    }
    if *was_paused {
        // Own what setup started, or quietly borrow what answers; decided once, when the pause ends.
        *was_paused = false;
        let owns = host
            .daemon()
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        watch.resume_after_pause(owns);
        return Tick::Continue;
    }
    let exited = if watch.borrowed {
        None // another copy's daemon: only its port is watched
    } else {
        match host.daemon().lock() {
            Ok(mut guard) => match guard.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(None) => None,
                    Ok(Some(status)) => {
                        note_previous_exit(status.code());
                        Some(exit_observation(status.code()))
                    }
                    Err(_) => Some(Observation::Exited),
                },
                // A restart that failed to launch: keep trying, unless it found another copy serving.
                None if host.take_borrow_at_start() => Some(Observation::AnotherCopy),
                None => Some(Observation::Exited),
            },
            Err(_) => return Tick::Stop,
        }
    };
    let seen = if let Some(exit) = exited {
        exit
    } else if watch.borrowed {
        if host.port_answers() {
            Observation::Answered
        } else {
            Observation::Silent
        }
    } else if host.daemon_answers() {
        Observation::Answered
    } else {
        Observation::Silent
    };
    let step = watch.observe(seen, elapsed);
    if let Some(line) = notice(step) {
        host.notice(&line);
    }
    match step {
        Step::Wait => {}
        Step::KillHung => {
            if let Ok(mut guard) = host.daemon().lock() {
                if let Some(child) = guard.as_mut() {
                    kill_tree(child);
                    note_hung();
                }
            }
        }
        Step::Restart { after } => {
            wait_unless(
                after,
                Duration::from_millis(250),
                || host.paused() || host.exiting(),
                |duration| host.sleep(duration),
            );
            if host.exiting() {
                return Tick::Stop;
            }
            if !may_launch(host.paused(), host.exiting()) {
                return Tick::Continue; // an update or repair began during the back-off: setup starts the daemon
            }
            if let Ok(mut guard) = host.daemon().lock() {
                if let Some(mut old) = guard.take() {
                    // Only a live process is killed: an exited one's id may already belong to another.
                    if matches!(old.try_wait(), Ok(None)) {
                        kill_tree(&mut old);
                    } else {
                        let _ = old.wait();
                    }
                }
                if host.port_taken() {
                    // Someone else started a daemon meanwhile (for example from a terminal): leave it, and watch
                    // it as a borrowed one.
                    println!("[Heliox OS] Another daemon is listening; not starting one");
                    watch.borrow();
                    return Tick::Continue;
                }
                *guard = host.launch();
            }
            watch.started();
        }
        Step::Borrow => {
            if let Ok(mut guard) = host.daemon().lock() {
                if let Some(mut old) = guard.take() {
                    let _ = old.wait(); // it has exited: reap it
                }
            }
        }
        Step::TakeOver => {
            if host.exiting() {
                return Tick::Stop;
            }
            if !may_launch(host.paused(), host.exiting()) {
                return Tick::Continue;
            }
            if let Ok(mut guard) = host.daemon().lock() {
                *guard = host.launch();
            }
            watch.started();
        }
    }
    Tick::Continue
}

/// Right before the supervisor starts a daemon (after a restart's back-off, or to take over): only when setup has
/// not paused it and the app is not exiting. A restart decided before an update began must not start a daemon from
/// the environment the update is replacing: on Windows its files would be locked and the update rolled back, and
/// elsewhere it would import a half-installed environment and then hold the port and the lock, so the updated
/// daemon would never start.
pub(crate) fn may_launch(paused: bool, exiting: bool) -> bool {
    !paused && !exiting
}

/// Sleep up to `total` in short steps, giving up as soon as `stop()` says so. Returns whether the whole time passed.
pub(crate) fn wait_unless(
    total: Duration,
    step: Duration,
    mut stop: impl FnMut() -> bool,
    mut sleep: impl FnMut(Duration),
) -> bool {
    let mut left = total;
    while !left.is_zero() {
        if stop() {
            return false;
        }
        let now = step.min(left);
        sleep(now);
        left -= now;
    }
    !stop()
}

/// 2, 4, 8, 16, 32, then 60 s.
pub(crate) fn restart_delay(restarts_in_row: u32) -> Duration {
    Duration::from_secs((2u64 << restarts_in_row.min(5)).min(60))
}

/// The one line the window shows.
pub(crate) fn notice(step: Step) -> Option<String> {
    match step {
        Step::Wait => None,
        Step::KillHung => Some(
            "Heliox's background service stopped answering, so it is being restarted.".to_string(),
        ),
        Step::Restart { after, .. } => Some(format!(
            "Heliox's background service stopped. Starting it again in {} s.",
            after.as_secs()
        )),
        Step::Borrow => Some(
            "Heliox's background service is already running from another copy; this window is using it."
                .to_string(),
        ),
        Step::TakeOver => Some(
            "The other copy of Heliox's background service stopped, so this app is starting its own."
                .to_string(),
        ),
    }
}

/// Kill the daemon and every process under it, at any depth: on Windows a virtual environment's python.exe is a
/// launcher whose child is the interpreter, and the daemon's workers (the voice worker, Gnomon's worker, the neural
/// sidecar) are its children. Their own parent watch cannot run while they are suspended or stuck, so they are not
/// left to end themselves.
///   macOS and Linux: the daemon's process group (own_group) gets SIGKILL, which also reaches a stopped process and
///   children that are no longer in its tree; then every process found under it by the parent links.
///   Windows: every process found under it by the parent links (each started after its parent, so a recycled
///   process id is not followed), then the daemon.
/// Before 13 November 2026 only the daemon's direct children were ended.
pub(crate) fn kill_tree(child: &mut Child) {
    use sysinfo::{Pid, ProcessesToUpdate, System};

    let root = child.id();
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::All, true);
    let links: Vec<Link> = system
        .processes()
        .iter()
        .map(|(pid, process)| Link {
            pid: pid.as_u32(),
            parent: process.parent().map(|parent| parent.as_u32()),
            started: process.start_time(),
        })
        .collect();
    let tree = descendants(&links, root);
    #[cfg(unix)]
    {
        let leader = root as libc::pid_t;
        // SAFETY: getpgid and killpg take plain ids and cannot fail in a way that harms this process; the group is
        // signalled only when the daemon leads it, so the app's own group is never signalled.
        unsafe {
            if libc::getpgid(leader) == leader {
                libc::killpg(leader, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
    for pid in tree {
        if let Some(process) = system.process(Pid::from_u32(pid)) {
            process.kill();
        }
    }
    let _ = child.wait();
}

/// One process as kill_tree sees it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Link {
    pub pid: u32,
    pub parent: Option<u32>,
    /// Seconds since the epoch.
    pub started: u64,
}

/// Every process under `root` at any depth, from the parent links. A process started before its recorded parent
/// is not its child (the parent's id was reused), so it is not followed. Pure.
pub(crate) fn descendants(links: &[Link], root: u32) -> Vec<u32> {
    let started = |pid: u32| {
        links
            .iter()
            .find(|link| link.pid == pid)
            .map(|link| link.started)
    };
    let mut found = Vec::new();
    let mut seen = std::collections::HashSet::from([root]);
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(parent) = queue.pop_front() {
        let parent_started = started(parent).unwrap_or(0);
        for link in links {
            if link.parent == Some(parent)
                && link.started >= parent_started
                && seen.insert(link.pid)
            {
                found.push(link.pid);
                queue.push_back(link.pid);
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exited_daemon_is_restarted_with_a_growing_back_off() {
        let mut watch = Watch::default();
        let delays: Vec<u64> = (0..7)
            .map(|_| match watch.observe(Observation::Exited, CHECK_EVERY) {
                Step::Restart { after, .. } => after.as_secs(),
                other => panic!("expected a restart, got {other:?}"),
            })
            .collect();
        assert_eq!(delays, vec![2, 4, 8, 16, 32, 60, 60]);
    }

    #[test]
    fn a_daemon_that_stays_up_resets_the_back_off() {
        let mut watch = Watch::default();
        watch.observe(Observation::Exited, CHECK_EVERY);
        watch.observe(Observation::Exited, CHECK_EVERY);
        watch.started();
        watch.observe(Observation::Answered, STABLE_AFTER);
        assert_eq!(watch.restarts_in_row, 0);
    }

    #[test]
    fn silence_during_start_up_is_not_a_hang() {
        let mut watch = Watch::default();
        watch.started();
        let mut elapsed = Duration::ZERO;
        while elapsed < STARTUP_GRACE - CHECK_EVERY {
            assert_eq!(watch.observe(Observation::Silent, CHECK_EVERY), Step::Wait);
            elapsed += CHECK_EVERY;
        }
    }

    #[test]
    fn the_hang_rule_counts_wall_time_and_a_sleep_counts_for_one_check_at_most() {
        let limits = Limits::default();
        let past_the_grace = || {
            let mut watch = Watch::default();
            watch.started();
            let mut counted = Duration::ZERO;
            while counted < STARTUP_GRACE {
                assert_eq!(
                    watch.observe(Observation::Answered, CHECK_EVERY),
                    Step::Wait
                );
                counted += CHECK_EVERY;
            }
            watch
        };
        // Unanswered checks each take the pause and the ping's whole limit: a minute is four of them, not twelve.
        let mut watch = past_the_grace();
        let silent = limits.counted(CHECK_EVERY + PING_TIMEOUT);
        let steps: Vec<Step> = (0..4)
            .map(|_| watch.observe(Observation::Silent, silent))
            .collect();
        assert_eq!(
            steps,
            vec![Step::Wait, Step::Wait, Step::Wait, Step::KillHung]
        );
        // A sleep and wake: one gap of hours counts as one long check, never as a minute of silence.
        let mut watch = past_the_grace();
        let slept = limits.counted(Duration::from_secs(8 * 3600));
        assert_eq!(slept, LONGEST_COUNTED_CHECK);
        assert_eq!(watch.observe(Observation::Silent, slept), Step::Wait);
        assert_eq!(
            watch.observe(Observation::Answered, CHECK_EVERY),
            Step::Wait
        );
    }

    #[test]
    fn a_minute_without_an_answer_after_start_up_is_a_hang() {
        let mut watch = Watch::default();
        watch.started();
        watch.observe(Observation::Answered, STARTUP_GRACE);
        let mut steps = Vec::new();
        for _ in 0..(HANG_AFTER.as_secs() / CHECK_EVERY.as_secs()) {
            steps.push(watch.observe(Observation::Silent, CHECK_EVERY));
        }
        assert_eq!(steps.last(), Some(&Step::KillHung));
        assert!(steps[..steps.len() - 1]
            .iter()
            .all(|step| *step == Step::Wait));
        // One answer in between starts the count again.
        watch.observe(Observation::Answered, CHECK_EVERY);
        assert_eq!(watch.observe(Observation::Silent, CHECK_EVERY), Step::Wait);
    }

    #[test]
    fn the_whole_tree_is_found_at_any_depth_and_a_reused_id_is_not_followed() {
        let link = |pid, parent, started| Link {
            pid,
            parent,
            started,
        };
        let links = [
            link(10, Some(1), 100),  // the daemon (a launcher on Windows)
            link(11, Some(10), 101), // the interpreter
            link(12, Some(11), 105), // the voice worker
            link(13, Some(12), 106), // something it started
            link(14, Some(11), 107), // Gnomon's worker
            link(20, Some(10), 50), // started before 10: its parent was an older process with id 10
            link(21, Some(20), 60),
            link(30, Some(1), 100), // unrelated
        ];
        let mut found = descendants(&links, 10);
        found.sort_unstable();
        assert_eq!(found, vec![11, 12, 13, 14]);
        let cycle = [link(5, Some(6), 1), link(6, Some(5), 1)];
        assert_eq!(descendants(&cycle, 5), vec![6], "a cycle in the links ends");
        assert!(descendants(&links, 99).is_empty());
    }

    #[cfg(unix)]
    fn suspend(pid: u32) {
        // SAFETY: a plain signal to a process this test started.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGSTOP) };
    }

    /// NtSuspendProcess, as psutil's suspend() (the real check's wrapper) does it.
    #[cfg(windows)]
    fn suspend(pid: u32) {
        use std::ffi::c_void;
        #[link(name = "kernel32")]
        extern "system" {
            fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }
        #[link(name = "ntdll")]
        extern "system" {
            fn NtSuspendProcess(handle: *mut c_void) -> i32;
        }
        const PROCESS_SUSPEND_RESUME: u32 = 0x0800;
        // SAFETY: a handle to a process this test started, used once and closed.
        unsafe {
            let handle = OpenProcess(PROCESS_SUSPEND_RESUME, 0, pid);
            if !handle.is_null() {
                NtSuspendProcess(handle);
                CloseHandle(handle);
            }
        }
    }

    /// A real tree three levels deep, every level suspended (as a frozen daemon's tree is): kill_tree ends all of
    /// it. Before 13 November 2026 the grandchild outlived it.
    #[test]
    fn kill_tree_ends_a_real_tree_three_levels_deep() {
        use sysinfo::{Pid, ProcessesToUpdate, System};
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("cmd");
            command.args(["/C", "cmd /C ping -n 120 127.0.0.1 >nul & rem"]);
            command
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", "sh -c 'sleep 120; :' & sleep 120; :"]);
            command
        };
        let mut child = command
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("a shell");
        let tree_of = |root: u32| {
            let mut system = System::new();
            system.refresh_processes(ProcessesToUpdate::All, true);
            let links: Vec<Link> = system
                .processes()
                .iter()
                .filter(|(_, process)| {
                    !matches!(
                        process.status(),
                        sysinfo::ProcessStatus::Zombie | sysinfo::ProcessStatus::Dead
                    )
                })
                .map(|(pid, process)| Link {
                    pid: pid.as_u32(),
                    parent: process.parent().map(|parent| parent.as_u32()),
                    started: process.start_time(),
                })
                .collect();
            let alive: Vec<u32> = links.iter().map(|link| link.pid).collect();
            (descendants(&links, root), alive)
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut tree = Vec::new();
        while std::time::Instant::now() < deadline {
            tree = tree_of(child.id()).0;
            if tree.len() >= 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(tree.len() >= 2, "the tree under the shell: {tree:?}");
        // Every level suspended, as the real check's wrapper freezes a hung daemon's tree.
        for pid in tree.iter().copied().chain([child.id()]) {
            suspend(pid);
        }

        kill_tree(&mut child);

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut left = tree.clone();
        while std::time::Instant::now() < deadline && !left.is_empty() {
            let (_, alive) = tree_of(0);
            left.retain(|pid| alive.contains(pid));
            std::thread::sleep(Duration::from_millis(100));
        }
        let mut system = System::new();
        system.refresh_processes(ProcessesToUpdate::All, true);
        let names: Vec<String> = left
            .iter()
            .filter_map(|pid| system.process(Pid::from_u32(*pid)))
            .map(|process| format!("{} {}", process.pid(), process.name().to_string_lossy()))
            .collect();
        for pid in &left {
            if let Some(process) = system.process(Pid::from_u32(*pid)) {
                process.kill(); // leave nothing behind, whatever the result
            }
        }
        assert!(left.is_empty(), "still running after kill_tree: {names:?}");
    }

    #[test]
    fn only_a_daemon_that_took_the_stop_and_did_not_answer_is_said_to_have_stopped_answering() {
        use crate::commands::StopAnswer;
        assert_eq!(note_for(StopAnswer::NoAnswer), Some(NO_ANSWER_AT_STOP));
        assert_eq!(
            note_for(StopAnswer::NotReached),
            None,
            "still starting or gone"
        );
        assert_eq!(note_for(StopAnswer::Stopping), None);
        let folder = std::env::temp_dir().join(format!("heliox-note-{}", std::process::id()));
        let file = folder.join("state").join("ended_by_app");
        leave_note(&file, NO_ANSWER_AT_STOP);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), NO_ANSWER_AT_STOP);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn the_next_daemon_learns_once_what_ended_the_last_one() {
        // One test for the one slot: tests run in parallel threads.
        let _ = take_previous_exit();
        note_previous_exit(Some(ANOTHER_COPY_EXIT_CODE));
        assert_eq!(
            take_previous_exit(),
            None,
            "another copy serving is not an end of ours"
        );
        note_previous_exit(Some(1));
        assert_eq!(take_previous_exit().as_deref(), Some("1"));
        assert_eq!(take_previous_exit(), None, "taken once");
        note_previous_exit(None);
        assert_eq!(take_previous_exit().as_deref(), Some("signal"));
        note_hung();
        assert_eq!(take_previous_exit().as_deref(), Some("hung"));
        // A hang the supervisor ended: the next tick sees the process it killed exit, which is not a second end.
        note_hung();
        note_previous_exit(Some(1));
        assert_eq!(take_previous_exit().as_deref(), Some("hung"));
        note_previous_exit(Some(1));
        assert_eq!(
            take_previous_exit().as_deref(),
            Some("1"),
            "a later crash is said as itself"
        );
    }

    #[test]
    fn exit_code_3_is_another_copy_not_a_crash() {
        assert_eq!(exit_observation(Some(3)), Observation::AnotherCopy);
        assert_eq!(exit_observation(Some(1)), Observation::Exited);
        assert_eq!(exit_observation(None), Observation::Exited); // ended by a signal
        let mut watch = Watch::default();
        watch.started();
        assert_eq!(
            watch.observe(Observation::AnotherCopy, CHECK_EVERY),
            Step::Borrow
        );
        assert!(watch.borrowed);
        assert_eq!(watch.restarts_in_row, 0);
        // While the other copy answers: no restart and no line, however long.
        for _ in 0..1000 {
            let step = watch.observe(Observation::Answered, CHECK_EVERY);
            assert_eq!(step, Step::Wait);
            assert_eq!(notice(step), None);
        }
    }

    #[test]
    fn a_borrowed_daemon_that_goes_silent_is_taken_over_once_after_the_limit() {
        let mut watch = Watch::default();
        watch.borrow();
        let checks = TAKE_OVER_AFTER.as_secs() / CHECK_EVERY.as_secs();
        let steps: Vec<Step> = (0..checks)
            .map(|_| watch.observe(Observation::Silent, CHECK_EVERY))
            .collect();
        assert_eq!(steps.last(), Some(&Step::TakeOver));
        assert!(steps[..steps.len() - 1]
            .iter()
            .all(|step| *step == Step::Wait));
        // A port that answers in between starts the count again (the other copy restarting itself).
        watch.borrow();
        watch.observe(Observation::Silent, CHECK_EVERY);
        watch.observe(Observation::Answered, CHECK_EVERY);
        assert_eq!(watch.observe(Observation::Silent, CHECK_EVERY), Step::Wait);
        // Once this app starts its own, it owns it again: an exit is a restart with the usual back-off.
        watch.started();
        assert!(!watch.borrowed);
        assert!(matches!(
            watch.observe(Observation::Exited, CHECK_EVERY),
            Step::Restart { .. }
        ));
    }

    #[test]
    fn an_update_that_starts_during_a_back_off_is_never_raced_by_a_restart() {
        // The daemon exited; the supervisor decided to restart it after the back-off.
        let mut watch = Watch::default();
        watch.started();
        let Step::Restart { after } = watch.observe(Observation::Exited, CHECK_EVERY) else {
            panic!("an exit is a restart");
        };
        // Setup pauses the supervisor for an update one second into that back-off.
        let mut slept = Duration::ZERO;
        let paused = std::cell::Cell::new(false);
        let waited = wait_unless(
            after,
            Duration::from_millis(250),
            || paused.get(),
            |step| {
                slept += step;
                paused.set(slept >= Duration::from_secs(1));
            },
        );
        assert!(
            !waited,
            "the back-off ends as soon as the update pauses the supervisor"
        );
        assert!(
            !may_launch(paused.get(), false),
            "no daemon is started from the environment being replaced"
        );
        assert!(slept < after);
        // Exiting the app also stops a launch; only a running, unpaused app launches.
        assert!(!may_launch(false, true));
        assert!(may_launch(false, false));
    }

    #[test]
    fn a_back_off_that_is_not_interrupted_waits_its_whole_time() {
        let mut slept = Duration::ZERO;
        let waited = wait_unless(
            Duration::from_secs(4),
            Duration::from_millis(250),
            || false,
            |step| slept += step,
        );
        assert!(waited);
        assert_eq!(slept, Duration::from_secs(4));
    }

    #[test]
    fn after_an_update_the_supervisor_owns_what_setup_started_or_quietly_borrows() {
        // Borrowing before the update; setup found the port answering afterwards and started nothing.
        let mut watch = Watch::default();
        watch.borrow();
        watch.resume_after_pause(false);
        assert!(watch.borrowed);
        let step = watch.observe(Observation::Answered, CHECK_EVERY);
        assert_eq!(step, Step::Wait);
        assert_eq!(notice(step), None); // no "stopped, starting it again" line
                                        // Owning before the update; setup started the updated daemon.
        let mut watch = Watch::default();
        watch.started();
        watch.resume_after_pause(true);
        assert!(!watch.borrowed);
        assert!(matches!(
            watch.observe(Observation::Exited, CHECK_EVERY),
            Step::Restart { .. }
        ));
    }

    #[test]
    fn the_notice_is_one_plain_line() {
        assert!(notice(Step::Borrow)
            .unwrap()
            .contains("this window is using it"));
        assert!(notice(Step::TakeOver).unwrap().contains("starting its own"));
        assert_eq!(notice(Step::Wait), None);
        assert_eq!(
            notice(Step::Restart {
                after: Duration::from_secs(4)
            })
            .unwrap(),
            "Heliox's background service stopped. Starting it again in 4 s."
        );
        assert!(notice(Step::KillHung)
            .unwrap()
            .contains("stopped answering"));
    }
}
