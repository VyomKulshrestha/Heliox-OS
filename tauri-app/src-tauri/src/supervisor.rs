//! Keeps the daemon the app started alive.
//!
//! The app starts the Python daemon at launch (`spawn_daemon` in main.rs). Before this, nothing watched it: if it
//! crashed or hung, the window showed "Reconnecting..." until the person restarted the app. Now a thread checks the
//! daemon the app owns every few seconds. A daemon that exited is started again after a back-off (2 s, doubling to
//! 60 s, reset after five quiet minutes); one that stops answering its RPC ping for a minute after its start-up grace
//! is killed and then started again. Each step is announced on the `daemon-supervisor` event in one plain line, and
//! the window reconnects on its own as it always did.
//!
//! A daemon the app did not start (for example one run from a terminal during development, found already listening
//! at launch) is never touched: `DaemonProcess` holds `None` and the supervisor only watches its port ("borrowed").
//!
//! A daemon that exits with code 3 found another copy serving the port (`AlreadyRunningError` in the daemon). That is
//! not a crash: there is no restart and no alarm, one calm line says the window is using the copy that is running,
//! and the supervisor borrows that copy too. If a borrowed copy's port stays silent for TAKE_OVER_AFTER, it has gone
//! and this app starts its own daemon.

use std::process::Child;
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
        *slot = Some(seen);
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
/// it marks its run as ended cleanly, give it STOP_ON_PURPOSE_WAIT, then end whatever is left.
pub(crate) fn stop_on_purpose(child: &mut Child, reason: &str) {
    if tauri::async_runtime::block_on(crate::commands::ask_daemon_to_stop(reason)) {
        let waited = std::time::Instant::now();
        while waited.elapsed() < STOP_ON_PURPOSE_WAIT {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    kill_tree(child);
}

pub(crate) const CHECK_EVERY: Duration = Duration::from_secs(5);
/// First start-up can load speech models for well over a minute on a slow or busy machine.
pub(crate) const STARTUP_GRACE: Duration = Duration::from_secs(300);
/// The daemon answers ping from its event loop; a minute without an answer means that loop is stuck.
pub(crate) const HANG_AFTER: Duration = Duration::from_secs(60);
/// After this long without a restart, the back-off starts again from 2 s.
pub(crate) const STABLE_AFTER: Duration = Duration::from_secs(300);
pub(crate) const PING_TIMEOUT: Duration = Duration::from_secs(10);
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

/// Pure bookkeeping for one owned daemon: what each observation means. Tested without processes or sockets.
#[derive(Debug, Default)]
pub(crate) struct Watch {
    pub(crate) restarts_in_row: u32,
    /// Watching a daemon another copy started; observations are "its port answers" or not.
    pub(crate) borrowed: bool,
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
                    if self.since_answer >= TAKE_OVER_AFTER {
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
                if self.since_start >= STABLE_AFTER {
                    self.restarts_in_row = 0;
                }
                Step::Wait
            }
            Observation::Silent => {
                self.since_answer += elapsed;
                if self.since_start >= STARTUP_GRACE && self.since_answer >= HANG_AFTER {
                    Step::KillHung
                } else {
                    Step::Wait
                }
            }
        }
    }
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

/// Kill the daemon and the processes it started directly. On Windows a virtual environment's python.exe is a
/// launcher whose child is the real interpreter; killing only the launcher would leave the daemon running. The
/// daemon's own workers end themselves when it does (closed pipes and a parent watch).
pub(crate) fn kill_tree(child: &mut Child) {
    use sysinfo::{Pid, ProcessesToUpdate, System};

    let root = Pid::from_u32(child.id());
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::All, true);
    for process in system.processes().values() {
        if process.parent() == Some(root) {
            process.kill();
        }
    }
    let _ = child.kill();
    let _ = child.wait();
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
    fn the_next_daemon_learns_once_what_ended_the_last_one() {
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
