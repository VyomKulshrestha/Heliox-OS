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
//! at launch) is never touched: `DaemonProcess` holds `None` and the supervisor only watches.

use std::process::Child;
use std::time::Duration;

pub(crate) const CHECK_EVERY: Duration = Duration::from_secs(5);
/// First start-up can load speech models for well over a minute on a slow or busy machine.
pub(crate) const STARTUP_GRACE: Duration = Duration::from_secs(300);
/// The daemon answers ping from its event loop; a minute without an answer means that loop is stuck.
pub(crate) const HANG_AFTER: Duration = Duration::from_secs(60);
/// After this long without a restart, the back-off starts again from 2 s.
pub(crate) const STABLE_AFTER: Duration = Duration::from_secs(300);
pub(crate) const PING_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Observation {
    Exited,
    Answered,
    Silent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    Wait,
    Restart { after: Duration },
    KillHung,
}

/// Pure bookkeeping for one owned daemon: what each observation means. Tested without processes or sockets.
#[derive(Debug, Default)]
pub(crate) struct Watch {
    pub(crate) restarts_in_row: u32,
    since_start: Duration,
    since_answer: Duration,
}

impl Watch {
    pub(crate) fn started(&mut self) {
        self.since_start = Duration::ZERO;
        self.since_answer = Duration::ZERO;
    }

    pub(crate) fn observe(&mut self, seen: Observation, elapsed: Duration) -> Step {
        self.since_start += elapsed;
        match seen {
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
    fn the_notice_is_one_plain_line() {
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
