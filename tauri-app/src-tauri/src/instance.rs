//! One copy of the app per person: a second launch brings the first window forward and exits.
//!
//! Without this, opening Heliox twice gave two windows, two trays and two supervisors fighting over one daemon.
//! The same rules on Windows, macOS and Linux, with no extra crate:
//!
//! - The first copy holds an exclusive lock on `app.lock` in the person's own Heliox folder (`~/.config/heliox-os`),
//!   so it is per user: two people signed in to one computer each get their own Heliox. The operating system releases
//!   the lock when the process ends, however it ends, so a crash never leaves a stale lock.
//! - The first copy listens on a loopback port the system picks and writes that port with a random secret to
//!   `app-instance.json` beside the lock. A second launch that cannot take the lock sends the secret to that port
//!   and exits; the first copy shows, un-minimises and focuses its window. Anything else sent to the port is ignored.
//! - If the lock cannot be taken for another reason (an unwritable folder), the app runs as before rather than not
//!   at all.
//!
//! macOS already re-activates a running app bundle on a second launch from Finder or the Dock; this also covers
//! `open -n` and running the binary directly.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

const LOCK_FILE: &str = "app.lock";
const INSTANCE_FILE: &str = "app-instance.json";
/// How long a second launch keeps trying to reach the first copy (it may still be starting).
const REACH_FIRST_FOR: Duration = Duration::from_secs(5);

/// This copy holds the lock: keep it for the life of the process.
pub(crate) struct FirstCopy {
    _lock: File,
    folder: PathBuf,
}

pub(crate) enum Launch {
    First(FirstCopy),
    /// Another copy holds the lock.
    Second,
    /// The lock could not be checked; run as before.
    Unknown(String),
}

pub(crate) fn claim(folder: &Path) -> Launch {
    if let Err(error) = std::fs::create_dir_all(folder) {
        return Launch::Unknown(format!("{}: {error}", folder.display()));
    }
    let file = match OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(folder.join(LOCK_FILE))
    {
        Ok(file) => file,
        Err(error) => return Launch::Unknown(format!("{}: {error}", folder.display())),
    };
    match file.try_lock() {
        Ok(()) => Launch::First(FirstCopy {
            _lock: file,
            folder: folder.to_path_buf(),
        }),
        Err(std::fs::TryLockError::WouldBlock) => Launch::Second,
        Err(std::fs::TryLockError::Error(error)) => {
            Launch::Unknown(format!("{}: {error}", folder.display()))
        }
    }
}

fn new_secret() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    // RandomState is seeded from the system's random source once per process; two of them give 128 bits.
    let mut parts = Vec::new();
    for salt in 0..2u64 {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(salt);
        hasher.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or_default(),
        );
        hasher.write_u32(std::process::id());
        parts.push(format!("{:016x}", hasher.finish()));
    }
    parts.concat()
}

impl FirstCopy {
    /// Listen for second launches and call `bring_forward` for each one that sends the secret.
    pub(crate) fn listen(&self, bring_forward: impl Fn() + Send + 'static) -> std::io::Result<()> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let secret = new_secret();
        let record =
            serde_json::json!({ "port": port, "secret": secret, "pid": std::process::id() });
        let path = self.folder.join(INSTANCE_FILE);
        let partial = self.folder.join(format!("{INSTANCE_FILE}.partial"));
        std::fs::write(&partial, record.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&partial, &path)?;
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                if asks_to_show(stream, &secret) {
                    bring_forward();
                }
            }
        });
        Ok(())
    }
}

fn asks_to_show(stream: TcpStream, secret: &str) -> bool {
    use std::io::Read;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut line = String::new();
    // At most one short line: a stranger cannot make the first copy read without end.
    if BufReader::new(stream.take(256))
        .read_line(&mut line)
        .is_err()
    {
        return false;
    }
    line.trim() == format!("show {secret}")
}

/// Second launch: ask the first copy to show its window. Returns whether it was reached.
pub(crate) fn ask_first_to_show(folder: &Path) -> bool {
    let deadline = std::time::Instant::now() + REACH_FIRST_FOR;
    loop {
        if let Some((port, secret)) = read_instance(folder) {
            let address = SocketAddr::from(([127, 0, 0, 1], port));
            if let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_secs(1)) {
                if stream
                    .write_all(format!("show {secret}\n").as_bytes())
                    .is_ok()
                {
                    return true;
                }
            }
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn read_instance(folder: &Path) -> Option<(u16, String)> {
    let text = std::fs::read_to_string(folder.join(INSTANCE_FILE)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let port = u16::try_from(value.get("port")?.as_u64()?).ok()?;
    let secret = value.get("secret")?.as_str()?.to_string();
    Some((port, secret))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn folder(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "heliox-instance-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn a_second_claim_is_refused_while_the_first_holds_the_lock_and_allowed_after() {
        let dir = folder("lock");
        let first = claim(&dir);
        assert!(matches!(first, Launch::First(_)));
        assert!(matches!(claim(&dir), Launch::Second));
        drop(first);
        assert!(matches!(claim(&dir), Launch::First(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_launch_brings_the_first_window_forward_and_a_wrong_secret_does_not() {
        let dir = folder("show");
        let Launch::First(first) = claim(&dir) else {
            panic!("the first claim must win");
        };
        let shown = Arc::new(AtomicUsize::new(0));
        let counter = shown.clone();
        first
            .listen(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            })
            .expect("listen");

        assert!(ask_first_to_show(&dir));
        let (port, _secret) = read_instance(&dir).expect("instance file");
        let mut stranger =
            TcpStream::connect(SocketAddr::from(([127, 0, 0, 1], port))).expect("connect");
        stranger.write_all(b"show not-the-secret\n").expect("write");
        drop(stranger);
        assert!(ask_first_to_show(&dir));

        // Ordered by the listener's own accept loop: wait for the two good requests, then check nothing else counted.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while shown.load(Ordering::SeqCst) < 2 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(shown.load(Ordering::SeqCst), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_launch_with_no_first_copy_reachable_gives_up() {
        let dir = folder("absent");
        std::fs::create_dir_all(&dir).expect("dir");
        // A record left by a copy that has ended: nothing listens there any more.
        let unused = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let port = unused.local_addr().expect("addr").port();
        drop(unused);
        std::fs::write(
            dir.join(INSTANCE_FILE),
            format!(r#"{{"port":{port},"secret":"x"}}"#),
        )
        .expect("write");
        assert!(!ask_first_to_show(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
