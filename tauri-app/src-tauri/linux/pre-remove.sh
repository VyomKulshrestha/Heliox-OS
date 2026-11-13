#!/bin/sh
# Before the .deb or .rpm is removed: end Heliox for everyone who has it open, so nothing keeps running from a
# removed install. The app runs as the person; the daemon it started runs from the person's own environment
# (~/.config/heliox-os/env), which this package does not own and does not delete. The app is ended first, so its
# supervisor cannot start the daemon again; the daemon saves and stops on SIGTERM.
#
# dpkg passes "remove", "upgrade" or "deconfigure"; rpm passes how many versions stay installed (0 on erase, 1 or
# more on upgrade). On an upgrade both keep running: the old app works until the person reopens Heliox.
# Only /proc is read, so nothing beyond the package's declared dependencies is needed.

case "$1" in
    remove | purge | 0) ;;
    *) exit 0 ;;
esac

end() { # end <pid>: SIGTERM, ignoring a process that has already gone
    kill -TERM "$1" 2>/dev/null || true
}

for dir in /proc/[0-9]*; do
    pid="${dir#/proc/}"
    if [ "$(cat "$dir/comm" 2>/dev/null)" = "heliox-os" ]; then
        end "$pid"
    fi
done
sleep 1
for dir in /proc/[0-9]*; do
    pid="${dir#/proc/}"
    command_line="$(tr '\000' ' ' < "$dir/cmdline" 2>/dev/null)"
    case "$command_line" in
        */heliox-os/env/bin/python*" -m pilot.server"*) end "$pid" ;;
    esac
done
exit 0
