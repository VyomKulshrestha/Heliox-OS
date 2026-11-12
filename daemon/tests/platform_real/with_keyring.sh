#!/bin/sh
# Run a command with an unlocked GNOME keyring (the Secret Service) on the current D-Bus session, for Linux CI:
#   dbus-run-session -- sh daemon/tests/platform_real/with_keyring.sh <command> [args...]
# The keyring's password is a throwaway for this session; the keyring lives in the runner's home folder.
set -e
if [ "${GITHUB_ACTIONS:-}" != "true" ] || [ -z "${RUNNER_TEMP:-}" ]; then
  echo "NOT RUN: this script unlocks a GNOME keyring with a throwaway password and starts a keyring daemon on this session, so it runs only on a CI runner (GITHUB_ACTIONS=true and RUNNER_TEMP set) and changed nothing here."
  exit 3
fi
printf "heliox-ci" | gnome-keyring-daemon --unlock --components=secrets >"${RUNNER_TEMP:-/tmp}/keyring.env"
gnome-keyring-daemon --start --components=secrets >>"${RUNNER_TEMP:-/tmp}/keyring.env"
while IFS= read -r line; do
  case "$line" in
    *=*) export "$line" ;;
  esac
done <"${RUNNER_TEMP:-/tmp}/keyring.env"
echo "Secret Service owner: $(dbus-send --session --print-reply --dest=org.freedesktop.DBus /org/freedesktop/DBus \
  org.freedesktop.DBus.GetNameOwner string:org.freedesktop.secrets 2>&1 | tail -1)"
exec "$@"
