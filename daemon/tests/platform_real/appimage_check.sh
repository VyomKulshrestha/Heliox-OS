#!/bin/bash
# The AppImage a person downloads, on a system that has never seen Heliox: set up through its own window, then
# removed through Settings' "Remove Heliox's files", the only removal an AppImage has.
#
#   appimage_check.sh <path to the .AppImage> <folder with tauri-driver>
#
# Run as root inside a fresh ubuntu:24.04 container (the linux-package workflow starts one), never anywhere else. An
# AppImage declares no dependencies, so only the test's own tools are installed: a virtual display, a session bus,
# Python for the check, WebKitWebDriver, and the certificate store every desktop has. Neither curl nor wget is
# installed, so first-run setup downloads uv with Python (first-run.sh, fetch). The container has no FUSE, so the AppImage runs extracted
# (APPIMAGE_EXTRACT_AND_RUN=1), which its runtime supports and which still sets APPIMAGE. Each result is a CHECK
# line named appimage/<check>; the exit code is the number of failed checks.
if [ "${GITHUB_ACTIONS:-}" != "true" ] || [ -z "${RUNNER_TEMP:-}" ]; then
    echo "NOT RUN: this check installs packages, so it runs only as root in a CI runner's throwaway container."
    exit 3
fi
if [ ! -f /.dockerenv ] || [ "$(id -u)" != 0 ]; then
    echo "NOT RUN: this check installs packages, so it runs only as root in a CI runner's throwaway container."
    exit 3
fi
set -u
export DEBIAN_FRONTEND=noninteractive
appimage="$1"
tools="$2"
timeout 300 apt-get update -q > /tmp/apt-update.log 2>&1
if ! timeout 600 apt-get install -y -q --no-install-recommends xvfb dbus-x11 webkit2gtk-driver python3 ca-certificates \
    > /tmp/harness.log 2>&1; then
    echo "CHECK appimage/test-tools: FAIL - $(tail -5 /tmp/harness.log | tr '\n' ' ')"
    exit 1
fi
useradd -m -s /bin/bash sam
mkdir -p /home/sam/Apps
cp "$appimage" /home/sam/Apps/
chmod 755 /home/sam/Apps/"$(basename "$appimage")"
# The daemon the setup installs: this commit's, as in the first-run job (HELIOX_DAEMON_SOURCE).
cp -r "$(dirname "$0")/../.." /home/sam/daemon-src
rm -rf /home/sam/daemon-src/.venv
chown -R sam:sam /home/sam/Apps /home/sam/daemon-src
python3 "$(dirname "$0")/appimage_check.py" "/home/sam/Apps/$(basename "$appimage")" "$tools"
