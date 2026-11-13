#!/bin/bash
# The released Linux package, installed the way a person installs it, on a system that has never seen Heliox.
#
#   deb_install_check.sh <path to the .deb> <folder with tauri-driver> <label> <full|offer>
#
# Run as root inside a fresh Ubuntu container (the linux-package workflow starts one per release), never on a
# runner's own image or a person's computer: it installs packages and creates a user. Part one is here, before
# anything but the package is installed: apt installs the .deb the way a person's apt does (its Depends and, as apt
# does by default, its Recommends), and every library the app binary links against must then resolve. Only after
# that does it add what the test itself needs (a virtual display, a session bus, Python for the check,
# WebKitWebDriver to read and press the setup screen) and hand over to deb_install_check.py: "full" sets Heliox up
# and removes it again, "offer" only reads what the setup screen offers. Each result is a CHECK line named
# <label>/<check>; the exit code is the number of failed checks.
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
deb="$1"
tools="$2"
label="$3"
mode="$4"
failed=0
check() { # check <name> <PASS|FAIL> <detail>
    echo "CHECK $label/$1: $2 - $3"
    [ "$2" = PASS ] || failed=$((failed + 1))
}

package="$(dpkg-deb -f "$deb" Package)"
echo "DIAG $(grep PRETTY_NAME /etc/os-release); $package $(dpkg-deb -f "$deb" Version)"
echo "DIAG Depends: $(dpkg-deb -f "$deb" Depends); Recommends: $(dpkg-deb -f "$deb" Recommends)"
timeout 300 apt-get update -q > /tmp/apt-update.log 2>&1
before="$(dpkg-query -W -f '${Package}\n' | sort)"
if timeout 900 apt-get install -y -q "$deb" > /tmp/install.log 2>&1; then
    added="$(comm -13 <(echo "$before") <(dpkg-query -W -f '${Package}\n' | sort))"
    python="$(command -v python3 > /dev/null && python3 --version 2>&1 || echo 'no python3')"
    check installs-with-declared-dependencies PASS \
        "apt installed it and $(($(echo "$added" | wc -l) - 1)) packages from its Depends and Recommends, nothing by hand; system Python: $python"
else
    check installs-with-declared-dependencies FAIL "$(tail -8 /tmp/install.log | tr '\n' ' ')"
    exit "$failed"
fi

binary="$(dpkg -L "$package" | grep -m1 '^/usr/bin/')"
missing="$(ldd "$binary" 2>&1 | grep 'not found')"
if [ -n "$binary" ] && [ -z "$missing" ]; then
    check libraries-resolve PASS "every library $binary links against is installed"
else
    check libraries-resolve FAIL "${binary:-no binary in /usr/bin}: $(echo "$missing" | tr '\n' ' ')"
fi
dpkg -L "$package" > /tmp/package-files.txt

# From here on only what the test needs, never what the app needs: a virtual display and a session bus for the
# person's desktop, Python for the check itself, and WebKitWebDriver, which tauri-driver drives.
if ! timeout 600 apt-get install -y -q --no-install-recommends xvfb dbus-x11 webkit2gtk-driver python3 \
    > /tmp/harness.log 2>&1; then
    check test-tools FAIL "$(tail -5 /tmp/harness.log | tr '\n' ' ')"
    exit "$failed"
fi
useradd -m -s /bin/bash sam
# The daemon the setup installs: this commit's, as in the first-run job (HELIOX_DAEMON_SOURCE), since the app
# built here can be newer than any release on PyPI.
cp -r "$(dirname "$0")/../.." /home/sam/daemon-src
rm -rf /home/sam/daemon-src/.venv
chown -R sam:sam /home/sam/daemon-src
python3 "$(dirname "$0")/deb_install_check.py" "$binary" "$package" "$tools" "$label" "$mode"
exit $((failed + $?))
