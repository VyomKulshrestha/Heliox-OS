#!/bin/bash
# The same build's .rpm on a clean Fedora, as far as the .deb's first three lines: dnf installs it with its declared
# dependencies and nothing else, every library the binary links against resolves, and the binary starts under a
# virtual display and is still running 20 s later. First-run setup and removal on Fedora are not checked here.
#
#   rpm_install_check.sh <path to the .rpm>
#
# Run as root inside a fresh fedora container (the CI job starts one), never anywhere else.
if [ "${GITHUB_ACTIONS:-}" != "true" ] || [ -z "${RUNNER_TEMP:-}" ]; then
    echo "NOT RUN: this check installs packages, so it runs only as root in a CI runner's throwaway container."
    exit 3
fi
if [ ! -f /.dockerenv ] || [ "$(id -u)" != 0 ]; then
    echo "NOT RUN: this check installs packages, so it runs only as root in a CI runner's throwaway container."
    exit 3
fi
set -u
rpm="$1"
failed=0
check() { # check <name> <PASS|FAIL> <detail>
    echo "CHECK rpm/$1: $2 - $3"
    [ "$2" = PASS ] || failed=$((failed + 1))
}

package="$(rpm -qp --queryformat '%{NAME}' "$rpm" 2>/dev/null)"
echo "DIAG $package, Requires: $(rpm -qp --requires "$rpm" 2>/dev/null | grep -v '^rpmlib(' | tr '\n' ' ')"
if timeout 900 dnf install -y -q "$rpm" > /tmp/install.log 2>&1; then
    check installs-with-declared-dependencies PASS "dnf installed it with what it requires, nothing by hand"
else
    check installs-with-declared-dependencies FAIL "$(tail -8 /tmp/install.log | tr '\n' ' ')"
    exit "$failed"
fi
binary="$(rpm -ql "$package" | grep -m1 '^/usr/bin/')"
missing="$(ldd "$binary" 2>&1 | grep 'not found')"
if [ -n "$binary" ] && [ -z "$missing" ]; then
    check libraries-resolve PASS "every library $binary links against is installed"
else
    check libraries-resolve FAIL "${binary:-no binary in /usr/bin}: $(echo "$missing" | tr '\n' ' ')"
fi

# Only what the test needs from here on: a virtual display and a session bus.
if ! timeout 600 dnf install -y -q --setopt=install_weak_deps=False xorg-x11-server-Xvfb dbus-daemon > /tmp/harness.log 2>&1; then
    check test-tools FAIL "$(tail -5 /tmp/harness.log | tr '\n' ' ')"
    exit "$failed"
fi
useradd -m sam
Xvfb :99 -ac -screen 0 1280x800x24 > /dev/null 2>&1 &
sleep 3
runuser -u sam -- env -i HOME=/home/sam USER=sam PATH=/usr/bin:/bin DISPLAY=:99 LANG=C.UTF-8 \
    dbus-run-session -- "$binary" > /tmp/app.log 2>&1 &
sleep 20
if [ "$(cat /proc/[0-9]*/comm 2>/dev/null | grep -c '^heliox-os$')" -gt 0 ] \
    && ! grep -q 'error while loading shared libraries' /tmp/app.log; then
    check app-starts PASS "running after 20 s under a virtual display"
else
    check app-starts FAIL "not running after 20 s: $(tail -c 400 /tmp/app.log | tr '\n' ' ')"
fi
exit "$failed"
