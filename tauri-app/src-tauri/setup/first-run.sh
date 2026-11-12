#!/bin/sh
# Heliox first-run setup for macOS and Linux. The desktop app runs this (src/setup.rs) and CI runs it from an
# empty HOME (the "first-run" job). It never touches the system Python and never installs anything the person
# did not choose: the core, then each optional group passed in HELIOX_GROUPS.
#
# Input (environment):
#   HELIOX_SETUP_DIR   the Heliox data folder (the app uses ~/.config/heliox-os)
#   HELIOX_CORE_REQ    the daemon requirement, e.g. "pilot-daemon==0.16.0" or a local path in CI
#   HELIOX_GROUPS      optional groups as "id=requirement;id=requirement" (may be empty)
#   HELIOX_PYTHON      "auto" (a suitable Python on this machine, else a private one) or "private"
#   HELIOX_PYTHON_MIN, HELIOX_PYTHON_MAX, HELIOX_PYTHON_PRIVATE
#                      the Python versions Heliox is tested with and the one it downloads, from
#                      setup/python-versions.json (the app passes them; CI reads the same file)
#   $1                 "probe" prints only the Python plan; "run" (default) does the setup
#
# Output: lines starting with "@@" are for the app; everything else is detail shown as progress.
#   @@python system <version> <path>      a suitable Python was found
#   @@python download <bytes> <url>       uv will be downloaded to fetch a private Python
#   @@step <id> <running|done|failed|skipped> <message>
#   @@hint <id> <what the person can do>
#   @@size <id> <KiB>                     with HELIOX_MEASURE_PIP_CACHE set (CI): pip's download cache so far
#   @@disk <id> <KiB>                     with HELIOX_MEASURE_PIP_CACHE set (CI): Heliox's folder so far
set -u

UV_VERSION="0.13.5"
DIR="${HELIOX_SETUP_DIR:?HELIOX_SETUP_DIR is required}"
MODE="${1:-run}"
PY_MIN="${HELIOX_PYTHON_MIN:?HELIOX_PYTHON_MIN is required}"
PY_MAX="${HELIOX_PYTHON_MAX:?HELIOX_PYTHON_MAX is required}"
PY_PRIVATE="${HELIOX_PYTHON_PRIVATE:?HELIOX_PYTHON_PRIVATE is required}"
# Inside the tested range only: an older Python lacks what Heliox uses, and a newer one may have no prebuilt wheels
# for the core's dependencies, so pip would try to compile them (CI on macOS, 12 November: Python 3.15 and scipy).
CHECK="import sys, venv, ensurepip; v = sys.version_info[:2]; lo = tuple(int(x) for x in '$PY_MIN'.split('.')); hi = tuple(int(x) for x in '$PY_MAX'.split('.')); sys.exit(0 if lo <= v <= hi else 1)"

say() { printf '%s\n' "$*"; }

# CI measures sizes: pip's download cache (PIP_CACHE_DIR pointed at an empty folder) and Heliox's folder, cumulative.
measure() {
  if [ -n "${HELIOX_MEASURE_PIP_CACHE:-}" ] && [ -d "$HELIOX_MEASURE_PIP_CACHE" ]; then
    say "@@size $1 $(du -sk "$HELIOX_MEASURE_PIP_CACHE" | cut -f1)"
    say "@@disk $1 $(du -sk "$DIR" | cut -f1)"
  fi
}

uv_asset() {
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) say "uv-aarch64-apple-darwin.tar.gz 14365303 743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73" ;;
    Darwin-x86_64) say "uv-x86_64-apple-darwin.tar.gz 17167423 b0f52b0e3945b4107d89024ea38e625683011fbfc05a0d98f9fa2d6dcee28c66" ;;
    Linux-x86_64) say "uv-x86_64-unknown-linux-gnu.tar.gz 16682639 f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f" ;;
    Linux-aarch64 | Linux-arm64) say "uv-aarch64-unknown-linux-gnu.tar.gz 15914298 4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403" ;;
    *) say "" ;;
  esac
}

# A Python in the tested range with venv and ensurepip, never /usr/bin/python3 on macOS: there it is a stub that
# opens Apple's developer-tools installer, and the Python it installs is 3.9.
find_python() {
  search="${PATH}:/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin"
  saved_ifs="$IFS"
  IFS=':'
  for dir in $search; do
    IFS="$saved_ifs"
    if [ "$(uname -s)" = "Darwin" ] && [ "$dir" = "/usr/bin" ]; then IFS=':'; continue; fi
    for name in python3.13 python3.12 python3.11 python3; do
      exe="$dir/$name"
      if [ -x "$exe" ] && "$exe" -c "$CHECK" >/dev/null 2>&1; then
        say "$exe"
        return 0
      fi
    done
    IFS=':'
  done
  IFS="$saved_ifs"
  return 1
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fail() { # id message hint
  say "@@step $1 failed $2"
  [ -n "${3:-}" ] && say "@@hint $1 $3"
  exit 1
}

PY=""
if [ "${HELIOX_PYTHON:-auto}" = "auto" ]; then PY="$(find_python || true)"; fi
ASSET="$(uv_asset)"
URL=""
if [ -n "$ASSET" ]; then
  set -- $ASSET
  ASSET_NAME="$1"; ASSET_SIZE="$2"; ASSET_SHA="$3"
  URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${ASSET_NAME}"
fi

if [ "$MODE" = "probe" ]; then
  if [ -n "$PY" ]; then
    say "@@python system $("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])') $PY"
  elif [ -n "$URL" ]; then
    say "@@python download ${ASSET_SIZE} ${URL}"
  else
    say "@@python none 0 -"
  fi
  exit 0
fi

mkdir -p "$DIR" || fail python "Cannot create $DIR." "Check that your home folder is writable."
ENV_DIR="$DIR/env"

env_from_system() {
  say "@@step python running Using $PY"
  "$PY" -m venv "$ENV_DIR" || fail python "Could not create the environment with $PY." \
    "On Debian or Ubuntu install python3-venv, or remove that Python from PATH so Heliox uses its own."
  say "@@step python done Using $("$ENV_DIR/bin/python" -c 'import sys; print(sys.version.split()[0])') from $PY"
}

env_from_private() {
  [ -n "$URL" ] || fail python "No Python ${PY_MIN} to ${PY_MAX} here, and no private Python is available for $(uname -s) $(uname -m)." \
    "Install Python ${PY_PRIVATE} and open Heliox again."
  TOOLS="$DIR/tools"
  mkdir -p "$TOOLS"
  ARCHIVE="$TOOLS/$ASSET_NAME"
  say "@@step uv running Downloading uv ${UV_VERSION} (${ASSET_SIZE} bytes) from github.com/astral-sh/uv"
  if [ ! -f "$ARCHIVE" ] || [ "$(sha256_of "$ARCHIVE")" != "$ASSET_SHA" ]; then
    rm -f "$ARCHIVE"
    curl -fL --retry 2 --connect-timeout 20 -o "$ARCHIVE" "$URL" 2>&1 | tr '\r' '\n' || true
  fi
  [ -f "$ARCHIVE" ] || fail uv "The download from github.com did not arrive." "Check the internet connection and press Retry."
  GOT="$(sha256_of "$ARCHIVE")"
  if [ "$GOT" != "$ASSET_SHA" ]; then
    rm -f "$ARCHIVE"
    fail uv "The downloaded uv does not match its published checksum ($GOT)." "Press Retry; if it happens again, report it."
  fi
  mkdir -p "$TOOLS/uv-$UV_VERSION"
  tar -xzf "$ARCHIVE" -C "$TOOLS/uv-$UV_VERSION" || fail uv "Could not unpack uv." "Press Retry."
  UV="$(find "$TOOLS/uv-$UV_VERSION" -type f -name uv | head -n 1)"
  [ -n "$UV" ] || fail uv "uv was not found in the download." "Press Retry."
  say "@@step uv done uv ${UV_VERSION}, checksum verified"
  export UV_PYTHON_INSTALL_DIR="$DIR/python" UV_CACHE_DIR="$DIR/uv-cache" UV_NO_CONFIG=1
  say "@@step python running Downloading a private Python ${PY_PRIVATE} for Heliox (python-build-standalone)"
  "$UV" python install "$PY_PRIVATE" 2>&1 || fail python "uv could not download Python." "Check the internet connection and press Retry."
  "$UV" venv --seed --python "$PY_PRIVATE" --python-preference only-managed "$ENV_DIR" 2>&1 || fail python "Could not create the environment." "Press Retry."
  say "@@step python done Private Python $("$ENV_DIR/bin/python" -c 'import sys; print(sys.version.split()[0])') in $DIR/python"
}

install_core() {
  "$ENV_DIR/bin/python" -m pip --disable-pip-version-check install --progress-bar off "$HELIOX_CORE_REQ" 2>&1
}

# ── Python ──
USED_SYSTEM=0
if [ -x "$ENV_DIR/bin/python" ]; then
  say "@@step python skipped Heliox's environment already exists."
elif [ -n "$PY" ]; then
  env_from_system
  USED_SYSTEM=1
else
  env_from_private
fi

# ── Core: if it cannot be installed on this computer's Python, try once more with Heliox's own ──
say "@@step core running Installing Heliox's core"
if ! install_core; then
  if [ "$USED_SYSTEM" = "1" ] && [ -n "$URL" ]; then
    say "@@step core running The core could not be installed with $PY; trying again with Heliox's own Python"
    rm -rf "$ENV_DIR"
    env_from_private
    install_core || fail core "The core could not be installed, with this computer's Python or with Heliox's own." \
      "Check the internet connection and press Retry. The full log is in the setup details."
  else
    fail core "The core could not be installed." \
      "Check the internet connection and press Retry. The full log is in the setup details."
  fi
fi
say "@@step core done Core installed"
measure core

PIP="$ENV_DIR/bin/python -m pip --disable-pip-version-check"

# ── Optional groups: a failure is reported and the others carry on ──
status=0
OLDIFS="$IFS"; IFS=';'
for pair in ${HELIOX_GROUPS:-}; do
  IFS="$OLDIFS"
  [ -n "$pair" ] || continue
  id="${pair%%=*}"; req="${pair#*=}"
  say "@@step $id running Installing"
  if $PIP install --progress-bar off "$req" 2>&1; then
    if [ "$id" = "browser" ]; then
      say "Downloading Chromium for browser automation (playwright.azureedge.net, about 170 MB)"
      if "$ENV_DIR/bin/python" -m playwright install chromium 2>&1; then
        say "@@step $id done Installed, with Chromium"
      else
        say "@@step $id failed Chromium could not be downloaded."
        say "@@hint $id Press Retry; the rest of Heliox works without it."
        status=2
      fi
    else
      say "@@step $id done Installed"
    fi
    measure "$id"
  else
    say "@@step $id failed Could not be installed."
    say "@@hint $id Press Retry; the rest of Heliox works without it."
    status=2
  fi
  IFS=';'
done
IFS="$OLDIFS"
exit $status
