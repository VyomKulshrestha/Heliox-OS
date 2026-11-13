#!/bin/bash
# Before a package workflow's long release build: are the build's inputs consistent? Each install the build is about
# to run is tried in its cheapest form, so a broken input fails in a minute rather than after the Rust cache restore
# (the first linux-package run, on a13a496, failed forty minutes in on a lock file out of step with package.json).
# Changes nothing: npm ci --dry-run installs nothing, cargo metadata only reads (it may fetch the crate index into the
# runner's cargo home). Run from the repository root on a CI runner. Prints CHECK lines; the exit code is the number
# of failures.
set -u
failed=0
check() { # check <name> <PASS|FAIL> <detail>
    echo "CHECK build-inputs/$1: $2 - $3"
    [ "$2" = PASS ] || failed=$((failed + 1))
}

# The page's dependencies, exactly as the build installs them (npm ci in tauri-app/ui, as the release does).
if out="$(cd tauri-app/ui && npm ci --dry-run --ignore-scripts 2>&1)"; then
    check ui-lock PASS "tauri-app/ui: npm ci would install from package-lock.json as it is"
else
    check ui-lock FAIL "tauri-app/ui: $(echo "$out" | grep -m3 -i 'error' | tr '\n' ' ')"
fi

# The app's crates, exactly as Cargo.lock pins them.
if out="$(cd tauri-app/src-tauri && cargo metadata --locked --format-version 1 2>&1 > /dev/null)"; then
    check cargo-lock PASS "tauri-app/src-tauri: Cargo.lock matches Cargo.toml"
else
    check cargo-lock FAIL "tauri-app/src-tauri: $(echo "$out" | tail -3 | tr '\n' ' ')"
fi

# The bundle settings parse.
if node -e 'JSON.parse(require("fs").readFileSync("tauri-app/src-tauri/tauri.conf.json", "utf8"))' 2>/dev/null; then
    check bundle-settings PASS "tauri-app/src-tauri/tauri.conf.json parses"
else
    check bundle-settings FAIL "tauri-app/src-tauri/tauri.conf.json does not parse"
fi

# The Tauri CLI, as tauri-app's lock file pins it: tauri-action runs tauri-app's "tauri" script, the CLI that npm ci
# installs there from that lock (test_tauri_cli_pinned.py keeps the manifest, the lock and the tauri crate together).
if out="$(cd tauri-app && npm ci --dry-run --ignore-scripts 2>&1)"; then
    pinned="$(node -e 'console.log(require("./tauri-app/package-lock.json").packages["node_modules/@tauri-apps/cli"].version)')"
    check cli-lock PASS "tauri-app: npm ci would install the Tauri CLI $pinned from package-lock.json as it is"
else
    check cli-lock FAIL "tauri-app: $(echo "$out" | grep -m3 -i 'error' | tr '\n' ' ')"
fi
exit "$failed"
