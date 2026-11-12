#!/usr/bin/env bash
# The credential-store and removal parts of the real-removal CI job, after a first run and a repair through the
# app's own code. One shell, so that on Linux every part shares the D-Bus session whose GNOME keyring
# with_keyring.sh unlocked. Run from the repository root with ENVPY (the environment's Python, while it exists) and
# SYSPY (the runner's Python, with keyring) set. The exit code is the number of failed parts.
#
# Order: the daemon's cleanup against the real store; test keys, then "Remove everything" (the keys go through the
# daemon's code while its environment exists) and what is left in the home folder; set up again; test keys, then
# "keep my data" (keys kept), then "everything" with the environment already gone (the keys are then listed by
# service, not removed), this check's own keys dropped, and what is left in the home folder again.
set -u
if [ "${GITHUB_ACTIONS:-}" != "true" ] || [ -z "${RUNNER_TEMP:-}" ]; then
  echo "NOT RUN: this script removes a Heliox install and its credentials from the home folder and the credential store, so it runs only on a CI runner (GITHUB_ACTIONS=true and RUNNER_TEMP set) and changed nothing here."
  exit 3
fi
ROOT=$(pwd)
STATE="$RUNNER_TEMP/heliox-test-keys.json"
CHECK="daemon/tests/platform_real/credential_store_check.py"
HOME_CHECK="daemon/tests/platform_real/home_snapshot.py"
# The checks' own state files: on Linux runners $RUNNER_TEMP is inside the home folder, so they are left out of
# the home listing by their exact paths.
failed=0

part() {
  local title=$1
  shift
  echo "::group::$title"
  "$@"
  local status=$?
  echo "::endgroup::"
  if [ $status -ne 0 ]; then
    echo "PART FAILED ($title): exit $status"
    failed=$((failed + 1))
  fi
  return 0
}

removal() {
  (cd "$ROOT/tauri-app/src-tauri" && HELIOX_REAL_REMOVAL="$1" cargo test --all-targets -- --ignored --exact \
    uninstall::tests::real_removal_from_a_set_up_home --nocapture)
}

setup_again() {
  (cd "$ROOT/tauri-app/src-tauri" && HELIOX_REAL_SETUP=1 cargo test --all-targets -- --ignored --exact \
    setup::tests::real_first_run_installs_through_the_apps_code --nocapture)
}

part "Credential cleanup against the real store" "$ENVPY" "$CHECK" full
part "Test keys for the removal" "$ENVPY" "$CHECK" write "$STATE"
part "Remove everything: the daemon's code removes the keys" removal everything
part "No test key and no key the daemon made is left" "$SYSPY" "$CHECK" gone "$STATE"
part "Nothing of Heliox left in the home folder" "$SYSPY" "$HOME_CHECK" left "$RUNNER_TEMP/home-before.txt" \
  --ignore "$STATE" --ignore "$RUNNER_TEMP/heliox-credential-check.json"
part "Set up again through the app's code" setup_again
part "Test keys again" "$ENVPY" "$CHECK" write "$STATE"
part "Remove the app's files, keep my data" removal keep
part "The keys stay with the data" "$SYSPY" "$CHECK" kept "$STATE"
part "Then remove everything, with the environment already gone" removal everything
part "Drop the recorded keys (the check's own, and the daemon's on a runner)" "$SYSPY" "$CHECK" drop "$STATE"
part "Nothing of Heliox left in the home folder" "$SYSPY" "$HOME_CHECK" left "$RUNNER_TEMP/home-before.txt" \
  --ignore "$STATE" --ignore "$RUNNER_TEMP/heliox-credential-check.json"
echo "real removal: $failed failed parts"
exit $failed
