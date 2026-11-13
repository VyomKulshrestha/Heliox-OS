#!/bin/bash
# What the bundler produced, as lines two runs can be compared by (a run before and after a change of the Tauri CLI,
# which carries the bundler): every line about the output starts with "BUNDLE", sorted, with no timestamps or paths
# that change between runs. Also one CHECK line: the asset names match the patterns the release's publish step
# requires for this system (release.yml, require_assets). Reads only. Run from the repository root.
#
#   bundle_facts.sh linux|windows
set -u
system="$1"
bundle=tauri-app/src-tauri/target/release/bundle
failed=0
check() { # check <name> <PASS|FAIL> <detail>
    echo "CHECK bundle-facts/$1: $2 - $3"
    [ "$2" = PASS ] || failed=$((failed + 1))
}
digest() { sha256sum "$1" | cut -c1-16; }

# The assets, by name (the version in the name is the app's, not the bundler's).
names="$(find "$bundle" -maxdepth 2 -type f \( -name '*.deb' -o -name '*.rpm' -o -name '*.AppImage' -o -name '*.exe' -o -name '*.msi' \) -printf '%f\n' 2>/dev/null | sort)"
echo "$names" | sed 's/^/BUNDLE asset /'

# The release's own patterns for this system, and how many of each it requires.
case "$system" in
    linux) wanted='\.AppImage$ 1;\.deb$ 1;\.rpm$ 1' ;;
    windows) wanted='\.exe$ 1' ;; # the NSIS installer; the .msi is not built by this workflow
esac
missing=""
IFS=';' read -ra pairs <<< "$wanted"
for pair in "${pairs[@]}"; do
    pattern="${pair% *}"
    count="${pair##* }"
    found="$(grep -Ec "$pattern" <<< "$names" || true)"
    [ "$found" = "$count" ] || missing="$missing $pattern=$found(wanted $count)"
done
if [ -z "$missing" ]; then
    check release-asset-names PASS "every pattern release.yml requires for $system matches: $wanted"
else
    check release-asset-names FAIL "against release.yml's patterns:$missing"
fi

if [ "$system" = linux ]; then
    deb="$(find "$bundle/deb" -name '*.deb' -print -quit)"
    if [ -n "$deb" ]; then
        # Control fields, except the ones that change with the build itself.
        dpkg-deb -f "$deb" | grep -v -E '^(Installed-Size|Version):' | sed 's/^/BUNDLE deb-control /'
        scripts="$(mktemp -d)"
        dpkg-deb -e "$deb" "$scripts"
        for file in "$scripts"/*; do
            echo "BUNDLE deb-member $(basename "$file") $(digest "$file")"
        done
        for name in preinst postinst prerm postrm; do
            [ -f "$scripts/$name" ] && sed "s/^/BUNDLE deb-$name | /" "$scripts/$name"
        done
        dpkg-deb -c "$deb" | awk '{print $6}' | sort | sed 's/^/BUNDLE deb-file /'
        rm -rf "$scripts"
    fi
    rpm="$(find "$bundle/rpm" -name '*.rpm' -print -quit)"
    if [ -n "$rpm" ] && command -v rpm > /dev/null; then
        rpm -qp --requires --recommends "$rpm" 2>/dev/null | sort | sed 's/^/BUNDLE rpm-dependency /'
        rpm -qp --scripts "$rpm" 2>/dev/null | sed 's/^/BUNDLE rpm-scripts | /'
    elif [ -n "$rpm" ]; then
        echo "BUNDLE rpm-scripts | not read: the rpm tool is not on this runner"
    fi
else
    script="tauri-app/src-tauri/target/release/nsis/x64/installer.nsi"
    if [ -f "$script" ]; then
        # The script's own hooks, defines and deletions, with the build's paths taken out.
        grep -E '^\s*(!include|!define (BUNDLEID|PRODUCTNAME|MAINBINARYNAME|INSTALLMODE)|!insertmacro NSIS_HOOK_|RmDir /r|RMDir /r)' "$script" \
            | sed -E 's#"[A-Za-z]:[^"]*[\\/]([^\\/"]+)"#"<build>/\1"#g; s/^\s+//' | sort | sed 's/^/BUNDLE nsis /'
        for language in $(grep -oE '!include "[^"]*\.nsh"' "$script" | sed -E 's/!include "(.*)"/\1/'); do
            file="$(cygpath -u "$language" 2>/dev/null || echo "$language")"
            [ -f "$file" ] || continue
            grep -oE '^LangString [A-Za-z0-9_]+' "$file" | sort | sed "s/^/BUNDLE nsis-strings $(basename "$file") /"
        done
    else
        echo "BUNDLE nsis | no installer.nsi at $script"
    fi
fi
exit "$failed"
