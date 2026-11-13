#!/bin/sh
# After the .deb or .rpm is removed: say, on the terminal the person ran apt or dnf in, what Heliox left in each
# home folder and the one command that deletes it. Nothing is deleted here: a package never deletes inside home
# folders. What is named is Heliox's own program files there (its Python environment, private Python, uv and
# setup files, in ~/.config/heliox-os), which Settings > "Remove Heliox's files" would have deleted while the app
# was installed, and the app's working files (logs, web view storage), which are safe to delete; the person's
# settings and data beside them are theirs and are only mentioned.
#
# dpkg passes "remove" (then "purge" for a purge), "upgrade" and others; rpm passes how many versions stay
# installed (0 on erase). The message is shown once, on removal, never on an upgrade.

case "$1" in
    remove | 0) ;;
    *) exit 0 ;;
esac

while IFS=: read -r _ _ uid _ _ home _; do
    [ "$uid" = 0 ] || [ "$uid" -ge 1000 ] 2>/dev/null || continue
    folder="$home/.config/heliox-os"
    [ -d "$folder/env" ] || [ -d "$folder/python" ] || [ -d "$folder/tools" ] || continue
    parts=""
    megabytes=0
    for name in env python tools uv-cache setup setup.json; do
        if [ -e "$folder/$name" ]; then
            parts="$parts \"$folder/$name\""
            size="$(du -sm "$folder/$name" 2>/dev/null | cut -f 1)"
            megabytes=$((megabytes + ${size:-0}))
        fi
    done
    echo "Heliox was removed, but its own program files are still in $folder (about $megabytes MB):"
    echo "its Python environment and tools. Your Heliox settings and data beside them are kept."
    echo "To delete the program files:  rm -rf$parts"
    # The app's working files (docs/FIRST_RUN.md's words): logs, the web view's storage and its cache, at their
    # default places; safe to delete, and not the person's data.
    working=""
    for path in "$home/.local/state/heliox-os" "$home/.local/share/com.helioxos.app" "$home/.cache/com.helioxos.app"; do
        [ -e "$path" ] && working="$working $path"
    done
    if [ -n "$working" ]; then
        echo "The app's working files (logs, the web view's storage and cache) are safe to delete too:$working"
    fi
done < /etc/passwd
exit 0
