"""Media Control Plugin — Spotify, system volume, YouTube, and media keys.

Cross-platform media control:
  - Spotify: Uses Spotify Web API (requires OAuth token in vault)
  - Volume:  Uses OS-native commands (PowerShell/amixer/osascript)
  - YouTube: Opens searches/URLs in default browser
  - Media Keys: Simulates hardware media key presses
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import webbrowser
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.plugins.media")

SPOTIFY_API = "https://api.spotify.com/v1"


def _spotify_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _spotify_request(endpoint: str, method: str = "GET", token: str = "", body: bytes | None = None) -> dict[str, Any]:
    """Make a Spotify Web API request."""
    if not token:
        return {"error": "No Spotify token. Store one via: store_api_key spotify <token>"}
    url = f"{SPOTIFY_API}{endpoint}"
    req = Request(url, headers=_spotify_headers(token), method=method, data=body)
    try:
        with urlopen(req, timeout=10) as resp:
            if resp.status == 204:
                return {"status": "ok"}
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": str(e)}


# ── Spotify Tools ──


async def spotify_play(query: str, token: str = "") -> dict[str, Any]:
    """Search for a track and start playback."""
    # Search
    search_url = f"/search?q={quote(query)}&type=track&limit=1"
    search = _spotify_request(search_url, token=token)
    if "error" in search:
        return search

    tracks = search.get("tracks", {}).get("items", [])
    if not tracks:
        return {"error": f"No results for '{query}'"}

    track = tracks[0]
    uri = track["uri"]

    # Start playback
    play_body = json.dumps({"uris": [uri]}).encode()
    result = _spotify_request("/me/player/play", method="PUT", token=token, body=play_body)
    if "error" in result:
        return result

    return {
        "track_name": track["name"],
        "artist": ", ".join(a["name"] for a in track["artists"]),
        "status": "playing",
    }


async def spotify_pause(token: str = "") -> dict[str, Any]:
    """Pause playback."""
    result = _spotify_request("/me/player/pause", method="PUT", token=token)
    return result if "error" in result else {"status": "paused"}


async def spotify_next(token: str = "") -> dict[str, Any]:
    """Skip to next track."""
    result = _spotify_request("/me/player/next", method="POST", token=token)
    if "error" in result:
        return result
    # Fetch new track info
    current = _spotify_request("/me/player/currently-playing", token=token)
    item = current.get("item", {})
    return {"track_name": item.get("name", "Unknown"), "status": "skipped"}


async def spotify_now_playing(token: str = "") -> dict[str, Any]:
    """Get currently playing track."""
    current = _spotify_request("/me/player/currently-playing", token=token)
    if "error" in current:
        return current
    if not current or not current.get("item"):
        return {"status": "nothing_playing"}

    item = current["item"]
    return {
        "track_name": item.get("name", ""),
        "artist": ", ".join(a["name"] for a in item.get("artists", [])),
        "album": item.get("album", {}).get("name", ""),
        "progress_ms": current.get("progress_ms", 0),
        "is_playing": current.get("is_playing", False),
    }


# ── Volume Tools ──


async def volume_set(level: int) -> dict[str, Any]:
    """Set system volume (0-100)."""
    level = max(0, min(100, int(level)))
    os_name = platform.system()

    try:
        if os_name == "Windows":
            # PowerShell volume control via audio endpoint
            ps_cmd = f"$vol = [Audio.Volume]::Volume; [Audio.Volume]::Volume = {level / 100:.2f}"
            # Fallback: use nircmd if available, otherwise PowerShell COM
            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"(New-Object -ComObject WScript.Shell).SendKeys([char]173); "
                    f"$wshShell = New-Object -ComObject WScript.Shell; "
                    f"1..50 | ForEach-Object {{ $wshShell.SendKeys([char]174) }}; "
                    f"1..{level // 2} | ForEach-Object {{ $wshShell.SendKeys([char]175) }}",
                ],
                capture_output=True,
                timeout=10,
            )
        elif os_name == "Darwin":
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                capture_output=True,
                timeout=5,
            )
        else:
            subprocess.run(
                ["amixer", "sset", "Master", f"{level}%"],
                capture_output=True,
                timeout=5,
            )
        return {"current_volume": level}
    except Exception as e:
        return {"error": str(e)}


async def volume_mute() -> dict[str, Any]:
    """Toggle mute."""
    os_name = platform.system()
    try:
        if os_name == "Windows":
            subprocess.run(
                ["powershell", "-Command", "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"],
                capture_output=True,
                timeout=5,
            )
        elif os_name == "Darwin":
            subprocess.run(
                ["osascript", "-e", "set volume with output muted"],
                capture_output=True,
                timeout=5,
            )
        else:
            subprocess.run(
                ["amixer", "sset", "Master", "toggle"],
                capture_output=True,
                timeout=5,
            )
        return {"muted": True}
    except Exception as e:
        return {"error": str(e)}


# ── YouTube ──


async def youtube_open(query: str) -> dict[str, Any]:
    """Open a YouTube search or video URL."""
    if query.startswith("http"):
        url = query
    else:
        url = f"https://www.youtube.com/results?search_query={quote(query)}"
    webbrowser.open(url)
    return {"url": url}


# ── Media Keys ──

_KEY_MAP = {
    "play_pause": 179,
    "next_track": 176,
    "prev_track": 177,
    "volume_up": 175,
    "volume_down": 174,
    "mute": 173,
}


async def media_key(key: str) -> dict[str, Any]:
    """Simulate a media key press."""
    key_lower = key.lower().replace(" ", "_")
    vk = _KEY_MAP.get(key_lower)
    if vk is None:
        return {"error": f"Unknown key: {key}. Valid: {', '.join(_KEY_MAP)}"}

    os_name = platform.system()
    try:
        if os_name == "Windows":
            subprocess.run(
                ["powershell", "-Command", f"(New-Object -ComObject WScript.Shell).SendKeys([char]{vk})"],
                capture_output=True,
                timeout=5,
            )
        elif os_name == "Darwin":
            # macOS media key simulation via osascript
            apple_keys = {
                "play_pause": "playpause",
                "next_track": "next",
                "prev_track": "previous",
            }
            ak = apple_keys.get(key_lower, key_lower)
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to key code {vk}'],
                capture_output=True,
                timeout=5,
            )
        else:
            # Linux: xdotool
            xdo_keys = {
                "play_pause": "XF86AudioPlay",
                "next_track": "XF86AudioNext",
                "prev_track": "XF86AudioPrev",
                "volume_up": "XF86AudioRaiseVolume",
                "volume_down": "XF86AudioLowerVolume",
                "mute": "XF86AudioMute",
            }
            xk = xdo_keys.get(key_lower, key_lower)
            subprocess.run(
                ["xdotool", "key", xk],
                capture_output=True,
                timeout=5,
            )
        return {"status": "ok", "key": key_lower}
    except Exception as e:
        return {"error": str(e)}


TOOL_HANDLERS = {
    "spotify_play": spotify_play,
    "spotify_pause": spotify_pause,
    "spotify_next": spotify_next,
    "spotify_now_playing": spotify_now_playing,
    "volume_set": volume_set,
    "volume_mute": volume_mute,
    "youtube_open": youtube_open,
    "media_key": media_key,
}
