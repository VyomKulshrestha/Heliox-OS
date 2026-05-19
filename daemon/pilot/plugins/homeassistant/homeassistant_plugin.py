"""Home Assistant Plugin — IoT smart device control via HA REST API.

Controls smart lights, switches, climate devices, and scenes
through a local or cloud Home Assistant instance.

Configuration:
  Store your HA long-lived access token via:
    store_api_key homeassistant <token>
  Set your HA URL (default: http://homeassistant.local:8123):
    store_api_key homeassistant_url http://192.168.1.100:8123
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from pilot.utils.logger import get_logger

logger = get_logger( "pilot.plugins.homeassistant")

DEFAULT_HA_URL = "http://homeassistant.local:8123"


def _ha_request(
    endpoint: str,
    method: str = "GET",
    token: str = "",
    ha_url: str = DEFAULT_HA_URL,
    body: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    """Make a Home Assistant REST API request."""
    if not token:
        return {"error": "No HA token. Store one via: store_api_key homeassistant <token>"}

    url = f"{ha_url.rstrip('/')}/api{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    data = json.dumps(body).encode() if body else None
    req = Request(url, headers=headers, method=method, data=data)

    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": f"HA API error: {e}"}


def _call_service(
    domain: str,
    service: str,
    entity_id: str,
    token: str = "",
    ha_url: str = DEFAULT_HA_URL,
    extra_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Home Assistant service."""
    body: dict[str, Any] = {"entity_id": entity_id}
    if extra_data:
        body.update(extra_data)

    result = _ha_request(
        f"/services/{domain}/{service}",
        method="POST",
        token=token,
        ha_url=ha_url,
        body=body,
    )
    if isinstance(result, dict) and "error" in result:
        return result
    return {"status": "ok", "entity_id": entity_id}


# ── Light Controls ──


async def ha_light_toggle(entity_id: str, token: str = "", ha_url: str = DEFAULT_HA_URL) -> dict[str, Any]:
    """Toggle a light on/off."""
    result = _call_service("light", "toggle", entity_id, token, ha_url)
    if "error" not in result:
        result["state"] = "toggled"
    return result


async def ha_light_brightness(
    entity_id: str, brightness: int, token: str = "", ha_url: str = DEFAULT_HA_URL
) -> dict[str, Any]:
    """Set light brightness (0-255)."""
    brightness = max(0, min(255, int(brightness)))
    result = _call_service(
        "light",
        "turn_on",
        entity_id,
        token,
        ha_url,
        extra_data={"brightness": brightness},
    )
    if "error" not in result:
        result["state"] = "on"
        result["brightness"] = brightness
    return result


async def ha_light_color(
    entity_id: str,
    r: int,
    g: int,
    b: int,
    token: str = "",
    ha_url: str = DEFAULT_HA_URL,
) -> dict[str, Any]:
    """Set light RGB color."""
    color = [max(0, min(255, int(c))) for c in (r, g, b)]
    result = _call_service(
        "light",
        "turn_on",
        entity_id,
        token,
        ha_url,
        extra_data={"rgb_color": color},
    )
    if "error" not in result:
        result["state"] = "on"
        result["color"] = color
    return result


# ── Switch Controls ──


async def ha_switch_toggle(entity_id: str, token: str = "", ha_url: str = DEFAULT_HA_URL) -> dict[str, Any]:
    """Toggle a switch on/off."""
    result = _call_service("switch", "toggle", entity_id, token, ha_url)
    if "error" not in result:
        result["state"] = "toggled"
    return result


# ── Climate Controls ──


async def ha_climate_set(
    entity_id: str,
    temperature: float,
    hvac_mode: str = "heat",
    token: str = "",
    ha_url: str = DEFAULT_HA_URL,
) -> dict[str, Any]:
    """Set climate target temperature and mode."""
    result = _call_service(
        "climate",
        "set_temperature",
        entity_id,
        token,
        ha_url,
        extra_data={"temperature": float(temperature), "hvac_mode": hvac_mode},
    )
    if "error" not in result:
        result["state"] = hvac_mode
        result["temperature"] = temperature
    return result


# ── State & Discovery ──


async def ha_get_state(entity_id: str, token: str = "", ha_url: str = DEFAULT_HA_URL) -> dict[str, Any]:
    """Get entity state."""
    result = _ha_request(f"/states/{entity_id}", token=token, ha_url=ha_url)
    if isinstance(result, dict) and "error" in result:
        return result
    if isinstance(result, dict):
        return {
            "entity_id": entity_id,
            "state": result.get("state", "unknown"),
            "attributes": result.get("attributes", {}),
        }
    return {"error": "Unexpected response"}


async def ha_scene_activate(scene_id: str, token: str = "", ha_url: str = DEFAULT_HA_URL) -> dict[str, Any]:
    """Activate a scene."""
    if not scene_id.startswith("scene."):
        scene_id = f"scene.{scene_id}"
    return _call_service("scene", "turn_on", scene_id, token, ha_url)


async def ha_list_devices(domain_filter: str = "", token: str = "", ha_url: str = DEFAULT_HA_URL) -> dict[str, Any]:
    """List all entities, optionally filtered by domain."""
    result = _ha_request("/states", token=token, ha_url=ha_url)
    if isinstance(result, dict) and "error" in result:
        return result

    devices: list[dict[str, str]] = []
    if isinstance(result, list):
        for entity in result:
            eid = entity.get("entity_id", "")
            if domain_filter and not eid.startswith(f"{domain_filter}."):
                continue
            devices.append(
                {
                    "entity_id": eid,
                    "state": entity.get("state", ""),
                    "friendly_name": entity.get("attributes", {}).get("friendly_name", eid),
                }
            )
    return {"devices": devices, "count": len(devices)}


TOOL_HANDLERS = {
    "ha_light_toggle": ha_light_toggle,
    "ha_light_brightness": ha_light_brightness,
    "ha_light_color": ha_light_color,
    "ha_switch_toggle": ha_switch_toggle,
    "ha_climate_set": ha_climate_set,
    "ha_get_state": ha_get_state,
    "ha_scene_activate": ha_scene_activate,
    "ha_list_devices": ha_list_devices,
}
