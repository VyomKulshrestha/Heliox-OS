"""neural_status says whether the sidecar is linked, and after a drop that it reconnects and must be armed again."""

from pilot.config import PilotConfig
from pilot.server import PilotServer


def test_sidecar_state_linked_lost_and_never_started():
    server = PilotServer(PilotConfig())
    assert server._neural_sidecar_state() == {"linked": False, "message": "The neural sidecar is not running."}

    server._neural_sidecar_lost_at = 1_800_000_000.0
    lost = server._neural_sidecar_state()
    assert lost["linked"] is False and "reconnects by itself" in lost["message"] and "Arm it again" in lost["message"]

    server._neural_sidecar_client = object()
    assert server._neural_sidecar_state() == {"linked": True, "message": ""}
