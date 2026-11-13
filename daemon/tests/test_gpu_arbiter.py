"""The GPU arbiter: readings from each kind of machine, the pure decision, the session fault count and the lease.

No test needs a GPU, nvidia-smi, torch or a Mac: every reading is made up.
"""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from pilot.system import gpu_arbiter
from pilot.system.gpu_arbiter import (
    GIB,
    GpuArbiter,
    GpuInfo,
    GpuLease,
    Limits,
    MachineReading,
    Needs,
    decide,
    lease_holder,
    parse_nvidia_smi,
    probe_nvidia,
    read_machine,
    visible_from_env,
)


@pytest.fixture(autouse=True)
def _cards_visible(monkeypatch):
    """The suite hides every card (conftest: CUDA_VISIBLE_DEVICES=-1); these tests decide over made-up cards, so the
    variable is unset here unless a test sets it itself."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


GNOMON = Needs(gpu_bytes=int(3.5 * GIB), host_bytes=int(2.5 * GIB), requires_cuda=True)
VOICE = gpu_arbiter.VOICE_NEEDS


def _nvidia(*cards: GpuInfo, free_ram: float = 8.0, total_ram: float = 16.0, error: str = "") -> MachineReading:
    return MachineReading("nvidia", cards, int(free_ram * GIB), int(total_ram * GIB), error)


def _card(index: int = 0, temperature: float | None = 60.0, free_gb: float | None = 7.0) -> GpuInfo:
    return GpuInfo(index, "RTX", temperature, None if free_gb is None else int(free_gb * GIB), 8 * GIB)


# ── Readings ──────────────────────────────────────────────────────────────────


def test_nvidia_smi_output_with_several_cards_and_missing_values():
    cards, error = parse_nvidia_smi(
        "0, NVIDIA GeForce RTX 4060 Laptop GPU, 85, 6412, 8188\n1, Tesla T4, [N/A], [N/A], 15360\n"
    )
    assert error == ""
    assert [card.index for card in cards] == [0, 1]
    assert cards[0].temperature_c == 85.0 and cards[0].free_bytes == 6412 * 1024**2
    assert cards[1].temperature_c is None and cards[1].free_bytes is None  # the driver did not say: cannot tell


def test_nvidia_smi_absent_means_no_nvidia_card_and_failures_mean_cannot_tell():
    assert probe_nvidia(which=lambda _name: None) is None

    def hangs(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)

    assert probe_nvidia(which=lambda _name: "nvidia-smi", run=hangs)[1].startswith("nvidia-smi did not answer")
    failed = SimpleNamespace(returncode=9, stdout="", stderr="GPU is lost")
    assert "GPU is lost" in probe_nvidia(which=lambda _name: "nvidia-smi", run=lambda *a, **k: failed)[1]


def test_read_machine_tells_apple_silicon_nvidia_and_cpu_only_apart():
    host = lambda: (4 * GIB, 8 * GIB)  # noqa: E731
    assert read_machine(nvidia=lambda: None, apple=lambda: True, host=host).kind == "apple"
    assert read_machine(nvidia=lambda: None, apple=lambda: False, host=host).kind == "cpu"
    reading = read_machine(nvidia=lambda: ((_card(),), ""), apple=lambda: False, host=host)
    assert reading.kind == "nvidia" and reading.host_total_bytes == 8 * GIB
    assert gpu_arbiter.is_apple_silicon("Darwin", "arm64") and not gpu_arbiter.is_apple_silicon("Darwin", "x86_64")
    assert not gpu_arbiter.is_apple_silicon("Linux", "aarch64")


def test_cuda_visible_devices():
    assert visible_from_env(None) is None and visible_from_env("") is None
    assert visible_from_env("1,0") == (1, 0)
    assert visible_from_env("GPU-8d1f") is None  # UUIDs: cannot tell, so every card is considered


# ── The decision, per kind of machine ─────────────────────────────────────────


def test_nvidia_cool_card_with_room_is_used():
    decision = decide("gnomon", GNOMON, _nvidia(_card()))
    assert decision.code == "ok" and decision.device == "cuda" and decision.gpu_index == 0


def test_nvidia_card_at_the_limit_is_not_used():
    decision = decide("gnomon", GNOMON, _nvidia(_card(temperature=83.0)))
    assert decision.code == "hot" and decision.blocked
    assert decision.reason == "The graphics card is at 83 °C (limit 83 °C)."
    assert decide("voice", VOICE, _nvidia(_card(temperature=82.9))).code == "ok"
    assert decide("gnomon", GNOMON, _nvidia(_card(temperature=90)), Limits(max_temperature_c=95)).code == "ok"


def test_nvidia_temperature_not_reported_is_not_a_reason_to_refuse():
    assert decide("gnomon", GNOMON, _nvidia(_card(temperature=None))).code == "ok"


def test_nvidia_short_of_graphics_memory():
    decision = decide("gnomon", GNOMON, _nvidia(_card(free_gb=3.0)))
    assert decision.code == "low_memory"
    assert decision.reason == "Needs 4.0 GB of free graphics memory, 3.0 GB free now."


def test_nvidia_card_held_by_another_heliox_process_is_busy():
    decision = decide("gnomon", GNOMON, _nvidia(_card()), holder={"component": "voice", "pid": 7})
    assert decision.code == "busy" and decision.reason == "The graphics card is in use by the voice."
    assert decide("voice", VOICE, _nvidia(_card()), holder={"component": "voice", "pid": 7}).code == "ok"


def test_nvidia_driver_that_does_not_answer_is_unreadable():
    decision = decide("gnomon", GNOMON, _nvidia(error="nvidia-smi did not answer within 3 s"))
    assert decision.code == "unreadable" and decision.blocked


def test_several_cards_the_coolest_with_most_room_is_chosen():
    hot = _card(0, temperature=90, free_gb=7.5)
    small = _card(1, temperature=50, free_gb=2.0)
    roomy = _card(2, temperature=70, free_gb=6.0)
    decision = decide("gnomon", GNOMON, _nvidia(hot, small, roomy))
    assert decision.code == "ok" and decision.gpu_index == 2
    assert decide("gnomon", GNOMON, _nvidia(hot, small, roomy), Limits(visible=(1, 0))).code == "low_memory"
    # The voice cannot be pointed at another card: only the first visible one counts.
    assert decide("voice", VOICE, _nvidia(hot, roomy)).code == "hot"


def test_apple_silicon_gnomon_is_not_supported_and_says_why():
    reading = MachineReading("apple", host_free_bytes=12 * GIB, host_total_bytes=16 * GIB)
    decision = decide("gnomon", GNOMON, reading)
    assert decision.code == "unsupported"
    assert decision.reason.startswith("Not supported on this system")


def test_apple_silicon_unified_memory_counts_the_model_against_free_system_memory():
    voice = Needs(gpu_bytes=1 * GIB, host_bytes=int(1.5 * GIB))
    roomy = MachineReading("apple", host_free_bytes=6 * GIB, host_total_bytes=16 * GIB)
    tight = MachineReading("apple", host_free_bytes=2 * GIB, host_total_bytes=8 * GIB)
    assert decide("voice", voice, roomy).code == "ok" and decide("voice", voice, roomy).device == "mps"
    decision = decide("voice", voice, tight)
    assert decision.code == "low_host_memory"
    assert decision.reason == "Needs 2.5 GB of free memory, 2.0 GB free now."


@pytest.mark.parametrize(
    ("total_gb", "free_gb", "code"),
    [(8, 1.8, "unsupported"), (8, 3.0, "unsupported"), (16, 2.0, "unsupported"), (32, 20.0, "unsupported")],
)
def test_cpu_only_machines_8_16_and_32_gb(total_gb, free_gb, code):
    """Without an NVIDIA card Gnomon is not supported, said at once whatever the memory (no worker is started to
    find out); the voice has nothing to arbitrate ("no_gpu")."""
    reading = MachineReading("cpu", host_free_bytes=int(free_gb * GIB), host_total_bytes=total_gb * GIB)
    assert decide("gnomon", GNOMON, reading).code == code
    assert decide("voice", VOICE, reading).code == "no_gpu"


def test_disabled_for_the_session_wins_over_a_cool_card():
    decision = decide("gnomon", GNOMON, _nvidia(_card()), disabled_reason="Stopped using the graphics card.")
    assert decision.code == "disabled"


# ── The arbiter: faults, settings, voice device ───────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_three_faults_in_ten_minutes_switch_the_gpu_off_for_the_session():
    clock = _Clock()
    arbiter = GpuArbiter(read=lambda: _nvidia(_card()), holder=lambda: None, clock=clock, wall=lambda: 0.0)
    assert arbiter.record_fault("gnomon", "exit") is False
    clock.now += 700  # outside the window
    assert arbiter.record_fault("voice", "exit") is False
    clock.now += 60
    assert arbiter.record_fault("gnomon", "driver") is False
    assert arbiter.check("gnomon", GNOMON).code == "ok"
    clock.now += 60
    assert arbiter.record_fault("gnomon", "hang") is True
    decision = arbiter.check("gnomon", GNOMON)
    assert decision.code == "disabled"
    assert decision.reason == (
        "Stopped using the graphics card for this session after 3 faults in 10 minutes. Restart Heliox to use it again."
    )
    assert arbiter.status()["faults_this_session"] == 4
    assert arbiter.last_fault("voice").message == "exit"


def test_settings_out_of_range_keep_their_defaults():
    arbiter = GpuArbiter(read=lambda: _nvidia(_card()), holder=lambda: None)
    arbiter.configure(SimpleNamespace(max_temperature_c=500, vram_headroom_mb=256, fault_limit=0))
    assert arbiter.max_temperature_c == 83.0 and arbiter.headroom_bytes == 256 * 1024**2 and arbiter.fault_limit == 3


def test_a_probe_that_raises_never_takes_the_caller_down():
    def broken():
        raise RuntimeError("driver gone")

    assert GpuArbiter(read=broken, holder=lambda: None).check("voice", VOICE).code == "no_gpu"


@pytest.mark.parametrize(
    ("reading", "configured", "expected"),
    [
        (_nvidia(_card(temperature=88)), "auto", ("cpu", False)),
        (_nvidia(_card()), "auto", ("auto", True)),
        (_nvidia(_card()), "cpu", ("cpu", False)),
        (MachineReading("apple", host_free_bytes=GIB, host_total_bytes=8 * GIB), "auto", ("auto", False)),
        (MachineReading("cpu"), "cuda", ("cuda", False)),
    ],
)
def test_voice_device_per_machine(monkeypatch, reading, configured, expected):
    monkeypatch.setattr(gpu_arbiter, "_shared", GpuArbiter(read=lambda: reading, holder=lambda: None))
    assert gpu_arbiter.voice_device(configured) == expected


def test_voice_moves_to_the_cpu_while_gnomon_holds_the_card(monkeypatch):
    holder = {"component": "gnomon", "pid": 1}
    monkeypatch.setattr(gpu_arbiter, "_shared", GpuArbiter(read=lambda: _nvidia(_card()), holder=lambda: holder))
    assert gpu_arbiter.voice_device("auto") == ("cpu", False)


# ── The lease (real file locks, on every operating system) ────────────────────


def test_one_lease_at_a_time_and_release(tmp_path):
    first, second = GpuLease(tmp_path), GpuLease(tmp_path)
    assert first.try_acquire("gnomon")
    assert not second.try_acquire("voice")
    holder = lease_holder(tmp_path)
    assert holder["component"] == "gnomon"
    first.release()
    assert lease_holder(tmp_path) is None
    assert second.try_acquire("voice")
    second.release()


def test_a_process_that_dies_holding_the_lease_frees_it(tmp_path):
    code = (
        "import sys, time\n"
        "from pilot.system.gpu_arbiter import GpuLease\n"
        "lease = GpuLease(sys.argv[1])\n"
        "assert lease.try_acquire('gnomon')\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert lease_holder(tmp_path)["component"] == "gnomon"
        child.kill()
        child.wait(timeout=10)
        deadline = time.monotonic() + 5
        while lease_holder(tmp_path) is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert lease_holder(tmp_path) is None  # the operating system dropped the lock
    finally:
        if child.poll() is None:
            child.kill()
        child.stdout.close()


def test_the_voice_worker_falls_back_to_the_cpu_when_the_lease_is_taken(tmp_path, monkeypatch):
    from pilot.system import local_tts_worker

    monkeypatch.setenv("HELIOX_GPU_LOCK_DIR", str(tmp_path))
    gnomon = GpuLease(tmp_path)
    assert gnomon.try_acquire("gnomon")
    assert local_tts_worker._take_gpu_lease("cuda") == ("cpu", None)
    gnomon.release()
    device, lease = local_tts_worker._take_gpu_lease("cuda")
    assert device == "cuda" and lease_holder(tmp_path)["component"] == "voice"
    lease.release()
