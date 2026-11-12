"""Gnomon in a supervised worker process: crashes, hangs, garbage and driver errors cost a review, never the daemon.

The supervisor is tested with a fake worker process (a thread that answers like the real one) and, for the parts
that only a real process shows (pipes, a kill, a parent that dies), with a small real Python process running the
worker's own ``serve`` loop and a fake engine. Nothing needs a GPU, torch or bitsandbytes.
"""

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from pilot.actions import Action, ActionPlan, ActionType, ApiRequestParams
from pilot.config import PilotConfig
from pilot.intelligence import gnomon_reviewer, gnomon_worker
from pilot.intelligence.gnomon_reviewer import GnomonReviewer
from pilot.intelligence.gnomon_worker import GnomonWorkerEngine, GnomonWorkerFault, GpuUnavailableError
from pilot.system.gpu_arbiter import GIB, GpuArbiter, GpuInfo, MachineReading

COOL = MachineReading("nvidia", (GpuInfo(0, "RTX", 60.0, 7 * GIB, 8 * GIB),), 8 * GIB, 16 * GIB)
LAST_WORDS = "Error an illegal memory access was encountered at line 81 in file ops.cu"
EMAIL = ActionPlan(
    actions=[
        Action(
            action_type=ActionType.API_SEND_EMAIL,
            target="all@example.com",
            parameters=ApiRequestParams(to="all@example.com", subject="Report", message="Q3 numbers"),
        )
    ],
    explanation="p",
)
ANSWER = {
    "status": "ok",
    "answer": {"yes_probability": 0.9, "probabilities": {}, "most_likely": "Read-only; no change"},
}


class _Stdout:
    def __init__(self) -> None:
        self.lines: queue.Queue[bytes] = queue.Queue()

    def readline(self) -> bytes:
        return self.lines.get()

    def close(self) -> None:
        pass


class _Stdin:
    def __init__(self, handle) -> None:
        self._handle = handle
        self._buffer = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.closed:
            raise BrokenPipeError("worker gone")
        self._buffer += data

    def flush(self) -> None:
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._handle(json.loads(line))

    def close(self) -> None:
        self.closed = True


class FakeWorker:
    """A worker process double. ``fail_on`` is the 1-based score request that misbehaves as ``mode``:
    "crash" (exits), "hang" (never answers), "garbage" (prints junk) or "driver" (reports a CUDA error)."""

    _pids = iter(range(5000, 6000))

    def __init__(self, mode: str = "ok", fail_on: int = 0, load_error: dict | None = None) -> None:
        self.mode, self.fail_on, self.load_error = mode, fail_on, load_error
        self.pid = next(self._pids)
        self.returncode: int | None = None
        self.scores = 0
        self.stdout = _Stdout()
        self.stdin = _Stdin(self._handle)

    def _reply(self, payload: dict) -> None:
        self.stdout.lines.put((json.dumps(payload) + "\n").encode())

    def _handle(self, request: dict) -> None:
        rid, op = request["id"], request["op"]
        if op == "ping":
            return self._reply({"id": rid, "status": "ok", "pid": self.pid, "rss_bytes": 50 * 1024**2})
        if op == "load":
            if self.load_error:
                return self._reply({"id": rid, "status": "error", **self.load_error})
            return self._reply({"id": rid, "status": "ok", "load_seconds": 0.1, "rss_bytes": 900 * 1024**2})
        self.scores += 1
        if self.scores == self.fail_on:
            if self.mode == "crash":
                return self.exit(3221225477)  # 0xC0000005, what Windows reports for an access violation
            if self.mode == "hang":
                return None
            if self.mode == "garbage":
                return self.stdout.lines.put(b"\x00\x01 not json\n")
            if self.mode == "driver":
                return self._reply({"id": rid, "status": "error", "driver": True,
                                    "error": "CUDA error: an illegal memory access was encountered"})  # fmt: skip
        self._reply({"id": rid, "status": "ok", "result": ANSWER})

    def exit(self, code: int) -> None:
        self.returncode = code
        self.stdin.closed = True
        self.stdout.lines.put(b"")

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        if self.returncode is None:
            self.exit(-9)

    def wait(self, timeout=None):
        return self.returncode


class Spawner:
    def __init__(self, *workers: FakeWorker) -> None:
        self.workers = list(workers)
        self.started: list[FakeWorker] = []
        self.gpu_indexes: list[int | None] = []

    def __call__(self, gpu_index=None) -> FakeWorker:
        self.gpu_indexes.append(gpu_index)
        worker = self.workers.pop(0) if self.workers else FakeWorker()
        self.started.append(worker)
        return worker


def _arbiter(reading: MachineReading = COOL, **kwargs) -> GpuArbiter:
    return GpuArbiter(read=lambda: reading, holder=lambda: None, **kwargs)


def _engine(spawner: Spawner, arbiter: GpuArbiter | None = None, **kwargs) -> GnomonWorkerEngine:
    return GnomonWorkerEngine(
        {"scoring_mode": "trained_head"},
        spawn=spawner,
        arbiter=arbiter or _arbiter(),
        stderr_tail=lambda: LAST_WORDS,
        score_timeout=kwargs.pop("score_timeout", 0.3),
        start_timeout=1.0,
        load_timeout=1.0,
        **kwargs,
    )


async def _until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


@pytest.fixture(autouse=True)
def _fast_restarts(monkeypatch):
    monkeypatch.setattr(gnomon_worker, "RESTART_BASE_SECONDS", 0.2)


# ── The supervisor on its own ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "kind"),
    [("crash", "exit"), ("hang", "hang"), ("garbage", "garbage"), ("driver", "driver")],
)
def test_a_misbehaving_worker_is_killed_reported_and_counted(mode, kind):
    arbiter = _arbiter()
    spawner = Spawner(FakeWorker(mode, fail_on=2))
    engine = _engine(spawner, arbiter)
    engine.load()
    assert engine.score({}, "noul", "q", ["no", "yes"]) == ANSWER

    with pytest.raises(GnomonWorkerFault) as raised:
        engine.score({}, "noul", "q", ["no", "yes"])

    assert raised.value.kind == kind
    assert spawner.started[0].returncode is not None  # killed, or exited by itself
    assert not engine.loaded and not engine.alive
    assert engine.last_fault["kind"] == kind
    assert arbiter.last_fault("gnomon").message.startswith(kind)
    if mode == "crash":
        assert "exit code 3221225477" in str(raised.value) and LAST_WORDS in str(raised.value)


def test_it_recovers_on_a_new_worker_with_back_off():
    spawner = Spawner(FakeWorker("crash", fail_on=1), FakeWorker())
    engine = _engine(spawner)
    assert engine.restart_delay() == 0.0
    engine.load()
    with pytest.raises(GnomonWorkerFault):
        engine.score({}, "noul", "q", ["no", "yes"])
    assert engine.restart_delay() == pytest.approx(0.2)
    engine.load()
    assert engine.score({}, "noul", "q", ["no", "yes"]) == ANSWER
    assert engine.starts == 2 and engine.worker_status()["pid"] == spawner.started[1].pid


def test_a_worker_killed_while_idle_is_noticed_once():
    spawner = Spawner(FakeWorker())
    engine = _engine(spawner)
    engine.load()
    spawner.started[0].exit(1)
    fault = engine.note_exit()
    assert isinstance(fault, GnomonWorkerFault) and fault.kind == "exit"
    assert engine.note_exit() is None


def test_the_hot_card_is_never_touched_and_the_reason_is_kept():
    hot = MachineReading("nvidia", (GpuInfo(0, "RTX", 88.0, 7 * GIB, 8 * GIB),), 8 * GIB, 16 * GIB)
    spawner = Spawner()
    with pytest.raises(GpuUnavailableError) as raised:
        _engine(spawner, _arbiter(hot)).load()
    assert raised.value.code == "hot" and raised.value.retry_seconds == gnomon_worker.GPU_WAIT_RETRY_SECONDS
    assert spawner.started == []  # no process was started


def test_memory_shortages_use_the_loaders_own_words_so_the_resource_retry_applies():
    low_ram = MachineReading("nvidia", COOL.gpus, int(1.8 * GIB), 8 * GIB)
    with pytest.raises(RuntimeError) as raised:
        _engine(Spawner(), _arbiter(low_ram)).load()
    shortage = gnomon_reviewer.classify_resource_shortage(raised.value)
    assert shortage is not None and shortage.resource == "memory" and shortage.required_bytes == int(2.5 * GIB)

    low_vram = MachineReading("nvidia", (GpuInfo(0, "RTX", 60.0, 2 * GIB, 8 * GIB),), 8 * GIB, 16 * GIB)
    with pytest.raises(RuntimeError) as raised:
        _engine(Spawner(), _arbiter(low_vram)).load()
    assert gnomon_reviewer.classify_resource_shortage(raised.value).resource == "graphics memory"


def test_several_cards_the_worker_sees_only_the_chosen_one():
    cards = (GpuInfo(0, "A", 90.0, 7 * GIB, 8 * GIB), GpuInfo(1, "B", 50.0, 7 * GIB, 8 * GIB))
    spawner = Spawner()
    _engine(spawner, _arbiter(MachineReading("nvidia", cards, 8 * GIB, 16 * GIB))).load()
    assert spawner.gpu_indexes == [1]
    env = gnomon_worker.worker_environment(1, base={"PYTHONPATH": "elsewhere"})
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[1] == "elsewhere"
    assert os.path.isdir(os.path.join(env["PYTHONPATH"].split(os.pathsep)[0], "pilot"))


def test_no_cuda_in_the_worker_means_not_supported_never_retried():
    # An NVIDIA card, but a torch build without CUDA: the worker finds out and says so once.
    spawner = Spawner(FakeWorker(load_error={"error": "CUDA is required for this opt-in local research model"}))
    with pytest.raises(GpuUnavailableError) as raised:
        _engine(spawner, _arbiter(COOL)).load()
    assert raised.value.code == "unsupported" and raised.value.retry_seconds is None


def test_a_machine_without_an_nvidia_card_starts_no_worker_to_find_out():
    spawner = Spawner()
    with pytest.raises(GpuUnavailableError) as raised:
        _engine(spawner, _arbiter(MachineReading("cpu", host_free_bytes=3 * GIB, host_total_bytes=8 * GIB))).load()
    assert raised.value.code == "unsupported" and raised.value.retry_seconds is None
    assert spawner.started == []


# ── Through the reviewer: fail closed, restart, switch the GPU off ─────────────


async def _ready(engine) -> GnomonReviewer:
    reviewer = GnomonReviewer(lambda: engine)
    reviewer.warm()
    assert await _until(lambda: reviewer.status()["state"] == "ready")
    return reviewer


async def test_worker_dying_mid_review_answers_not_reviewed_then_restarts():
    # Warm-up asks 3 questions; the review's harm question (the 5th) kills the worker.
    spawner = Spawner(FakeWorker("crash", fail_on=5), FakeWorker())
    engine = _engine(spawner)
    reviewer = await _ready(engine)

    verdict = await reviewer.review_plan("Email the Q3 report to the team", EMAIL, "")

    assert verdict is None  # the review did not happen: never a pass
    status = reviewer.status()
    assert status["state"] == "unavailable"
    assert status["reason"].startswith("Gnomon's process stopped. Restarting it in")
    assert status["last_fault"]["kind"] == "exit" and LAST_WORDS in status["last_fault"]["message"]
    assert status["process"] == "separate"

    assert await _until(lambda: reviewer.status()["state"] == "ready")
    assert len(spawner.started) == 2
    assert reviewer.status()["device"] == "cuda"
    await reviewer.close()


async def test_an_escalation_found_before_the_worker_died_still_asks_for_approval(monkeypatch):
    class Escalating(FakeWorker):
        def _handle(self, request):
            if request["op"] == "score" and request.get("question") == gnomon_reviewer.SCOPE_QUESTION:
                self.scores += 1
                low = {"status": "ok", "answer": {"yes_probability": 0.01}}
                return self._reply({"id": request["id"], "status": "ok", "result": low})
            return super()._handle(request)

    spawner = Spawner(Escalating("crash", fail_on=5), FakeWorker())
    reviewer = await _ready(_engine(spawner))
    verdict = await reviewer.review_plan("Email the Q3 report to the team", EMAIL, "")
    assert verdict.escalate is True and verdict.complete is False
    assert gnomon_reviewer.STOPPED_DURING_REVIEW in verdict.notes
    await reviewer.close()


async def test_hang_past_the_reviewers_own_limit_is_still_noticed():
    spawner = Spawner(FakeWorker("hang", fail_on=4), FakeWorker())
    engine = _engine(spawner, score_timeout=0.4)
    reviewer = GnomonReviewer(lambda: engine, call_timeout=0.1)
    reviewer.warm()
    assert await _until(lambda: reviewer.status()["state"] == "ready")
    assert await reviewer.review_completion("t", "e", "r", "x") is None  # the reviewer gave up first
    assert await _until(lambda: reviewer.status()["state"] != "ready", timeout=2.0)  # then the worker was killed
    assert reviewer.status()["last_fault"]["kind"] == "hang"
    assert await _until(lambda: reviewer.status()["state"] == "ready")
    await reviewer.close()


async def test_repeated_faults_switch_the_gpu_off_for_the_session_and_say_so():
    arbiter = _arbiter()
    spawner = Spawner(*[FakeWorker("driver", fail_on=4) for _ in range(5)])
    engine = _engine(spawner, arbiter)
    reviewer = await _ready(engine)
    for _ in range(3):
        await _until(lambda: reviewer.status()["state"] == "ready")
        await reviewer.review_completion("t", "e", "r", "x")
    assert await _until(lambda: reviewer.status()["gpu_disabled"])
    status = reviewer.status()
    assert status["state"] == "unavailable" and status["retrying"] is False
    assert status["reason"].startswith("Stopped using the graphics card for this session after 3 faults")
    assert status["reason"].endswith(gnomon_reviewer.GPU_OFF_SUFFIX)
    started = len(spawner.started)
    reviewer.warm()
    await asyncio.sleep(0.3)
    assert len(spawner.started) == started  # no more starts this session
    await reviewer.close()


async def test_apple_silicon_says_not_supported_and_never_starts_a_worker():
    spawner = Spawner()
    mac = MachineReading("apple", host_free_bytes=12 * GIB, host_total_bytes=16 * GIB)
    reviewer = GnomonReviewer(lambda: _engine(spawner, _arbiter(mac)))
    reviewer.warm()
    assert await _until(lambda: reviewer.status()["state"] == "unavailable")
    status = reviewer.status()
    assert status["supported"] is False and status["gpu_disabled"] is False
    assert status["reason"].startswith("Not supported on this system")
    assert spawner.started == []
    await reviewer.close()


async def test_the_request_review_reads_as_not_reviewed_in_the_dialog():
    """Through the server: the plan card gets state "unavailable" and the reason, which the approval dialog shows as
    "Not loaded, did not review this plan. <reason>" (PlanChecks.svelte); nothing reads as a pass."""
    from pilot.server import PilotServer

    spawner = Spawner(FakeWorker("crash", fail_on=4), FakeWorker())
    reviewer = await _ready(_engine(spawner))
    config = PilotConfig()
    config.typed_decision.enabled = True
    server = PilotServer(config)
    server._gnomon_reviewer = reviewer

    escalate, reasons, review = await server._gnomon_review_request("Email the Q3 report to the team", EMAIL, "")

    assert escalate is False and reasons == []
    assert review["state"] == "unavailable"
    assert review["reason"].startswith("Gnomon's process stopped.")
    assert review["execution_authority"] is False
    await reviewer.close()


# ── A real process: pipes, kill, parent watch (all three systems) ──────────────

FAKE_WORKER = """
import sys, time
from pilot.intelligence import gnomon_worker

class Engine:
    def __init__(self, spec):
        pass
    def load(self):
        pass
    def score(self, state, kind, question, options):
        if question == "die":
            import os
            os._exit(7)
        if question == "sleep":
            time.sleep(30)
        print("library noise on stdout")
        return {"status": "ok", "answer": {"yes_probability": 0.5}}

class Lease:
    def try_acquire(self, component):
        return True
    def release(self):
        pass

if "--protect-stdout" in sys.argv:
    import os
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
else:
    protocol = sys.stdout
sys.exit(gnomon_worker.serve(sys.stdin, protocol, engine_factory=Engine, lease_factory=Lease))
"""


def _real_spawn(_gpu_index=None):
    return subprocess.Popen(
        [sys.executable, "-c", FAKE_WORKER, "--protect-stdout"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=gnomon_worker.worker_environment(None),
    )


def test_real_process_answers_survives_library_noise_and_a_crash():
    engine = GnomonWorkerEngine(
        {}, spawn=_real_spawn, arbiter=_arbiter(), stderr_tail=lambda: "", score_timeout=10.0, start_timeout=30.0
    )
    engine.load()
    try:
        assert engine.score({}, "noul", "q", ["no", "yes"])["status"] == "ok"
        assert engine.score({}, "noul", "q", ["no", "yes"])["status"] == "ok"
        with pytest.raises(GnomonWorkerFault) as raised:
            engine.score({}, "noul", "die", ["no", "yes"])
        assert raised.value.kind == "exit" and "exit code 7" in str(raised.value)
        engine.load()  # a fresh process
        assert engine.score({}, "noul", "q", ["no", "yes"])["status"] == "ok"
    finally:
        engine.unload()


def test_real_process_that_hangs_is_killed():
    engine = GnomonWorkerEngine(
        {}, spawn=_real_spawn, arbiter=_arbiter(), stderr_tail=lambda: "", score_timeout=1.0, start_timeout=30.0
    )
    engine.load()
    process = engine._process
    with pytest.raises(GnomonWorkerFault) as raised:
        engine.score({}, "noul", "sleep", ["no", "yes"])
    assert raised.value.kind == "hang"
    assert process.wait(timeout=10) is not None


def test_the_parent_watch_fires_when_the_watched_process_ends():
    short = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    gone = threading.Event()
    gnomon_worker.watch_parent(short.pid, gone.set, poll=0.05)
    assert not gone.is_set()
    short.wait(timeout=10)
    assert gone.wait(timeout=10)


def test_a_worker_whose_daemon_dies_exits_on_its_own(tmp_path):
    """The daemon is killed (no clean shutdown): its worker must not outlive it on any system."""
    daemon = (
        "import subprocess, sys\n"
        "worker = subprocess.Popen([sys.executable, '-m', 'pilot.intelligence.gnomon_worker', '--serve',"
        " '--parent-pid', str(__import__('os').getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE)\n"
        "print(worker.pid, flush=True)\n"
        "import time; time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", daemon],
        stdout=subprocess.PIPE,
        text=True,
        env=gnomon_worker.worker_environment(None),
    )
    worker_pid = int(parent.stdout.readline())
    parent.kill()
    parent.wait(timeout=10)
    parent.stdout.close()
    import psutil

    def gone() -> bool:
        try:
            return psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:  # also when it exits between two looks
            return True

    # An upper limit only: a worker still importing on a loaded machine notices its dead parent once it is up.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if gone():
            break
        time.sleep(0.1)
    else:
        try:
            psutil.Process(worker_pid).kill()
        except psutil.NoSuchProcess:
            pass
        pytest.fail("the worker outlived its daemon")


def test_status_fields_say_where_gnomon_runs():
    engine = _engine(Spawner())
    engine.load()
    status = engine.worker_status()
    assert status["process"] == "separate" and status["device"] == "cuda" and status["starts"] == 1
    assert status["start_seconds"] is not None and status["rss_mb"] == 900
    assert SimpleNamespace(**status).last_fault is None
