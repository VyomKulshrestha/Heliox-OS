"""The co-adaptation and approval documentation must describe what the code does.

Each test pins a runtime behaviour (or reads it from the code) and checks that SECURITY.md, IPC_MESSAGE_FORMATS.md,
docs/COADAPTATION.md and docs/INTELLIGENCE_EVALUATION.md state it, so a change on either side fails here.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchmarks import coadaptation_suite
from pilot.coadaptation import service as service_module
from pilot.coadaptation import store as store_module
from pilot.coadaptation.core import special
from pilot.coadaptation.core.study_plan import PROTOCOL_HYPERPARAMETERS
from pilot.config import CO_ADAPTATION_FLOAT_RANGES, CoAdaptationConfig
from pilot.intelligence.experience import ExperienceLedger
from pilot.workflows import manual_workflow_hook
from pilot.workflows.durable_tasks import ApprovalStatus, DurableTaskStore
from tests.test_coadaptation_service import begin_params, make_harness, new_participant
from tests.test_pending_resolution_lock import _decisions, _durable_task, _Run, _server, _socket_confirm

REPO = Path(__file__).resolve().parents[2]


def _raw(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _flat(text: str) -> str:
    """Whitespace-insensitive view of Markdown prose, so line wrapping does not matter."""
    return " ".join(text.split())


def _doc(name: str) -> str:
    return _flat(_raw(name))


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    stop = text.find(end, begin + len(start))
    return text[begin : stop if stop != -1 else len(text)]


# ── spoken denial versus the locked denial helper ────────────────────────────


@pytest.fixture
async def durable(tmp_path):
    store = DurableTaskStore(tmp_path / "durable.db")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def ledger(tmp_path):
    instance = ExperienceLedger(tmp_path / "experience.db")
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


async def test_voice_listener_denial_does_not_override_an_unconsumed_approval_as_documented(durable, ledger):
    await _durable_task(durable)
    server = _server(durable=durable, ledger=ledger)
    server._speak_voice_response = AsyncMock()
    run = _Run(server)
    pending = await run.start()

    assert await _socket_confirm(server, pending, True) == {"status": "ok", "confirmed": True}
    assert not pending.event.is_set() and "plan-1" in server._pending_confirms  # approved, not yet consumed

    await server._voice_command_dispatch("No.")
    messages = [
        call.args[1]["message"]
        for call in server._broadcast_notification.await_args_list
        if call.args[0] == "voice_result"
    ]
    # The dialog's reply was written, as _process_request does after the socket send.
    pending.acknowledged = True
    pending.event.set()
    confirmed, _approved, _required = await run.task

    assert messages == [
        "That approval was already answered, so nothing was denied. Say or press Stop to cancel the task."
    ]
    assert confirmed is True and run.executed == [[0, 1]]
    assert (await durable.get_approval("plan-1")).status == ApprovalStatus.APPROVED
    assert await _decisions(ledger) == ["approved"]

    security = _doc("SECURITY.md")
    assert "a typed or spoken denial all deny" not in security
    assert "The always-on voice listener does not use that helper." in security
    assert "the spoken denial does not override it" in security
    ipc = _doc("IPC_MESSAGE_FORMATS.md")
    assert "it does not override an approval the waiting task has not acted on yet" in ipc


# ── erasure: the keyring exception and the expiry lag ────────────────────────


async def test_degraded_erasure_keeps_the_key_pending_until_the_keyring_returns_as_documented(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    pk = await new_participant(h)
    await h.svc.close()

    h.keys.set_available(False)
    h.svc = h.new_service()
    await h.svc.start()
    try:
        revoked = await h.svc.revoke_consent({"participant_key": pk})
        degraded_status = await h.svc.status({})  # its prune retries the list and the keyring is still locked
    finally:
        await h.svc.close()
    assert revoked["status"] == "ok"
    assert revoked["erased"]["key_destroyed"] is False and revoked["erased"]["pending"] is True
    assert degraded_status["erasure_pending"] is True
    pending_file = h.db_path.with_name(store_module.PENDING_ERASURE_FILE)
    assert pk in h.keys and pk in pending_file.read_text(encoding="utf-8")  # the rows are gone, the key is not

    h.keys.set_available(True)
    h.svc = h.new_service()
    await h.svc.start()  # the start retries the pending-erasure list
    try:
        assert (await h.svc.status({}))["erasure_pending"] is False
    finally:
        await h.svc.close()
    assert pk not in h.keys and not pending_file.exists()

    pending_name = f"`{store_module.PENDING_ERASURE_FILE}`"
    security = _doc("SECURITY.md")
    assert "`key_destroyed: false` and `pending: true`" in security
    assert pending_name in security
    assert "Heliox retries it at every daemon start, `co_adaptation_status` read and prune" in security
    assert "crypto-shredding takes effect only when the keyring can be reached again" in security
    coadaptation = _doc("docs/COADAPTATION.md")
    assert "`key_destroyed: false` and `pending: true`" in coadaptation
    assert pending_name in coadaptation
    ipc = _doc("IPC_MESSAGE_FORMATS.md")
    assert "`key_destroyed` is `false` when the OS keyring could not be reached" in ipc
    assert pending_name in ipc
    assert "The daemon retries that list at every service start" in ipc
    for text in (security, coadaptation, ipc):
        assert "nothing retries" not in text and "nothing records the key" not in text
        assert "`erasure_pending: true`" in text or "`erasure_pending` is true" in text


async def test_every_status_field_is_in_the_documented_example(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    try:
        await new_participant(h)
        reply = await h.svc.status({})
    finally:
        await h.svc.close()
    section = _between(_raw("IPC_MESSAGE_FORMATS.md"), "#### `co_adaptation_status`", "\n#### ")
    example = json.loads(_between(section, "```json", "\n```").removeprefix("```json"))
    assert set(reply) == set(example), sorted(set(reply) ^ set(example))


async def test_expiry_erases_lazily_within_the_documented_prune_interval(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    try:
        pk = await new_participant(h, retention_days=30)
        h.clock.advance(days=31)
        # Expired but not yet pruned: no start, status, session_begin or open-session call has run.
        assert pk in await h.svc._store.participants() and pk in h.keys
        assert (await h.svc.status({}))["participants"] == []  # a status read runs the prune
        assert pk not in await h.svc._store.participants() and pk not in h.keys
    finally:
        await h.svc.close()

    hours = service_module.PRUNE_INTERVAL_S // 3600
    for name in ("SECURITY.md", "docs/COADAPTATION.md", "IPC_MESSAGE_FORMATS.md"):
        text = _doc(name)
        assert f"every {hours} hours" in text, name
        assert f"up to {hours} hours" in text, name
    assert "Consent duration equals retention" not in _doc("docs/COADAPTATION.md")


# ── refusals and the rules file ──────────────────────────────────────────────


def _service_refusal_reasons() -> set[str]:
    tree = ast.parse(inspect.getsource(service_module))
    reasons: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_refused":
            if node.args:
                reasons |= {c.value for c in ast.walk(node.args[0]) if isinstance(c, ast.Constant)}
        elif isinstance(node, ast.Dict):
            pairs = {
                k.value: v.value
                for k, v in zip(node.keys, node.values, strict=True)
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
            }
            if pairs.get("status") == "refused" and isinstance(pairs.get("reason"), str):
                reasons.add(pairs["reason"])
        elif isinstance(node, ast.AsyncFunctionDef) and node.name == "_consent_for":
            for ret in (n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None):
                reasons |= {c.value for c in ast.walk(ret.value) if isinstance(c, ast.Constant)}
    return {reason for reason in reasons if isinstance(reason, str)}


async def test_every_study_service_refusal_is_documented(tmp_path):
    h = make_harness(tmp_path)
    await h.svc.start()
    pk = await new_participant(h)
    h.preferences_path.write_text("{not json", encoding="utf-8")
    await h.restart()
    try:
        reply = await h.svc.begin_session(begin_params(pk))
    finally:
        await h.svc.close()
    assert reply == {"status": "refused", "reason": "preferences_unreadable"}

    reasons = _service_refusal_reasons()
    assert {"preferences_unreadable", "keyring_unavailable", "consent_expired", "no_consent"} <= reasons
    listed = _between(_doc("IPC_MESSAGE_FORMATS.md"), "The study service itself may refuse with", ".")
    missing = sorted(reason for reason in reasons if f"`{reason}`" not in listed)
    assert missing == [], f"IPC_MESSAGE_FORMATS.md does not list these refusals: {missing}"
    ipc = _doc("IPC_MESSAGE_FORMATS.md")
    assert "no study session starts until the file is fixed or removed" in ipc
    assert "restart Heliox after fixing or removing it" in ipc
    assert "`preferences_unreadable`" in _doc("docs/COADAPTATION.md")


# ── study settings pinned by the protocol ────────────────────────────────────


def _config_rows() -> dict[str, str]:
    section = _between(_raw("IPC_MESSAGE_FORMATS.md"), "#### `co_adaptation_config_update`", "\n#### ")
    rows = re.findall(r"^\| `(\w+)` \| \w+ \| `[^`]*` \| (.*) \|$", section, flags=re.MULTILINE)
    return dict(rows)


def test_config_table_says_which_study_settings_the_protocol_pins(tmp_path):
    h = make_harness(tmp_path)
    effective = service_module._Effective(adaptive=False, cross_session=False, gaze=False, hand=False, export=False)
    rows = _config_rows()
    coadaptation = _doc("docs/COADAPTATION.md")
    pinned: list[str] = []
    for name, (low, high) in CO_ADAPTATION_FLOAT_RANGES.items():
        protocol = float(PROTOCOL_HYPERPARAMETERS[name])
        used = {}
        for value in (low, high):
            view = dataclasses.replace(CoAdaptationConfig(), **{name: value})
            settings, deviations = h.svc._derive_settings(effective, view)
            used[value] = float(getattr(settings, name))
            assert (name in deviations) == (value != protocol)  # a saved value that differs is always recorded
        row = rows[name]
        if used[low] == used[high] == protocol:
            pinned.append(name)
            assert f"always use the protocol's {protocol:g}" in row, (name, row)
            assert f"`{name}`" in _between(coadaptation, "**Study sessions follow the protocol.**", " - **")
        else:
            assert "stricter" in row, (name, row)
            if used[high] == protocol < high:
                assert f"uses at most {protocol:g}" in row, (name, row)
            if used[low] == protocol > low:
                assert f"uses at least {protocol:g}" in row, (name, row)
    assert pinned  # the protocol pins at least one setting
    ipc = _doc("IPC_MESSAGE_FORMATS.md")
    assert "but it does not change what a session runs with" in ipc
    assert "also when the effective value did not change" in ipc


# ── manual recording and the window title ────────────────────────────────────


def test_manual_recording_docs_name_only_the_titles_the_check_matches(monkeypatch):
    tauri = json.loads((REPO / "tauri-app" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    desktop_title = tauri["app"]["windows"][0]["title"]
    page_title = re.search(r"<title>(.*?)</title>", (REPO / "tauri-app" / "ui" / "index.html").read_text("utf-8"))[1]
    assert manual_workflow_hook._is_heliox_window_title(desktop_title)
    browser_matches = manual_workflow_hook._is_heliox_window_title(f"{page_title} - Google Chrome")
    mention_matches = manual_workflow_hook._is_heliox_window_title(f"server.py - {desktop_title} - Visual Studio Code")
    monkeypatch.setattr(sys, "platform", "linux")
    linux_skips = manual_workflow_hook._is_heliox_foreground()

    security = _between(_doc("SECURITY.md"), "## Manual workflow recording", " ## ")
    ipc = _between(_doc("IPC_MESSAGE_FORMATS.md"), "manual-input recording skips", "### ")
    for name, text in (("SECURITY.md", security), ("IPC_MESSAGE_FORMATS.md", ipc)):
        assert f'"{desktop_title}"' in text, name
        assert '"Heliox OS"' not in text, name  # no Heliox page or window carries this title
        if not browser_matches:
            assert f'"{page_title}"' in text and "browser" in text, name
        if not mention_matches:  # the whole title must match; a mention is someone else's window
            assert 'looks for "heliox os"' not in text, name
            assert "whole title" in text and "only mentions the project is recorded" in text, name
        if not linux_skips:
            assert "on Linux and macOS nothing is skipped" in text, name


# ── simulated benchmark reproduction ─────────────────────────────────────────


def _documented_suite_commands() -> list[list[str]]:
    section = _between(_raw("docs/INTELLIGENCE_EVALUATION.md"), "### Simulated-user benchmark", "\n### ")
    block = _between(section, "```powershell", "\n```")
    marker = "-m benchmarks.coadaptation_suite"
    return [line.split(marker, 1)[1].split() for line in block.splitlines() if marker in line]


def test_documented_benchmark_commands_run_in_order_with_one_holdout_look(tmp_path, monkeypatch):
    def fake_benchmark(**_kwargs):
        return {"metrics": {}, "claim_boundary": "simulated users only; no claim about people"}

    monkeypatch.setattr(coadaptation_suite, "benchmark", fake_benchmark)
    monkeypatch.setattr(coadaptation_suite, "_power_report", lambda _protocol, _digest: {"claim_boundary": "x"})
    commands = _documented_suite_commands()
    assert any("--full" in args for args in commands) and any("--power" in args for args in commands)
    for args in commands:
        try:
            code = coadaptation_suite.main([*args, "--state-dir", str(tmp_path)])
        except SystemExit as exc:  # parser.exit(2) on a refused second holdout look
            pytest.fail(f"documented command {args} exited with {exc.code}")
        assert code == 0, args

    looks = (tmp_path / coadaptation_suite.LOOKS_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["extra"] for line in looks] == [False]
    digest = coadaptation_suite.protocol_sha256(coadaptation_suite.load_protocol())[:12]
    doc = _doc("docs/INTELLIGENCE_EVALUATION.md")
    for path in tmp_path.glob("coadaptation_*.json"):
        assert f"`{path.name.replace(digest, '<sha12>')}`" in doc, path.name


def test_quick_benchmark_text_matches_a_fresh_run_and_the_registered_artifact():
    # The quick run is deterministic on one platform and takes under a minute. Floating-point differences between
    # platforms' math libraries can send the simulated engine down slightly different choices, so the exact numbers
    # in the doc are checked against the registered artifact (on the machine that registered it, the artifact must
    # equal a fresh run), and a fresh run anywhere must keep every invariant and ordering the prose states.
    report = coadaptation_suite.benchmark(quick=True)
    digest = coadaptation_suite.protocol_sha256(coadaptation_suite.load_protocol())[:12]
    artifact = coadaptation_suite.DEFAULT_STATE_DIR / f"coadaptation_quick_{digest}.json"
    registered = None
    if artifact.exists():  # .bench-state is git-ignored; when present, the registered result must be current
        registered = json.loads(artifact.read_text(encoding="utf-8"))
        for key in ("metrics", "agency_audit"):
            assert registered[key] == json.loads(json.dumps(report[key])), f"{artifact.name} is stale: rerun --quick"
    doc = _doc("docs/INTELLIGENCE_EVALUATION.md")
    assert f"for protocol hash `{digest}`" in doc

    def cell(summary: dict) -> str:
        low, high = summary["ci95"]
        return f"{summary['mean']:.2f} [{low:.2f}, {high:.2f}]"

    families = report["metrics"]["by_family"]
    means: dict[str, dict[str, float]] = {}
    for family in ("F1", "F3"):
        policies = families[family]["policies"]
        utility = {name: policies[name]["realized_utility_per_trial"] for name in policies}
        means[family] = {name: summary["mean"] for name, summary in utility.items()}
        if registered is not None:
            documented = registered["metrics"]["by_family"][family]["policies"]
            cells = " | ".join(
                cell(documented[name]["realized_utility_per_trial"])
                for name in ("1_fixed_assistance_off", "4_uniform_random", "9_engine_full")
            )
            assert f"| {family} | {cells} |" in doc, family
        for name, policy in policies.items():
            assert policy["constraint_violations"] == 0 and policy["acting_under_ambiguity"] == 0, (family, name)
    # The prose reads the table: on F1 the engine did not beat assistance off, on F3 it scored highest.
    assert means["F1"]["9_engine_full"] < means["F1"]["1_fixed_assistance_off"]
    assert means["F3"]["9_engine_full"] == max(means["F3"].values())
    for name, audit in report["agency_audit"].items():
        hard = {key: value for key, value in audit["counts"].items() if key not in ("A7", "A8")}
        assert audit["passed"] and not any(hard.values()), name

    gains = []
    for family in ("F1", "F3"):
        gain = families[family]["prediction"]["full_vs_no_today"]
        low, high = gain["ci95"]
        gains.append(f"{gain['mean']:+.3f} [{low:.3f}, {high:.3f}] on {family}")
        assert low <= 0.0  # "the day term did not improve prediction in either tuning family"
    if registered is not None:
        documented_gains = []
        for family in ("F1", "F3"):
            gain = registered["metrics"]["by_family"][family]["prediction"]["full_vs_no_today"]
            low, high = gain["ci95"]
            documented_gains.append(f"{gain['mean']:+.3f} [{low:.3f}, {high:.3f}] on {family}")
        assert " and ".join(documented_gains) in doc


def test_ope_validation_text_matches_the_code_and_the_registered_artifact():
    doc = _doc("docs/INTELLIGENCE_EVALUATION.md")
    if "vectorize" not in inspect.getsource(special.norm_cdf_np) + inspect.getsource(special.erfc_np):
        assert "np.vectorize(math.erfc)" not in doc

    digest = coadaptation_suite.protocol_sha256(coadaptation_suite.load_protocol())[:12]
    artifact = coadaptation_suite.DEFAULT_STATE_DIR / f"coadaptation_ope_validation_{digest}.json"
    if not artifact.exists():
        pytest.skip("no registered OPE validation for the current simulation protocol (.bench-state is git-ignored)")
    result = json.loads(artifact.read_text(encoding="utf-8"))["ope_validation"]
    assert f"`{digest}`" in doc

    def row(label: str, target: dict) -> str:
        bias = "not estimated" if target["mean_bias"] is None else f"{target['mean_bias']:+.3f}"
        return (
            f"| {label} | {target['covered']} of {target['studies']} | {target['refusal_rate']:.0%} "
            f"| {target['median_ess']:.1f} | {bias} |"
        )

    assert row("In-sample `S_2`", result["in_sample"]["S_2"]) in doc
    assert row("In-sample `S_3`", result["in_sample"]["S_3"]) in doc
    assert row("Out-of-sample `S_2`", result["out_of_sample_S_2"]) in doc
    s2, s3 = result["in_sample"]["S_2"]["mean_bias"], result["in_sample"]["S_3"]["mean_bias"]
    assert f"{s2:+.3f} utility per trial for `S_2` and {s3:+.3f} for `S_3`" in doc
    errors = result["estimator_check"]["mean_errors"]
    assert (
        f"{errors['dr']['mean']:+.3f} for doubly robust, {errors['ips']['mean']:+.3f} for clipped IPS and "
        f"{errors['snips']['mean']:+.3f} for SNIPS"
    ) in doc
    assert f"median ESS {result['estimator_check']['median_ess']:.0f}" in doc
    if result["elapsed_s"] < 900:
        assert "does not finish under `--timeout=900`" not in doc
        assert "inside the `--timeout=900`" in doc
