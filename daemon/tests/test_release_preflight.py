"""Tests for the repository-wide release metadata gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
assert SPEC and SPEC.loader
check_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_release)


def test_current_repository_release_versions_are_synchronized():
    assert check_release.validate_release(ROOT, "v0.16.0") == "0.16.0"


def test_release_gate_rejects_version_mismatch(monkeypatch):
    monkeypatch.setattr(
        check_release,
        "collect_versions",
        lambda _root: {"daemon": "0.10.0", "desktop": "0.9.0"},
    )

    with pytest.raises(ValueError, match="inconsistent"):
        check_release.validate_release(ROOT)


def test_release_gate_rejects_wrong_tag(monkeypatch):
    monkeypatch.setattr(
        check_release,
        "collect_versions",
        lambda _root: {"daemon": "0.10.0", "desktop": "0.10.0"},
    )

    with pytest.raises(ValueError, match="does not match"):
        check_release.validate_release(ROOT, "v0.10.1")


def test_release_workflow_holds_no_signing_certificate_or_notarization():
    """Signing keys never come into GitHub: Windows signing goes through SignPath, whose certificate and key stay in
    SignPath (docs/WINDOWS_SIGNING.md), and there is no macOS signing or notarisation yet."""
    release_text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    forbidden = (
        "WINDOWS_CERTIFICATE",
        "APPLE_CERTIFICATE",
        "APPLE_SIGNING_IDENTITY",
        "certificateThumbprint",
        "notariz",
    )

    for token in forbidden:
        assert token not in release_text


# --- Windows release signing (SignPath), off until the repository variable WINDOWS_RELEASE_SIGNING is "signpath" ---

RELEASE = ROOT / ".github" / "workflows" / "release.yml"
SIGNPATH_TEST = ROOT / ".github" / "workflows" / "signpath-test.yml"
ON = "signpath"


def _workflow(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _gh(expression: str, context: dict) -> object:
    """Evaluate the GitHub expressions this workflow uses (names, strings, ==, !=, !, &&, ||, always()) the way
    GitHub does: && and || return an operand, and '' counts as false."""
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    text = " ".join(text.split())
    text = text.replace("!=", " __NE__ ").replace("!", " not ").replace("__NE__", "!=")
    text = text.replace("&&", " and ").replace("||", " or ").replace("always()", "True")

    class Names(dict):
        def __missing__(self, key):
            return _Dotted(context, key)

    return eval(text, {"__builtins__": {}}, Names())  # noqa: S307 - a test evaluating the workflow's own text


class _Dotted:
    def __init__(self, context: dict, name: str) -> None:
        self.context, self.name = context, name

    def __getattr__(self, part: str):
        key = f"{self.name}.{part}"
        if key in self.context:
            return self.context[key]
        if any(name.startswith(key + ".") for name in self.context):
            return _Dotted(self.context, key)
        return ""


def _context(signing: str, platform: str = "windows-latest", **results: str) -> dict:
    context = {
        "vars.WINDOWS_RELEASE_SIGNING": signing,
        "matrix.platform": platform,
        "github.event_name": "push",
        "github.ref_name": "v1.2.3",
        "inputs.release_tag": "",
    }
    for job, result in {
        "release-gate": "success",
        "publish-daemon-to-pypi": "success",
        "build-tauri": "success",
        **results,
    }.items():
        context[f"needs.{job}.result"] = result
    return context


def _needs(job: str) -> str:
    return job.replace("-", "_")


def _eval_with_needs(expression: str, context: dict) -> object:
    # Python names cannot hold "-": rename the jobs on both sides.
    renamed = {
        key.replace("sign-windows", "sign_windows")
        .replace("build-tauri", "build_tauri")
        .replace("release-gate", "release_gate")
        .replace("publish-daemon-to-pypi", "publish_daemon_to_pypi"): value
        for key, value in context.items()
    }
    for job in ("sign-windows", "build-tauri", "release-gate", "publish-daemon-to-pypi"):
        expression = expression.replace(job, _needs(job))
    return _gh(expression, renamed)


def test_signing_is_off_by_default_and_only_the_exact_value_turns_it_on():
    jobs = _workflow(RELEASE)["jobs"]
    sign_if = jobs["sign-windows"]["if"]
    assert not _eval_with_needs(sign_if, _context(""))
    assert not _eval_with_needs(sign_if, _context("true"))
    assert not _eval_with_needs(sign_if, _context("on"))
    assert _eval_with_needs(sign_if, _context(ON))
    assert not _eval_with_needs(sign_if, _context(ON, **{"build-tauri": "failure"}))


def test_the_windows_build_attaches_its_installers_only_while_signing_is_off():
    steps = _workflow(RELEASE)["jobs"]["build-tauri"]["steps"]
    tauri = next(step for step in steps if step.get("id") == "tauri")
    tag = tauri["with"]["tagName"]
    # Off: every leg, Windows included, hands tauri-action the tag, so it attaches what it built (unsigned, as before).
    assert _gh(tag, _context("")) == "v1.2.3"
    assert _gh(tag, _context("", platform="ubuntu-22.04")) == "v1.2.3"
    # On: the Windows leg gets no tag, so tauri-action attaches nothing; the other legs attach as before.
    assert _gh(tag, _context(ON)) == ""
    assert _gh(tag, _context(ON, platform="macos-latest")) == "v1.2.3"
    staging = [step for step in steps if "SignPath" in step.get("name", "") or "signing job" in step.get("name", "")]
    assert len(staging) == 2, [step.get("name") for step in steps]
    for step in staging:
        assert _gh(step["if"], _context(ON))
        assert not _gh(step["if"], _context(""))
        assert not _gh(step["if"], _context(ON, platform="ubuntu-22.04"))


def test_publication_waits_for_the_signed_pair_while_signing_is_on():
    publish_if = _workflow(RELEASE)["jobs"]["publish-release"]["if"]
    assert _workflow(RELEASE)["jobs"]["publish-release"]["needs"][-1] == "sign-windows"
    # Off: the signing job is skipped and publication goes ahead as before.
    assert _eval_with_needs(publish_if, _context("", **{"sign-windows": "skipped"}))
    # On: only a successful signing job lets the release out; skipped, failed or cancelled stop it.
    assert _eval_with_needs(publish_if, _context(ON, **{"sign-windows": "success"}))
    for result in ("skipped", "failure", "cancelled"):
        assert not _eval_with_needs(publish_if, _context(ON, **{"sign-windows": result})), result


def test_the_signing_job_uses_the_release_policy_verifies_trust_and_fails_closed():
    job = _workflow(RELEASE)["jobs"]["sign-windows"]
    steps = job["steps"]
    submit = next(step for step in steps if step.get("id") == "signpath")
    test_submit = next(
        step
        for step in _workflow(SIGNPATH_TEST)["jobs"]["sign-windows-test-artifacts"]["steps"]
        if step.get("id") == "signpath"
    )
    assert submit["uses"] == test_submit["uses"]
    assert submit["with"]["signing-policy-slug"] == "release-signing"
    for same in ("organization-id", "project-slug", "artifact-configuration-slug", "api-token"):
        assert submit["with"][same] == test_submit["with"][same], same
    assert "test-signing" not in RELEASE.read_text(encoding="utf-8")

    verify = next(step for step in steps if step.get("name", "").startswith("Verify a trusted signature"))["run"]
    # Trusted, not merely present: inside the function that stops the job, a Valid status, the expected signer and
    # a timestamp, each one a throw; and that function is applied to both installers and the application in the MSI.
    gate = verify[verify.index("function Assert-Trusted") : verify.index("Assert-Trusted $exe[0].FullName")]
    for required in ('if ($signature.Status -ne "Valid") {', "$env:EXPECTED_SIGNER", "TimeStamperCertificate"):
        assert required in gate, required
    assert gate.count("throw ") == 3
    assert job["env"]["EXPECTED_SIGNER"] == "SignPath Foundation"
    for applied in (
        'Assert-Trusted $exe[0].FullName "the setup program"',
        'Assert-Trusted $msi[0].FullName "the MSI package"',
        'Assert-Trusted $program.FullName "the application inside the MSI"',
    ):
        assert applied in verify, applied
    assert "msiexec.exe" in verify

    why = next(step for step in steps if step.get("name", "").startswith("Say why the release stopped"))
    assert why["if"] == "failure() && steps.signpath.outcome == 'failure'"
    assert "Nothing unsigned is attached" in why["run"]

    # The signed pair is the only thing this job attaches, and only after the verification step.
    names = [step.get("name", "") for step in steps]
    attach = names.index("Attach only the signed pair to the release")
    assert attach > names.index("Verify a trusted signature on both installers and the application inside")
    attach_run = steps[attach]["run"]
    assert "signpath-signed" in attach_run and "signpath-unsigned" not in attach_run
    assert "may be unsigned" in attach_run  # refuses a release that already holds Windows installers
    for step in steps[:attach]:
        assert "releases/" not in str(step.get("run", "")), step.get("name")


def test_end_user_all_extra_contains_both_local_voice_engines():
    project = check_release._toml(ROOT / "daemon" / "pyproject.toml")
    extras = project["project"]["optional-dependencies"]
    voice = extras["voice"]
    all_extra = extras["all"]

    assert any(requirement.startswith("kokoro") for requirement in voice)
    assert any(requirement.startswith("pocket-tts") for requirement in voice)
    assert any(requirement.startswith("sounddevice") for requirement in voice)
    assert any("pilot-daemon[" in requirement and "voice" in requirement for requirement in all_extra)
