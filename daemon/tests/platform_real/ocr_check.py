"""Reading the screen for real, without the platform's own OCR, for the ``platform-readers`` job.

    python ocr_check.py linux                 # in an X session (Xvfb): a window with one line of large text
    python ocr_check.py render OUT.png        # macOS, with pyobjc: a PNG of the line, drawn with AppKit
    python ocr_check.py mac-fallback IN.png   # macOS, in a second venv without pyobjc

Linux: ``ocr_window.py`` (GTK 3, system Python) shows the line; ``screen.screenshot`` captures the screen with the
tool the daemon finds (scrot, gnome-screenshot or pyautogui), and ``vision.screen_ocr`` reads it with tesseract.

macOS, without pyobjc: ``vision._ocr_macos_native`` must take its swift route (Vision through the Xcode
command-line tools) and read the line, and the tesseract route must read it too when tesseract is installed.

Each check prints one ``CHECK`` line; the exit code is the number of failed checks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import one_loop  # noqa: E402  (next to this script)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LINE = "Heliox reads this line"
HERE = Path(__file__).resolve().parent
failures = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}{' - ' + detail if detail else ''}", flush=True)
    return ok


def found(text: str) -> bool:
    return LINE.casefold() in " ".join(text.split()).casefold()


def linux(system_python: str = "/usr/bin/python3") -> None:
    from pilot.system import screen, vision

    window = subprocess.Popen([system_python, str(HERE / "ocr_window.py")], stdout=subprocess.PIPE, text=True)
    try:
        assert window.stdout is not None
        check("ocr-linux/window-shown", window.stdout.readline().startswith("ready"))
        time.sleep(1.0)
        shot = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "heliox-ocr-shot.png"
        try:
            output = one_loop.run(screen.screenshot(str(shot)))
        except Exception as exc:
            output = f"{type(exc).__name__}: {exc}"
        tools = [name for name in ("scrot", "gnome-screenshot", "import") if shutil.which(name)]
        check(
            "ocr-linux/screenshot-written",
            shot.exists() and shot.stat().st_size > 1000,
            f"{output}; tools on PATH: {tools}",
        )
        start = time.perf_counter()
        try:
            text = one_loop.run(vision.screen_ocr(allow_slow_fallback=False))
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
        took = (time.perf_counter() - start) * 1000
        check("ocr-linux/tesseract-reads-the-line", found(text), f"{text[:160]!r} in {took:.0f} ms")
    finally:
        window.kill()


def render(out: Path) -> None:
    sys.path.insert(0, str(HERE))
    from macos_ax_check import rendered_text_png

    out.write_bytes(rendered_text_png(LINE))
    print(f"rendered {out} ({out.stat().st_size} bytes)")


def mac_fallback(png: Path) -> None:
    from pilot.system import vision

    try:
        import Vision  # noqa: F401

        check("ocr-macos/second-venv-has-no-pyobjc", False, "pyobjc's Vision imports here")
        return
    except ImportError:
        check("ocr-macos/second-venv-has-no-pyobjc", True)
    data = png.read_bytes()
    # The first read starts compiling the swift helper in the background and does not wait for it: it must come
    # back at once, saying a one-time preparation is under way. Once the helper exists (within
    # SWIFT_OCR_COMPILE_TIMEOUT), a read runs it within SWIFT_OCR_RUN_TIMEOUT. Both are timed and reported.
    start = time.perf_counter()
    try:
        text = one_loop.run(vision._ocr_macos_native(data))
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
    took = (time.perf_counter() - start) * 1000
    check(
        "ocr-macos/first-read-does-not-wait-for-the-compile",
        "one-time step" in text and took < 5000,
        f"{text[:160]!r} in {took:.0f} ms",
    )
    start = time.perf_counter()
    thread = vision._swift_state.thread
    if thread is not None:
        thread.join(vision.SWIFT_OCR_COMPILE_TIMEOUT + 10)
    compiled = (time.perf_counter() - start) * 1000
    check(
        "ocr-macos/swift-helper-compiled",
        vision._swift_helper_path().exists(),
        f"in {compiled:.0f} ms; failure remembered: {vision._swift_state.failed[:160]!r}",
    )
    start = time.perf_counter()
    try:
        text = one_loop.run(vision._ocr_macos_native(data))
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
    took = (time.perf_counter() - start) * 1000
    bound = vision.SWIFT_OCR_RUN_TIMEOUT
    check(
        "ocr-macos/vision-through-swift",
        found(text) and took <= bound * 1000,
        f"{text[:160]!r} in {took:.0f} ms (bound {bound} s)",
    )
    if shutil.which("tesseract"):
        try:
            text = one_loop.run(vision._ocr_tesseract_cli(data, "eng"))
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
        check("ocr-macos/tesseract-reads-the-line", found(text), repr(text[:160]))
    else:
        print("tesseract is not installed here; its route was not run")


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "linux"
    if mode == "render":
        render(Path(argv[1]))
    elif mode == "mac-fallback":
        mac_fallback(Path(argv[1]))
    else:
        linux()
    return failures


if __name__ == "__main__":
    from ci_only import require_runner

    require_runner("opens a full-screen window on this desktop")
    sys.exit(main(sys.argv[1:]))
