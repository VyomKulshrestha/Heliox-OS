"""Text recognition on macOS through Apple's Vision framework, in the daemon's process (pyobjc).

Before this, Heliox wrote a Swift script and ran it with ``swift``, which exists only when the Xcode command-line
tools are installed and compiles the script on every call (seconds). ``pyobjc-framework-Vision`` (part of the
``macos`` extra, a prebuilt wheel) calls the same ``VNRecognizeTextRequest`` directly. The Swift route stays as the
fallback when pyobjc is not installed.

Vision returns observations with normalised bounding boxes whose origin is the bottom-left corner; lines are put
in reading order (top to bottom, then left to right) by ``reading_order``, which is pure and tested everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable

LINE_TOLERANCE = 0.01  # boxes whose vertical centres differ by less than this share a line (normalised units)


def reading_order(items: Iterable[tuple[str, float, float, float]]) -> list[str]:
    """Texts in reading order from (text, x, y, height) boxes with a bottom-left origin."""
    boxes = [(text, x, y + height / 2) for text, x, y, height in items if text]
    boxes.sort(key=lambda box: (-box[2], box[1]))
    lines: list[list[tuple[str, float, float]]] = []
    for box in boxes:
        if lines and abs(lines[-1][0][2] - box[2]) < LINE_TOLERANCE:
            lines[-1].append(box)
        else:
            lines.append([box])
    return [" ".join(text for text, _x, _y in sorted(line, key=lambda b: b[1])) for line in lines]


def recognize_text(image: bytes, accurate: bool = True) -> str:
    """The text in an image (PNG or any format ImageIO reads). Raises ImportError without pyobjc's Vision."""
    import Foundation
    import Vision

    data = Foundation.NSData.dataWithBytes_length_(image, len(image))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelAccurate if accurate else Vision.VNRequestTextRecognitionLevelFast
    )
    request.setUsesLanguageCorrection_(True)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision text recognition failed: {error}")
    items = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        box = observation.boundingBox()
        items.append((str(candidates[0].string()), box.origin.x, box.origin.y, box.size.height))
    return "\n".join(reading_order(items))
