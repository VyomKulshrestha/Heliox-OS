"""Screen Understanding — OCR, element detection, screen analysis.

Combines screenshot capture with OCR (Tesseract/EasyOCR/Windows native)
and optional vision model analysis for true screen comprehension.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
from pathlib import Path

from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_powershell

logger = logging.getLogger("pilot.system.vision")


async def _capture_screenshot_bytes(region: tuple[int, int, int, int] | None = None) -> bytes:
    """Capture screenshot and return PNG bytes."""
    try:
        from io import BytesIO

        import pyautogui

        img = pyautogui.screenshot(region=region)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: capture via system command and read file
        tmp = os.path.join(tempfile.gettempdir(), f"pilot_screen_{os.getpid()}.png")
        from pilot.system.screen import screenshot

        await screenshot(tmp)
        data = Path(tmp).read_bytes()
        os.unlink(tmp)
        return data


async def screen_ocr(
    region: tuple[int, int, int, int] | None = None,
    language: str = "eng",
) -> str:
    """Extract ALL text from the screen (or a region) using OCR.

    region: (left, top, width, height) or None for full screen.
    Returns all detected text.
    """
    img_bytes = await _capture_screenshot_bytes(region)

    # Windows native OCR first (zero external dependencies)
    if CURRENT_PLATFORM == Platform.WINDOWS:
        try:
            result = await _ocr_windows_native(img_bytes)
            if result and not result.startswith("Windows OCR failed"):
                return result
        except Exception:
            pass

    # Try EasyOCR (better accuracy, GPU accelerated)
    try:
        return await _ocr_easyocr(img_bytes, language)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("EasyOCR failed: %s, trying Tesseract", e)

    # Try Tesseract (catches both import errors AND binary-not-found)
    try:
        return await _ocr_tesseract(img_bytes, language)
    except (ImportError, OSError, Exception) as e:
        err_str = str(e).lower()
        if "not installed" in err_str or "not in your path" in err_str or "import" in err_str:
            pass  # Fall through
        else:
            raise  # Real error, re-raise

    raise RuntimeError(
        "No OCR engine available. Options:\n"
        "  1. Windows native OCR (should be built-in on Win10+)\n"
        "  2. pip install easyocr  (recommended, GPU-accelerated)\n"
        "  3. pip install pytesseract + install Tesseract binary"
    )


async def _ocr_easyocr(img_bytes: bytes, language: str) -> str:
    from io import BytesIO

    import easyocr
    import numpy as np
    from PIL import Image

    # Map common language codes to easyocr format
    lang_map = {
        "eng": "en",
        "fra": "fr",
        "deu": "de",
        "spa": "es",
        "ita": "it",
        "por": "pt",
        "rus": "ru",
        "jpn": "ja",
        "kor": "ko",
        "chi_sim": "ch_sim",
    }
    lang_code = lang_map.get(language, language[:2] if len(language) >= 2 else "en")

    def _do():
        # Cache the reader on the module to avoid re-init
        cache_key = f"_easyocr_reader_{lang_code}"
        reader = getattr(_ocr_easyocr, cache_key, None)
        if reader is None:
            try:
                reader = easyocr.Reader([lang_code], gpu=True, verbose=False)
            except Exception:
                # GPU not available — fall back to CPU
                reader = easyocr.Reader([lang_code], gpu=False, verbose=False)
            setattr(_ocr_easyocr, cache_key, reader)
        img = Image.open(BytesIO(img_bytes))
        results = reader.readtext(np.array(img))
        lines = [text for (_, text, conf) in results if conf > 0.3]
        return "\n".join(lines)

    return await asyncio.to_thread(_do)


async def _ocr_tesseract(img_bytes: bytes, language: str) -> str:
    from io import BytesIO

    import pytesseract
    from PIL import Image

    def _do():
        img = Image.open(BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang=language)

    return await asyncio.to_thread(_do)


async def _ocr_windows_native(img_bytes: bytes) -> str:
    """Use Windows built-in OCR via PowerShell."""
    tmp = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    Path(tmp).write_bytes(img_bytes)
    try:
        code, out, err = await run_powershell(
            f"Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
            f"$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]; "
            f"$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages(); "
            f"$file = [Windows.Storage.StorageFile]::GetFileFromPathAsync('{tmp}').GetAwaiter().GetResult(); "
            f"$stream = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read).GetAwaiter().GetResult(); "
            f"$decoder = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream).GetAwaiter().GetResult(); "
            f"$bitmap = $decoder.GetSoftwareBitmapAsync().GetAwaiter().GetResult(); "
            f"$result = $engine.RecognizeAsync($bitmap).GetAwaiter().GetResult(); "
            f"$result.Text"
        )
        return out.strip() if code == 0 else f"Windows OCR failed: {err}"
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


async def screen_find_text(
    target_text: str,
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Find specific text on screen and return its approximate location.

    Uses OCR with bounding boxes to locate text elements.
    """
    img_bytes = await _capture_screenshot_bytes(region)

    try:
        from io import BytesIO

        import easyocr
        import numpy as np
        from PIL import Image

        def _do():
            reader = easyocr.Reader(["en"], gpu=True)
            img = Image.open(BytesIO(img_bytes))
            results = reader.readtext(np.array(img))
            matches = []
            for bbox, text, conf in results:
                if target_text.lower() in text.lower():
                    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    cx = int(sum(p[0] for p in bbox) / 4)
                    cy = int(sum(p[1] for p in bbox) / 4)
                    matches.append(
                        {
                            "text": text,
                            "center": (cx, cy),
                            "confidence": round(conf, 3),
                            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                        }
                    )
            return matches

        matches = await asyncio.to_thread(_do)
        if not matches:
            return f"Text '{target_text}' not found on screen"
        return json.dumps({"matches": matches, "count": len(matches)}, indent=2)

    except ImportError:
        pass

    try:
        from io import BytesIO

        import pytesseract
        from PIL import Image

        def _do():
            img = Image.open(BytesIO(img_bytes))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            matches = []
            for i, text in enumerate(data["text"]):
                if target_text.lower() in text.lower() and int(data["conf"][i]) > 30:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    matches.append(
                        {
                            "text": text,
                            "center": (x + w // 2, y + h // 2),
                            "confidence": int(data["conf"][i]) / 100,
                            "bbox": [x, y, x + w, y + h],
                        }
                    )
            return matches

        matches = await asyncio.to_thread(_do)
        if not matches:
            return f"Text '{target_text}' not found on screen"
        return json.dumps({"matches": matches, "count": len(matches)}, indent=2)

    except ImportError:
        raise RuntimeError("Install easyocr or pytesseract for text finding")


async def screen_analyze(
    prompt: str = "Describe what you see on the screen",
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Analyze the screen using a vision-capable LLM.

    Takes a screenshot, encodes it as base64, and sends it to the model
    for analysis. Requires a vision-capable model (e.g., llava, bakllava).
    """
    img_bytes = await _capture_screenshot_bytes(region)
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    # Try Ollama with vision model
    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            # Try llava or bakllava
            for model in ["llava:7b", "llava", "bakllava", "moondream"]:
                try:
                    resp = await client.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "images": [b64_image],
                            "stream": False,
                        },
                    )
                    if resp.status_code == 200:
                        return resp.json().get("response", "No response")
                except Exception:
                    continue

        # Fallback: just do OCR and describe
        ocr_text = await screen_ocr(region)
        return f"[Vision model not available — falling back to OCR]\nScreen text content:\n{ocr_text[:2000]}"
    except Exception as e:
        return f"Screen analysis failed: {e}"


async def screen_element_map(
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Create a map of interactive elements on screen.

    Identifies buttons, text fields, links, etc. using OCR + heuristics.
    Returns a JSON list of detected elements with positions.
    """
    img_bytes = await _capture_screenshot_bytes(region)

    try:
        from io import BytesIO

        import easyocr
        import numpy as np
        from PIL import Image

        def _do():
            reader = easyocr.Reader(["en"], gpu=True)
            img = Image.open(BytesIO(img_bytes))
            results = reader.readtext(np.array(img))
            elements = []
            for i, (bbox, text, conf) in enumerate(results):
                if conf < 0.3 or not text.strip():
                    continue
                cx = int(sum(p[0] for p in bbox) / 4)
                cy = int(sum(p[1] for p in bbox) / 4)
                w = int(max(p[0] for p in bbox) - min(p[0] for p in bbox))
                h = int(max(p[1] for p in bbox) - min(p[1] for p in bbox))

                # Heuristic element type detection
                elem_type = "text"
                text_lower = text.lower().strip()
                if text_lower in (
                    "ok",
                    "cancel",
                    "save",
                    "close",
                    "yes",
                    "no",
                    "apply",
                    "submit",
                    "next",
                    "back",
                    "done",
                    "open",
                    "delete",
                    "remove",
                    "install",
                    "run",
                ):
                    elem_type = "button"
                elif w > 100 and h < 30:
                    elem_type = "label"
                elif text_lower.startswith("http") or text_lower.startswith("www"):
                    elem_type = "link"

                elements.append(
                    {
                        "id": i,
                        "type": elem_type,
                        "text": text,
                        "center": {"x": cx, "y": cy},
                        "size": {"w": w, "h": h},
                        "confidence": round(conf, 3),
                    }
                )
            return elements

        elements = await asyncio.to_thread(_do)
        return json.dumps(
            {
                "elements": elements,
                "count": len(elements),
                "note": "Use mouse_click with center coordinates to interact",
            },
            indent=2,
        )

    except ImportError:
        return "Install easyocr for element detection: pip install easyocr"
