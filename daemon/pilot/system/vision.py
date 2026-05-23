"""Screen Understanding — OCR, element detection, screen analysis.

Combines screenshot capture with OCR engines per platform:
  - Windows: WinRT OCR (built-in Win10+), tesseract CLI, EasyOCR
  - macOS:   Vision framework (built-in), tesseract CLI, EasyOCR
  - Linux:   tesseract CLI, EasyOCR
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pilot.config import PilotConfig
from pilot.system.http_client import create_httpx_client
from pilot.system.platform_detect import CURRENT_PLATFORM, Platform, run_command

logger = logging.getLogger("pilot.system.vision")

# ──────────────────────────────────────────────────────────────────────
#  Screenshot capture
# ──────────────────────────────────────────────────────────────────────


async def _capture_screenshot_bytes(region: tuple[int, int, int, int] | None = None) -> bytes:
    """Capture screenshot and return PNG bytes."""
    try:
        from io import BytesIO

        import pyautogui

        img = pyautogui.screenshot(region=region)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.debug(f"pyautogui screenshot failed ({e}), falling back to system command")
        # Fallback: capture via system command and read file
        tmp = os.path.join(tempfile.gettempdir(), f"pilot_screen_{os.getpid()}.png")
        from pilot.system.screen import screenshot

        await screenshot(tmp)
        data = Path(tmp).read_bytes()
        os.unlink(tmp)
        return data


# ──────────────────────────────────────────────────────────────────────
#  Main OCR entry point
# ──────────────────────────────────────────────────────────────────────


async def screen_ocr(
    region: tuple[int, int, int, int] | None = None,
    language: str = "eng",
) -> str:
    """Extract ALL text from the screen (or a region) using OCR.

    region: (left, top, width, height) or None for full screen.
    Returns all detected text.

    Tries platform-native OCR first, then EasyOCR, then Tesseract.
    """
    img_bytes = await _capture_screenshot_bytes(region)
    errors: list[str] = []

    # ── Platform-native OCR ──────────────────────────────────────────
    if CURRENT_PLATFORM == Platform.WINDOWS:
        try:
            result = await _ocr_windows_native(img_bytes)
            if result and result.strip():
                logger.info("OCR succeeded via Windows native (WinRT)")
                return result
        except Exception as e:
            errors.append(f"Windows native OCR: {e}")
            logger.debug("Windows native OCR failed: %s", e)

    elif CURRENT_PLATFORM == Platform.MACOS:
        try:
            result = await _ocr_macos_native(img_bytes)
            if result and result.strip():
                logger.info("OCR succeeded via macOS Vision framework")
                return result
        except Exception as e:
            errors.append(f"macOS Vision OCR: {e}")
            logger.debug("macOS Vision OCR failed: %s", e)

    # ── Tesseract CLI (cross-platform, no Python wrapper needed) ─────
    try:
        result = await _ocr_tesseract_cli(img_bytes, language)
        if result and result.strip():
            logger.info("OCR succeeded via tesseract CLI")
            return result
    except Exception as e:
        errors.append(f"Tesseract CLI: {e}")
        logger.debug("Tesseract CLI failed: %s", e)

    # ── EasyOCR (GPU accelerated) ────────────────────────────────────
    try:
        result = await _ocr_easyocr(img_bytes, language)
        if result and result.strip():
            logger.info("OCR succeeded via EasyOCR")
            return result
    except ImportError:
        errors.append("EasyOCR: not installed")
    except Exception as e:
        errors.append(f"EasyOCR: {e}")
        logger.debug("EasyOCR failed: %s", e)

    # ── pytesseract Python wrapper ───────────────────────────────────
    try:
        result = await _ocr_pytesseract(img_bytes, language)
        if result and result.strip():
            logger.info("OCR succeeded via pytesseract")
            return result
    except ImportError:
        errors.append("pytesseract: not installed")
    except Exception as e:
        errors.append(f"pytesseract: {e}")
        logger.debug("pytesseract failed: %s", e)

    # ── All engines failed ───────────────────────────────────────────
    install_hint = _get_install_hint()
    logger.warning("No OCR engine available.")
    return f"[OCR unavailable on this system. Tried: {'; '.join(errors)}. To fix: {install_hint}]"


def _get_install_hint() -> str:
    """Return platform-specific install instructions."""
    if CURRENT_PLATFORM == Platform.WINDOWS:
        return (
            "Windows 10/11 native OCR should work automatically. "
            "If it fails, install Tesseract: "
            "winget install UB-Mannheim.TesseractOCR  OR  "
            "pip install easyocr"
        )
    elif CURRENT_PLATFORM == Platform.MACOS:
        return (
            "macOS Vision framework should work automatically (10.15+). "
            "If it fails: brew install tesseract  OR  pip install easyocr"
        )
    else:
        return (
            "Install tesseract: sudo apt install tesseract-ocr  OR  sudo dnf install tesseract  OR  pip install easyocr"
        )


# ──────────────────────────────────────────────────────────────────────
#  Windows native OCR (WinRT)
# ──────────────────────────────────────────────────────────────────────


async def _ocr_windows_native(img_bytes: bytes) -> str:
    """Use Windows 10/11 built-in OCR via PowerShell WinRT APIs.

    Writes the PS script to a temp file to avoid escaping issues.
    """
    tmp_img = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    tmp_ps1 = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.ps1")
    Path(tmp_img).write_bytes(img_bytes)

    # PowerShell script that uses WinRT OCR
    ps_script = f"""
try {{
    Add-Type -AssemblyName System.Runtime.WindowsRuntime

    # Load WinRT types
    [void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
    [void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
    [void][Windows.Storage.StorageFile, Windows.Foundation, ContentType=WindowsRuntime]

    # Helper to await WinRT async operations
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{
        $_.Name -eq 'AsTask' -and
        $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    }})[0]

    Function Await($WinRtTask, $ResultType) {{
        $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
        $netTask = $asTask.Invoke($null, @($WinRtTask))
        $netTask.Wait(-1) | Out-Null
        $netTask.Result
    }}

    # Open image file
    $imgPath = '{tmp_img.replace(chr(92), chr(92) + chr(92))}'
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imgPath)) ([Windows.Storage.StorageFile])

    # Open stream and create decoder
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

    # Create OCR engine and recognize
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    if ($engine -eq $null) {{
        Write-Error "OCR engine could not be created"
        exit 1
    }}
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

    # Output text
    Write-Output $result.Text

    # Cleanup
    $stream.Dispose()
}} catch {{
    Write-Error $_.Exception.Message
    exit 1
}}
"""
    Path(tmp_ps1).write_text(ps_script, encoding="utf-8")

    try:
        code, out, err = await run_command(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", tmp_ps1],
            timeout=15,
        )
        if code == 0 and out.strip():
            return out.strip()
        # If exit code is 0 but no output, might be empty screen
        if code == 0:
            return ""
        raise RuntimeError(f"Windows OCR failed (exit {code}): {err.strip()}")
    finally:
        for f in (tmp_img, tmp_ps1):
            try:
                os.unlink(f)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
#  macOS native OCR (Vision framework)
# ──────────────────────────────────────────────────────────────────────


async def _ocr_macos_native(img_bytes: bytes) -> str:
    """Use macOS Vision framework for OCR (macOS 10.15+)."""
    tmp_img = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    Path(tmp_img).write_bytes(img_bytes)

    # Swift script that uses Vision framework
    swift_script = f"""
import Vision
import AppKit

let imgPath = "{tmp_img}"
guard let image = NSImage(contentsOfFile: imgPath),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {{
    fputs("Failed to load image", stderr)
    exit(1)
}}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([request])

guard let observations = request.results else {{
    exit(0)
}}

for observation in observations {{
    if let candidate = observation.topCandidates(1).first {{
        print(candidate.string)
    }}
}}
"""

    tmp_swift = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.swift")
    Path(tmp_swift).write_text(swift_script)

    try:
        code, out, err = await run_command(
            ["swift", tmp_swift],
            timeout=20,
        )
        if code == 0:
            return out.strip()
        raise RuntimeError(f"macOS Vision OCR failed: {err.strip()}")
    finally:
        for f in (tmp_img, tmp_swift):
            try:
                os.unlink(f)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
#  Tesseract CLI (cross-platform, no Python wrapper)
# ──────────────────────────────────────────────────────────────────────


async def _ocr_tesseract_cli(img_bytes: bytes, language: str) -> str:
    """Use tesseract binary directly without pytesseract wrapper."""
    # Check if tesseract is installed
    tesseract_bin = shutil.which("tesseract")
    if not tesseract_bin:
        raise RuntimeError("tesseract binary not found in PATH")

    tmp_img = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}.png")
    tmp_out = os.path.join(tempfile.gettempdir(), f"pilot_ocr_{os.getpid()}_out")
    Path(tmp_img).write_bytes(img_bytes)

    try:
        code, out, err = await run_command(
            [tesseract_bin, tmp_img, tmp_out, "-l", language],
            timeout=30,
        )
        result_file = Path(f"{tmp_out}.txt")
        if result_file.exists():
            text = result_file.read_text(encoding="utf-8", errors="replace")
            return text.strip()
        if code != 0:
            raise RuntimeError(f"tesseract failed (exit {code}): {err.strip()}")
        return ""
    finally:
        for f in (tmp_img, f"{tmp_out}.txt"):
            try:
                os.unlink(f)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
#  EasyOCR
# ──────────────────────────────────────────────────────────────────────


async def _ocr_easyocr(img_bytes: bytes, language: str) -> str:
    from io import BytesIO

    try:
        import easyocr
    except ImportError:
        logger.info("easyocr not installed, auto-installing cross-platform fallback...")
        code, out, err = await run_command([sys.executable, "-m", "pip", "install", "easyocr"], timeout=300)
        if code != 0:
            raise RuntimeError(f"Failed to auto-install easyocr: {err.strip()}")
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
        cache_key = f"_easyocr_reader_{lang_code}"
        reader = getattr(_ocr_easyocr, cache_key, None)
        if reader is None:
            try:
                reader = easyocr.Reader([lang_code], gpu=True, verbose=False)
            except Exception:
                reader = easyocr.Reader([lang_code], gpu=False, verbose=False)
            setattr(_ocr_easyocr, cache_key, reader)
        img = Image.open(BytesIO(img_bytes))
        results = reader.readtext(np.array(img))
        lines = [text for (_, text, conf) in results if conf > 0.3]
        return "\n".join(lines)

    return await asyncio.to_thread(_do)


# ──────────────────────────────────────────────────────────────────────
#  pytesseract (Python wrapper)
# ──────────────────────────────────────────────────────────────────────


async def _ocr_pytesseract(img_bytes: bytes, language: str) -> str:
    from io import BytesIO

    import pytesseract
    from PIL import Image

    def _do():
        img = Image.open(BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang=language)

    return await asyncio.to_thread(_do)


# ──────────────────────────────────────────────────────────────────────
#  screen_find_text
# ──────────────────────────────────────────────────────────────────────


async def screen_find_text(
    target_text: str,
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Find specific text on screen and return its approximate location."""
    img_bytes = await _capture_screenshot_bytes(region)

    try:
        from io import BytesIO

        import easyocr
        import numpy as np
        from PIL import Image

        def _do():
            cache_key = "_easyocr_reader_en"
            reader = getattr(_ocr_easyocr, cache_key, None)
            if reader is None:
                try:
                    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                except Exception:
                    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                setattr(_ocr_easyocr, cache_key, reader)
            img = Image.open(BytesIO(img_bytes))
            results = reader.readtext(np.array(img))
            matches = []
            for bbox, text, conf in results:
                if target_text.lower() in text.lower():
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

    # Fallback: use OCR and search in text
    try:
        all_text = await screen_ocr(region)
        if target_text.lower() in all_text.lower():
            return json.dumps(
                {
                    "found": True,
                    "text": target_text,
                    "note": "Text found in OCR output but exact coordinates unavailable without easyocr",
                    "context": all_text[:500],
                },
                indent=2,
            )
        return f"Text '{target_text}' not found on screen"
    except Exception as e:
        return f"Text search failed: {e}"


# ──────────────────────────────────────────────────────────────────────
#  screen_analyze (vision LLM)
# ──────────────────────────────────────────────────────────────────────


async def screen_analyze(
    prompt: str = "Describe what you see on the screen",
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Analyze the screen using a vision-capable LLM.

    Priority: Cloud Vision (Gemini/OpenAI/Claude) → Local Ollama → OCR fallback.
    Includes retry logic for transient 503/429 errors.
    """
    img_bytes = await _capture_screenshot_bytes(region)
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    cloud_errors: list[str] = []

    # ── 1. Cloud Vision (Gemini / OpenAI / Claude) ───────────────────
    try:
        from pilot.security.vault import KeyVault

        config = PilotConfig.load()
        if config.model.provider == "cloud" and config.model.cloud_provider:
            vault = KeyVault(config)
            api_key = await vault.get_key(config.model.cloud_provider)
            if api_key:
                provider = config.model.cloud_provider

                if provider == "gemini":
                    result = await _gemini_vision(api_key, b64_image, prompt, config.model.cloud_model)
                    if result is not None:
                        return result
                    cloud_errors.append("Gemini Vision: all models returned errors")

                elif provider == "openai":
                    result = await _openai_vision(api_key, b64_image, prompt, config.model.cloud_model)
                    if result is not None:
                        return result
                    cloud_errors.append("OpenAI Vision: request failed")

                elif provider == "claude":
                    result = await _claude_vision(api_key, b64_image, prompt, config.model.cloud_model)
                    if result is not None:
                        return result
                    cloud_errors.append("Claude Vision: request failed")

                else:
                    cloud_errors.append(f"Unsupported cloud provider for vision: {provider}")
            else:
                cloud_errors.append(f"No API key stored for {config.model.cloud_provider}")
    except Exception as e:
        cloud_errors.append(f"Cloud vision init error: {e}")
        logger.warning("Cloud vision exception: %s", e)

    # ── 2. Local Ollama with vision models ───────────────────────────
    try:
        ollama_url = "http://127.0.0.1:11434"
        try:
            ollama_url = PilotConfig.load().model.ollama_base_url or ollama_url
        except Exception:
            pass

        async with create_httpx_client(PilotConfig.load(), timeout=60) as client:
            for model_name in ["llava:7b", "llava", "bakllava", "moondream"]:
                try:
                    resp = await client.post(
                        f"{ollama_url}/api/generate",
                        json={"model": model_name, "prompt": prompt, "images": [b64_image], "stream": False},
                    )
                    if resp.status_code == 200:
                        return resp.json().get("response", "No response")
                except Exception:
                    continue
    except Exception:
        pass

    # ── 3. Fallback: OCR text dump ───────────────────────────────────
    try:
        ocr_text = await screen_ocr(region)
        if ocr_text and not ocr_text.startswith("[OCR unavailable"):
            return f"[Vision model not available — falling back to OCR]\nScreen text content:\n{ocr_text[:2000]}"
    except Exception:
        pass

    # ── All methods failed — return actionable error ─────────────────
    detail = "; ".join(cloud_errors) if cloud_errors else "No cloud provider configured"
    return (
        f"[Screen analysis unavailable. {detail}. "
        f"Local Ollama vision models not found. OCR also unavailable. "
        f"Please check your API key in Settings or install a local vision model.]"
    )


async def _gemini_vision(api_key: str, b64_image: str, prompt: str, configured_model: str | None = None) -> str | None:
    """Call Gemini Vision API with model fallback and retry on 503/429."""
    models_to_try = []
    primary = configured_model or "gemini-2.5-flash"
    models_to_try.append(primary)
    # Add fallbacks that are different from the primary
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    base_url = "https://generativelanguage.googleapis.com/v1beta"
    try:
        from pilot.models.cloud import PROVIDER_ENDPOINTS

        base_url = PROVIDER_ENDPOINTS.get("gemini", base_url)
    except Exception:
        pass

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": b64_image}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1},
    }

    async with create_httpx_client(PilotConfig.load(), timeout=90) as client:
        for model_name in models_to_try:
            endpoint = f"{base_url}/models/{model_name}:generateContent?key={api_key}"
            # Retry up to 2 times on transient errors (503, 429)
            for attempt in range(3):
                try:
                    resp = await client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                        logger.warning("Gemini returned empty candidates for %s", model_name)
                        break  # Try next model
                    elif resp.status_code in (429, 503):
                        wait = 2.0 * (2**attempt)
                        logger.info(
                            "Gemini %s returned %d, retrying in %.1fs (attempt %d/3)",
                            model_name,
                            resp.status_code,
                            wait,
                            attempt + 1,
                        )
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.warning("Gemini %s failed: %d %s", model_name, resp.status_code, resp.text[:150])
                        break  # Try next model
                except Exception as e:
                    logger.warning("Gemini %s request error: %s", model_name, e)
                    break
    return None


async def _openai_vision(api_key: str, b64_image: str, prompt: str, configured_model: str | None = None) -> str | None:
    """Call OpenAI Vision API."""
    model = configured_model or "gpt-4o"
    endpoint = "https://api.openai.com/v1/chat/completions"
    try:
        from pilot.models.cloud import PROVIDER_ENDPOINTS

        endpoint = PROVIDER_ENDPOINTS.get("openai", endpoint)
    except Exception:
        pass

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
    }

    async with create_httpx_client(PilotConfig.load(), timeout=90) as client:
        for attempt in range(3):
            try:
                resp = await client.post(endpoint, json=payload, headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                elif resp.status_code in (429, 503):
                    wait = 2.0 * (2**attempt)
                    logger.info("OpenAI returned %d, retrying in %.1fs", resp.status_code, wait)
                    await asyncio.sleep(wait)
                    continue
                else:
                    logger.warning("OpenAI Vision failed: %d %s", resp.status_code, resp.text[:150])
                    return None
            except Exception as e:
                logger.warning("OpenAI request error: %s", e)
                return None
    return None


async def _claude_vision(api_key: str, b64_image: str, prompt: str, configured_model: str | None = None) -> str | None:
    """Call Anthropic Claude Vision API."""
    model = configured_model or "claude-3-5-sonnet-20241022"
    endpoint = "https://api.anthropic.com/v1/messages"
    try:
        from pilot.models.cloud import PROVIDER_ENDPOINTS

        endpoint = PROVIDER_ENDPOINTS.get("claude", endpoint)
    except Exception:
        pass

    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    async with create_httpx_client(PilotConfig.load(), timeout=90) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["content"][0]["text"]
                elif resp.status_code in (429, 529):
                    wait = 2.0 * (2**attempt)
                    logger.info("Claude returned %d, retrying in %.1fs", resp.status_code, wait)
                    await asyncio.sleep(wait)
                    continue
                else:
                    logger.warning("Claude Vision failed: %d %s", resp.status_code, resp.text[:150])
                    return None
            except Exception as e:
                logger.warning("Claude request error: %s", e)
                return None
    return None


# ──────────────────────────────────────────────────────────────────────
#  screen_element_map
# ──────────────────────────────────────────────────────────────────────


async def screen_element_map(
    region: tuple[int, int, int, int] | None = None,
) -> str:
    """Create a map of interactive elements on screen."""
    img_bytes = await _capture_screenshot_bytes(region)

    try:
        import sys
        from io import BytesIO

        try:
            import easyocr
        except ImportError:
            logger.info("easyocr not installed, auto-installing cross-platform fallback for element mapping...")
            code, out, err = await run_command([sys.executable, "-m", "pip", "install", "easyocr"], timeout=300)
            if code != 0:
                return (
                    f"Install easyocr for element detection: pip install easyocr (Auto-install failed: {err.strip()})"
                )
            import easyocr

        import numpy as np
        from PIL import Image

        def _do():
            cache_key = "_easyocr_reader_en"
            reader = getattr(_ocr_easyocr, cache_key, None)
            if reader is None:
                try:
                    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                except Exception:
                    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                setattr(_ocr_easyocr, cache_key, reader)
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


# ──────────────────────────────────────────────────────────────────────
#  screen_detect_elements — VLM zero-shot element detection
# ──────────────────────────────────────────────────────────────────────

# Structured prompt sent to the VLM.  The model is asked to return ONLY
# a JSON array so we can parse it reliably without post-processing prose.
_DETECT_ELEMENTS_PROMPT = """\
You are a UI element detector. Analyse the screenshot and return a JSON array \
of every interactive element you can see (buttons, links, input fields, \
checkboxes, dropdowns, icons, etc.).

For EACH element output exactly this object:
{
  "label": "<short human-readable name>",
  "type": "<button|input|link|checkbox|dropdown|icon|text|other>",
  "action": "<click|type|select|scroll|other>",
  "bbox": [x, y, width, height],
  "confidence": <0.0-1.0>
}

Rules:
- bbox values are pixel coordinates relative to the top-left of the image.
- Return ONLY the JSON array, no markdown fences, no explanation.
- If you cannot detect any elements return an empty array: []
"""

_DETECT_ELEMENTS_FILTER_PROMPT = """\
You are a UI element detector. Analyse the screenshot and find the element \
that best matches this description: "{description}".

Return a JSON array containing ONLY the matching element(s) in this format:
{{
  "label": "<short human-readable name>",
  "type": "<button|input|link|checkbox|dropdown|icon|text|other>",
  "action": "<click|type|select|scroll|other>",
  "bbox": [x, y, width, height],
  "confidence": <0.0-1.0>
}}

Rules:
- bbox values are pixel coordinates relative to the top-left of the image.
- Return ONLY the JSON array, no markdown fences, no explanation.
- If nothing matches return an empty array: []
"""


async def screen_detect_elements(
    description: str = "",
    region: tuple[int, int, int, int] | None = None,
    max_elements: int = 20,
    action_filter: str = "",
) -> str:
    """Detect interactive UI elements via zero-shot VLM inference.

    Sends a screenshot to the configured vision-capable LLM (Gemini,
    GPT-4o, Claude, or local LLaVA/Ollama) with a structured prompt that
    asks it to return bounding-box coordinates for every interactive element
    it can see — no DOM parsing, no accessibility APIs required.

    Parameters
    ----------
    description:
        Optional natural-language filter, e.g. ``"the login button"``.
        When provided, the VLM is asked to return only matching elements.
    region:
        ``(left, top, width, height)`` crop, or ``None`` for full screen.
    max_elements:
        Maximum number of elements to return (applied after VLM response).
    action_filter:
        ``"click"``, ``"type"``, or ``""`` (all actions).

    Returns
    -------
    str
        JSON string with keys ``elements``, ``count``, ``source``, and
        ``note``.  Each element has ``label``, ``type``, ``action``,
        ``bbox`` ``[x, y, w, h]``, and ``confidence``.
    """
    img_bytes = await _capture_screenshot_bytes(region)
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        _DETECT_ELEMENTS_FILTER_PROMPT.format(description=description)
        if description.strip()
        else _DETECT_ELEMENTS_PROMPT
    )

    raw_response: str | None = None
    source = "unknown"

    # ── 1. Cloud VLM (Gemini / OpenAI / Claude) ──────────────────────
    try:
        from pilot.config import PilotConfig
        from pilot.security.vault import KeyVault

        config = PilotConfig.load()
        if config.model.provider == "cloud" and config.model.cloud_provider:
            vault = KeyVault(config)
            api_key = await vault.get_key(config.model.cloud_provider)
            if api_key:
                provider = config.model.cloud_provider
                if provider == "gemini":
                    raw_response = await _gemini_vision(api_key, b64_image, prompt, config.model.cloud_model)
                    source = "gemini"
                elif provider == "openai":
                    raw_response = await _openai_vision(api_key, b64_image, prompt, config.model.cloud_model)
                    source = "openai"
                elif provider == "claude":
                    raw_response = await _claude_vision(api_key, b64_image, prompt, config.model.cloud_model)
                    source = "claude"
    except Exception as exc:
        logger.warning("screen_detect_elements: cloud VLM error: %s", exc)

    # ── 2. Local Ollama vision model ──────────────────────────────────
    if raw_response is None:
        try:
            ollama_url = "http://127.0.0.1:11434"
            try:
                ollama_url = PilotConfig.load().model.ollama_base_url or ollama_url
            except Exception:
                pass

            async with create_httpx_client(PilotConfig.load(), timeout=90) as client:
                for model_name in ["llava:7b", "llava", "bakllava", "moondream"]:
                    try:
                        resp = await client.post(
                            f"{ollama_url}/api/generate",
                            json={
                                "model": model_name,
                                "prompt": prompt,
                                "images": [b64_image],
                                "stream": False,
                            },
                        )
                        if resp.status_code == 200:
                            raw_response = resp.json().get("response", "")
                            source = f"ollama/{model_name}"
                            break
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning("screen_detect_elements: Ollama error: %s", exc)

    # ── 3. Parse VLM response ─────────────────────────────────────────
    if raw_response:
        elements = _parse_element_response(raw_response)
        if elements is not None:
            # Apply filters
            if action_filter:
                elements = [e for e in elements if e.get("action", "") == action_filter]
            elements = elements[:max_elements]
            return json.dumps(
                {
                    "elements": elements,
                    "count": len(elements),
                    "source": source,
                    "note": "Use mouse_click with bbox center coordinates to interact",
                },
                indent=2,
            )
        logger.warning("screen_detect_elements: VLM returned unparseable response, falling back to EasyOCR")

    # ── 4. Fallback: EasyOCR element map ─────────────────────────────
    logger.info("screen_detect_elements: falling back to EasyOCR element map")
    ocr_result = await screen_element_map(region)
    try:
        ocr_data = json.loads(ocr_result)
        # Normalise EasyOCR output to the same schema
        normalised = []
        for el in ocr_data.get("elements", []):
            cx = el.get("center", {}).get("x", 0)
            cy = el.get("center", {}).get("y", 0)
            w = el.get("size", {}).get("w", 0)
            h = el.get("size", {}).get("h", 0)
            normalised.append(
                {
                    "label": el.get("text", ""),
                    "type": el.get("type", "text"),
                    "action": "click" if el.get("type") in ("button", "link") else "type",
                    "bbox": [cx - w // 2, cy - h // 2, w, h],
                    "confidence": el.get("confidence", 0.5),
                }
            )
        if action_filter:
            normalised = [e for e in normalised if e.get("action") == action_filter]
        normalised = normalised[:max_elements]
        return json.dumps(
            {
                "elements": normalised,
                "count": len(normalised),
                "source": "easyocr_fallback",
                "note": "VLM unavailable — coordinates from OCR (less accurate)",
            },
            indent=2,
        )
    except Exception:
        return ocr_result


def _parse_element_response(raw: str) -> list[dict] | None:
    """Extract and validate a JSON element array from a VLM response.

    The VLM is instructed to return only JSON, but may wrap it in markdown
    fences, add conversational prefixes, or include square brackets in prose
    (e.g. ``"Checking [window title]: [...]"``).  A simple ``find("[")``
    would match those incidental brackets and produce invalid JSON.

    This function uses a regex to find the first ``[`` that is immediately
    followed by ``{`` or whitespace+``{`` — i.e. the start of a JSON array
    of objects — which is far more robust against conversational noise.

    Returns ``None`` if the response cannot be parsed into a valid list.
    """
    import re as _re

    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(inner).strip()

    # Find the first '[' that starts a JSON array of objects: '[' followed
    # (with optional whitespace) by '{' or ']' (empty array).
    # This skips incidental brackets like "[window title]" in prose.
    array_start = _re.search(r"\[\s*(\{|\])", text)
    if array_start is None:
        return None

    start = array_start.start()
    # Find the matching closing ']' by scanning from the end
    end = text.rfind("]", start)
    if end == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    validated: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bbox = [int(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        validated.append(
            {
                "label": str(item.get("label", "")),
                "type": str(item.get("type", "other")),
                "action": str(item.get("action", "click")),
                "bbox": bbox,
                "confidence": float(item.get("confidence", 0.5)),
            }
        )

    return validated
