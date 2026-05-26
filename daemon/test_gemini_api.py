"""Minimal test: hit Gemini Vision API directly, no pyautogui."""

import asyncio
import base64
import struct
import zlib

import httpx

API_KEY = "AIzaSyBxPVeNINvLsMKLkUqXoZpHFmwUNs-FOwg"

# Create a tiny 1x1 red PNG (valid image)


def make_tiny_png():
    raw = b"\x00\xff\x00\x00\xff"  # filter byte + RGBA

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


img_bytes = make_tiny_png()
b64_image = base64.b64encode(img_bytes).decode()


async def test_models():
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro-vision",
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        # First: list available models
        print("=== Listing available vision models ===")
        resp = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}")
        if resp.status_code == 200:
            data = resp.json()
            for m in data.get("models", []):
                name = m.get("name", "")
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    print(f"  {name} -> {methods}")
        else:
            print(f"  List models failed: {resp.status_code} {resp.text[:200]}")

        print()

        # Now try each model with vision
        for model in models_to_try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            payload = {
                "contents": [
                    {"parts": [{"text": "Say hello"}, {"inlineData": {"mimeType": "image/png", "data": b64_image}}]}
                ],
                "generationConfig": {"temperature": 0.1},
            }

            resp = await client.post(endpoint, json=payload)
            status = resp.status_code
            if status == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✅ {model}: {status} -> {text[:80]}")
            else:
                print(f"❌ {model}: {status} -> {resp.text[:120]}")


asyncio.run(test_models())
