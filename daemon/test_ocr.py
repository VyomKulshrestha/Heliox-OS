"""Quick test of Windows native OCR."""

import asyncio
import sys

sys.path.insert(0, ".")

from pilot.system.vision import screen_ocr


async def test():
    print("Testing universal screen OCR...")
    try:
        txt = await screen_ocr()
        print(f"SUCCESS: got {len(txt)} chars")
        print("--- First 300 chars ---")
        print(txt[:300])
    except Exception as e:
        print(f"FAILED: {e}")


asyncio.run(test())
