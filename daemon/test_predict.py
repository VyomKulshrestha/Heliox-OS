import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from pilot.cognitive.tribe_engine import TribeEngine

    e = TribeEngine.get_instance()
    print("Loading model...")
    success = await e.load_model()
    print(f"Model loaded: {success}")

    print("Predicting cognitive state...")
    try:
        # We wrap with a timeout to see if it hangs
        snapshot = await asyncio.wait_for(e.predict_cognitive_state("hello world testing"), timeout=60)
        print("Prediction SUCCESS:", snapshot)
    except asyncio.TimeoutError:
        print("Prediction TIMEOUT - it hung!")
    except Exception as exc:
        print("Prediction ERROR:", exc)


if __name__ == "__main__":
    asyncio.run(main())
