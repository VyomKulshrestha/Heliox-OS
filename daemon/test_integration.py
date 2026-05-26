import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
import sys

sys.path.insert(0, r"c:\Users\marcu\Videos\cursor-os\pilot\daemon")


async def main():
    print("Initializing test modules individually...")
    from pilot.actions import Action, ActionType
    from pilot.agents.executor import Executor
    from pilot.cognitive.attention_scorer import AttentionAwareUI
    from pilot.cognitive.intent_predictor import IntentPredictor
    from pilot.cognitive.stress_gate import StressGate
    from pilot.cognitive.tribe_engine import TribeEngine
    from pilot.config import PilotConfig
    from pilot.multimodal.fusion import InputEvent, ModalityType, MultimodalFusionEngine
    from pilot.server import PilotServer

    config = PilotConfig.load()
    server = PilotServer(config)

    tribe = TribeEngine.get_instance()
    await tribe.load_model()

    server._attention_ui = AttentionAwareUI(tribe)
    server._stress_gate = StressGate(tribe)
    server._intent_predictor = IntentPredictor(tribe)

    server._fusion = MultimodalFusionEngine()
    server._fusion._intent_predictor = server._intent_predictor

    server._executor = Executor(config, None, None, None)
    server._executor._stress_gate = server._stress_gate

    print("\n--- TEST: Broadcast Notification (AttentionAwareUI) ---")
    mock_clients = []

    class MockClient:
        async def send(self, msg):
            mock_clients.append(msg)

    server._clients.add(MockClient())
    await server._broadcast_notification("status", {"message": "testing pipeline"})
    print("Broadcast output:")
    for msg in mock_clients:
        if isinstance(msg, str):
            import json

            print(" ", json.loads(msg))
        else:
            print(" ", msg)

    print("\n--- TEST: Intent Prediction (FusionEngine) ---")
    voice_event = InputEvent(
        modality=ModalityType.VOICE, transcript="take a screenshot", voice_confidence=0.9, is_final=True
    )
    gesture_event = InputEvent(modality=ModalityType.GESTURE, gesture_name="finger_gun", gesture_confidence=0.85)
    await server._fusion.on_gesture_event(gesture_event)
    intent = await server._fusion.on_voice_event(voice_event)
    print("Fused intent:", intent.command if intent else None)
    if intent:
        print("Cognitive metadata attached inside Fusion:", intent.metadata)

    print("\n--- TEST: Stress Gating (Executor) ---")
    action = Action(action_type=ActionType.FILE_DELETE, parameters={"path": "critical.sys"})
    result = await server._executor._execute_single(action, "12345")
    print("Action Result:", result.success, "| Error:", result.error)


if __name__ == "__main__":
    asyncio.run(main())
