import os

from neuralset.extractors.huggingface import HuggingFaceText

ext = HuggingFaceText(model_name="unsloth/Llama-3.2-3B", device="cpu")
print("Loading model...")
try:
    m = ext.model
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
