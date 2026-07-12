import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print("Testing 8-bit load directly")
try:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-3B", device_map="auto", quantization_config=quant_config
    )
    print("Success! Memory used:", torch.cuda.memory_allocated() / 1e9, "GB")
except Exception as e:
    import traceback

    traceback.print_exc()
