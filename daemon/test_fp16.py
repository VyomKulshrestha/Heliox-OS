import os

import torch
from transformers import AutoModelForCausalLM

print("Testing fp16 load without device_map=auto")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-3B", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    print("Moving to cuda...")
    model.to("cuda")
    print("Success! Memory used:", torch.cuda.memory_allocated() / 1e9, "GB")
except Exception as e:
    import traceback

    traceback.print_exc()
