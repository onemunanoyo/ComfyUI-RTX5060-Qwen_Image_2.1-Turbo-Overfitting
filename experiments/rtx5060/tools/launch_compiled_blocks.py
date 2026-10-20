import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
os.environ["COMFY_GGUF_FUSED_DEQUANT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRITON_CACHE_DIR"] = str(root / ".uv-cache/triton")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / ".uv-cache/inductor-qwenblocks")

import main
import nodes
import torch
import torch._inductor.utils
from comfy.ldm.qwen_image21.model import QwenImage21TransformerBlock

torch._inductor.utils.is_big_gpu = lambda index_or_device=0: True
compiled_forward = torch.compile(
    QwenImage21TransformerBlock.forward,
    mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False,
)
QwenImage21TransformerBlock.forward = torch._dynamo.config.patch(
    recompile_limit=int(os.environ.get("BLOCK_RECOMPILE_LIMIT", "8")),
)(compiled_forward)

loop, prompt_server, start = main.start_comfyui()
if main.args.fp16_unet:
    gguf_ops = sys.modules[nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"].__module__]
    gguf_ops.GGMLOps.Linear.comfy_force_forward = True
try:
    loop.run_until_complete(start())
except KeyboardInterrupt:
    pass
finally:
    prompt_server.asset_manager.shutdown()
