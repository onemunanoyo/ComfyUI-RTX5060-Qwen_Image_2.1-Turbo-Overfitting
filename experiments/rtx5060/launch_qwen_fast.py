import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import main
import nodes

loop, prompt_server, start = main.start_comfyui()
gguf_ops = sys.modules[nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"].__module__]
# Generic activation fusion cannot interpret GGMLTensor's packed weight layout.
gguf_ops.GGMLOps.Linear.comfy_force_forward = True
try:
    loop.run_until_complete(start())
except KeyboardInterrupt:
    pass
finally:
    prompt_server.asset_manager.shutdown()
