import argparse
import cProfile
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
options, comfy_args = parser.parse_known_args()
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
sys.argv = ["profile_encoder_loading", "--offline", *comfy_args]

import main  # noqa: E402, F401 -- initializes ComfyUI paths and device policy
import nodes  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import comfy.memory_management  # noqa: E402

process = psutil.Process()
results = {"args": comfy_args, "dynamic_vram": comfy.memory_management.aimdo_enabled, "os_cache": "not controlled", "phases": []}


def measure(name, operation):
    profile = cProfile.Profile()
    io_before = process.io_counters()
    start = time.perf_counter()
    value = profile.runcall(operation)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    io_after = process.io_counters()
    stats = profile.getstats()
    reads = [entry for entry in stats if hasattr(entry.code, "co_name") and entry.code.co_name == "read_file_slice"]
    result = {
        "name": name,
        "seconds": elapsed,
        "read_file_slice_seconds": sum(entry.totaltime for entry in reads),
        "read_file_slice_calls": sum(entry.callcount for entry in reads),
        "process_read_bytes": io_after.read_bytes - io_before.read_bytes,
        "ram_available_bytes": psutil.virtual_memory().available,
    }
    results["phases"].append(result)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return value


clip = measure("load_encoder", lambda: nodes.CLIPLoader().load_clip("qwen3vl_8b_int8_convrot.safetensors", "qwen_image")[0])
for index, prompt in enumerate(("A watercolor cabin beside a lake.", "A watercolor cabin beside a lake with pine trees.")):
    tokens = clip.tokenize(prompt, keep_vision=True, prevent_empty_text=True)
    conditioning = measure(f"encode_{index}", lambda: clip.encode_from_tokens_scheduled(tokens))
    torch.save(conditioning[0][0].cpu(), options.output.with_suffix(f".encode-{index}.pt"))
