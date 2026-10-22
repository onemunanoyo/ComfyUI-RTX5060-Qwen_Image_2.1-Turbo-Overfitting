import argparse
import asyncio
import cProfile
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

started = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--read-ahead", action="store_true")
options, comfy_args = parser.parse_known_args()
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
os.environ["COMFY_GGUF_FUSED_DEQUANT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRITON_CACHE_DIR"] = str(root / ".uv-cache/triton")
sys.argv = ["profile_qwen_startup", "--offline", "--fp16-unet", "--use-ck-attention", "--fast", "fp16_accumulation", *comfy_args]

import main  # noqa: E402, F401 -- initializes ComfyUI paths and device policy
import nodes  # noqa: E402
import folder_paths  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import comfy.model_management  # noqa: E402
from comfy_extras.nodes_qwen import TextEncodeQwenImage21  # noqa: E402
from comfy_extras.nodes_custom_sampler import SamplerCustom, KSamplerSelect, ManualSigmas  # noqa: E402

asyncio.run(nodes.load_custom_node(str(root / "custom_nodes/ComfyUI-GGUF")))
gguf_nodes = sys.modules[nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"].__module__]
gguf_nodes.GGMLOps.Linear.comfy_force_forward = True
process = psutil.Process()
results = {"args": comfy_args, "read_ahead": options.read_ahead, "os_cache": "not controlled", "import_seconds": time.perf_counter() - started, "phases": []}


def measure(name, operation):
    profile = cProfile.Profile()
    before = process.io_counters()
    start = time.perf_counter()
    value = profile.runcall(operation)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    after = process.io_counters()
    entries = sorted(profile.getstats(), key=lambda entry: entry.totaltime, reverse=True)
    result = {
        "name": name, "seconds": elapsed,
        "process_read_bytes": after.read_bytes - before.read_bytes,
        "ram_available_bytes": psutil.virtual_memory().available,
        "top_calls": [{"function": str(entry.code), "seconds": entry.totaltime, "calls": entry.callcount} for entry in entries[:20]],
    }
    results["phases"].append(result)
    results["wall_seconds"] = time.perf_counter() - started
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    profile.dump_stats(str(options.output.with_suffix(f".{name}.prof")))
    print(json.dumps({k: v for k, v in result.items() if k != "top_calls"}), flush=True)
    return value


def read_models():
    start = time.perf_counter()
    buffer = bytearray(8 * 1024 * 1024)
    count = 0
    for category, filename in (
        ("text_encoders", "qwen3vl_8b_int8_convrot.safetensors"),
        ("diffusion_models", "Qwen-Image-2.1-Turbo-AD-Q6_K.gguf"),
        ("vae", "qwen_image_2.1_vae_bf16.safetensors"),
    ):
        with open(folder_paths.get_full_path_or_raise(category, filename), "rb", buffering=0) as source:
            while size := source.readinto(buffer):
                count += size
    return {"seconds": time.perf_counter() - start, "bytes": count}


executor = ThreadPoolExecutor(max_workers=1) if options.read_ahead else None
prefetch = executor.submit(read_models) if executor is not None else None
try:
    vae = measure("load_vae", lambda: nodes.VAELoader().load_vae("qwen_image_2.1_vae_bf16.safetensors")[0])
    clip = measure("load_encoder", lambda: nodes.CLIPLoader().load_clip("qwen3vl_8b_int8_convrot.safetensors", "qwen_image")[0])
    prompt = "A watercolor illustration of a small wooden cabin beside a mountain lake, pine trees, morning sunlight, detailed landscape."
    positive, negative, latent = measure("encode_0", lambda: TextEncodeQwenImage21.execute(clip, prompt, "", resolution=1024).result)
    model = measure("load_gguf", lambda: nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet("Qwen-Image-2.1-Turbo-AD-Q6_K.gguf")[0])
    sampler = KSamplerSelect.execute("euler")[0]
    sigmas = ManualSigmas.execute("1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0")[0]
    for index in range(3):
        if index == 2:
            positive, negative, latent = measure("encode_changed", lambda: TextEncodeQwenImage21.execute(clip, prompt + " Soft clouds in the sky.", "", resolution=1024).result)
        sampled = measure(f"sample_{index}", lambda: SamplerCustom.execute(model, True, 12345 + index, 1.0, positive, negative, sampler, sigmas, latent)[0])
        image = measure(f"decode_{index}", lambda: vae.decode(sampled["samples"]))
        torch.save({"latent": sampled["samples"].cpu(), "image": image.cpu()}, options.output.with_suffix(f".run-{index}.pt"))
    if prefetch is not None:
        results["prefetch"] = prefetch.result()
    options.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
finally:
    if executor is not None:
        executor.shutdown(wait=True)
    comfy.model_management.unload_all_models()
