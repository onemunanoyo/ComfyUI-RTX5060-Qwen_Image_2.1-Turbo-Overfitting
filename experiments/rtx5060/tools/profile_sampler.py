import asyncio
import cProfile
import faulthandler
import json
import os
import pstats
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[3]
faulthandler.dump_traceback_later(90, repeat=True)
sys.path.insert(0, str(root))
os.environ["COMFY_GGUF_FUSED_DEQUANT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRITON_CACHE_DIR"] = str(root / ".uv-cache/triton")
sys.argv = ["profile_sampler", "--offline"] + sys.argv[1:]

import main  # noqa: F401 -- initializes ComfyUI paths and device policy
import nodes
import torch
import comfy.model_management
from comfy_extras.nodes_qwen import TextEncodeQwenImage21
from comfy_extras.nodes_custom_sampler import SamplerCustom, KSamplerSelect, ManualSigmas

asyncio.run(nodes.load_custom_node(str(root / "custom_nodes/ComfyUI-GGUF")))
clip = nodes.CLIPLoader().load_clip("qwen3vl_8b_int8_convrot.safetensors", "qwen_image")[0]
prompt = "A watercolor illustration of a small wooden cabin beside a mountain lake, pine trees, morning sunlight, detailed landscape."
positive, negative, latent = TextEncodeQwenImage21.execute(clip, prompt, "", resolution=1024).result
model = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet("Qwen-Image-2.1-Turbo-AD-Q6_K.gguf")[0]
sampler = KSamplerSelect.execute("euler")[0]
sigmas = ManualSigmas.execute("1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0")[0]

def sample(seed):
    return SamplerCustom.execute(model, True, seed, 1.0, positive, negative, sampler, sigmas, latent)

sample(12345)
torch.cuda.synchronize()
if os.environ.get("SAMPLER_COMPARE") in ("1", "pinned"):
    tensor_class = type(model.model.diffusion_model.transformer_blocks[0].attn.to_q.weight)
    original_to = tensor_class.to

    def async_to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if self.device.type == "cpu" and isinstance(device, torch.device) and device.type == "cuda":
            kwargs.setdefault("non_blocking", True)
        return original_to(self, *args, **kwargs)

    results = []
    reference = None
    pin_trial = os.environ.get("SAMPLER_COMPARE") == "pinned"
    modes = ["baseline", "pinned_async", "pinned_async"] if pin_trial else ["baseline", "async", "baseline", "async"]
    for mode in modes:
        if mode == "pinned_async":
            comfy.model_management.PINNING_ALLOWED_TYPES.add(tensor_class.__name__)
            for name, parameter in model.model.named_parameters():
                if parameter.device.type == "cpu" and name not in model.pinned:
                    model.pin_weight_to_device(name)
            print("Pinned weight keys:", len(model.pinned), flush=True)
        tensor_class.to = original_to if mode == "baseline" else async_to
        start = time.perf_counter()
        output = sample(12346)[0]["samples"]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = output.clone()
        result = {"mode": mode, "seconds": elapsed, "equal": torch.equal(output, reference), "max_abs": (output - reference).abs().max().item()}
        results.append(result)
        print(json.dumps(result), flush=True)
        filename = "sampler-pinned-results.json" if pin_trial else "sampler-async-results.json"
        Path("experiments/rtx5060/results", filename).write_text(json.dumps(results, indent=2))
    tensor_class.to = original_to
    faulthandler.cancel_dump_traceback_later()
    comfy.model_management.unload_all_models()
    sys.exit(0)
placement = []
for name, module in model.model.diffusion_model.named_modules():
    weight = getattr(module, "weight", None)
    if weight is not None:
        placement.append({"name": name, "shape": list(weight.shape), "bytes": weight.nelement() * weight.element_size(), "device": str(weight.device), "pinned": weight.is_pinned(), "type": str(getattr(weight, "tensor_type", ""))})
Path("experiments/rtx5060/profiles/qwen-placement.json").write_text(json.dumps(placement, indent=2))
cpu_profile = cProfile.Profile()
start = time.perf_counter()
cpu_profile.runcall(sample, 12346)
torch.cuda.synchronize()
print("CPU profile wall:", time.perf_counter() - start, flush=True)
with Path("experiments/rtx5060/profiles/sampler-cpu.txt").open("w") as output:
    pstats.Stats(cpu_profile, stream=output).sort_stats("cumulative").print_stats(65)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
    sample(12347)
    torch.cuda.synchronize()
Path("experiments/rtx5060/profiles/sampler-gpu.txt").write_text(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=55))
profile.export_chrome_trace("experiments/rtx5060/profiles/sampler-trace.json")
print("Profiler complete", flush=True)
faulthandler.cancel_dump_traceback_later()
comfy.model_management.unload_all_models()
