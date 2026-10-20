import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import aiohttp
import psutil


def workflow(model, seed, prompt, steps, sigmas=None):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-turbo-v1.1.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "chroma", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"benchmark/{model}"}},
    }
    if model == "qwen":
        graph["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Qwen-Image-2.1-Turbo-AD-Q6_K.gguf"}}
        graph["2"]["inputs"].update(clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image")
        graph["3"]["inputs"]["vae_name"] = "qwen_image_2.1_vae_bf16.safetensors"
        graph["4"] = {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}}
        del graph["5"]
        del graph["6"]
        graph["7"]["inputs"].update(negative=["4", 1], latent_image=["4", 2])
    if sigmas is not None:
        inputs = graph["7"]["inputs"]
        graph["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}}
        graph["11"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        graph["7"] = {"class_type": "SamplerCustom", "inputs": {"model": inputs["model"], "add_noise": True, "noise_seed": seed, "cfg": inputs["cfg"], "positive": inputs["positive"], "negative": inputs["negative"], "latent_image": inputs["latent_image"], "sampler": ["11", 0], "sigmas": ["10", 0]}}
    return graph


async def run(args):
    base = f"http://127.0.0.1:{args.port}"
    results = []
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index in range(args.runs):
            prompt = f"{args.prompt} The cabin has {index + 1} windows." if args.vary_prompt else args.prompt
            graph = workflow(args.model, args.seed + index, prompt, args.steps, args.sigmas)
            client_id = str(uuid.uuid4())
            async with session.ws_connect(f"{base}/ws?clientId={client_id}") as ws:
                start = time.perf_counter()
                async with session.post(f"{base}/prompt", json={"prompt": graph, "client_id": client_id}) as response:
                    submitted = await response.json()
                    if response.status != 200:
                        raise RuntimeError(submitted)
                prompt_id = submitted["prompt_id"]
                events = []
                async with asyncio.timeout(900):
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        data = event["data"]
                        if data.get("prompt_id") != prompt_id:
                            continue
                        kind = event["type"]
                        elapsed = time.perf_counter() - start
                        if kind in ("executing", "execution_cached", "execution_error", "execution_success"):
                            events.append({"seconds": elapsed, "ram_available": psutil.virtual_memory().available, **event})
                        if kind == "execution_error":
                            raise RuntimeError(data)
                        if kind == "executing" and data.get("node") is None:
                            break
                async with session.get(f"{base}/history/{prompt_id}") as response:
                    history = (await response.json())[prompt_id]
                result = {"model": args.model, "run": index, "seed": args.seed + index, "steps": args.steps,
                          "wall_seconds": time.perf_counter() - start, "events": events, "history": history, "workflow": graph}
                results.append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(json.dumps({"model": args.model, "run": index, "wall_seconds": result["wall_seconds"], "events": events}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["anima", "qwen"], required=True)
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sigmas", help="Explicit sigma list, including the final zero; overrides --steps")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--vary-prompt", action="store_true")
    parser.add_argument("--prompt", default="A watercolor illustration of a small wooden cabin beside a mountain lake, pine trees, morning sunlight, detailed landscape.")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
