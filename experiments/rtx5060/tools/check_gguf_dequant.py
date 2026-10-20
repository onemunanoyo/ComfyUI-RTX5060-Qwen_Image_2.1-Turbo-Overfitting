import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "custom_nodes" / "ComfyUI-GGUF"))
os.environ["COMFY_GGUF_FUSED_DEQUANT"] = "0"
import dequant
from dequant_triton import dequantize_cuda


def measure(function):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    reader = gguf.GGUFReader(args.model)
    checked = 0
    benchmarks = {}
    for tensor in reader.tensors:
        kind = tensor.tensor_type
        if kind not in (gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q8_0):
            continue
        block_size, byte_size = gguf.GGML_QUANT_SIZES[kind]
        blocks = tensor.data.reshape(-1, byte_size)
        for start in (0, max(0, len(blocks) // 2 - 32), max(0, len(blocks) - 64)):
            data = torch.from_numpy(np.array(blocks[start:start + 64], copy=True)).cuda()
            shape = (data.numel() // byte_size * block_size,)
            expected = dequant.dequantize(data, kind, shape)
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                actual = dequantize_cuda(data, shape, dtype, 6 if kind == gguf.GGMLQuantizationType.Q6_K else 8)
                torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
                checked += 1
        if kind.name not in benchmarks:
            for count in (1, 3):
                data = torch.from_numpy(np.array(blocks[:count], copy=True)).cuda()
                shape = (len(data) * block_size,)
                expected = dequant.dequantize(data, kind, shape).to(torch.bfloat16)
                actual = dequantize_cuda(data, shape, torch.bfloat16, 6 if kind == gguf.GGMLQuantizationType.Q6_K else 8)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                checked += 1
            limit = min(len(blocks), 4 * 1024 * 1024 // block_size)
            data = torch.from_numpy(np.array(blocks[:limit], copy=True)).cuda()
            shape = (limit * block_size,)
            eager = lambda: dequant.dequantize(data, kind, shape).to(torch.bfloat16)
            fused = lambda: dequantize_cuda(data, shape, torch.bfloat16, 6 if kind == gguf.GGMLQuantizationType.Q6_K else 8)
            torch.testing.assert_close(fused(), eager(), rtol=0, atol=0)
            benchmarks[kind.name] = {"elements": shape[0], "eager_ms": measure(eager), "fused_ms": measure(fused)}
    result = {"exact_comparisons": checked, "benchmarks": benchmarks}
    Path("experiments/rtx5060/results/dequant-check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
