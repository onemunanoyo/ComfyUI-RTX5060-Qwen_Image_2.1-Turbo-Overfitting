import json
from pathlib import Path

import torch

torch.manual_seed(12345)
results = []
for n, k in [(4096, 4096), (24576, 4096), (4096, 12288)]:
    x = torch.randn(4096, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for dtype, fast in [(torch.bfloat16, False), (torch.float16, False), (torch.float16, True)]:
        inp, weight = x.to(dtype), w.to(dtype)
        torch.backends.cuda.matmul.allow_fp16_accumulation = fast
        for _ in range(5):
            output = torch.nn.functional.linear(inp, weight)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            output = torch.nn.functional.linear(inp, weight)
        end.record()
        end.synchronize()
        result = {"n": n, "k": k, "dtype": str(dtype), "fp16_accumulation": fast, "ms": start.elapsed_time(end) / 20, "finite": torch.isfinite(output).all().item()}
        results.append(result)
        print(json.dumps(result), flush=True)
        Path("experiments/rtx5060/results/accumulation.json").write_text(json.dumps(results, indent=2))
    del x, w, inp, weight, output
    torch.cuda.empty_cache()
