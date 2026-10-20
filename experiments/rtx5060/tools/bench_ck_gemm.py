import json
from pathlib import Path

import torch
import comfy_kitchen as ck

torch.manual_seed(12345)
torch.backends.cuda.matmul.allow_fp16_accumulation = True
results = []
for n, k in [(4096, 4096), (24576, 4096), (4096, 12288)]:
    x = torch.randn(4096, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    reference = torch.nn.functional.linear(x, w)
    for name, operation in [("torch", torch.nn.functional.linear), ("kitchen", ck.fp16_linear)]:
        for _ in range(5):
            output = operation(x, w)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            output = operation(x, w)
        end.record()
        end.synchronize()
        result = {"n": n, "k": k, "backend": name, "ms": start.elapsed_time(end) / 20, "equal": torch.equal(output, reference), "max_abs": (output - reference).abs().max().item(), "finite": torch.isfinite(output).all().item()}
        results.append(result)
        print(json.dumps(result), flush=True)
        Path("experiments/rtx5060/results/kitchen-fp16-gemm.json").write_text(json.dumps(results, indent=2))
    del x, w, reference, output
    torch.cuda.empty_cache()
