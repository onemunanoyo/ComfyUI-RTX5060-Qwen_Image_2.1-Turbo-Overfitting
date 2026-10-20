import json
import os
import time
from pathlib import Path

os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(__file__).resolve().parents[3] / ".uv-cache/inductor-matrix-suite")

import torch
import torch._inductor.utils

torch._inductor.utils.is_big_gpu = lambda index_or_device=0: True
torch.manual_seed(12345)
compiled = torch.compile(torch.nn.functional.linear, mode="max-autotune-no-cudagraphs", fullgraph=True)
results = []
for n, k in [(4096, 4096), (24576, 4096), (4096, 12288)]:
    x = torch.randn(4096, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    wt = w.T.contiguous()
    reference = torch.nn.functional.linear(x, w)
    operations = {
        "linear": lambda x=x, w=w: torch.nn.functional.linear(x, w),
        "swapped": lambda x=x, w=w: (w @ x.T).T,
        "weight_transposed": lambda x=x, wt=wt: x @ wt,
        "transpose_each_call": lambda x=x, w=w: x @ w.T.contiguous(),
        "compiled": lambda x=x, w=w: compiled(x, w),
    }
    for name, operation in operations.items():
        setup_start = time.perf_counter()
        for _ in range(5):
            output = operation()
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_start
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            output = operation()
        end.record()
        end.synchronize()
        result = {"m": 4096, "n": n, "k": k, "variant": name, "setup_seconds": setup_seconds, "ms": start.elapsed_time(end) / 20, "equal": torch.equal(output, reference), "max_abs": (output - reference).abs().max().item()}
        results.append(result)
        print(json.dumps(result), flush=True)
        Path("experiments/rtx5060/results/matrix-suite.json").write_text(json.dumps(results, indent=2))
    del operations, operation, x, w, wt, reference, output
    torch.cuda.empty_cache()
