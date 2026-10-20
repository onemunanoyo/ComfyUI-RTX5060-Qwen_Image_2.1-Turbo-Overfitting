import json
import os
import time
import torch

force_tune = os.environ.get("GEMM_FORCE_TUNE") == "1"
if force_tune:
    import torch._inductor.utils
    torch._inductor.utils.is_big_gpu = lambda index_or_device=0: True

torch.manual_seed(12345)
compiled = torch.compile(torch.nn.functional.linear, mode="max-autotune", fullgraph=True)
x = torch.randn(1, 4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.randn(24576, 4096, device="cuda", dtype=torch.bfloat16)
results = []
reference = None
for name, operation in [("eager", torch.nn.functional.linear), ("compiled", compiled)]:
    start_wall = time.perf_counter()
    for _ in range(3):
        output = operation(x, w)
    torch.cuda.synchronize()
    setup = time.perf_counter() - start_wall
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        output = operation(x, w)
    end.record()
    end.synchronize()
    if reference is None:
        reference = output.clone()
    result = {"backend": name, "setup_seconds": setup, "ms": start.elapsed_time(end) / 20, "equal": torch.equal(output, reference), "max_abs": (output - reference).abs().max().item()}
    results.append(result)
    print(json.dumps(result), flush=True)
    with open("experiments/rtx5060/results/compiled-gemm-forced.json" if force_tune else "experiments/rtx5060/results/compiled-gemm.json", "w") as file:
        json.dump(results, file, indent=2)
