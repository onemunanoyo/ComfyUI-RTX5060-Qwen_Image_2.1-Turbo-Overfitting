import json
import torch

torch.manual_seed(12345)
results = []
for n, k in [(4096, 4096), (24576, 4096), (4096, 12288)]:
    x = torch.randn(1, 4096, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    reference = None
    for backend in ["cublas", "cublaslt"]:
        torch.backends.cuda.preferred_blas_library(backend)
        for _ in range(5):
            output = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            output = torch.nn.functional.linear(x, w)
        end.record()
        end.synchronize()
        if reference is None:
            reference = output.clone()
        result = {"n": n, "k": k, "m": 4096, "backend": backend, "ms": start.elapsed_time(end) / 20, "equal": torch.equal(reference, output), "max_abs": (reference - output).abs().max().item()}
        results.append(result)
        print(json.dumps(result), flush=True)
    del x, w, output, reference
    torch.cuda.empty_cache()
with open("experiments/rtx5060/results/gemm-backends.json", "w") as file:
    json.dump(results, file, indent=2)
