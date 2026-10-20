import json
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(12345)
q, k, v = [torch.randn(1, 4096, 32, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3)]
results = []
reference = None
for backend in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]:
    if backend == SDPBackend.FLASH_ATTENTION and not torch.backends.cuda.is_flash_attention_available():
        results.append({"backend": str(backend), "available": False})
        continue
    with sdpa_kernel(backend):
        for _ in range(5):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        end.record()
        end.synchronize()
        if reference is None:
            reference = output.clone()
        result = {"backend": str(backend), "ms": start.elapsed_time(end) / 20, "equal": torch.equal(output, reference), "max_abs": (output - reference).abs().max().item()}
        results.append(result)
        print(json.dumps(result), flush=True)
with open("experiments/rtx5060/results/attention-backends.json", "w") as file:
    json.dump(results, file, indent=2)
