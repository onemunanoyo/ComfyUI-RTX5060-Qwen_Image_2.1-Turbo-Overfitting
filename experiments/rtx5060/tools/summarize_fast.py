import json
import statistics
from pathlib import Path

files = {
    "bf16_control": "qwen-final-control.json",
    "bf16_compiled": "qwen-compiled-blocks.json",
    "fp16_accumulation": "qwen-fp16-accum.json",
    "fp16_accumulation_ck": "qwen-fp16-accum-ck.json",
    "fp16_compiled_limit8": "qwen-fp16-compiled.json",
    "fp16_compiled_limit64": "qwen-fp16-compiled64.json",
    "fast_launcher_recheck": "qwen-fast-final.json",
}
results = {}
for name, filename in files.items():
    path = Path("experiments/rtx5060/results", filename)
    if not path.exists():
        continue
    rows = json.loads(path.read_text())
    warm = rows[1:]
    if not warm:
        continue
    sampler_times = []
    for row in warm:
        events = {e["data"].get("node"): e["seconds"] for e in row["events"] if e["type"] == "executing"}
        sampler_times.append(events["8"] - events["7"])
    results[name] = {"file": str(path), "cold_seconds": rows[0]["wall_seconds"], "warm_seconds": [r["wall_seconds"] for r in warm], "warm_median": statistics.median(r["wall_seconds"] for r in warm), "sampler_median": statistics.median(sampler_times), "all_completed": all(r["history"]["status"]["completed"] for r in rows)}
Path("experiments/rtx5060/results/qwen-fast-results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
