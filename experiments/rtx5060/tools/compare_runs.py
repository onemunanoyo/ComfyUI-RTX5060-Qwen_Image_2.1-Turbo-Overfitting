import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
baseline = {row["seed"]: row for row in json.loads(args.baseline.read_text())}
candidate = json.loads(args.candidate.read_text())
results = []
for row in candidate:
    paths = []
    arrays = []
    for run in [baseline[row["seed"]], row]:
        image = run["history"]["outputs"]["9"]["images"][0]
        path = Path("output") / image["subfolder"] / image["filename"]
        paths.append(str(path.resolve()))
        arrays.append(np.asarray(Image.open(path).convert("RGB")).astype(np.float32))
    diff = arrays[0] - arrays[1]
    result = {"seed": row["seed"], "baseline_path": paths[0], "candidate_path": paths[1], "equal": bool(np.array_equal(*arrays)), "mae": float(np.abs(diff).mean()), "rmse": float(np.square(diff).mean() ** 0.5), "candidate_mean": float(arrays[1].mean()), "candidate_std": float(arrays[1].std())}
    results.append(result)
args.output.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
