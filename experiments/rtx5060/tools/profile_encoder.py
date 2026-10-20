import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.argv = ["profile_encoder", "--offline"] + sys.argv[1:]
import main  # noqa: F401 -- initializes ComfyUI paths and device policy
import nodes
import torch

clip = nodes.CLIPLoader().load_clip("qwen3vl_8b_int8_convrot.safetensors", "qwen_image")[0]
result = []
for index, prompt in enumerate(("A watercolor cabin beside a lake.", "A watercolor cabin beside a lake with pine trees.")):
    tokens = clip.tokenize(prompt, keep_vision=True, prevent_empty_text=True)
    profile = cProfile.Profile()
    start = time.perf_counter()
    cond = profile.runcall(clip.encode_from_tokens_scheduled, tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    profile.dump_stats(f"experiments/rtx5060/profiles/encoder-{index}.prof")
    with Path(f"experiments/rtx5060/profiles/encoder-{index}-profile.txt").open("w") as output:
        pstats.Stats(profile, stream=output).sort_stats("cumulative").print_stats(45)
    result.append({"run": index, "seconds": elapsed, "shape": list(cond[0][0].shape)})
    print(json.dumps(result[-1]), flush=True)
Path("experiments/rtx5060/profiles/encoder-profile-results.json").write_text(json.dumps(result, indent=2))
