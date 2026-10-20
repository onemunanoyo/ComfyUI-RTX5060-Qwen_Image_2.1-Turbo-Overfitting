# RTX 5060 inference experiments

Setup, model filenames, launch modes and the measured speedup are documented in the [root README](../../README.md). The original ComfyUI documentation is preserved in [README.upstream.md](../../README.upstream.md).

## Layout

Run all commands from the checkout root.

- `launch_qwen_fast.py`: optional FP16 execution launcher, selected by `run_experiment.ps1 -QwenFast`.
- `requirements.txt`: tested experiment dependencies, including the root requirements.
- `benchmark-summary.json`: curated public measurements and limitations.
- `tools/`: measurement and comparison scripts, including rejected experiments.
- `patches/gguf-rtx5060.patch`: complete GGUF node changes against city96 commit `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`.
- `patches/host-cache.patch`: core host-cache change snapshot; the fork already includes this change.
- `results/`, `logs/`, `profiles/`: local artifacts, excluded from Git.

Images stay in `output/benchmark/`. Historical local records are preserved unchanged and can contain paths from the previous layout. Local organization manifests and machine-specific notes are not published.

## Behavior

The GGUF patch adds Qwen Image 2.1 architecture recognition and fused Q6_K/Q8_0 dequantization. The fused path is enabled by `COMFY_GGUF_FUSED_DEQUANT=1`, which the root launcher sets by default. It retains the reference FP16 intermediate rounding before casting to the requested output dtype. CPU, unsupported block formats and explicit nondefault dequantization precision use the original implementation.

The core memory-management change keeps a dynamic model registered with its existing host buffers when freeing GPU residency for a legacy model. Existing RAM-pressure eviction still applies. Explicit full unload still destroys the cache. The four regression cases cover mixed-model eviction, ordinary legacy eviction, explicit unload and low-RAM release.

Fast mode selects FP16 denoiser computation and FP16 accumulation. The launcher marks GGUF Linear as owning its forward path so generic activation fusion does not reinterpret packed weights. Kitchen Attention is an additional CLI option and uses INT8 internally. Neither fast mode is pixel-identical to BF16.

## Validation

```powershell
.venv/Scripts/python.exe -m pytest tests-unit/comfy_test/test_mixed_model_offload.py -q
.venv/Scripts/python.exe experiments/rtx5060/tools/check_gguf_dequant.py PATH_TO_Q6_K_GGUF
.venv/Scripts/python.exe experiments/rtx5060/tools/benchmark_local.py --model qwen --runs 4 --sigmas '1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0' --output experiments/rtx5060/results/qwen.json
```

The actual model passed 1732 exact dequantization comparisons over Q6_K/Q8_0 tensors, three output dtypes and partial blocks. That validates weight expansion, not the changed arithmetic of the fast modes. Earlier BF16 fused-dequantization images matched the original expansion path exactly; fast-mode images do not match BF16.

Measurements use one landscape prompt, 1024x1024, batch 1, CFG 1, Euler and eight steps. Seed changes force sampling, decode and saving to execute while the fixed prompt reuses conditioning. The benchmark's generic default schedule remains `simple`; pass the explicit Turbo sigma list above for Qwen. Initial loading and filesystem caches are not controlled for cold-start comparisons.

## Rejected experiments

- Nonblocking transfers alone and pinning plus nonblocking transfers did not improve the standalone sampler time.
- Three representative BF16 GEMM shapes showed at most a small tuning gain. Comfy Kitchen FP16 GEMM was slower than PyTorch in the tested shapes.
- Block compilation offered little steady-state benefit and could trigger approximately 55 seconds of recompilation after model swapping. It is not enabled by the fast launcher.
- Anima prefetch did not improve steady-state generation; memory compilation failed. Neither Anima experiment was retained.

GPU performance and image quality on other machines, prompts, resolutions, editing workflows and models remain unvalidated. CI runs CPU regression checks, not a GPU speed or image-quality benchmark.
