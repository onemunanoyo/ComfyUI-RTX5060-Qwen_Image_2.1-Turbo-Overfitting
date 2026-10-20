param(
    [int]$Port = 8190,
    [switch]$Baseline,
    [switch]$QwenFast,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:COMFY_GGUF_FUSED_DEQUANT = if ($Baseline) { '0' } else { '1' }
$env:TRITON_CACHE_DIR = "$PSScriptRoot\.uv-cache\triton"
$entryPoint = 'main.py'
if ($QwenFast) {
    $entryPoint = 'experiments/rtx5060/launch_qwen_fast.py'
    $ExtraArgs = @('--fp16-unet', '--fast', 'fp16_accumulation') + $ExtraArgs
}
& "$PSScriptRoot\.venv\Scripts\python.exe" $entryPoint --listen 127.0.0.1 --port $Port --offline --disable-auto-launch @ExtraArgs
exit $LASTEXITCODE
