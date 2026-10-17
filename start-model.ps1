param([int]$Port = 8081)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$engine = '.tools\llama-b11714\llama-server.exe'
$weights = 'models\qwen3-vl-2b\Qwen3VL-2B-Instruct-Q4_K_M.gguf'
$vision = 'models\qwen3-vl-2b\mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'
foreach ($file in @($engine, $weights, $vision)) {
    if (-not (Test-Path -LiteralPath $file)) { throw 'Run setup-model.ps1 first.' }
}
Write-Host 'Loading the local model. Wait for: model loaded / listening.'
Write-Host 'Keep this terminal open. The model runs offline, on this computer.'
& $engine -m $weights --mmproj $vision --alias clearity-local --host 127.0.0.1 --port $Port --threads 6 --threads-batch 6 --ctx-size 4096 --parallel 1 --cache-ram 0 --image-min-tokens 64 --image-max-tokens 256 --n-gpu-layers 0 --no-mmproj-offload --offline --no-webui
