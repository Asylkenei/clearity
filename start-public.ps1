# Start a separate upload-only server. Training data stays in the local app.
param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
& '.\.venv\Scripts\python.exe' -m app.public_setup
if ($LASTEXITCODE -ne 0) { throw 'Could not configure the public access key.' }
$env:FORTE_PUBLIC_MODE = '1'
Write-Host "Upload-only server: http://127.0.0.1:$Port . Keep this terminal open."
Write-Host 'Use FORTE_PUBLIC_API_KEY from .env for the shared site.'
& '.\.venv\Scripts\python.exe' -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port $Port
