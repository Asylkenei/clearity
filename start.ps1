param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath '.env')) { throw 'Run setup.ps1 first to create .env.' }
Write-Host "Open http://127.0.0.1:$Port . Press Ctrl+C to stop."
& '.\.venv\Scripts\python.exe' -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port $Port
