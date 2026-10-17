$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
& '.\.venv\Scripts\python.exe' -m app.train_run --setup-only
if ($LASTEXITCODE -ne 0) { throw 'Encoder installation failed.' }
