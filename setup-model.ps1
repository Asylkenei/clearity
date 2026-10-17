$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
& '.\.venv\Scripts\python.exe' -m app.model_setup
if ($LASTEXITCODE -ne 0) { throw 'Model installation failed. Check the connection and run again.' }
