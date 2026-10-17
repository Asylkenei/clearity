param([string]$RunName = 'baseline-v1')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -m app.train_run --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Read the error above; no model has been activated.' }
