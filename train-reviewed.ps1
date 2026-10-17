param([string]$RunName = 'reviewed-v2', [string]$BaseRun = 'baseline-v1')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
& '.\.venv\Scripts\python.exe' -m app.train_run --use-reviewed --base-run $BaseRun --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw 'Training did not complete. Check saved annotations and the message above.' }
