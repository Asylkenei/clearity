# Установка библиотек в отдельное окружение. Запускайте из PowerShell.
param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (-not $PythonPath) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundledPython) { $PythonPath = $bundledPython }
        else { $PythonPath = 'python' }
    }
    & $PythonPath -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Run setup.ps1 -PythonPath FULL_PATH_TO_PYTHON.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath '.env')) {
    & '.\.venv\Scripts\python.exe' -c "import secrets; from pathlib import Path; Path('.env').write_text('FORTE_API_KEY=' + secrets.token_urlsafe(32) + '\n', encoding='utf-8')"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .env.' }
}
Write-Host 'Setup complete. Open .env to copy your local access key, then run start.ps1.'
