# Creates a temporary HTTPS link. Requires start-public.ps1 in another terminal.
param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$fortePublicHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
if ($fortePublicHealth.public_mode -ne $true) {
    throw 'Refusing to expose a local review server. Start start-public.ps1 first.'
}
$forteTunnelVersion = '2026.10.1'
$forteTunnelHash = '4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6'
$forteTunnelExe = Join-Path $PSScriptRoot ".tools\cloudflared-$forteTunnelVersion.exe"
if (-not (Test-Path -LiteralPath $forteTunnelExe)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '.tools') | Out-Null
    Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/download/$forteTunnelVersion/cloudflared-windows-amd64.exe" -OutFile $forteTunnelExe -UseBasicParsing
}
if ((Get-FileHash -LiteralPath $forteTunnelExe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $forteTunnelHash) {
    throw 'The cloudflared download does not match the official release checksum.'
}
Write-Host 'Open the https://...trycloudflare.com address printed below. Keep both terminals open.'
& $forteTunnelExe tunnel --url "http://127.0.0.1:$Port" --protocol http2 --no-autoupdate
