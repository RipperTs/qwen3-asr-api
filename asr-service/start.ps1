#Requires -Version 5.1
<#
.SYNOPSIS
    Qwen3-ASR Service Launcher (PowerShell)
.DESCRIPTION
    Detects Python environment (portable first, then venv) and launches the ASR service.
    All extra arguments are passed to the Python service.
.EXAMPLE
    .\start.ps1 --model-size 1.7b --enable-align
    .\start.ps1 --device cpu --model-size 0.6b
    .\start.ps1 --model-source huggingface
#>

param(
    [Parameter(ValueFromRemainingArguments)]
    $ExtraArgs
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$env:PYTHONPATH = $PSScriptRoot
$env:PATH = "$PSScriptRoot\bin;$PSScriptRoot\bin\python;$($env:PATH)"

$pythonBin = $null

if (Test-Path 'bin\python\python.exe') {
    $pythonBin = Join-Path $PSScriptRoot 'bin\python\python.exe'
    Write-Host '[INFO] Using portable Python' -ForegroundColor Cyan
}
elseif (Test-Path 'venv\Scripts\python.exe') {
    $pythonBin = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    Write-Host '[INFO] Using venv virtual environment' -ForegroundColor Cyan
}
else {
    Write-Host '[ERROR] No Python environment detected - neither portable nor venv exists' -ForegroundColor Red
    Write-Host 'Please run setup.ps1 first to configure the environment.' -ForegroundColor Red
    Write-Host
    Write-Host 'Press Enter to exit...' -ForegroundColor DarkGray
    Read-Host
    exit 1
}

& $pythonBin -m app.main @ExtraArgs
