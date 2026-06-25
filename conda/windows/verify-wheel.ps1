<#
.SYNOPSIS
Verify that a built PySCF wheel works on Windows.

.DESCRIPTION
Usage:
  powershell -ExecutionPolicy Bypass -File conda\windows\verify-wheel.ps1 -Phase all
  powershell -ExecutionPolicy Bypass -File conda\windows\verify-wheel.ps1 -InstallWheel -WheelPath dist\pyscf-2.13.1-py3-none-win_amd64.whl -Phase all -OutputJson .tmp\verify-wheel-all.json

The verification phases are:
  artifact     Wheel artifact content checks
  import       Basic import checks
  smoke        Minimal numerical sanity checks
  examples     Small representative examples
  packaging    Historical packaging regression examples
  diagnostics  Expected non-packaging failures that should keep failing in the same way
  all          Run every phase above
#>

param(
    [string]$PythonExe = "",
    [string]$WheelPath = "",
    [string]$RepoRoot = "",
    [ValidateSet("artifact", "import", "smoke", "examples", "packaging", "diagnostics", "all")]
    [string]$Phase = "all",
    [string]$OutputJson = "",
    [switch]$InstallWheel
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$ConfiguredValue)
    if ($ConfiguredValue) {
        if (-not (Test-Path $ConfiguredValue)) {
            throw "Missing Python interpreter: $ConfiguredValue"
        }
        return (Resolve-Path $ConfiguredValue).Path
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Python was not found on PATH. Activate the target conda environment or pass -PythonExe explicitly."
    }
    return $cmd.Source
}

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$PythonExe = Resolve-PythonExe $PythonExe

if ($InstallWheel) {
    if (-not $WheelPath) {
        throw "WheelPath is required when InstallWheel is set"
    }
    if (-not (Test-Path $WheelPath)) {
        throw "Missing wheel file: $WheelPath"
    }
    & $PythonExe -m pip install --force-reinstall $WheelPath
}

$runnerPath = Join-Path $PSScriptRoot "verify-wheel.py"
if (-not (Test-Path $runnerPath)) {
    throw "Missing runner script: $runnerPath"
}

if ($OutputJson) {
    $outputDir = Split-Path -Parent $OutputJson
    if ($outputDir) {
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    }
}

$args = @(
    $runnerPath,
    "--repo-root", $RepoRoot,
    "--phase", $Phase,
    "--python-exe", $PythonExe
)
if ($WheelPath) {
    $args += @("--wheel-path", (Resolve-Path $WheelPath).Path)
}
if ($OutputJson) {
    $args += @("--output-json", $OutputJson)
}

& $PythonExe @args
exit $LASTEXITCODE
