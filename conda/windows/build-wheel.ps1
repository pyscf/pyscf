<#
.SYNOPSIS
Build a Windows wheel for PySCF and prepare the required external runtime dependencies.

.DESCRIPTION
Usage:
  powershell -ExecutionPolicy Bypass -File conda\windows\build-wheel.ps1
  powershell -ExecutionPolicy Bypass -File conda\windows\build-wheel.ps1 -ParallelLevel 8
  powershell -ExecutionPolicy Bypass -File conda\windows\build-wheel.ps1 -RuntimeDllDir <path-to-ucrt64-bin> -Clean

The script synchronizes external runtime dependencies into
  pyscf\lib\deps\win64\bin
sets PYSCF_WINDOWS_RUNTIME_DLL_DIR to that location, and then runs
  python -m build --wheel --no-isolation

The staged DLLs are local build artifacts and are not intended to be tracked
in git.

If third_party sources are not present, the CMake ExternalProject rules in
PySCF will download them during the build.
#>

param(
    [string]$PythonExe = "",
    [string]$RuntimeDllDir = "",
    [string]$RepoRoot = "",
    [int]$ParallelLevel = 0,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$buildStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$RequiredRuntimeDlls = @(
    "libgcc_s_seh-1.dll",
    "libgomp-1.dll",
    "libgfortran-5.dll",
    "libopenblas.dll",
    "libquadmath-0.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll"
)

function Sync-RuntimeDlls {
    param(
        [string]$SourceDir,
        [string]$TargetDir,
        [string[]]$DllNames
    )
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    foreach ($dllName in $DllNames) {
        $sourcePath = Join-Path $SourceDir $dllName
        if (-not (Test-Path $sourcePath)) {
            throw "Missing runtime DLL: $sourcePath"
        }
        $targetPath = Join-Path $TargetDir $dllName
        if (Test-Path $targetPath) {
            Remove-Item -LiteralPath $targetPath -Force
        }
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Require-Path {
    param(
        [string]$Path,
        [string]$Message
    )
    if (-not (Test-Path $Path)) {
        throw $Message
    }
}

function Require-Command {
    param(
        [string]$Name,
        [string]$Message
    )
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw $Message
    }
    return $cmd.Source
}

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

function Require-PythonBuildFrontend {
    param(
        [string]$PythonExe,
        [string]$WorkingDirectory
    )
    $code = "import build; import build.__main__; print(build.__file__)"
    Push-Location $WorkingDirectory
    try {
        $output = & $PythonExe -c $code 2>&1
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Missing Python build frontend 'build'. Install it into the active environment before packaging. Output: $output"
    }
}

function Get-FirstExistingDirectory {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Get-FirstExistingPath {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Find-RuntimeDllDirectory {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (-not $candidate -or -not (Test-Path $candidate)) {
            continue
        }
        if (Test-Path (Join-Path $candidate "libgcc_s_seh-1.dll")) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Resolve-ParallelLevel {
    param([int]$ConfiguredValue)
    if ($ConfiguredValue -gt 0) {
        return $ConfiguredValue
    }
    $logicalCores = [Environment]::ProcessorCount
    if ($logicalCores -lt 1) {
        return 4
    }
    return [Math]::Min($logicalCores, 8)
}

function Assert-UnlockedBuildOutputs {
    param([string]$RootDir)
    $locked = @()
    Get-ChildItem $RootDir -Filter *.dll -File -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $stream = [System.IO.File]::Open($_.FullName, 'Open', 'ReadWrite', 'None')
            $stream.Close()
        }
        catch {
            $locked += $_.FullName
        }
    }
    if ($locked.Count -gt 0) {
        $details = $locked -join [Environment]::NewLine
        throw "A running process is locking existing build outputs under $RootDir. Close Python interpreters, notebooks, IDE test sessions, or terminals that imported PySCF from this source tree and retry. Locked files:`n$details"
    }
}

$PythonExe = Resolve-PythonExe $PythonExe
$pythonDir = Split-Path $PythonExe -Parent
$envRoot = if ((Split-Path $pythonDir -Leaf) -ieq "Scripts") {
    Split-Path $pythonDir -Parent
} else {
    $pythonDir
}
$libraryBin = Join-Path $envRoot "Library\bin"
$scriptsDir = Join-Path $envRoot "Scripts"
$repoRuntimeDir = Join-Path $RepoRoot "pyscf\lib\deps\win64\bin"
New-Item -ItemType Directory -Force -Path $repoRuntimeDir | Out-Null

$bootstrapDirs = @(
    $repoRuntimeDir,
    $libraryBin,
    $scriptsDir,
    $pythonDir,
    "D:\Program Files\Git\cmd",
    "C:\Program Files\Git\cmd",
    "C:\msys64\ucrt64\bin",
    "D:\msys64\ucrt64\bin"
) | Where-Object { $_ -and (Test-Path $_) }
if ($bootstrapDirs.Count -gt 0) {
    $env:PATH = ($bootstrapDirs + $env:PATH.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)) -join ';'
}

$gitCmd = Require-Command "git" "Missing git. Install Git for Windows first."
$cmakeCmd = Require-Command "cmake" "Missing cmake. Install CMake >= 3.22 first."
$ninjaCmd = Require-Command "ninja" "Missing ninja. Install Ninja first."
$gccCmd = Require-Command "gcc" "Missing gcc. Install MSYS2 UCRT64 GCC first."
$gxxCmd = Require-Command "g++" "Missing g++. Install MSYS2 UCRT64 GCC first."

$buildInvocationDir = Split-Path $RepoRoot -Parent
Require-PythonBuildFrontend -PythonExe $PythonExe -WorkingDirectory $buildInvocationDir

$cmakeVersion = & $cmakeCmd --version | Select-Object -First 1
if ($cmakeVersion -match '(\d+)\.(\d+)\.(\d+)') {
    $version = [Version]::new([int]$Matches[1], [int]$Matches[2], [int]$Matches[3])
    if ($version -lt [Version]::new(3, 22, 0)) {
        throw "CMake version is too old: $cmakeVersion"
    }
}

if (-not $RuntimeDllDir) {
    $RuntimeDllDir = Find-RuntimeDllDirectory @(
        $repoRuntimeDir,
        (Split-Path $gccCmd -Parent),
        (Split-Path $gxxCmd -Parent),
        $libraryBin
    )
}
Require-Path $RuntimeDllDir "Missing runtime DLL directory: $RuntimeDllDir"

Sync-RuntimeDlls -SourceDir $RuntimeDllDir -TargetDir $repoRuntimeDir -DllNames $RequiredRuntimeDlls

$pythonVersion = & $PythonExe -c "import sys; print(sys.version)"
$pythonVersionInfo = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pythonVersionInfo -ne "3.13") {
    throw "Unsupported Python version for the Windows release path: $pythonVersionInfo. Use Python 3.13."
}
$blasLibrary = Get-FirstExistingPath @(
    (Join-Path $envRoot "Library\lib\openblas.lib"),
    (Join-Path (Split-Path $RuntimeDllDir -Parent) "lib\libopenblas.dll.a")
)
Write-Host "Python:  $PythonExe"
Write-Host $pythonVersion
Write-Host "Repo:    $RepoRoot"
Write-Host "git:     $gitCmd"
Write-Host "cmake:   $cmakeCmd"
Write-Host "ninja:   $ninjaCmd"
Write-Host "gcc:     $gccCmd"
Write-Host "g++:     $gxxCmd"
Write-Host "runtime: $repoRuntimeDir"
if ($blasLibrary) {
    Write-Host "blas:    $blasLibrary"
}

if (-not (Test-Path (Join-Path $RepoRoot "third_party\libcint\CMakeLists.txt"))) {
    Write-Host "Info: third_party/libcint not found. CMake will download libcint during the build."
}
if (-not (Test-Path (Join-Path $RepoRoot "third_party\libxc-7.0.0\CMakeLists.txt"))) {
    Write-Host "Info: third_party/libxc-7.0.0 not found. CMake will download libxc during the build."
}
if (-not (Test-Path (Join-Path $RepoRoot "third_party\xcfun\CMakeLists.txt"))) {
    Write-Host "Info: third_party/xcfun not found. CMake will download xcfun during the build."
}

$env:PATH = @(
    (Split-Path $gitCmd -Parent),
    (Join-Path $RepoRoot "pyscf\lib"),
    (Join-Path $RepoRoot "pyscf\lib\deps\bin"),
    (Join-Path $RepoRoot "pyscf\lib\deps\lib"),
    (Join-Path $RepoRoot "pyscf\lib\deps\win64\bin"),
    $libraryBin,
    $scriptsDir,
    $pythonDir,
    $env:PATH
) -join ';'

$env:CC = "gcc"
$env:CXX = "g++"
$configureArgs = @(
    "-G",
    "Ninja",
    "-DCMAKE_C_COMPILER=gcc",
    "-DCMAKE_CXX_COMPILER=g++"
)
if ($blasLibrary) {
    $configureArgs += "-DBLAS_LIBRARIES=$blasLibrary"
}
$env:CMAKE_CONFIGURE_ARGS = [string]::Join(" ", $configureArgs)
$ParallelLevel = Resolve-ParallelLevel $ParallelLevel
$env:CMAKE_BUILD_PARALLEL_LEVEL = $ParallelLevel.ToString()
$env:PYSCF_WINDOWS_RUNTIME_DLL_DIR = $repoRuntimeDir
Write-Host "parallel: $ParallelLevel"

Push-Location $RepoRoot
try {
    if ($Clean) {
        Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem pyscf\lib -Filter *.dll -File -ErrorAction SilentlyContinue | Remove-Item -Force
        Get-ChildItem pyscf\lib -Filter *.pdb -File -ErrorAction SilentlyContinue | Remove-Item -Force
        foreach ($dllName in $RequiredRuntimeDlls) {
            Remove-Item (Join-Path $repoRuntimeDir $dllName) -Force -ErrorAction SilentlyContinue
        }
    }
    else {
        Remove-Item (Join-Path $RepoRoot "build\lib") -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem (Join-Path $RepoRoot "build") -Directory -Filter "bdist.*" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    Assert-UnlockedBuildOutputs -RootDir (Join-Path $RepoRoot "pyscf\lib")

    Push-Location $buildInvocationDir
    try {
        & $PythonExe -m build --wheel --no-isolation --outdir (Join-Path $RepoRoot "dist") $RepoRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Wheel build failed"
        }
    }
    finally {
        Pop-Location
    }

    $wheel = Get-ChildItem dist\pyscf-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $wheel) {
        throw "No wheel file was produced in dist\"
    }
    Write-Host "Wheel:   $($wheel.FullName)"
}
finally {
    $buildStopwatch.Stop()
    $elapsed = $buildStopwatch.Elapsed
    Write-Host "Build time: $($elapsed.ToString("hh\:mm\:ss"))"
    Pop-Location
}
