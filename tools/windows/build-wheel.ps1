param(
    [string]$PythonExe = "",
    [string]$RuntimeDllDir = "",
    [string]$RepoRoot = "",
    [int]$ParallelLevel = 8,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$RuntimeDlls = @(
    "libgcc_s_seh-1.dll",
    "libgomp-1.dll",
    "libgfortran-5.dll",
    "libopenblas.dll",
    "libquadmath-0.dll",
    "libstdc++-6.dll",
    "libwinpthread-1.dll"
)

$SupportDlls = @(
    "libcint.dll",
    "libxc.dll"
)

function Resolve-RepoRoot {
    param([string]$ConfiguredValue)
    if ($ConfiguredValue) {
        return (Resolve-Path $ConfiguredValue).Path
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Resolve-PythonExe {
    param([string]$ConfiguredValue)
    if ($ConfiguredValue) {
        return (Resolve-Path $ConfiguredValue).Path
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Python was not found on PATH. Activate the target conda environment or pass -PythonExe explicitly."
    }
    return $cmd.Source
}

function Require-Command {
    param(
        [string]$Name,
        [string]$Hint = ""
    )
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        if ($Hint) {
            throw "Missing required command: $Name. $Hint"
        }
        throw "Missing required command: $Name"
    }
    return $cmd.Source
}

function Resolve-NinjaExe {
    param(
        [string]$LibraryBin,
        [string]$ScriptsDir
    )
    $candidates = @(
        (Join-Path $LibraryBin "ninja.exe"),
        (Join-Path $ScriptsDir "ninja.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "Missing required command: ninja. Install Ninja into the target conda environment."
}

function Resolve-RuntimeDllDir {
    param([string]$ConfiguredValue)
    if ($ConfiguredValue) {
        return (Resolve-Path $ConfiguredValue).Path
    }

    $gccCmd = Get-Command gcc -ErrorAction SilentlyContinue
    if ($gccCmd) {
        return (Split-Path $gccCmd.Source -Parent)
    }

    $candidates = @(
        "D:\msys64\ucrt64\bin",
        "C:\msys64\ucrt64\bin"
    )
    foreach ($candidate in $candidates) {
        if ((Test-Path (Join-Path $candidate "gcc.exe")) -and
            (Test-Path (Join-Path $candidate "libgcc_s_seh-1.dll"))) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Unable to locate the MSYS2 UCRT64 runtime directory. Pass -RuntimeDllDir explicitly or install MSYS2 UCRT64 at D:\msys64\ucrt64\bin or C:\msys64\ucrt64\bin."
}

function Copy-RequiredDlls {
    param(
        [string]$RuntimeDllDir,
        [string]$LibDir
    )
    # Keep runtime DLLs next to the extension modules so the wheel can load
    # them through the normal pyscf\lib search path.
    foreach ($name in $RuntimeDlls) {
        $source = Join-Path $RuntimeDllDir $name
        if (-not (Test-Path $source)) {
            throw "Missing runtime DLL: $source"
        }
        Copy-Item -LiteralPath (Join-Path $RuntimeDllDir $name) -Destination (Join-Path $LibDir $name) -Force
    }
}

function Copy-SupportDlls {
    param(
        [string]$DepsBinDir,
        [string]$LibDir
    )
    # Stage the bundled support DLLs in pyscf\lib because PySCF loads them
    # through its main library directory rather than from deps\bin. Duplicate
    # copies under deps\bin are excluded from the wheel by MANIFEST.in.
    foreach ($name in $SupportDlls) {
        $source = Join-Path $DepsBinDir $name
        if (-not (Test-Path $source)) {
            throw "Missing support DLL: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $LibDir $name) -Force
    }
}

try {
    $RepoRoot = Resolve-RepoRoot $RepoRoot
    $PythonExe = Resolve-PythonExe $PythonExe
    $PythonDir = Split-Path $PythonExe -Parent
    if ((Split-Path $PythonDir -Leaf) -ieq "Scripts") {
        $EnvRoot = Split-Path $PythonDir -Parent
    }
    else {
        $EnvRoot = $PythonDir
    }
    $LibraryBin = Join-Path $EnvRoot "Library\\bin"
    $ScriptsDir = Join-Path $EnvRoot "Scripts"
    $LibDir = Join-Path $RepoRoot 'pyscf\lib'
    $DepsBinDir = Join-Path $RepoRoot 'pyscf\lib\deps\bin'

    $RuntimeDllDir = Resolve-RuntimeDllDir -ConfiguredValue $RuntimeDllDir
    $BootstrapPaths = @(
        $RuntimeDllDir,
        $LibraryBin,
        $ScriptsDir,
        $PythonDir,
        $env:PATH
    )
    $BootstrapPaths = $BootstrapPaths | Where-Object { $_ } | Select-Object -Unique
    $env:PATH = $BootstrapPaths -join ';'

    Require-Command "cmake" | Out-Null
    $NinjaExe = Resolve-NinjaExe -LibraryBin $LibraryBin -ScriptsDir $ScriptsDir
    Require-Command "gcc" | Out-Null
    Require-Command "g++" | Out-Null

    Copy-RequiredDlls -RuntimeDllDir $RuntimeDllDir -LibDir $LibDir
    Copy-SupportDlls -DepsBinDir $DepsBinDir -LibDir $LibDir

    if ($Clean) {
        Remove-Item (Join-Path $RepoRoot "build") -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $RepoRoot "dist") -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        Remove-Item (Join-Path $RepoRoot "build\\lib") -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem (Join-Path $RepoRoot "build") -Directory -Filter "bdist.*" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    $env:CC = "gcc"
    $env:CXX = "g++"
    $env:CMAKE_BUILD_PARALLEL_LEVEL = $ParallelLevel.ToString()
    $env:CMAKE_CONFIGURE_ARGS = "-G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBLAS_LIBRARIES=$RuntimeDllDir\\..\\lib\\libopenblas.dll.a -DENABLE_XCFUN=OFF -DBUILD_XCFUN=OFF"

    Push-Location $RepoRoot
    try {
        & $PythonExe -m build -x --wheel --no-isolation --outdir dist .
        if ($LASTEXITCODE -ne 0) {
            throw "Wheel build failed"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $Stopwatch.Stop()
    Write-Host ("Total build time: {0:hh\:mm\:ss} ({1:N1} s)" -f $Stopwatch.Elapsed, $Stopwatch.Elapsed.TotalSeconds)
}
