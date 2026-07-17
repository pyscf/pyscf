param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeDllDir,
    [string]$RepoRoot = '',
    [string]$ReportDirectory = 'tmp/windows-installed-wheel'
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot '..\..\..'
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$RuntimeDllDir = (Resolve-Path -LiteralPath $RuntimeDllDir).Path
$reportDir = Join-Path $RepoRoot $ReportDirectory
$libDir = Join-Path $RepoRoot 'pyscf\lib'
$depsBinDir = Join-Path $libDir 'deps\bin'
$buildLog = Join-Path $reportDir 'build.log'

function Invoke-WheelBuild([string]$Label) {
    "=== $Label ===" | Tee-Object -FilePath $buildLog -Append
    & python -m build --wheel --no-isolation --outdir dist . 2>&1 |
        Tee-Object -FilePath $buildLog -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
    if (Select-String -LiteralPath $buildLog -Pattern 'corrupt patch', 'patch failed', 'error: patch failed' -Quiet) {
        throw "$Label reported an XCFun patch failure"
    }
}

function Copy-RequiredFile([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required file was not generated: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
Set-Content -LiteralPath $buildLog -Value '' -Encoding utf8

Push-Location $RepoRoot
try {
    $env:PATH = "$RuntimeDllDir;$env:PATH"
    $env:CC = Join-Path $RuntimeDllDir 'gcc.exe'
    $env:CXX = Join-Path $RuntimeDllDir 'g++.exe'
    $env:CMAKE_BUILD_PARALLEL_LEVEL = '4'
    $openBlasImportLibrary = Join-Path $RuntimeDllDir '..\lib\libopenblas.dll.a'
    $env:CMAKE_CONFIGURE_ARGS = "-G Ninja -DCMAKE_C_COMPILER=$env:CC -DCMAKE_CXX_COMPILER=$env:CXX -DBLAS_LIBRARIES=$openBlasImportLibrary -DENABLE_XCFUN=ON -DBUILD_XCFUN=ON"

    Remove-Item -LiteralPath 'build', 'dist' -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $libDir -Filter '*.dll' -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Get-ChildItem -LiteralPath $depsBinDir -Filter '*.dll' -File -ErrorAction SilentlyContinue |
        Remove-Item -Force

    foreach ($name in @(
        'libgcc_s_seh-1.dll',
        'libgomp-1.dll',
        'libgfortran-5.dll',
        'libopenblas.dll',
        'libquadmath-0.dll',
        'libstdc++-6.dll',
        'libwinpthread-1.dll'
    )) {
        Copy-RequiredFile (Join-Path $RuntimeDllDir $name) (Join-Path $libDir $name)
    }

    Invoke-WheelBuild 'Bootstrap wheel build'

    Copy-RequiredFile (Join-Path $depsBinDir 'libcint.dll') (Join-Path $libDir 'libcint.dll')
    Copy-RequiredFile (Join-Path $depsBinDir 'libxc.dll') (Join-Path $libDir 'libxc.dll')
    $xcfun = @('libxcfun.dll', 'xcfun.dll') |
        ForEach-Object { Join-Path $depsBinDir $_ } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if (-not $xcfun) {
        throw 'XCFun runtime DLL was not generated'
    }
    Copy-Item -LiteralPath $xcfun -Destination (Join-Path $libDir 'libxcfun.dll') -Force

    Remove-Item -LiteralPath 'dist' -Recurse -Force
    Invoke-WheelBuild 'Final wheel build'

    $wheels = @(Get-ChildItem -LiteralPath 'dist' -Filter 'pyscf-*.whl' -File)
    if ($wheels.Count -ne 1) {
        throw "Expected exactly one wheel, found $($wheels.Count)"
    }
    $wheel = $wheels[0]
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($wheel.FullName)
    try {
        $entries = @($archive.Entries.FullName)
    }
    finally {
        $archive.Dispose()
    }
    $entries | Sort-Object | Set-Content -LiteralPath (Join-Path $reportDir 'wheel-contents.txt') -Encoding utf8
    $distInfoRoots = @($entries |
        Where-Object { $_ -match '^([^/]+\.dist-info)/' } |
        ForEach-Object { $Matches[1] } |
        Sort-Object -Unique)
    if ($distInfoRoots.Count -ne 1) {
        throw "Expected exactly one dist-info directory, found $($distInfoRoots.Count)"
    }
    foreach ($suffix in @('.dist-info/METADATA', '.dist-info/WHEEL', '.dist-info/RECORD')) {
        if (-not ($entries -like "*$suffix")) {
            throw "Wheel is missing $suffix"
        }
    }
    foreach ($name in @(
        'libnp_helper.dll',
        'libcgto.dll',
        'libcint.dll',
        'libxc_itrf.dll',
        'libxcfun_itrf.dll',
        'libxc.dll',
        'libxcfun.dll',
        'libopenblas.dll'
    )) {
        if ($entries -notcontains "pyscf/lib/$name") {
            throw "Wheel is missing pyscf/lib/$name"
        }
    }
    foreach ($name in @(
        'pyscf/lib/deps/bin/libcint.dll',
        'pyscf/lib/deps/bin/libxc.dll',
        'pyscf/lib/deps/bin/libxcfun.dll',
        'pyscf/lib/deps/bin/xcfun.dll',
        'pyscf/lib/xc.dll',
        'pyscf/lib/xcfun.dll'
    )) {
        if ($entries -contains $name) {
            throw "Wheel contains duplicate $name"
        }
    }
    $wheel.Length | Set-Content -LiteralPath (Join-Path $reportDir 'wheel-size.txt') -Encoding ascii
    if ($wheel.Length -ge 120MB) {
        throw "Wheel is unexpectedly large: $($wheel.Length) bytes"
    }
    $wheelHash = (Get-FileHash -LiteralPath $wheel.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $wheelHash | Set-Content -LiteralPath (Join-Path $reportDir 'wheel-sha256.txt') -Encoding ascii
    "Wheel: $($wheel.FullName)" | Tee-Object -FilePath $buildLog -Append
    "Wheel bytes: $($wheel.Length)" | Tee-Object -FilePath $buildLog -Append
    "Wheel SHA256: $wheelHash" | Tee-Object -FilePath $buildLog -Append
    "Wheel entries: $($entries.Count)" | Tee-Object -FilePath $buildLog -Append
}
finally {
    Pop-Location
}
