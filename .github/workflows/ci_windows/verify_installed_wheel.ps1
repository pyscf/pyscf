param(
    [string]$RepoRoot = '',
    [string]$ReportDirectory = 'tmp/windows-installed-wheel'
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = Join-Path $PSScriptRoot '..\..\..'
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$reportDir = Join-Path $RepoRoot $ReportDirectory
$runnerTemp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$runRoot = Join-Path $runnerTemp 'pyscf-installed-wheel'
$launchRoot = Join-Path $runRoot 'launch'
$venv = Join-Path $runRoot '.venv'

function Add-PackageMarker([string]$Directory) {
    $marker = Join-Path $Directory '__init__.py'
    if (-not (Test-Path -LiteralPath $marker)) {
        New-Item -ItemType File -Path $marker | Out-Null
    }
}

New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
Remove-Item -LiteralPath $runRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $launchRoot -Force | Out-Null

Push-Location $RepoRoot
try {
    python -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to create installed-wheel test environment'
    }
    $python = Join-Path $venv 'Scripts\python.exe'
    $wheels = @(Get-ChildItem -LiteralPath 'dist' -Filter 'pyscf-*.whl' -File)
    if ($wheels.Count -ne 1) {
        throw "Expected exactly one wheel, found $($wheels.Count)"
    }
    & $python -m pip install $wheels[0].FullName 'pytest<9' geometric spglib 'git+https://github.com/jhrmnn/pyberny.git@36a4be9' 2>&1 |
        Tee-Object -FilePath (Join-Path $reportDir 'install.log')
    if ($LASTEXITCODE -ne 0) {
        throw 'Installed-wheel test environment setup failed'
    }

    $sourcePackage = (Resolve-Path 'pyscf').Path
    $installedPackage = Join-Path $venv 'Lib\site-packages\pyscf'
    $sitePackages = Split-Path $installedPackage -Parent
    if (-not (Test-Path -LiteralPath $installedPackage -PathType Container)) {
        throw "Installed package was not found: $installedPackage"
    }
    Get-ChildItem -LiteralPath $sourcePackage -Directory -Recurse |
        Where-Object Name -eq 'test' |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($sourcePackage, $_.FullName)
            $destination = Join-Path $installedPackage $relative
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
            Copy-Item -Path (Join-Path $_.FullName '*') -Destination $destination -Recurse -Force
            $packageDirectory = $destination
            while ($packageDirectory -ne $installedPackage) {
                Add-PackageMarker $packageDirectory
                $packageDirectory = Split-Path $packageDirectory -Parent
            }
            Get-ChildItem -LiteralPath $destination -Directory -Recurse |
                ForEach-Object { Add-PackageMarker $_.FullName }
        }

    $pytestConfig = (Resolve-Path 'pytest.ini').Path
    $pyscfConfig = Join-Path $launchRoot '.pyscf_conf.py'
    @(
        'pbc_tools_pbc_fft_engine = "NUMPY+BLAS"',
        'scf_hf_SCF_mute_chkfile = True'
    ) | Set-Content -LiteralPath $pyscfConfig -Encoding utf8

    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYSCF_EXT_PATH -ErrorAction SilentlyContinue
    $env:PYSCF_CONFIG_FILE = $pyscfConfig
    & $python -VV 2>&1 | Set-Content -LiteralPath (Join-Path $reportDir 'environment.txt') -Encoding utf8
    & $python -m pip freeze | Add-Content -LiteralPath (Join-Path $reportDir 'environment.txt') -Encoding utf8

    Push-Location $sitePackages
    try {
        & $python -c "import pathlib,pyscf,sys; p=pathlib.Path(pyscf.__file__).resolve(); source=pathlib.Path(r'$RepoRoot').resolve(); print(p); assert 'site-packages' in str(p); assert all(pathlib.Path(x or '.').resolve() != source for x in sys.path); from pyscf import gto,scf; assert abs(scf.RHF(gto.M(atom='H 0 0 0; H 0 0 1.4',unit='Bohr',basis='sto-3g',verbose=0)).kernel()+1.116714325062551)<1e-9" 2>&1 |
            Tee-Object -FilePath (Join-Path $reportDir 'import-smoke.log')
        if ($LASTEXITCODE -ne 0) {
            throw 'Installed-wheel import smoke failed'
        }
        & $python -m pytest pyscf -s -c $pytestConfig --rootdir . --import-mode=prepend --durations=20 2>&1 |
            Tee-Object -FilePath (Join-Path $reportDir 'pytest.log')
        $pytestExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        Remove-Item Env:PYSCF_CONFIG_FILE -ErrorAction SilentlyContinue
    }
    $pytestExitCode | Set-Content -LiteralPath (Join-Path $reportDir 'pytest-exit-code.txt') -Encoding ascii
    if ($pytestExitCode -ne 0) {
        throw "Installed-wheel pytest failed with exit code $pytestExitCode"
    }
}
finally {
    Pop-Location
}
