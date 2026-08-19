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
$savedPytestAddopts = $env:PYTEST_ADDOPTS

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
    $pipCheckOutput = & $python -m pip check 2>&1
    $pipCheckExitCode = $LASTEXITCODE
    $pipCheckOutput | Set-Content -LiteralPath (Join-Path $reportDir 'pip-check.txt') -Encoding utf8
    if ($pipCheckExitCode -ne 0) {
        throw "Installed-wheel dependency check failed with exit code $pipCheckExitCode"
    }

    $sourcePackage = (Resolve-Path 'pyscf').Path
    $installedPackage = Join-Path $venv 'Lib\site-packages\pyscf'
    $sitePackages = Split-Path $installedPackage -Parent
    if (-not (Test-Path -LiteralPath $installedPackage -PathType Container)) {
        throw "Installed package was not found: $installedPackage"
    }
    # Wheels do not ship tests. Overlay only test directories onto the installed
    # package so production modules and DLLs still come from the wheel.
    Get-ChildItem -LiteralPath $sourcePackage -Directory -Recurse |
        Where-Object Name -eq 'test' |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($sourcePackage, $_.FullName)
            $destination = Join-Path $installedPackage $relative
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
            Copy-Item -Path (Join-Path $_.FullName '*') -Destination $destination -Recurse -Force
            # Keep relative imports working without copying source package modules.
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

    # Inherited paths and pytest options can shadow the wheel or change the
    # repository's regular test selection.
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYSCF_EXT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTEST_ADDOPTS -ErrorAction SilentlyContinue
    $env:PYSCF_CONFIG_FILE = $pyscfConfig
    & $python -VV 2>&1 | Set-Content -LiteralPath (Join-Path $reportDir 'environment.txt') -Encoding utf8
    & $python -m pip freeze | Add-Content -LiteralPath (Join-Path $reportDir 'environment.txt') -Encoding utf8

    $junitPath = Join-Path $reportDir 'pytest-results.xml'
    $pytestArgs = @(
        'pyscf',
        '-s',
        '-c', $pytestConfig,
        '--rootdir', '.',
        '--import-mode=prepend',
        '--durations=20',
        "--junitxml=$junitPath"
    )
    @(
        "pytest.ini SHA256: $((Get-FileHash -LiteralPath $pytestConfig -Algorithm SHA256).Hash.ToLowerInvariant())",
        "python -m pytest $($pytestArgs -join ' ')"
    ) | Set-Content -LiteralPath (Join-Path $reportDir 'pytest-selection.txt') -Encoding utf8
    Copy-Item -LiteralPath $pytestConfig -Destination (Join-Path $reportDir 'pytest.ini') -Force

    # Launch outside the checkout so cwd cannot shadow the installed package.
    Push-Location $sitePackages
    try {
        & $python -c "import pathlib,pyscf,sys; p=pathlib.Path(pyscf.__file__).resolve(); source=pathlib.Path(r'$RepoRoot').resolve(); print(p); assert 'site-packages' in str(p); assert all(pathlib.Path(x or '.').resolve() != source for x in sys.path); from pyscf import gto,scf; from pyscf.dft import libxc,xcfun; assert libxc.__file__ and xcfun.__file__; assert abs(scf.RHF(gto.M(atom='H 0 0 0; H 0 0 1.4',unit='Bohr',basis='sto-3g',verbose=0)).kernel()+1.116714325062551)<1e-9" 2>&1 |
            Tee-Object -FilePath (Join-Path $reportDir 'import-smoke.log')
        if ($LASTEXITCODE -ne 0) {
            throw 'Installed-wheel import smoke failed'
        }
        & $python -m pytest @pytestArgs 2>&1 |
            Tee-Object -FilePath (Join-Path $reportDir 'pytest.log')
        $pytestExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        Remove-Item Env:PYSCF_CONFIG_FILE -ErrorAction SilentlyContinue
    }
    $pytestExitCode | Set-Content -LiteralPath (Join-Path $reportDir 'pytest-exit-code.txt') -Encoding ascii
    $pytestSummary = Get-Content -LiteralPath (Join-Path $reportDir 'pytest.log') |
        Where-Object { $_ -match '\d+ (passed|failed|error|errors|skipped|deselected)' } |
        Select-Object -Last 1
    if (-not $pytestSummary) {
        $pytestSummary = 'pytest summary not found; inspect pytest.log'
    }
    $wheelBytes = (Get-Content -LiteralPath (Join-Path $reportDir 'wheel-size.txt') -Raw).Trim()
    $wheelHash = (Get-Content -LiteralPath (Join-Path $reportDir 'wheel-sha256.txt') -Raw).Trim()
    $summary = @(
        '# CI Windows summary',
        '',
        "- Wheel: $($wheels[0].Name)",
        "- Wheel bytes: $wheelBytes",
        "- Wheel SHA256: $wheelHash",
        '- pip check: passed',
        "- pytest exit code: $pytestExitCode",
        "- pytest: $pytestSummary",
        '- JUnit: pytest-results.xml'
    )
    $summaryPath = Join-Path $reportDir 'summary.md'
    $summary | Set-Content -LiteralPath $summaryPath -Encoding utf8
    if ($env:GITHUB_STEP_SUMMARY) {
        $summary | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Encoding utf8
    }
    if ($pytestExitCode -ne 0) {
        throw "Installed-wheel pytest failed with exit code $pytestExitCode"
    }
}
finally {
    if ($null -eq $savedPytestAddopts) {
        Remove-Item Env:PYTEST_ADDOPTS -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTEST_ADDOPTS = $savedPytestAddopts
    }
    Pop-Location
}
