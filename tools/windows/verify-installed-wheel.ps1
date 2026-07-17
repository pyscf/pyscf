param(
    [string]$PythonExe = "",
    [string]$RuntimeDllDir = "",
    [string]$RepoRoot = "",
    [string]$ReportDir = "",
    [string[]]$TestRoots = @(),
    [string[]]$ExcludeTestRoots = @(),
    [switch]$SkipBuild,
    [switch]$SkipInstall,
    [switch]$KeepRunRoot,
    [switch]$SkipPbc
)

$ErrorActionPreference = "Stop"
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

function Resolve-RepoRoot {
    param([string]$ConfiguredValue)
    if ($ConfiguredValue) {
        return (Resolve-Path $ConfiguredValue).Path
    }
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Normalize-TestRootsBinding {
    param(
        [string]$RepoRoot,
        [string]$ConfiguredPythonExe,
        [string[]]$ConfiguredTestRoots
    )

    $normalizedPythonExe = $ConfiguredPythonExe
    $normalizedTestRoots = @($ConfiguredTestRoots)

    if ($normalizedPythonExe -and -not $normalizedPythonExe.Trim().ToLowerInvariant().EndsWith(".exe")) {
        $candidatePath = if ([System.IO.Path]::IsPathRooted($normalizedPythonExe)) {
            $normalizedPythonExe
        }
        else {
            Join-Path $RepoRoot $normalizedPythonExe
        }

        if (Test-Path $candidatePath -PathType Container) {
            $normalizedTestRoots = @($normalizedTestRoots + $normalizedPythonExe)
            $normalizedPythonExe = ""
        }
    }

    return [pscustomobject]@{
        PythonExe = $normalizedPythonExe
        TestRoots = $normalizedTestRoots
    }
}

function Resolve-PythonExe {
    param([string]$ConfiguredValue)
    if ($ConfiguredValue) {
        $resolved = (Resolve-Path $ConfiguredValue).Path
        if (Test-Path $resolved -PathType Container) {
            throw "PythonExe resolved to a directory: $resolved. If you are passing multiple -TestRoots values through powershell -File, use comma-separated values such as -TestRoots 'pyscf\dft\test','pyscf\tdscf\test', or pass -PythonExe explicitly."
        }
        return $resolved
    }
    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPython) {
            return (Resolve-Path $condaPython).Path
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Python was not found on PATH. Activate the target conda environment or pass -PythonExe explicitly."
    }
    return $cmd.Source
}

function Invoke-ExternalCommandCapture {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    try {
        & $FilePath @ArgumentList 1> $stdoutPath 2> $stderrPath
        $exitCode = $LASTEXITCODE
        $stdout = @()
        $stderr = @()
        if (Test-Path $stdoutPath) {
            $stdout = @(Get-Content $stdoutPath -ErrorAction SilentlyContinue)
        }
        if (Test-Path $stderrPath) {
            $stderr = @(Get-Content $stderrPath -ErrorAction SilentlyContinue)
        }
        return [pscustomobject]@{
            ExitCode = $exitCode
            StdOut = $stdout
            StdErr = $stderr
            AllOutput = @($stdout + $stderr)
        }
    }
    finally {
        Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
        Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-PythonSnippetCapture {
    param(
        [string]$PythonExe,
        [string]$Code
    )
    $snippetPath = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".py")
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    try {
        [System.IO.File]::WriteAllText($snippetPath, $Code, $utf8NoBom)
        return Invoke-ExternalCommandCapture -FilePath $PythonExe -ArgumentList @($snippetPath)
    }
    finally {
        Remove-Item $snippetPath -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-Pytest {
    param([string]$PythonExe)
    $result = Invoke-ExternalCommandCapture -FilePath $PythonExe -ArgumentList @(
        "-m",
        "pytest",
        "--version"
    )
    if ($result.ExitCode -ne 0) {
        throw "pytest is not available in the target environment. Install pytest into that environment before running verify-installed-wheel.ps1. Output: $($result.AllOutput -join ' ')"
    }
    return ($result.AllOutput | Select-Object -Last 1).ToString().Trim()
}

function Write-PytestConfig {
    param(
        [string]$RunRoot
    )

    $pytestIni = Join-Path $RunRoot "pytest.ini"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($pytestIni, @(
        '[pytest]',
        'addopts = --import-mode=importlib',
        '  -k "not _high_cost and not _skip"',
        '  --ignore=examples',
        '  --ignore-glob="*_slow*.py"',
        '  --ignore-glob="*test_kproxy*.py"',
        '  --ignore-glob="*test_proxy*.py"',
        '  --ignore-glob="*test_bz*"',
        '  --ignore-glob="*pbc/cc/test/*test_h_*.py"',
        '  --ignore-glob="*test_ks_noimport*.py"'
    ), $utf8NoBom)
    return $pytestIni
}

function Expand-PathArguments {
    param(
        [string]$RepoRoot,
        [string[]]$Paths
    )

    $expanded = New-Object System.Collections.Generic.List[string]
    foreach ($pathValue in $Paths) {
        if (-not $pathValue) {
            continue
        }
        $candidatePath = if ([System.IO.Path]::IsPathRooted($pathValue)) {
            $pathValue
        }
        else {
            Join-Path $RepoRoot $pathValue
        }

        if ((-not (Test-Path $candidatePath)) -and $pathValue.Contains(",")) {
            foreach ($part in $pathValue.Split(',')) {
                $trimmed = $part.Trim()
                if ($trimmed) {
                    $expanded.Add($trimmed)
                }
            }
            continue
        }

        $expanded.Add($pathValue)
    }

    return @($expanded)
}

function Resolve-PathList {
    param(
        [string]$RepoRoot,
        [string[]]$Paths
    )
    $expandedPaths = Expand-PathArguments -RepoRoot $RepoRoot -Paths $Paths
    return $expandedPaths |
        ForEach-Object {
            if ([System.IO.Path]::IsPathRooted($_)) {
                (Resolve-Path $_).Path
            }
            else {
                (Resolve-Path (Join-Path $RepoRoot $_)).Path
            }
        }
}

function Get-LogicalPath {
    param(
        [string]$RepoRoot,
        [string]$TargetPath
    )
    return Get-RelativePath -BasePath $RepoRoot -TargetPath $TargetPath
}

function Stage-TestDirectory {
    param(
        [string]$SourceDirectory,
        [string]$RunRoot,
        [string]$RepoRoot
    )

    $relative = Get-LogicalPath -RepoRoot $RepoRoot -TargetPath $SourceDirectory
    $stageRoot = Join-Path $RunRoot "tests"
    $stagedDirectory = Join-Path $stageRoot (Sanitize-Name $relative)
    if (Test-Path -LiteralPath $stagedDirectory) {
        Remove-Item -LiteralPath $stagedDirectory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagedDirectory -Force | Out-Null
    Copy-Item -Path (Join-Path $SourceDirectory '*') -Destination $stagedDirectory -Recurse -Force

    return [pscustomobject]@{
        source_directory = $SourceDirectory
        staged_directory = $stagedDirectory
        logical_directory = $relative
    }
}

function Get-TestDirectories {
    param(
        [string]$RepoRoot,
        [string[]]$ConfiguredRoots,
        [string[]]$ExcludedRoots,
        [switch]$SkipPbc
    )
    if ($ConfiguredRoots.Count -gt 0) {
        $testDirs = Resolve-PathList -RepoRoot $RepoRoot -Paths $ConfiguredRoots
    }
    else {
        $testDirs = Get-ChildItem -Directory -Recurse (Join-Path $RepoRoot "pyscf") |
            Where-Object { $_.Name -eq 'test' } |
            Select-Object -ExpandProperty FullName
    }

    $excluded = @()
    if ($ExcludedRoots.Count -gt 0) {
        $excluded += Resolve-PathList -RepoRoot $RepoRoot -Paths $ExcludedRoots
    }
    if ($SkipPbc) {
        $excluded += (Resolve-Path (Join-Path $RepoRoot "pyscf\pbc")).Path
    }

    if ($excluded.Count -gt 0) {
        $excluded = $excluded |
            ForEach-Object { [System.IO.Path]::GetFullPath($_).TrimEnd('\', '/') } |
            Sort-Object -Unique
        $testDirs = $testDirs |
            Where-Object {
                $candidate = [System.IO.Path]::GetFullPath($_).TrimEnd('\', '/')
                -not ($excluded | Where-Object {
                    $candidate -eq $_ -or
                    $candidate.StartsWith($_ + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
                })
            }
    }

    return $testDirs | Sort-Object -Unique
}

function Invoke-BuildWheel {
    param(
        [string]$RepoRoot,
        [string]$PythonExe,
        [string]$RuntimeDllDir
    )
    $buildScript = Join-Path $RepoRoot "tools\windows\build-wheel.ps1"
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $buildScript,
        "-PythonExe", $PythonExe,
        "-Clean"
    )
    if ($RuntimeDllDir) {
        $args += @("-RuntimeDllDir", $RuntimeDllDir)
    }
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "build-wheel.ps1 failed"
    }
}

function Get-LatestWheel {
    param([string]$RepoRoot)
    $wheel = Get-ChildItem (Join-Path $RepoRoot "dist\pyscf-*.whl") |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $wheel) {
        throw "No wheel was found under dist/. Build the wheel first or omit -SkipBuild."
    }
    return $wheel
}

function Assert-WheelContents {
    param(
        [string]$WheelPath,
        [string]$ReportDir
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($WheelPath)
    try {
        $entries = @($archive.Entries | ForEach-Object { $_.FullName })
    }
    finally {
        $archive.Dispose()
    }

    New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
    $entries | Sort-Object | Set-Content -LiteralPath (Join-Path $ReportDir "wheel-contents.txt") -Encoding UTF8
    foreach ($name in @("libnp_helper.dll", "libcgto.dll", "libcint.dll", "libxc_itrf.dll", "libxcfun_itrf.dll", "libopenblas.dll")) {
        if ($entries -notcontains "pyscf/lib/$name") {
            throw "Wheel is missing pyscf/lib/$name"
        }
    }
    if (($entries -notcontains "pyscf/lib/libxc.dll") -and ($entries -notcontains "pyscf/lib/xc.dll")) {
        throw 'Wheel is missing pyscf/lib/libxc.dll or pyscf/lib/xc.dll'
    }
    if (($entries -notcontains "pyscf/lib/libxcfun.dll") -and ($entries -notcontains "pyscf/lib/xcfun.dll")) {
        throw 'Wheel is missing pyscf/lib/libxcfun.dll or pyscf/lib/xcfun.dll'
    }
}

function Install-Wheel {
    param(
        [string]$PythonExe,
        [string]$WheelPath
    )
    $result = Invoke-ExternalCommandCapture -FilePath $PythonExe -ArgumentList @(
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        $WheelPath
    )
    foreach ($line in $result.AllOutput) {
        Write-Host $line
    }
    if ($result.ExitCode -ne 0) {
        throw "Wheel installation failed: $WheelPath`n$($result.AllOutput -join [Environment]::NewLine)"
    }
}

function New-RunRoot {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $RunRoot = Join-Path $env:TEMP "pyscf-installed-wheel-$stamp"
    New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
    return $RunRoot
}

function Initialize-RunRoot {
    param([string]$RunRoot)
    $tmpDir = Join-Path $RunRoot "pyscftmpdir"
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
    $configPath = Join-Path $RunRoot ".pyscf_conf.py"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($configPath, @(
        'pbc_tools_pbc_fft_engine = "NUMPY+BLAS"',
        'scf_hf_SCF_mute_chkfile = True',
        'TMPDIR = "./pyscftmpdir"'
    ), $utf8NoBom)
}

function Sanitize-Name {
    param([string]$Value)
    return ($Value -replace '[\\/:*?"<>| ]', '__')
}

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )
    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFull += [System.IO.Path]::DirectorySeparatorChar
    }
    $baseUri = [System.Uri]$baseFull
    $targetUri = [System.Uri]$targetFull
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)
    $relative = [System.Uri]::UnescapeDataString($relativeUri.ToString())
    return $relative -replace '/', [System.IO.Path]::DirectorySeparatorChar
}

function Get-PytestSummary {
    param([string[]]$OutputLines)

    $summaryLine = $null
    $reversedOutput = @($OutputLines)
    [array]::Reverse($reversedOutput)
    foreach ($line in $reversedOutput) {
        if ($line -match '\bin\b' -and $line -match '(subtests passed|subtests failed|passed|failed|error|errors|skipped|deselected|warning|warnings|xfailed|xpassed)') {
            $summaryLine = $line.Trim()
            break
        }
    }

    $counts = [ordered]@{
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        deselected = 0
        warnings = 0
        xfailed = 0
        xpassed = 0
        subtests_passed = 0
        subtests_failed = 0
    }

    if ($summaryLine) {
        $pattern = '(?<value>\d+)\s+(?<label>subtests passed|subtests failed|passed|failed|errors|error|skipped|deselected|warnings|warning|xfailed|xpassed)'
        foreach ($match in [regex]::Matches($summaryLine, $pattern)) {
            $value = [int]$match.Groups['value'].Value
            $label = $match.Groups['label'].Value
            switch ($label) {
                'passed' { $counts.passed = $value }
                'failed' { $counts.failed = $value }
                'error' { $counts.errors = $value }
                'errors' { $counts.errors = $value }
                'skipped' { $counts.skipped = $value }
                'deselected' { $counts.deselected = $value }
                'warning' { $counts.warnings = $value }
                'warnings' { $counts.warnings = $value }
                'xfailed' { $counts.xfailed = $value }
                'xpassed' { $counts.xpassed = $value }
                'subtests passed' { $counts.subtests_passed = $value }
                'subtests failed' { $counts.subtests_failed = $value }
            }
        }
    }

    return [pscustomobject]@{
        summary_line = $summaryLine
        counts = [pscustomobject]$counts
    }
}

function Invoke-PytestDirectory {
    param(
        [string]$PythonExe,
        [string]$Directory,
        [string]$PytestIni,
        [string]$LogPath
    )
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-ExternalCommandCapture -FilePath $PythonExe -ArgumentList @(
        "-m",
        "pytest",
        $Directory,
        "-c",
        $PytestIni,
        "-q"
    )
    $timer.Stop()
    $result.AllOutput | Set-Content -Path $LogPath -Encoding UTF8
    $pytestSummary = Get-PytestSummary -OutputLines $result.AllOutput
    return [pscustomobject]@{
        directory = $Directory
        exit_code = $result.ExitCode
        duration_seconds = [math]::Round($timer.Elapsed.TotalSeconds, 3)
        log_path = $LogPath
        status = if ($result.ExitCode -eq 0) { "passed" } else { "failed" }
        pytest_summary = $pytestSummary.summary_line
        pytest_counts = $pytestSummary.counts
    }
}

function Write-TestProgress {
    param(
        [int]$CompletedCount,
        [int]$TotalCount,
        [string]$LogicalDirectory,
        [string]$Status,
        [string]$LogPath
    )
    $statusLabel = if ($Status -eq "passed") { "pass" } else { "fail" }
    $resolvedLogPath = if (Test-Path $LogPath) {
        (Resolve-Path $LogPath).Path
    }
    else {
        [System.IO.Path]::GetFullPath($LogPath)
    }
    Write-Host ("[{0}/{1}] {2} completed: {3}. Log: {4}" -f $CompletedCount, $TotalCount, $LogicalDirectory, $statusLabel, $resolvedLogPath)
}

function Write-FailureSummary {
    param(
        [string]$RepoRoot,
        [psobject[]]$Results
    )

    $failed = @($Results | Where-Object status -eq "failed")
    if ($failed.Count -eq 0) {
        return
    }

    Write-Host ""
    Write-Host ("Failed test directories: {0}" -f $failed.Count)
    foreach ($result in $failed) {
        $resolvedLogPath = if (Test-Path $result.log_path) {
            (Resolve-Path $result.log_path).Path
        }
        else {
            [System.IO.Path]::GetFullPath($result.log_path)
        }
        $logicalDirectory = Get-LogicalPath -RepoRoot $RepoRoot -TargetPath $result.directory
        Write-Host ("- {0} | Log: {1}" -f $logicalDirectory, $resolvedLogPath)
    }
    Write-Host "Verification completed with failed directories. See installed-wheel-report.md for details."
}

function Write-Reports {
    param(
        [string]$RepoRoot,
        [string]$ReportDir,
        [psobject[]]$Results,
        [string]$WheelPath,
        [string]$PythonExe,
        [string]$RunRoot,
        [string]$PytestVersion,
        [psobject]$ImportInfo
    )
    New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
    $reportStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $jsonPath = Join-Path $ReportDir "installed-wheel-report.json"
    $mdPath = Join-Path $ReportDir "installed-wheel-report.md"
    $datedJsonPath = Join-Path $ReportDir "installed-wheel-report-$reportStamp.json"
    $datedMdPath = Join-Path $ReportDir "installed-wheel-report-$reportStamp.md"
    $wheelRelativePath = Get-RelativePath -BasePath $RepoRoot -TargetPath $WheelPath
    $runRootLabel = [System.IO.Path]::GetFileName($RunRoot)

    $summary = [pscustomobject]@{
        generated_at = (Get-Date).ToString("s")
        wheel = $WheelPath
        python = $PythonExe
        pytest = $PytestVersion
        run_root = $RunRoot
        environment_name = $ImportInfo.env_name
        report_json = $jsonPath
        report_md = $mdPath
        archived_report_json = $datedJsonPath
        archived_report_md = $datedMdPath
        passed = @($Results | Where-Object status -eq "passed").Count
        failed = @($Results | Where-Object status -eq "failed").Count
        total = @($Results).Count
        import_info = $ImportInfo
        results = @(
            foreach ($result in $Results) {
                $resolvedLog = (Resolve-Path $result.log_path).Path
                [pscustomobject]@{
                    directory = $result.directory
                    staged_directory = $result.staged_directory
                    relative_directory = Get-LogicalPath -RepoRoot $RepoRoot -TargetPath $result.directory
                    exit_code = $result.exit_code
                    duration_seconds = $result.duration_seconds
                    log_path = $result.log_path
                    relative_log_path = Get-RelativePath -BasePath $RepoRoot -TargetPath $resolvedLog
                    status = $result.status
                    pytest_summary = $result.pytest_summary
                    pytest_counts = $result.pytest_counts
                }
            }
        )
    }
    $summaryJson = $summary | ConvertTo-Json -Depth 5
    $summaryJson | Set-Content -Path $jsonPath -Encoding UTF8
    $summaryJson | Set-Content -Path $datedJsonPath -Encoding UTF8

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Installed Wheel Verification Report")
    $lines.Add("")
    $lines.Add("- Generated at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
    $lines.Add("- Conda environment: $($ImportInfo.env_name)")
    $lines.Add("- pytest: $PytestVersion")
    $lines.Add("- Wheel: $wheelRelativePath")
    $lines.Add("- Run root: $runRootLabel")
    $lines.Add("- Report JSON: $(Get-RelativePath -BasePath $RepoRoot -TargetPath $jsonPath)")
    $lines.Add("- Report JSON (timestamped): $(Get-RelativePath -BasePath $RepoRoot -TargetPath $datedJsonPath)")
    $lines.Add("- Report Markdown: $(Get-RelativePath -BasePath $RepoRoot -TargetPath $mdPath)")
    $lines.Add("- Report Markdown (timestamped): $(Get-RelativePath -BasePath $RepoRoot -TargetPath $datedMdPath)")
    $lines.Add("- Summary: passed $($summary.passed) / total $($summary.total), failed $($summary.failed)")
    $lines.Add("")
    $lines.Add("## Installed Import")
    $lines.Add("")
    $lines.Add("- Conda environment: $($ImportInfo.env_name)")
    $lines.Add("- PySCF path: $($ImportInfo.package_path)")
    $lines.Add("- PySCF lib path: $($ImportInfo.lib_path)")
    foreach ($property in $ImportInfo.packages.PSObject.Properties) {
        $lines.Add("- $($property.Name)==$($property.Value)")
    }
    $lines.Add("")
    $lines.Add("## Per-Directory Results")
    $lines.Add("")
    $lines.Add("| Directory | Status | Seconds | Pytest Summary | Log |")
    $lines.Add("| --- | --- | ---: | --- | --- |")
    foreach ($result in $summary.results) {
        $pytestSummary = if ($result.pytest_summary) { $result.pytest_summary } else { "(no pytest summary captured)" }
        $lines.Add("| $($result.relative_directory) | $($result.status) | $($result.duration_seconds) | $pytestSummary | $($result.relative_log_path) |")
    }
    $lines | Set-Content -Path $mdPath -Encoding UTF8
    $lines | Set-Content -Path $datedMdPath -Encoding UTF8
}

try {
    $RepoRoot = Resolve-RepoRoot $RepoRoot
    $normalizedBinding = Normalize-TestRootsBinding -RepoRoot $RepoRoot -ConfiguredPythonExe $PythonExe -ConfiguredTestRoots $TestRoots
    $PythonExe = $normalizedBinding.PythonExe
    $TestRoots = @($normalizedBinding.TestRoots)
    $PythonExe = Resolve-PythonExe $PythonExe
    if (-not $ReportDir) {
        $ReportDir = Join-Path $RepoRoot "tools\windows\reports"
    }

    if (-not $SkipBuild) {
        Invoke-BuildWheel -RepoRoot $RepoRoot -PythonExe $PythonExe -RuntimeDllDir $RuntimeDllDir
    }

    $wheel = Get-LatestWheel -RepoRoot $RepoRoot
    Assert-WheelContents -WheelPath $wheel.FullName -ReportDir $ReportDir

    if (-not $SkipInstall) {
        Install-Wheel -PythonExe $PythonExe -WheelPath $wheel.FullName
    }

    $pytestVersion = Ensure-Pytest -PythonExe $PythonExe
    $testDirs = @(Get-TestDirectories -RepoRoot $RepoRoot -ConfiguredRoots $TestRoots -ExcludedRoots $ExcludeTestRoots -SkipPbc:$SkipPbc)
    if ($testDirs.Count -eq 0) {
        throw "No test directories were found under pyscf/."
    }

    $RunRoot = New-RunRoot
    $logsDir = Join-Path $ReportDir "logs"
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    Initialize-RunRoot -RunRoot $RunRoot
    $pytestIni = Write-PytestConfig -RunRoot $RunRoot

    $env:OMP_NUM_THREADS = "4"
    $env:OPENBLAS_NUM_THREADS = "4"
    $savedPythonPath = $env:PYTHONPATH
    $savedPyscfExtPath = $env:PYSCF_EXT_PATH
    $savedPytestAddopts = $env:PYTEST_ADDOPTS
    $savedPytestDisablePluginAutoload = $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYSCF_EXT_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTEST_ADDOPTS -ErrorAction SilentlyContinue
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

    $importInfo = $null
    $results = @()
    Push-Location $RunRoot
    try {
        $importResult = Invoke-PythonSnippetCapture -PythonExe $PythonExe -Code @'
import json
import importlib.metadata as metadata
import pathlib
import sys
import pyscf
import pyscf.lib
package_path = pathlib.Path(pyscf.__file__).resolve()
lib_path = pathlib.Path(pyscf.lib.__file__).resolve()
if "site-packages" not in {part.lower() for part in package_path.parts}:
    raise RuntimeError(f"pyscf was not imported from site-packages: {package_path}")
packages = {}
for name in ["pyscf", "numpy", "scipy", "h5py", "pytest", "pytest-cov", "pytest-timer", "geometric", "spglib", "pyberny"]:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
env_name = pathlib.Path(sys.executable).parent.name
print(json.dumps({"env_name": env_name, "package_path": str(package_path), "lib_path": str(lib_path), "packages": packages}, ensure_ascii=False))
'@
        if ($importResult.ExitCode -ne 0) {
            throw "Installed wheel import check failed: $($importResult.AllOutput -join ' ')"
        }
        $importInfo = ($importResult.AllOutput | Select-Object -Last 1 | ConvertFrom-Json)

        $totalTests = $testDirs.Count
        $completedCount = 0
        foreach ($dir in $testDirs) {
            $staged = Stage-TestDirectory -SourceDirectory $dir -RunRoot $RunRoot -RepoRoot $RepoRoot
            $logName = (Sanitize-Name $staged.logical_directory) + ".log"
            $logPath = Join-Path $logsDir $logName
            $pytestResult = Invoke-PytestDirectory `
                -PythonExe $PythonExe `
                -Directory $staged.staged_directory `
                -PytestIni $pytestIni `
                -LogPath $logPath
            $results += [pscustomobject]@{
                directory = $staged.source_directory
                staged_directory = $staged.staged_directory
                exit_code = $pytestResult.exit_code
                duration_seconds = $pytestResult.duration_seconds
                log_path = $pytestResult.log_path
                status = $pytestResult.status
                pytest_summary = $pytestResult.pytest_summary
                pytest_counts = $pytestResult.pytest_counts
            }
            $completedCount += 1
            Write-TestProgress `
                -CompletedCount $completedCount `
                -TotalCount $totalTests `
                -LogicalDirectory $staged.logical_directory `
                -Status $pytestResult.status `
                -LogPath $pytestResult.log_path
        }
    }
    finally {
        Pop-Location
        if ($null -ne $savedPythonPath) {
            $env:PYTHONPATH = $savedPythonPath
        }
        else {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        }
        if ($null -ne $savedPyscfExtPath) {
            $env:PYSCF_EXT_PATH = $savedPyscfExtPath
        }
        else {
            Remove-Item Env:PYSCF_EXT_PATH -ErrorAction SilentlyContinue
        }
        if ($null -ne $savedPytestAddopts) {
            $env:PYTEST_ADDOPTS = $savedPytestAddopts
        }
        else {
            Remove-Item Env:PYTEST_ADDOPTS -ErrorAction SilentlyContinue
        }
        if ($null -ne $savedPytestDisablePluginAutoload) {
            $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = $savedPytestDisablePluginAutoload
        }
        else {
            Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
        }
        if ((-not $KeepRunRoot) -and (Test-Path $RunRoot)) {
            Remove-Item $RunRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Reports `
        -RepoRoot $RepoRoot `
        -ReportDir $ReportDir `
        -Results $results `
        -WheelPath $wheel.FullName `
        -PythonExe $PythonExe `
        -RunRoot $RunRoot `
        -PytestVersion $pytestVersion `
        -ImportInfo $importInfo

    Write-FailureSummary -RepoRoot $RepoRoot -Results $results
    $failedCount = @($results | Where-Object status -eq "failed").Count
    if ($failedCount -gt 0) {
        throw "One or more test directories failed. See installed-wheel-report.md for details."
    }
}
finally {
    $Stopwatch.Stop()
    Write-Host ("Total verification time: {0:hh\:mm\:ss} ({1:N1} s)" -f $Stopwatch.Elapsed, $Stopwatch.Elapsed.TotalSeconds)
}
