import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VERIFY_WHEEL = REPO_ROOT / "tools" / "windows" / "verify-installed-wheel.ps1"


class VerifyInstalledWheelScriptTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("powershell") or shutil.which("pwsh"), "PowerShell is required")
    def test_staging_recreates_the_target_directory(self):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        command = r'''
$scriptPath = $env:STAGE_TEST_SCRIPT
$source = $env:STAGE_TEST_SOURCE
$runRoot = $env:STAGE_TEST_RUN_ROOT
$repoRoot = $env:STAGE_TEST_REPO_ROOT
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $scriptPath, [ref]$tokens, [ref]$errors)
foreach ($name in @('Get-LogicalPath', 'Get-RelativePath', 'Sanitize-Name', 'Stage-TestDirectory')) {
    $node = $ast.FindAll({
        param($item)
        $item -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq $name
    }, $true)[0]
    Invoke-Expression $node.Extent.Text
}
$first = Stage-TestDirectory -SourceDirectory $source -RunRoot $runRoot -RepoRoot $repoRoot
$sentinel = Join-Path $first.staged_directory 'sentinel.txt'
Set-Content -LiteralPath $sentinel -Value 'stale'
$null = Stage-TestDirectory -SourceDirectory $source -RunRoot $runRoot -RepoRoot $repoRoot
if (Test-Path -LiteralPath $sentinel) { throw 'stale staged file survived' }
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "repo" / "pyscf" / "demo" / "test"
            source.mkdir(parents=True)
            (source / "test_demo.py").write_text("pass\n", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "STAGE_TEST_SCRIPT": str(VERIFY_WHEEL),
                "STAGE_TEST_SOURCE": str(source),
                "STAGE_TEST_RUN_ROOT": str(root / "run"),
                "STAGE_TEST_REPO_ROOT": str(root / "repo"),
            })
            result = subprocess.run(
                [powershell, "-NoProfile", "-Command", command],
                capture_output=True, text=True, check=False, env=env,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_script_exists_and_reuses_build_wheel_entrypoint(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn("build-wheel.ps1", text)
        self.assertIn("param(", text)
        self.assertIn("[switch]$SkipBuild", text)

    def test_script_checks_pytest_before_running_suite(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn('"pytest"', text)
        self.assertIn('"--version"', text)
        self.assertIn("pytest is not available", text)
        self.assertIn("function Invoke-ExternalCommandCapture", text)

    def test_script_prefers_active_conda_python_when_available(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn('$env:CONDA_PREFIX', text)
        self.assertIn('Join-Path $env:CONDA_PREFIX "python.exe"', text)
        self.assertIn('Test-Path $condaPython', text)
        self.assertIn('function Normalize-TestRootsBinding', text)
        self.assertIn("PythonExe resolved to a directory", text)
        self.assertIn("comma-separated values", text)
        self.assertIn('function Expand-PathArguments', text)
        self.assertIn(".Split(',')", text)
        self.assertIn('Test-Path $candidatePath -PathType Container', text)

    def test_script_runs_from_temp_root_and_clears_pythonpath(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn("Remove-Item Env:PYTHONPATH", text)
        self.assertIn("Join-Path $env:TEMP", text)
        self.assertIn("Push-Location $RunRoot", text)
        self.assertIn("function Invoke-PythonSnippetCapture", text)
        self.assertIn("UTF8Encoding($false)", text)
        self.assertIn("PYTEST_DISABLE_PLUGIN_AUTOLOAD", text)
        self.assertIn("function Write-PytestConfig", text)

    def test_script_discovers_installed_test_directories(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn("function Stage-TestDirectory", text)
        self.assertIn("Get-ChildItem -Directory -Recurse (Join-Path $RepoRoot \"pyscf\")", text)
        self.assertIn("Where-Object { $_.Name -eq 'test' }", text)
        self.assertIn("Sort-Object -Unique", text)
        self.assertIn("function Get-RelativePath", text)
        self.assertNotIn("[System.IO.Path]::GetRelativePath", text)
        self.assertNotIn(".Replace($RepoRoot", text)
        self.assertIn("Copy-Item -Path (Join-Path $SourceDirectory '*')", text)
        self.assertIn("Join-Path $RunRoot \"tests\"", text)

    def test_script_installs_latest_wheel_and_writes_reports(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn("Get-ChildItem (Join-Path $RepoRoot \"dist\\pyscf-*.whl\")", text)
        self.assertIn('"pip"', text)
        self.assertIn('"install"', text)
        self.assertIn('"--force-reinstall"', text)
        self.assertIn("--no-deps", text)
        self.assertIn("installed-wheel-report.md", text)
        self.assertIn("installed-wheel-report.json", text)
        self.assertIn("installed-wheel-report-$reportStamp.md", text)
        self.assertIn("installed-wheel-report-$reportStamp.json", text)
        self.assertIn("pytest_summary", text)
        self.assertIn("Get-RelativePath -BasePath $RepoRoot -TargetPath $resolvedLog", text)
        self.assertNotIn('-PytestIni (Join-Path $RepoRoot "pytest.ini")', text)
        self.assertIn("staged_directory", text)
        self.assertIn("function Write-TestProgress", text)
        self.assertIn("function Write-FailureSummary", text)
        self.assertIn('Write-Host ("[{0}/{1}] {2} completed: {3}. Log: {4}"', text)
        self.assertIn('Write-Host ("Failed test directories: {0}" -f $failed.Count)', text)
        self.assertIn('Write-Host ("- {0} | Log: {1}" -f $logicalDirectory, $resolvedLogPath)', text)
        self.assertIn('Write-Host "Verification completed with failed directories. See installed-wheel-report.md for details."', text)
        self.assertIn("Resolve-Path $LogPath", text)
        self.assertIn("Write-TestProgress `", text)
        self.assertIn("Write-FailureSummary -RepoRoot $RepoRoot -Results $results", text)
        self.assertIn("-CompletedCount $completedCount", text)
        self.assertIn("-TotalCount $totalTests", text)
        self.assertNotIn('throw "One or more test directories failed. See installed-wheel-report.md for details."', text)

    def test_script_supports_optional_test_exclusions(self):
        text = VERIFY_WHEEL.read_text(encoding="utf-8")
        self.assertIn("[string[]]$ExcludeTestRoots", text)
        self.assertIn("[switch]$SkipPbc", text)
        self.assertIn("ExcludeTestRoots", text)
        self.assertIn("SkipPbc", text)
        self.assertIn('Join-Path $RepoRoot "pyscf\\pbc"', text)


class Win64PackageReadmeTests(unittest.TestCase):
    def test_readme_documents_verification_parameters(self):
        readme = (REPO_ROOT / "tools" / "windows" / "win64-package-readme.md").read_text(encoding="utf-8")
        self.assertIn("-TestRoots", readme)
        self.assertIn("-ExcludeTestRoots", readme)
        self.assertIn("-SkipBuild", readme)
        self.assertIn("-SkipInstall", readme)
        self.assertIn("-KeepRunRoot", readme)
        self.assertIn("-SkipPbc", readme)
        self.assertIn("installed-wheel-report-YYYYMMDD-HHMMSS.md", readme)
        self.assertIn("repository checkout", readme)
        self.assertIn("already-installed wheel", readme)
        self.assertIn("Unlike the upstream CI", readme)
        self.assertIn("source-tree pytest", readme)


if __name__ == "__main__":
    unittest.main()
