import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BUILD_WHEEL = REPO_ROOT / "conda" / "windows" / "build-wheel.ps1"
README = REPO_ROOT / "conda" / "windows" / "README.md"
CMAKELISTS = REPO_ROOT / "pyscf" / "lib" / "CMakeLists.txt"
SETUPPY = REPO_ROOT / "setup.py"


class BuildWheelScriptTests(unittest.TestCase):
    def test_build_script_exposes_parallel_level_parameter(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('[int]$ParallelLevel = 0', text)
        self.assertIn('$env:CMAKE_BUILD_PARALLEL_LEVEL = $ParallelLevel.ToString()', text)

    def test_build_script_reports_total_elapsed_time(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('[System.Diagnostics.Stopwatch]::StartNew()', text)
        self.assertIn('$elapsed.ToString("hh\\:mm\\:ss")', text)

    def test_build_script_checks_for_locked_output_dlls(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('function Assert-UnlockedBuildOutputs', text)
        self.assertIn('A running process is locking existing build outputs', text)
        self.assertIn('Assert-UnlockedBuildOutputs -RootDir (Join-Path $RepoRoot "pyscf\\lib")', text)

    def test_build_script_cleans_python_staging_dirs_without_full_clean(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('Remove-Item (Join-Path $RepoRoot "build\\lib") -Recurse -Force -ErrorAction SilentlyContinue', text)
        self.assertIn('Get-ChildItem (Join-Path $RepoRoot "build") -Directory -Filter "bdist.*"', text)

    def test_libxc_external_project_disables_build_testing(self):
        text = CMAKELISTS.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count('-DBUILD_TESTING=OFF'), 2)

    def test_readme_documents_parallel_level_build_command(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn('-ParallelLevel 8', text)
        self.assertIn('Use `-Clean` for a release rebuild or when you need to reset cached build state.', text)

    def test_readme_documents_result_indexing_layer(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("index-example-results.py", text)
        self.assertIn("report-example-results.py", text)
        self.assertIn(".tmp/windows/index/examples.db", text)

    def test_setup_build_py_reports_package_data_source_and_destination(self):
        text = SETUPPY.read_text(encoding="utf-8")
        self.assertIn('def build_package_data(self):', text)
        self.assertIn('Failed to stage package data during wheel packaging', text)
        self.assertIn('from {srcfile} to {target}', text)

    def test_setup_build_py_prints_traceback_for_run_failures(self):
        text = SETUPPY.read_text(encoding="utf-8")
        self.assertIn('import traceback', text)
        self.assertIn('traceback.print_exc()', text)
        self.assertIn("self.announce('build_py failed; printing traceback', level=4)", text)

    def test_setup_skips_copy_when_runtime_dll_source_matches_destination(self):
        text = SETUPPY.read_text(encoding="utf-8")
        self.assertIn('dst = os.path.join(dst_dir, dll_name)', text)
        self.assertIn('os.path.samefile(src, dst)', text)
        self.assertIn('if os.path.exists(dst) and os.path.samefile(src, dst):', text)


if __name__ == "__main__":
    unittest.main()
