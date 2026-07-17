import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BUILD_WHEEL = REPO_ROOT / "tools" / "windows" / "build-wheel.ps1"
ENVIRONMENT = REPO_ROOT / "tools" / "windows" / "environment.yml"
GITATTRIBUTES = REPO_ROOT / ".gitattributes"


class BuildWheelLayoutTests(unittest.TestCase):
    def test_clean_build_removes_staged_dlls_before_rebuilding(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn(
            "Get-ChildItem -LiteralPath $LibDir -Filter '*.dll' -File -ErrorAction SilentlyContinue",
            text,
        )
        self.assertIn(
            "Get-ChildItem -LiteralPath $DepsBinDir -Filter '*.dll' -File -ErrorAction SilentlyContinue",
            text,
        )

    def test_environment_file_declares_python_313_and_build_frontend(self):
        text = ENVIRONMENT.read_text(encoding="utf-8")
        self.assertIn("python=3.13", text)
        self.assertIn("cmake<4", text)
        self.assertIn("ninja", text)
        self.assertIn("- build", text)

    def test_build_script_stages_runtime_and_support_dlls_into_pyscf_lib(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn("$LibDir = Join-Path $RepoRoot 'pyscf\\lib'", text)
        self.assertIn("$DepsBinDir = Join-Path $RepoRoot 'pyscf\\lib\\deps\\bin'", text)
        self.assertIn("libgcc_s_seh-1.dll", text)
        self.assertIn("libopenblas.dll", text)
        self.assertIn('"libcint.dll"', text)
        self.assertIn('"libxc.dll"', text)
        self.assertIn("Copy-Item -LiteralPath (Join-Path $RuntimeDllDir $name) -Destination (Join-Path $LibDir $name) -Force", text)
        self.assertIn("Copy-Item -LiteralPath $source -Destination (Join-Path $LibDir $name) -Force", text)

    def test_build_script_keeps_xcfun_enabled_like_upstream_release_builds(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn("libopenblas.dll.a", text)
        self.assertIn('"xcfun.dll"', text)
        self.assertIn('"libxcfun.dll"', text)
        self.assertIn("-DENABLE_XCFUN=ON", text)
        self.assertIn("-DBUILD_XCFUN=ON", text)
        self.assertNotIn("-DENABLE_XCFUN=OFF", text)
        self.assertNotIn("-DBUILD_XCFUN=OFF", text)

    def test_build_script_bootstraps_runtime_paths_without_hardcoded_ninja_dirs(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('$RuntimeDllDir,', text)
        self.assertIn('$BootstrapPaths = @(', text)
        self.assertIn("$env:PATH = $BootstrapPaths -join ';'", text)
        self.assertNotIn('CommonExtensions\\Microsoft\\CMake\\Ninja', text)
        self.assertNotIn('C:\\Program Files\\Ninja', text)

    def test_build_script_resolves_ninja_from_conda_environment(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('function Resolve-NinjaExe', text)
        self.assertIn('Join-Path $LibraryBin "ninja.exe"', text)
        self.assertIn('Join-Path $ScriptsDir "ninja.exe"', text)
        self.assertIn('$NinjaExe = Resolve-NinjaExe -LibraryBin $LibraryBin -ScriptsDir $ScriptsDir', text)
        self.assertIn('Install Ninja into the target conda environment', text)

    def test_build_script_resolves_runtime_dir_from_common_msys2_locations(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('function Resolve-RuntimeDllDir', text)
        self.assertIn('D:\\msys64\\ucrt64\\bin', text)
        self.assertIn('C:\\msys64\\ucrt64\\bin', text)
        self.assertIn('$RuntimeDllDir = Resolve-RuntimeDllDir -ConfiguredValue $RuntimeDllDir', text)

    def test_build_script_handles_conda_python_from_env_root(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('(Split-Path $PythonDir -Leaf) -ieq "Scripts"', text)
        self.assertIn('$EnvRoot = $PythonDir', text)

    def test_build_script_reports_elapsed_time(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('[System.Diagnostics.Stopwatch]::StartNew()', text)
        self.assertIn('$Stopwatch.Stop()', text)
        self.assertIn('Total build time:', text)

    def test_build_script_stages_support_dlls_from_deps_bin(self):
        text = BUILD_WHEEL.read_text(encoding="utf-8")
        self.assertIn('function Copy-SupportDlls', text)
        self.assertIn('[switch]$AllowMissing', text)
        self.assertIn('Join-Path $DepsBinDir $candidate', text)
        self.assertIn('Missing support DLL. Checked:', text)
        self.assertIn('Missing support DLLs will be retried after the first wheel build pass.', text)
        self.assertIn('Copy-Item -LiteralPath $source -Destination (Join-Path $LibDir $name) -Force', text)

    def test_gitattributes_forces_lf_for_patch_files(self):
        text = GITATTRIBUTES.read_text(encoding="utf-8")
        self.assertIn("*.patch text eol=lf", text)

    def test_cmakelists_keeps_xcfun_build_flags_without_patch_fallback(self):
        text = (REPO_ROOT / "pyscf" / "lib" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('PATCH_COMMAND git apply --reject ${PROJECT_SOURCE_DIR}/libxcfun.patch || true', text)
        self.assertNotIn('${CMAKE_COMMAND} -E true', text)
        self.assertIn('-DCMAKE_CXX_FLAGS=-DM_PI=3.14159265358979323846', text)


if __name__ == "__main__":
    unittest.main()
