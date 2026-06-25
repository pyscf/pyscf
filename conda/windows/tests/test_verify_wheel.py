"""Unit tests for the Windows wheel verification helpers.

Usage:
    python conda/windows/tests/test_verify_wheel.py
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile


MODULE_PATH = Path(__file__).resolve().parents[1] / "verify" / "verify-wheel.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location("verify_wheel", MODULE_PATH)
verify_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_wheel)


class VerifyWheelRunnerTest(unittest.TestCase):
    def test_load_verification_manifest_contains_expected_sections(self):
        manifest = verify_wheel.load_verification_manifest()
        self.assertIn("examples", manifest)
        self.assertIn("packaging", manifest)
        self.assertIn("diagnostics", manifest)

    def test_filter_cases_by_phase(self):
        cases = [
            verify_wheel.Case("a", "import", lambda: None),
            verify_wheel.Case("b", "smoke", lambda: None),
            verify_wheel.Case("c", "examples", lambda: None),
        ]
        filtered = verify_wheel.filter_cases(cases, "smoke")
        self.assertEqual([case.name for case in filtered], ["b"])

    def test_summary_exit_code_fails_when_any_case_fails(self):
        results = [
            verify_wheel.Result("ok", "import", "PASS", 0.1, ""),
            verify_wheel.Result("bad", "smoke", "FAIL", 0.2, "boom"),
        ]
        summary = verify_wheel.build_summary(results)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["exit_code"], 1)
        self.assertEqual(summary["by_phase"]["import"]["PASS"], 1)
        self.assertEqual(summary["by_phase"]["smoke"]["FAIL"], 1)
        self.assertEqual(summary["failures"][0]["phase"], "smoke")

    def test_default_cases_cover_import_and_smoke(self):
        cases = verify_wheel.default_cases(repo_root="D:/codex-workspace/pyscf")
        names = {case.name for case in cases}
        self.assertIn("wheel_artifact", names)
        self.assertIn("import_pyscf", names)
        self.assertIn("import_core_modules", names)
        self.assertIn("minimal_rhf", names)
        self.assertIn("minimal_dft", names)
        self.assertIn("minimal_df", names)
        self.assertIn("momgfccsd_export", names)

    def test_example_manifest_contains_momgf_and_df_cases(self):
        manifest = verify_wheel.example_manifest()
        self.assertIn("examples/cc/50-simple_momgfccsd.py", manifest)
        self.assertIn("examples/cc/51-momgfccsd_hermiticity.py", manifest)
        self.assertIn("examples/df/00-with_df.py", manifest)
        self.assertIn("examples/dft/02-gks.py", manifest)

    def test_packaging_manifest_covers_historical_xcfun_and_momgf_cases(self):
        manifest = verify_wheel.packaging_regression_manifest()
        self.assertIn("examples/cc/52-momgfccsd_moment_input.py", manifest)
        self.assertIn("examples/cc/54-momgfccsd_self_energy.py", manifest)
        self.assertIn("examples/dft/12-camb3lyp.py", manifest)
        self.assertIn("examples/mcscf/13-restart.py", manifest)

    def test_diagnostic_manifest_is_currently_empty(self):
        manifest = verify_wheel.diagnostic_failure_manifest()
        self.assertEqual(manifest, {})
        self.assertNotIn("examples/gw/00-simple_gw.py", manifest)

    def test_build_summary_counts_by_status(self):
        results = [
            verify_wheel.Result("a", "examples", "PASS", 0.1, ""),
            verify_wheel.Result("b", "examples", "SKIP", 0.1, "x"),
            verify_wheel.Result("c", "examples", "FAIL", 0.1, "y"),
        ]
        summary = verify_wheel.build_summary(results)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["by_phase"]["examples"]["PASS"], 1)
        self.assertEqual(summary["by_phase"]["examples"]["FAIL"], 1)
        self.assertEqual(summary["by_phase"]["examples"]["SKIP"], 1)

    def test_format_summary_includes_phase_breakdown_and_failure_count(self):
        results = [
            verify_wheel.Result("artifact_ok", "artifact", "PASS", 0.1, ""),
            verify_wheel.Result("import_bad", "import", "FAIL", 0.2, "missing dependency"),
        ]
        summary = verify_wheel.build_summary(results)
        rendered = verify_wheel.format_summary(summary)
        self.assertIn("Summary:", rendered)
        self.assertIn("artifact", rendered)
        self.assertIn("import", rendered)
        self.assertIn("Failures:", rendered)

    def test_normalize_success_output_drops_verbose_stdout(self):
        self.assertEqual(verify_wheel.normalize_success_output("line1\nline2"), "")

    def test_merge_output_prefers_stderr_but_keeps_stdout(self):
        merged = verify_wheel.merge_process_output("stderr text", "stdout text")
        self.assertIn("stderr text", merged)
        self.assertIn("stdout text", merged)

    def test_assess_expected_failure_as_pass_when_pattern_matches(self):
        spec = verify_wheel.ExampleSpec(
            path="examples/gw/00-simple_gw.py",
            phase="diagnostics",
            expected_status="FAIL",
            expected_pattern="unexpected keyword argument",
        )
        result = verify_wheel.assess_example_outcome(
            spec,
            observed_status="FAIL",
            detail="TypeError: GWAC.kernel() got an unexpected keyword argument 'orbs'",
            elapsed_sec=1.2,
        )
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.observed_status, "FAIL")

    def test_assess_expected_failure_as_fail_when_pattern_mismatches(self):
        spec = verify_wheel.ExampleSpec(
            path="examples/agf2/06-adc2_solver.py",
            phase="diagnostics",
            expected_status="FAIL",
            expected_pattern="RADCIP",
        )
        result = verify_wheel.assess_example_outcome(
            spec,
            observed_status="FAIL",
            detail="AttributeError: module x has no attribute y",
            elapsed_sec=1.2,
        )
        self.assertEqual(result.status, "FAIL")

    def test_resolve_wheel_path_uses_latest_dist_wheel_when_path_not_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            dist_dir = repo_root / "dist"
            dist_dir.mkdir()
            older = dist_dir / "pyscf-2.13.0-py3-none-win_amd64.whl"
            newer = dist_dir / "pyscf-2.13.1-py3-none-win_amd64.whl"
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            resolved = verify_wheel.resolve_wheel_path(str(repo_root), "")
            self.assertEqual(Path(resolved), newer)

    def test_inspect_wheel_artifact_reports_missing_runtime_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_path = Path(tmpdir) / "pyscf-2.13.1-py3-none-win_amd64.whl"
            with ZipFile(wheel_path, "w") as archive:
                archive.writestr("pyscf/__init__.py", "")
                archive.writestr("pyscf/lib/libcgto.dll", "")
                archive.writestr("pyscf-2.13.1.dist-info/METADATA", "Metadata-Version: 2.1")
                archive.writestr("pyscf-2.13.1.dist-info/WHEEL", "Wheel-Version: 1.0")
                archive.writestr("pyscf-2.13.1.dist-info/RECORD", "")

            issues = verify_wheel.inspect_wheel_artifact(str(wheel_path))
            self.assertTrue(any("libxc.dll" in issue for issue in issues))
            self.assertTrue(any("libopenblas.dll" in issue for issue in issues))

    def test_inspect_wheel_artifact_accepts_expected_minimal_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_path = Path(tmpdir) / "pyscf-2.13.1-py3-none-win_amd64.whl"
            with ZipFile(wheel_path, "w") as archive:
                archive.writestr("pyscf/__init__.py", "")
                archive.writestr("pyscf/lib/libcgto.dll", "")
                archive.writestr("pyscf/lib/libxc_itrf.dll", "")
                archive.writestr("pyscf/lib/libxcfun_itrf.dll", "")
                archive.writestr("pyscf/lib/deps/bin/libcint.dll", "")
                archive.writestr("pyscf/lib/deps/bin/libxc.dll", "")
                archive.writestr("pyscf/lib/deps/bin/libxcfun.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libopenblas.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libgfortran-5.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libgcc_s_seh-1.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libgomp-1.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libquadmath-0.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libstdc++-6.dll", "")
                archive.writestr("pyscf/lib/deps/win64/bin/libwinpthread-1.dll", "")
                archive.writestr("pyscf-2.13.1.dist-info/METADATA", "Metadata-Version: 2.1")
                archive.writestr(
                    "pyscf-2.13.1.dist-info/WHEEL",
                    "Wheel-Version: 1.0\nTag: py3-none-win_amd64\n",
                )
                archive.writestr("pyscf-2.13.1.dist-info/RECORD", "")

            issues = verify_wheel.inspect_wheel_artifact(str(wheel_path))
            self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
