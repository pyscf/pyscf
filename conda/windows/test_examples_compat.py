import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(module_name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class ExampleVerificationRoutingTests(unittest.TestCase):
    def test_installed_example_runner_passes_windows_verify_flag(self):
        module = load_module("run_installed_examples", "conda/windows/run-installed-examples.py")

        def fake_run(cmd, **kwargs):
            self.assertIn("--pyscf-verify-windows", cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
            code, stdout, stderr, timed_out = module.run_script(
                REPO_ROOT / "examples/gw/00-simple_gw.py",
                "python",
                REPO_ROOT,
                30,
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        self.assertFalse(timed_out)

    def test_verify_wheel_runner_passes_windows_verify_flag(self):
        module = load_module("verify_wheel", "conda/windows/verify-wheel.py")
        spec = module.ExampleSpec("examples/gw/00-simple_gw.py", "examples")

        def fake_run(cmd, **kwargs):
            self.assertIn("--pyscf-verify-windows", cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
            result = module._run_example(spec, repo_root=REPO_ROOT, python_exe="python")

        self.assertEqual(result.status, "PASS")


class ExampleManifestTests(unittest.TestCase):
    def test_api_drift_examples_are_no_longer_diagnostics(self):
        manifest = json.loads((REPO_ROOT / "conda/windows/verify-wheel-manifest.json").read_text(encoding="utf-8"))
        diagnostics = {entry["path"] for entry in manifest["diagnostics"]}
        examples = {entry["path"] for entry in manifest["examples"]}

        repaired = {
            "examples/1-advanced/033-constrained_dft.py",
            "examples/agf2/06-adc2_solver.py",
            "examples/gw/00-simple_gw.py",
        }

        self.assertTrue(repaired.issubset(examples))
        self.assertTrue(repaired.isdisjoint(diagnostics))


if __name__ == "__main__":
    unittest.main()
