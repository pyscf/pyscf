import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TEST_ENV = REPO_ROOT / "tools" / "windows" / "environment-test.yml"
TEST_REQ = REPO_ROOT / "tools" / "windows" / "requirements-test.txt"


class WindowsTestEnvironmentYmlTests(unittest.TestCase):
    def test_test_environment_has_dedicated_name(self):
        text = TEST_ENV.read_text(encoding="utf-8")
        self.assertIn("name: pyscf-win313-test", text)

    def test_test_environment_includes_runtime_test_dependencies(self):
        text = TEST_ENV.read_text(encoding="utf-8")
        self.assertIn("python=3.13", text)
        self.assertIn("numpy!=1.16,!=1.17", text)
        self.assertIn("scipy!=1.5", text)
        self.assertIn("h5py", text)
        self.assertIn('pytest<9', text)
        self.assertIn("pytest-cov", text)
        self.assertIn("geometric", text)
        self.assertIn("spglib", text)

    def test_test_environment_includes_optional_test_helpers_from_ci(self):
        text = TEST_ENV.read_text(encoding="utf-8")
        self.assertNotIn("pip:", text)

    def test_test_environment_omits_unavailable_windows_py313_dispersion_wheel(self):
        text = TEST_ENV.read_text(encoding="utf-8")
        self.assertNotIn("pyscf-dispersion", text)

    def test_test_requirements_file_contains_pip_only_dependencies(self):
        text = TEST_REQ.read_text(encoding="utf-8")
        self.assertIn("pytest-timer", text)
        self.assertIn("git+https://github.com/jhrmnn/pyberny.git@36a4be9", text)
        self.assertNotIn("pyscf-dispersion", text)


if __name__ == "__main__":
    unittest.main()
