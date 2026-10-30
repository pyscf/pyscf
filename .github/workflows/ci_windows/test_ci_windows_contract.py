from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class CIWindowsContractTest(unittest.TestCase):
    def read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding='utf-8')

    def test_workflow_keeps_orchestration_small(self):
        workflow = self.read('.github/workflows/ci-windows.yml')
        self.assertIn('concurrency:', workflow)
        self.assertIn('cancel-in-progress: true', workflow)
        self.assertIn('python .github/workflows/ci_windows/test_ci_windows_contract.py', workflow)

    def test_build_records_hash_and_dist_info(self):
        build = self.read('.github/workflows/ci_windows/build_wheel.ps1')
        self.assertIn('Get-FileHash', build)
        self.assertIn('wheel-sha256.txt', build)
        for name in ('METADATA', 'WHEEL', 'RECORD'):
            self.assertIn(f'.dist-info/{name}', build)

    def test_verify_records_standard_evidence_without_nodeid_exceptions(self):
        verify = self.read('.github/workflows/ci_windows/verify_installed_wheel.ps1')
        for value in (
            'PYTEST_ADDOPTS',
            'pip check',
            '--junitxml',
            'pytest-results.xml',
            'pytest-selection.txt',
            'summary.md',
            'pyscf.dft import libxc,xcfun',
        ):
            self.assertIn(value, verify)
        self.assertNotIn('--deselect', verify)


if __name__ == '__main__':
    unittest.main()
