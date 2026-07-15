import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LINT_WORKFLOW = REPO_ROOT / '.github' / 'workflows' / 'lint.yml'


class WindowsToolsWorkflowTests(unittest.TestCase):
    def test_lint_workflow_runs_windows_tool_tests(self):
        text = LINT_WORKFLOW.read_text(encoding='utf-8')
        self.assertIn('windows-tools-tests:', text)
        self.assertIn('runs-on: windows-latest', text)
        self.assertIn(
            'python -m unittest discover -s tools/windows/tests -v', text)


if __name__ == '__main__':
    unittest.main()
