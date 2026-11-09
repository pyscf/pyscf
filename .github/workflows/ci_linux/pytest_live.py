"""Run pytest with failure details printed before a CI timeout can hide them."""

import sys

import pytest


class ImmediateFailures:
    def pytest_runtest_logreport(self, report):
        if report.failed:
            print(f'\nFAILURE: {report.nodeid} [{report.when}]', flush=True)
            print(report.longreprtext, flush=True)


if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:], plugins=[ImmediateFailures()]))
