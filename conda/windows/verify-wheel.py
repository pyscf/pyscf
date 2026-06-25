#!/usr/bin/env python
"""Verify that an installed PySCF wheel works on Windows.

Usage:
    python conda/windows/verify-wheel.py --repo-root . --phase all
    python conda/windows/verify-wheel.py --repo-root . --phase packaging --output-json .tmp/verify-wheel-packaging.json

The script must be run from a Python interpreter that can import the installed
wheel. It always runs the example scripts from outside the repository root so
that imports resolve to the installed package instead of the source tree.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from wheel_utils import (
    build_summary,
    format_summary,
    inspect_wheel_artifact,
    load_verification_manifest,
    resolve_wheel_path,
)


@dataclass
class Case:
    name: str
    phase: str
    fn: object


@dataclass
class Result:
    name: str
    phase: str
    status: str
    elapsed_sec: float
    detail: str
    observed_status: str = ""


@dataclass
class ExampleSpec:
    path: str
    phase: str
    expected_status: str = "PASS"
    expected_pattern: str = ""


def _verification_env():
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _run_python_code(code, cwd=None, python_exe=None):
    python_exe = python_exe or sys.executable
    return subprocess.run(
        [python_exe, "-c", code],
        cwd=cwd,
        env=_verification_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _execute_case(case, repo_root=None, python_exe=None, wheel_path=None):
    started = time.perf_counter()
    try:
        detail = case.fn(repo_root=repo_root, python_exe=python_exe, wheel_path=wheel_path)
        status = "PASS"
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        status = "FAIL"
    return Result(
        name=case.name,
        phase=case.phase,
        status=status,
        elapsed_sec=round(time.perf_counter() - started, 3),
        detail=detail,
    )


def filter_cases(cases, phase):
    if phase == "all":
        return list(cases)
    return [case for case in cases if case.phase == phase]


def normalize_success_output(text):
    return ""


def merge_process_output(stderr_text, stdout_text):
    parts = [text.strip() for text in (stderr_text, stdout_text) if text and text.strip()]
    return "\n".join(parts)


def _outside_repo_cwd(repo_root):
    return str(Path(repo_root).resolve().anchor)


def _check_wheel_artifact(repo_root=None, python_exe=None, wheel_path=None):
    resolved = resolve_wheel_path(repo_root, wheel_path)
    issues = inspect_wheel_artifact(resolved)
    if issues:
        raise RuntimeError("\n".join(issues))
    return resolved


def _import_pyscf(repo_root=None, python_exe=None, wheel_path=None):
    code = "import pyscf; print(pyscf.__file__)"
    proc = _run_python_code(code, cwd=_outside_repo_cwd(repo_root), python_exe=python_exe)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def _import_core_modules(repo_root=None, python_exe=None, wheel_path=None):
    code = (
        "from pyscf import lib, gto, scf, dft, df, cc; "
        "from pyscf.cc import MomGFCCSD; "
        "from pyscf.dft import libxc; "
        'print("ok")'
    )
    proc = _run_python_code(code, cwd=_outside_repo_cwd(repo_root), python_exe=python_exe)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def _minimal_rhf(repo_root=None, python_exe=None, wheel_path=None):
    code = (
        'from pyscf import gto, scf; '
        'mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"); '
        "mf = scf.RHF(mol); "
        "e = mf.kernel(); "
        "print(round(e, 12))"
    )
    proc = _run_python_code(code, cwd=_outside_repo_cwd(repo_root), python_exe=python_exe)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def _minimal_dft(repo_root=None, python_exe=None, wheel_path=None):
    code = (
        'from pyscf import gto, dft; '
        'mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"); '
        "mf = dft.RKS(mol); "
        'mf.xc = "lda,vwn"; '
        "e = mf.kernel(); "
        "print(round(e, 12))"
    )
    proc = _run_python_code(code, cwd=_outside_repo_cwd(repo_root), python_exe=python_exe)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def _minimal_df(repo_root=None, python_exe=None, wheel_path=None):
    code = (
        'from pyscf import gto, scf; '
        'mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="sto-3g"); '
        "mf = scf.RHF(mol).density_fit(); "
        "e = mf.kernel(); "
        "print(round(e, 12))"
    )
    proc = _run_python_code(code, cwd=_outside_repo_cwd(repo_root), python_exe=python_exe)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def _momgfccsd_export(repo_root=None, python_exe=None, wheel_path=None):
    code = 'from pyscf.cc import MomGFCCSD; print(MomGFCCSD.__name__)'
    proc = _run_python_code(code, cwd=_outside_repo_cwd(repo_root), python_exe=python_exe)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def default_cases(repo_root):
    return [
        Case("wheel_artifact", "artifact", _check_wheel_artifact),
        Case("import_pyscf", "import", _import_pyscf),
        Case("import_core_modules", "import", _import_core_modules),
        Case("momgfccsd_export", "smoke", _momgfccsd_export),
        Case("minimal_rhf", "smoke", _minimal_rhf),
        Case("minimal_dft", "smoke", _minimal_dft),
        Case("minimal_df", "smoke", _minimal_df),
    ]


def example_manifest():
    return [entry["path"] for entry in load_verification_manifest()["examples"]]


def packaging_regression_manifest():
    return [entry["path"] for entry in load_verification_manifest()["packaging"]]


def diagnostic_failure_manifest():
    return {
        entry["path"]: entry["expected_pattern"]
        for entry in load_verification_manifest()["diagnostics"]
    }


def assess_example_outcome(spec, observed_status, detail, elapsed_sec):
    detail = detail.strip()
    if spec.expected_status == "PASS":
        status = "PASS" if observed_status == "PASS" else "FAIL"
        return Result(spec.path, spec.phase, status, elapsed_sec, detail, observed_status)

    pattern = spec.expected_pattern
    matched = observed_status == spec.expected_status and (not pattern or pattern in detail)
    if matched:
        message = f"expected failure matched: {pattern}" if pattern else "expected failure matched"
        return Result(spec.path, spec.phase, "PASS", elapsed_sec, message, observed_status)

    return Result(spec.path, spec.phase, "FAIL", elapsed_sec, detail, observed_status)


def _run_example(spec, repo_root=None, python_exe=None):
    repo_root = Path(repo_root).resolve()
    script = repo_root / spec.path
    if not script.exists():
        return Result(spec.path, spec.phase, "SKIP", 0.0, "example not found", "SKIP")

    code = (
        "import runpy, sys; "
        "script = sys.argv[1]; "
        "sys.argv = [script] + sys.argv[2:]; "
        'runpy.run_path(script, run_name="__main__")'
    )
    started = time.perf_counter()
    proc = subprocess.run(
        [python_exe or sys.executable, "-c", code, str(script), "--pyscf-verify-windows"],
        cwd=_outside_repo_cwd(repo_root),
        env=_verification_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 3)
    if proc.returncode != 0:
        detail = merge_process_output(proc.stderr, proc.stdout)
        return assess_example_outcome(spec, "FAIL", detail, elapsed)
    return assess_example_outcome(
        spec,
        "PASS",
        normalize_success_output((proc.stdout or "").strip()),
        elapsed,
    )


def _example_specs():
    return [ExampleSpec(path, "examples") for path in example_manifest()]


def _packaging_specs():
    return [ExampleSpec(path, "packaging") for path in packaging_regression_manifest()]


def _diagnostic_specs():
    return [
        ExampleSpec(path, "diagnostics", "FAIL", pattern)
        for path, pattern in diagnostic_failure_manifest().items()
    ]


def run_examples(repo_root, python_exe=None, phase="examples"):
    specs = []
    if phase in ("examples", "all"):
        specs.extend(_example_specs())
    if phase in ("packaging", "all"):
        specs.extend(_packaging_specs())
    if phase in ("diagnostics", "all"):
        specs.extend(_diagnostic_specs())
    return [
        _run_example(spec, repo_root=repo_root, python_exe=python_exe)
        for spec in specs
    ]


def main():
    parser = argparse.ArgumentParser(description="Verify an installed PySCF wheel on Windows")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--phase",
        default="all",
        choices=("artifact", "import", "smoke", "examples", "packaging", "diagnostics", "all"),
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--wheel-path", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    results = []
    cases = filter_cases(default_cases(args.repo_root), args.phase)
    results.extend(
        _execute_case(case, repo_root=args.repo_root, python_exe=args.python_exe, wheel_path=args.wheel_path)
        for case in cases
    )
    if args.phase in ("examples", "packaging", "diagnostics", "all"):
        results.extend(run_examples(args.repo_root, python_exe=args.python_exe, phase=args.phase))
    for result in results:
        print(f"[{result.phase}] {result.name}: {result.status}")
        if result.detail:
            print(result.detail)

    summary = build_summary(results)
    print(format_summary(summary))
    if args.output_json:
        import json

        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
