#!/usr/bin/env python
"""Run the full PySCF example suite against an installed Windows wheel.

Usage:
    python conda/windows/examples/run-installed-examples.py --repo-root . --examples-root ./examples --output .tmp/wheel-examples.jsonl
    python conda/windows/examples/run-installed-examples.py --repo-root . --examples-root ./examples --output .tmp/wheel-examples-part1.jsonl --limit 228
    python conda/windows/examples/run-installed-examples.py --repo-root . --examples-root ./examples --output .tmp/wheel-examples-part2.jsonl --start-at examples/mcscf/13-restart.py

The script always runs the example processes from outside the repository root
so that imports resolve to the installed package, not to the local source tree.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def verification_env():
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def iter_example_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if not path.name.startswith("."):
            yield path


def resolve_script(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return path
    if "\n" not in text and text.endswith(".py") and not text.startswith("#") and "import " not in text:
        target = (path.parent / text).resolve()
        if target.is_file():
            return target
    return path


def classify(stderr: str, stdout: str, returncode: int, timed_out: bool):
    text = f"{stdout}\n{stderr}"
    if timed_out:
        return "TIMEOUT"
    if returncode == 0:
        return "PASS"
    lowered = text.lower()
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "MISSING_DEP"
    if "importerror" in lowered:
        return "IMPORT_ERROR"
    if "file not found" in lowered or "filenotfounderror" in lowered:
        return "MISSING_FILE"
    return "FAIL"


def run_script(script, python_exe, cwd, timeout):
    code = (
        "import runpy, sys; "
        "script = sys.argv[1]; "
        "sys.argv = [script] + sys.argv[2:]; "
        'runpy.run_path(script, run_name="__main__")'
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", code, str(script), "--pyscf-verify-windows"],
            cwd=str(cwd),
            env=verification_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or "", True


def main():
    parser = argparse.ArgumentParser(description="Run installed-wheel example validation on Windows")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-at", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    examples_root = Path(args.examples_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    all_files = list(iter_example_files(examples_root))
    if args.start_at:
        start_at = args.start_at.replace("\\", "/")
        all_files = [
            path for path in all_files
            if path.relative_to(repo_root).as_posix() >= start_at
        ]
    if args.limit:
        all_files = all_files[:args.limit]

    results = []
    outside_repo = Path(repo_root.anchor)
    for index, path in enumerate(all_files, 1):
        rel = path.relative_to(repo_root).as_posix()
        script = resolve_script(path)
        started = time.time()
        returncode, stdout, stderr, timed_out = run_script(
            script,
            args.python_exe,
            outside_repo,
            args.timeout,
        )
        status = classify(stderr, stdout, returncode, timed_out)
        elapsed = round(time.time() - started, 2)
        result = {
            "index": index,
            "file": rel,
            "resolved_script": script.relative_to(repo_root).as_posix(),
            "status": status,
            "returncode": returncode,
            "elapsed_sec": elapsed,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
        results.append(result)
        print(f"[{index}/{len(all_files)}] {status:11} {elapsed:7.2f}s {rel}", flush=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")

    summary = {}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    print(json.dumps({"total": len(results), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
