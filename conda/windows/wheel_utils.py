"""Shared helpers for Windows wheel packaging verification."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile


ARTIFACT_NATIVE_LIBS = (
    "pyscf/lib/libcgto.dll",
    "pyscf/lib/libxc_itrf.dll",
    "pyscf/lib/libxcfun_itrf.dll",
    "pyscf/lib/deps/bin/libcint.dll",
    "pyscf/lib/deps/bin/libxc.dll",
    "pyscf/lib/deps/bin/libxcfun.dll",
)

ARTIFACT_RUNTIME_LIBS = (
    "pyscf/lib/deps/win64/bin/libgcc_s_seh-1.dll",
    "pyscf/lib/deps/win64/bin/libgomp-1.dll",
    "pyscf/lib/deps/win64/bin/libgfortran-5.dll",
    "pyscf/lib/deps/win64/bin/libopenblas.dll",
    "pyscf/lib/deps/win64/bin/libquadmath-0.dll",
    "pyscf/lib/deps/win64/bin/libstdc++-6.dll",
    "pyscf/lib/deps/win64/bin/libwinpthread-1.dll",
)

MANIFEST_PATH = Path(__file__).with_name("verify-wheel-manifest.json")


def build_summary(results):
    by_phase = {}
    failures = []
    for result in results:
        phase_counts = by_phase.setdefault(result.phase, {"PASS": 0, "FAIL": 0, "SKIP": 0})
        phase_counts[result.status] = phase_counts.get(result.status, 0) + 1
        if result.status == "FAIL":
            failures.append(
                {
                    "phase": result.phase,
                    "name": result.name,
                    "detail": result.detail,
                    "observed_status": result.observed_status,
                }
            )
    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "by_phase": by_phase,
        "failures": failures,
        "results": [
            {
                "name": result.name,
                "phase": result.phase,
                "status": result.status,
                "elapsed_sec": result.elapsed_sec,
                "detail": result.detail,
                "observed_status": result.observed_status,
            }
            for result in results
        ],
        "exit_code": 1 if failed else 0,
    }


def format_summary(summary):
    lines = [
        "Summary:",
        f"  passed={summary['passed']} failed={summary['failed']} skipped={summary['skipped']}",
        "  by phase:",
    ]
    for phase in sorted(summary["by_phase"]):
        counts = summary["by_phase"][phase]
        lines.append(
            f"    - {phase}: PASS={counts.get('PASS', 0)} FAIL={counts.get('FAIL', 0)} SKIP={counts.get('SKIP', 0)}"
        )
    if summary["failures"]:
        lines.append("Failures:")
        for failure in summary["failures"]:
            lines.append(f"  - [{failure['phase']}] {failure['name']}")
    else:
        lines.append("Failures: none")
    return "\n".join(lines)


def load_verification_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_wheel_path(repo_root, wheel_path):
    if wheel_path:
        path = Path(wheel_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"wheel not found: {path}")
        return str(path)

    dist_dir = Path(repo_root).resolve() / "dist"
    candidates = sorted(dist_dir.glob("pyscf-*.whl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no wheel found under {dist_dir}")
    return str(candidates[0])


def inspect_wheel_artifact(wheel_path):
    wheel_path = Path(wheel_path).resolve()
    issues = []
    with ZipFile(wheel_path) as archive:
        names = {entry.filename for entry in archive.infolist()}
        dist_info_dirs = sorted({name.split("/", 1)[0] for name in names if ".dist-info/" in name})
        if len(dist_info_dirs) != 1:
            issues.append(f"expected exactly one dist-info directory, found {len(dist_info_dirs)}")
        else:
            dist_info_dir = dist_info_dirs[0]
            for metadata_name in ("METADATA", "WHEEL", "RECORD"):
                metadata_path = f"{dist_info_dir}/{metadata_name}"
                if metadata_path not in names:
                    issues.append(f"missing metadata file: {metadata_path}")
            wheel_metadata = f"{dist_info_dir}/WHEEL"
            if wheel_metadata in names:
                wheel_text = archive.read(wheel_metadata).decode("utf-8", errors="replace")
                if "Tag: py3-none-win_amd64" not in wheel_text:
                    issues.append("wheel metadata missing expected Windows platform tag")

        for required_name in ARTIFACT_NATIVE_LIBS + ARTIFACT_RUNTIME_LIBS:
            if required_name not in names:
                issues.append(f"missing wheel payload: {required_name}")
    return issues
