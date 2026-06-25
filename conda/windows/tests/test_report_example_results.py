import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_MODULE_PATH = REPO_ROOT / "conda" / "windows" / "index-example-results.py"
REPORT_MODULE_PATH = REPO_ROOT / "conda" / "windows" / "report-example-results.py"


def load_module(path, module_name):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


def prepare_example_db(tmpdir):
    index_module = load_module(INDEX_MODULE_PATH, "index_example_results")
    tmpdir = Path(tmpdir)
    db_path = tmpdir / "examples.db"
    jsonl_path = write_jsonl(
        tmpdir / "results.jsonl",
        [
            {
                "file": "examples/a.py",
                "resolved_script": "examples/a.py",
                "status": "PASS",
                "returncode": 0,
                "elapsed_sec": 1.0,
                "stdout_tail": "",
                "stderr_tail": "",
            },
            {
                "file": "examples/pbc/b.py",
                "resolved_script": "examples/pbc/b.py",
                "status": "FAIL",
                "returncode": 1,
                "elapsed_sec": 8.5,
                "stdout_tail": "",
                "stderr_tail": "malloc failed in CVHFallocate_JKArray",
            },
            {
                "file": "examples/c.py",
                "resolved_script": "examples/c.py",
                "status": "TIMEOUT",
                "returncode": 124,
                "elapsed_sec": 120.0,
                "stdout_tail": "",
                "stderr_tail": "",
            },
        ],
    )
    run_id = index_module.import_jsonl_run(
        db_path=db_path,
        inputs=[jsonl_path],
        source_kind="full",
        repo_root=REPO_ROOT,
    )
    return db_path, run_id


class ReportRenderingTests(unittest.TestCase):
    def test_summary_reports_status_counts(self):
        report_module = load_module(REPORT_MODULE_PATH, "report_example_results")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path, run_id = prepare_example_db(tmpdir)
            text = report_module.render_summary(db_path=db_path, run_id=run_id)
        self.assertIn("PASS", text)
        self.assertIn("FAIL", text)
        self.assertIn("TIMEOUT", text)

    def test_clusters_group_by_fingerprint(self):
        report_module = load_module(REPORT_MODULE_PATH, "report_example_results")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path, run_id = prepare_example_db(tmpdir)
            text = report_module.render_clusters(db_path=db_path, run_id=run_id, status="FAIL")
        self.assertIn("memory_pressure:cvhfallocate_jkarray", text)

    def test_diff_reports_status_changes(self):
        index_module = load_module(INDEX_MODULE_PATH, "index_example_results")
        report_module = load_module(REPORT_MODULE_PATH, "report_example_results")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "examples.db"
            base_path = write_jsonl(
                tmpdir / "base.jsonl",
                [
                    {
                        "file": "examples/a.py",
                        "resolved_script": "examples/a.py",
                        "status": "FAIL",
                        "returncode": 1,
                        "elapsed_sec": 1.0,
                        "stdout_tail": "",
                        "stderr_tail": "TypeError: boom",
                    }
                ],
            )
            target_path = write_jsonl(
                tmpdir / "target.jsonl",
                [
                    {
                        "file": "examples/a.py",
                        "resolved_script": "examples/a.py",
                        "status": "PASS",
                        "returncode": 0,
                        "elapsed_sec": 1.0,
                        "stdout_tail": "",
                        "stderr_tail": "",
                    }
                ],
            )
            base_run = index_module.import_jsonl_run(db_path=db_path, inputs=[base_path], source_kind="full", repo_root=REPO_ROOT)
            target_run = index_module.import_jsonl_run(db_path=db_path, inputs=[target_path], source_kind="full", repo_root=REPO_ROOT)
            text = report_module.render_diff(db_path=db_path, base_run_id=base_run, target_run_id=target_run)
        self.assertIn("FAIL -> PASS", text)
        self.assertIn("examples/a.py", text)


class HistoricalSweepTests(unittest.TestCase):
    def test_historical_full_sweep_rebuilds_known_status_distribution(self):
        index_module = load_module(INDEX_MODULE_PATH, "index_example_results")
        report_module = load_module(REPORT_MODULE_PATH, "report_example_results")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "examples.db"
            run_id = index_module.import_jsonl_run(
                db_path=db_path,
                inputs=[
                    REPO_ROOT / ".tmp" / "wheel-examples-part1.jsonl",
                    REPO_ROOT / ".tmp" / "wheel-examples-part2.jsonl",
                ],
                source_kind="chunked",
                repo_root=REPO_ROOT,
            )
            summary = report_module.load_status_summary(db_path=db_path, run_id=run_id)
        self.assertEqual(summary["PASS"], 323)
        self.assertEqual(summary["FAIL"], 62)
        self.assertEqual(summary["TIMEOUT"], 25)
        self.assertEqual(summary["IMPORT_ERROR"], 4)
        self.assertEqual(summary["MISSING_DEP"], 41)
        self.assertEqual(summary["MISSING_FILE"], 1)


if __name__ == "__main__":
    unittest.main()
