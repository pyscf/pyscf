import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "conda" / "windows" / "index-example-results.py"


def load_module():
    if str(MODULE_PATH.parent) not in sys.path:
        sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location("index_example_results", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_table_names(conn):
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def fetch_column_names(conn, table_name):
    return {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})")
    }


def count_rows(db_path, table_name):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    finally:
        conn.close()


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


class IndexSchemaTests(unittest.TestCase):
    def test_schema_contains_runs_and_example_results_tables(self):
        module = load_module()
        conn = sqlite3.connect(":memory:")
        try:
            module.ensure_schema(conn)
            tables = fetch_table_names(conn)
        finally:
            conn.close()
        self.assertIn("runs", tables)
        self.assertIn("example_results", tables)

    def test_schema_contains_expected_result_columns(self):
        module = load_module()
        conn = sqlite3.connect(":memory:")
        try:
            module.ensure_schema(conn)
            columns = fetch_column_names(conn, "example_results")
        finally:
            conn.close()
        self.assertIn("run_id", columns)
        self.assertIn("file", columns)
        self.assertIn("status", columns)
        self.assertIn("error_fingerprint", columns)
        self.assertIn("category", columns)


class IndexImportTests(unittest.TestCase):
    def test_import_jsonl_creates_one_run_and_multiple_results(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
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
                        "elapsed_sec": 1.2,
                        "stdout_tail": "",
                        "stderr_tail": "",
                    },
                    {
                        "file": "examples/b.py",
                        "resolved_script": "examples/b.py",
                        "status": "FAIL",
                        "returncode": 1,
                        "elapsed_sec": 2.3,
                        "stdout_tail": "",
                        "stderr_tail": "TypeError: boom",
                    },
                ],
            )
            run_id = module.import_jsonl_run(
                db_path=db_path,
                inputs=[jsonl_path],
                source_kind="full",
            )
            self.assertTrue(run_id)
            self.assertEqual(count_rows(db_path, "runs"), 1)
            self.assertEqual(count_rows(db_path, "example_results"), 2)

    def test_import_captures_git_and_input_metadata(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
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
                        "elapsed_sec": 1.2,
                        "stdout_tail": "",
                        "stderr_tail": "",
                    }
                ],
            )
            run_id = module.import_jsonl_run(
                db_path=db_path,
                inputs=[jsonl_path],
                source_kind="rerun",
                python_exe="python.exe",
                wheel_path="dist/pyscf.whl",
                timeout_sec=120,
                repo_root=REPO_ROOT,
            )
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT run_id, source_kind, python_exe, wheel_path, timeout_sec, git_branch, git_commit, source_files_json FROM runs"
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(row[0], run_id)
            self.assertEqual(row[1], "rerun")
            self.assertEqual(row[2], "python.exe")
            self.assertEqual(row[3], "dist/pyscf.whl")
            self.assertEqual(row[4], 120)
            self.assertTrue(row[5])
            self.assertTrue(row[6])
            self.assertIn("results.jsonl", row[7])

    def test_rerun_source_marks_rows_as_rerun(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "examples.db"
            jsonl_path = write_jsonl(
                tmpdir / "results.jsonl",
                [
                    {
                        "file": "examples/a.py",
                        "resolved_script": "examples/a.py",
                        "status": "FAIL",
                        "returncode": 1,
                        "elapsed_sec": 1.2,
                        "stdout_tail": "",
                        "stderr_tail": "TypeError: boom",
                    }
                ],
            )
            module.import_jsonl_run(
                db_path=db_path,
                inputs=[jsonl_path],
                source_kind="rerun",
            )
            conn = sqlite3.connect(db_path)
            try:
                is_rerun = conn.execute(
                    "SELECT is_rerun FROM example_results"
                ).fetchone()[0]
            finally:
                conn.close()
        self.assertEqual(is_rerun, 1)


class IndexClassificationTests(unittest.TestCase):
    def test_missing_dep_extracts_module_name(self):
        module = load_module()
        row = module.normalize_result_row(
            {
                "file": "examples/geomopt/01-geometric.py",
                "resolved_script": "examples/geomopt/01-geometric.py",
                "status": "MISSING_DEP",
                "returncode": 1,
                "elapsed_sec": 1.0,
                "stdout_tail": "",
                "stderr_tail": "ModuleNotFoundError: No module named 'geometric'",
            }
        )
        self.assertEqual(row["error_fingerprint"], "missing_dep:geometric")
        self.assertEqual(row["category"], "optional_dep")

    def test_memory_pressure_maps_to_stable_fingerprint(self):
        module = load_module()
        row = module.normalize_result_row(
            {
                "file": "examples/pbc/x.py",
                "resolved_script": "examples/pbc/x.py",
                "status": "FAIL",
                "returncode": 1,
                "elapsed_sec": 30.0,
                "stdout_tail": "",
                "stderr_tail": "malloc failed in CVHFallocate_JKArray",
            }
        )
        self.assertEqual(row["error_fingerprint"], "memory_pressure:cvhfallocate_jkarray")
        self.assertEqual(row["category"], "memory_pressure")

    def test_directory_group_uses_top_level_example_bucket(self):
        module = load_module()
        row = module.normalize_result_row(
            {
                "file": "examples/pbc/20-k_points_scf.py",
                "resolved_script": "examples/pbc/20-k_points_scf.py",
                "status": "PASS",
                "returncode": 0,
                "elapsed_sec": 5.0,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
        self.assertEqual(row["directory_group"], "examples/pbc")

    def test_non_pass_row_is_not_rerun_without_rerun_source(self):
        module = load_module()
        row = module.normalize_result_row(
            {
                "file": "examples/scf/x.py",
                "resolved_script": "examples/scf/x.py",
                "status": "FAIL",
                "returncode": 1,
                "elapsed_sec": 2.0,
                "stdout_tail": "",
                "stderr_tail": "TypeError: boom",
            }
        )
        self.assertEqual(row["is_rerun"], 0)


if __name__ == "__main__":
    unittest.main()
