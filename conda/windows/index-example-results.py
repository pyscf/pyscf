#!/usr/bin/env python
"""Index Windows installed-wheel example sweep results into SQLite.

Usage:
    python conda/windows/index-example-results.py --db .tmp/windows/index/examples.db --input .tmp/windows/examples/wheel-examples.jsonl --source-kind full
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


RULES_PATH = Path(__file__).with_name("example-triage-rules.json")


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            repo_root TEXT NOT NULL,
            git_commit TEXT,
            git_branch TEXT,
            python_exe TEXT,
            wheel_path TEXT,
            timeout_sec INTEGER,
            source_kind TEXT NOT NULL,
            source_files_json TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS example_results (
            run_id TEXT NOT NULL,
            file TEXT NOT NULL,
            resolved_script TEXT NOT NULL,
            status TEXT NOT NULL,
            returncode INTEGER,
            elapsed_sec REAL,
            stdout_tail TEXT,
            stderr_tail TEXT,
            first_error_line TEXT,
            error_fingerprint TEXT,
            category TEXT,
            directory_group TEXT,
            is_rerun INTEGER NOT NULL DEFAULT 0,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (run_id, file)
        );
        """
    )


def load_rules():
    return sorted(
        [rule for rule in json.loads(RULES_PATH.read_text(encoding="utf-8")) if rule.get("enabled", True)],
        key=lambda item: item.get("priority", 0),
        reverse=True,
    )


def generate_run_id():
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def git_output(repo_root, *args):
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def extract_first_error_line(stderr_tail, stdout_tail):
    combined = "\n".join(text for text in (stderr_tail, stdout_tail) if text)
    for line in combined.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if (
            "error" in lowered
            or "exception" in lowered
            or "traceback" in lowered
            or "failed" in lowered
            or "notimplementederror" in lowered
        ):
            return line
    for line in combined.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def directory_group(file_path):
    parts = Path(file_path).as_posix().split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return file_path


def _match_rule(rule, status, stderr_tail, stdout_tail, file_path):
    pattern = rule["pattern"]
    match_type = rule["match_type"]
    if match_type == "status_is":
        return status == pattern
    if match_type == "stderr_contains":
        return pattern in (stderr_tail or "")
    if match_type == "stdout_contains":
        return pattern in (stdout_tail or "")
    if match_type == "file_glob":
        return Path(file_path).match(pattern)
    return False


def _extract_missing_dep_name(text):
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    return ""


def _normalize_fingerprint_line(line):
    line = re.sub(r"0x[0-9A-Fa-f]+", "0xADDR", line)
    line = re.sub(r"\d+", "N", line)
    line = line.strip().lower().replace(" ", "_")
    return line[:120]


def classify_result(status, first_error_line, stderr_tail, stdout_tail, file_path, rules=None):
    rules = rules or load_rules()
    if status == "PASS":
        return "", "pass"
    if status == "MISSING_DEP":
        dep_name = _extract_missing_dep_name(stderr_tail or first_error_line or stdout_tail or "")
        return f"missing_dep:{dep_name or 'unknown'}", "optional_dep"
    for rule in rules:
        if _match_rule(rule, status, stderr_tail, stdout_tail, file_path):
            return rule["fingerprint"], rule["category"]
    if status == "FAIL":
        fingerprint_line = first_error_line or stderr_tail or stdout_tail or "unknown_fail"
        return _normalize_fingerprint_line(fingerprint_line), "unknown_fail"
    return status.lower(), status.lower()


def normalize_result_row(row, rules=None, is_rerun=False):
    stderr_tail = row.get("stderr_tail", "")
    stdout_tail = row.get("stdout_tail", "")
    first_error_line = extract_first_error_line(stderr_tail, stdout_tail)
    fingerprint, category = classify_result(
        row["status"],
        first_error_line,
        stderr_tail,
        stdout_tail,
        row["file"],
        rules=rules,
    )
    return {
        **row,
        "first_error_line": first_error_line,
        "error_fingerprint": fingerprint,
        "category": category,
        "directory_group": directory_group(row["file"]),
        "is_rerun": 1 if row.get("is_rerun", is_rerun) else 0,
    }


def insert_run(conn, *, run_id, repo_root, source_kind, inputs, python_exe="", wheel_path="", timeout_sec=0, notes=""):
    source_files_json = json.dumps([str(Path(path)) for path in inputs], ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO runs (
            run_id, created_at, repo_root, git_commit, git_branch, python_exe, wheel_path,
            timeout_sec, source_kind, source_files_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now_iso(),
            str(repo_root),
            git_output(repo_root, "rev-parse", "HEAD"),
            git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            python_exe,
            wheel_path,
            timeout_sec,
            source_kind,
            source_files_json,
            notes,
        ),
    )


def insert_result(conn, *, run_id, row):
    conn.execute(
        """
        INSERT INTO example_results (
            run_id, file, resolved_script, status, returncode, elapsed_sec, stdout_tail, stderr_tail,
            first_error_line, error_fingerprint, category, directory_group, is_rerun, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            row["file"],
            row["resolved_script"],
            row["status"],
            row.get("returncode"),
            row.get("elapsed_sec"),
            row.get("stdout_tail", ""),
            row.get("stderr_tail", ""),
            row.get("first_error_line", ""),
            row.get("error_fingerprint", ""),
            row.get("category", ""),
            row.get("directory_group", ""),
            int(bool(row.get("is_rerun", 0))),
            utc_now_iso(),
        ),
    )


def import_jsonl_run(*, db_path, inputs, source_kind, python_exe="", wheel_path="", timeout_sec=0, notes="", repo_root=None):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root or Path.cwd()).resolve()
    rules = load_rules()
    run_id = generate_run_id()
    rows = []
    for input_path in inputs:
        input_path = Path(input_path)
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        insert_run(
            conn,
            run_id=run_id,
            repo_root=repo_root,
            source_kind=source_kind,
            inputs=inputs,
            python_exe=python_exe,
            wheel_path=wheel_path,
            timeout_sec=timeout_sec,
            notes=notes,
        )
        is_rerun = source_kind == "rerun"
        for row in rows:
            insert_result(
                conn,
                run_id=run_id,
                row=normalize_result_row(row, rules=rules, is_rerun=is_rerun),
            )
        conn.commit()
    finally:
        conn.close()
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Index Windows example sweep results into SQLite")
    parser.add_argument("--db", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--source-kind", required=True, choices=("full", "chunked", "rerun"))
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--wheel-path", default="")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    run_id = import_jsonl_run(
        db_path=args.db,
        inputs=args.input,
        source_kind=args.source_kind,
        python_exe=args.python_exe,
        wheel_path=args.wheel_path,
        timeout_sec=args.timeout,
        notes=args.notes,
        repo_root=args.repo_root or None,
    )
    print(run_id)


if __name__ == "__main__":
    main()
