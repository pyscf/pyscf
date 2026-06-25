#!/usr/bin/env python
"""Render reports from indexed Windows installed-wheel example sweep results."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _connect(db_path):
    return sqlite3.connect(str(Path(db_path)))


def resolve_run_id(conn, run_id):
    if run_id != "latest":
        return run_id
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise ValueError("no runs indexed")
    return row[0]


def load_status_summary(*, db_path, run_id):
    conn = _connect(db_path)
    try:
        run_id = resolve_run_id(conn, run_id)
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM example_results WHERE run_id = ? GROUP BY status",
            (run_id,),
        ).fetchall()
        return {status: count for status, count in rows}
    finally:
        conn.close()


def render_summary(*, db_path, run_id):
    conn = _connect(db_path)
    try:
        run_id = resolve_run_id(conn, run_id)
        status_rows = conn.execute(
            "SELECT status, COUNT(*) FROM example_results WHERE run_id = ? GROUP BY status ORDER BY status",
            (run_id,),
        ).fetchall()
        dir_rows = conn.execute(
            "SELECT directory_group, COUNT(*) FROM example_results WHERE run_id = ? GROUP BY directory_group ORDER BY COUNT(*) DESC, directory_group LIMIT 10",
            (run_id,),
        ).fetchall()
        fingerprint_rows = conn.execute(
            """
            SELECT error_fingerprint, COUNT(*)
            FROM example_results
            WHERE run_id = ? AND error_fingerprint != ''
            GROUP BY error_fingerprint
            ORDER BY COUNT(*) DESC, error_fingerprint
            LIMIT 10
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    lines = [f"Run: {run_id}", "", "Status Summary:"]
    for status, count in status_rows:
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("Top Directory Groups:")
    for directory_group, count in dir_rows:
        lines.append(f"- {directory_group}: {count}")
    lines.append("")
    lines.append("Top Error Fingerprints:")
    for fingerprint, count in fingerprint_rows:
        lines.append(f"- {fingerprint}: {count}")
    return "\n".join(lines)


def render_clusters(*, db_path, run_id, status):
    conn = _connect(db_path)
    try:
        run_id = resolve_run_id(conn, run_id)
        rows = conn.execute(
            """
            SELECT error_fingerprint, category, COUNT(*)
            FROM example_results
            WHERE run_id = ? AND status = ?
            GROUP BY error_fingerprint, category
            ORDER BY COUNT(*) DESC, error_fingerprint
            """,
            (run_id, status),
        ).fetchall()
        lines = [f"Run: {run_id}", f"Status: {status}", "", "Clusters:"]
        for fingerprint, category, count in rows:
            lines.append(f"- {fingerprint} [{category}] x{count}")
            samples = conn.execute(
                """
                SELECT file
                FROM example_results
                WHERE run_id = ? AND status = ? AND error_fingerprint = ?
                ORDER BY file
                LIMIT 5
                """,
                (run_id, status, fingerprint),
            ).fetchall()
            for sample in samples:
                lines.append(f"  sample: {sample[0]}")
    finally:
        conn.close()
    return "\n".join(lines)


def render_diff(*, db_path, base_run_id, target_run_id):
    conn = _connect(db_path)
    try:
        base_run_id = resolve_run_id(conn, base_run_id)
        target_run_id = resolve_run_id(conn, target_run_id)
        base_rows = dict(
            conn.execute(
                "SELECT file, status FROM example_results WHERE run_id = ?",
                (base_run_id,),
            ).fetchall()
        )
        target_rows = dict(
            conn.execute(
                "SELECT file, status FROM example_results WHERE run_id = ?",
                (target_run_id,),
            ).fetchall()
        )
    finally:
        conn.close()
    rows = []
    for file_path in sorted(set(base_rows) | set(target_rows)):
        base_status = base_rows.get(file_path)
        target_status = target_rows.get(file_path)
        if (base_status or "") != (target_status or ""):
            rows.append((file_path, base_status, target_status))
    lines = [f"Base: {base_run_id}", f"Target: {target_run_id}", "", "Status Changes:"]
    grouped = {}
    for file_path, base_status, target_status in rows:
        label = f"{base_status or 'MISSING'} -> {target_status or 'MISSING'}"
        grouped.setdefault(label, []).append(file_path)
    for label, files in grouped.items():
        lines.append(f"- {label}")
        for file_path in files[:10]:
            lines.append(f"  sample: {file_path}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render reports from indexed Windows example sweep results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_summary = subparsers.add_parser("summary")
    parser_summary.add_argument("--db", required=True)
    parser_summary.add_argument("--run-id", default="latest")

    parser_clusters = subparsers.add_parser("clusters")
    parser_clusters.add_argument("--db", required=True)
    parser_clusters.add_argument("--run-id", default="latest")
    parser_clusters.add_argument("--status", required=True)

    parser_diff = subparsers.add_parser("diff")
    parser_diff.add_argument("--db", required=True)
    parser_diff.add_argument("--base", required=True)
    parser_diff.add_argument("--target", required=True)

    args = parser.parse_args()
    if args.command == "summary":
        print(render_summary(db_path=args.db, run_id=args.run_id))
    elif args.command == "clusters":
        print(render_clusters(db_path=args.db, run_id=args.run_id, status=args.status))
    elif args.command == "diff":
        print(render_diff(db_path=args.db, base_run_id=args.base, target_run_id=args.target))


if __name__ == "__main__":
    main()
