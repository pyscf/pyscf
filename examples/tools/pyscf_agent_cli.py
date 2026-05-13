#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pyscf_agent_backend as backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the PySCF agent from the command line')
    parser.add_argument('request', nargs='?', help='User request text or JSON payload')
    parser.add_argument('--request-file', help='Path to a text file containing the request')
    parser.add_argument('--channel', default='cli', help='Structured message channel label')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON response')
    return parser


def load_request(args: argparse.Namespace) -> str:
    if args.request_file:
        return pathlib.Path(args.request_file).read_text(encoding='utf-8')
    if args.request:
        return args.request
    return backend.example_request()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = backend.execute_request(load_request(args), channel=args.channel)['final_report']
    if args.pretty:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
