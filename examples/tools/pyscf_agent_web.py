#!/usr/bin/env python

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Pyscf_agent.web import *  # noqa: F403


if __name__ == '__main__':
    raise SystemExit(serve())
