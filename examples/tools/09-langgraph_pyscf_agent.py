#!/usr/bin/env python

from __future__ import annotations

import pathlib
import sys


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pyscf_agent_backend import *  # noqa: F401,F403
from pyscf_agent_backend import main


if __name__ == '__main__':
    main()
