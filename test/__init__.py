from __future__ import division
import os
import sys
import subprocess
import unittest
from glob import glob


def test(verbosity=1, testdir=None, stream=sys.stdout, files=None):

    ts = unittest.TestSuite()
    if files:
        files = [os.path.join(__path__[0], f) for f in files]
    else:
        files = glob(__path__[0] + '/*')

    print(files)
