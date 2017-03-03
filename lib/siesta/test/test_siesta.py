#!/usr/bin/env python

import os
import ctypes
import numpy
from pyscf import lib

# FIXME

libsiesta = lib.load_library('siesta')

if __name__ == '__main__':
    print('Full Tests for siesta')
    a = 1
    d = 1.0
    libsiesta(a,d)


