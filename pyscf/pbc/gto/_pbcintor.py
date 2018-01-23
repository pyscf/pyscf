#!/usr/bin/env python

import ctypes
import numpy
from pyscf import lib

libpbc = lib.load_library('libpbc')

class PBCOpt(object):
    def __init__(self, cell):
        self._this = ctypes.POINTER(_CPBCOpt)()
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        libpbc.PBCinit_optimizer(ctypes.byref(self._this),
                                 cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                 cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                 cell._env.ctypes.data_as(ctypes.c_void_p))

    def init_rcut_cond(self, cell, precision=None):
        if precision is None: precision = cell.precision
        rcut = numpy.array([cell.bas_rcut(ib, precision)
                            for ib in range(cell.nbas)])
        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        libpbc.PBCset_rcut_cond(self._this,
                                rcut.ctypes.data_as(ctypes.c_void_p),
                                cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                cell._env.ctypes.data_as(ctypes.c_void_p))
        return self

    def __del__(self):
        libpbc.PBCdel_optimizer(ctypes.byref(self._this))

class _CPBCOpt(ctypes.Structure):
    _fields_ = [('rrcut', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p)]

