#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import numpy
from pyscf import lib

libpbc = lib.load_library('libpbc')
def _fpointer(name):
    return ctypes.addressof(getattr(libpbc, name))

class PBCOpt:
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
        if cell.use_loose_rcut:
            rcut = cell.rcut_by_shells(precision)
            fn_set_rcut_cond = getattr(libpbc, 'PBCset_rcut_cond_loose')
        else:
            rcut = numpy.array([cell.bas_rcut(ib, precision)
                                for ib in range(cell.nbas)])
            fn_set_rcut_cond = getattr(libpbc, 'PBCset_rcut_cond')

        natm = ctypes.c_int(cell._atm.shape[0])
        nbas = ctypes.c_int(cell._bas.shape[0])
        fn_set_rcut_cond(self._this,
                         rcut.ctypes.data_as(ctypes.c_void_p),
                         cell._atm.ctypes.data_as(ctypes.c_void_p), natm,
                         cell._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                         cell._env.ctypes.data_as(ctypes.c_void_p))
        return self

    def del_rcut_cond(self):
        self._this.contents.fprescreen = _fpointer('PBCnoscreen')
        return self

    def __del__(self):
        try:
            libpbc.PBCdel_optimizer(ctypes.byref(self._this))
        except AttributeError:
            pass

class _CPBCOpt(ctypes.Structure):
    __slots__ = []
    _fields_ = [('rrcut', ctypes.c_void_p),
                ('rcut', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p)]
