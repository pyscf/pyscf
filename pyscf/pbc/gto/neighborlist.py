#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import ctypes
import numpy as np
from pyscf import lib
from pyscf.lib import logger

libpbc = lib.load_library('libpbc')

class _CNeighborPair(ctypes.Structure):
    __slots__ = []
    _fields_ = [("nimgs", ctypes.c_int),
                ("Ls_list", ctypes.POINTER(ctypes.c_int)),
                ("q_cond", ctypes.POINTER(ctypes.c_double)),
                ("center", ctypes.POINTER(ctypes.c_double))]


class _CNeighborList(ctypes.Structure):
    __slots__ = []
    _fields_ = [("nish", ctypes.c_int),
                ("njsh", ctypes.c_int),
                ("nimgs", ctypes.c_int),
                ("pairs", ctypes.POINTER(ctypes.POINTER(_CNeighborPair)))]


class _CNeighborListOpt(ctypes.Structure):
    __slots__ = []
    _fields_ = [("nl", ctypes.POINTER(_CNeighborList)),
                ('fprescreen', ctypes.c_void_p)]


def build_neighbor_list_for_shlpairs(cell, cell1=None, Ls=None,
                                     ish_rcut=None, jsh_rcut=None, hermi=0,
                                     precision=None):
    '''
    Build the neighbor list of shell pairs for periodic calculations.

    Arguments:
        cell : :class:`pbc.gto.cell.Cell`
            The :class:`Cell` instance for the bra basis functions.
        cell1 : :class:`pbc.gto.cell.Cell`, optional
            The :class:`Cell` instance for the ket basis functions.
            If not given, both bra and ket basis functions come from cell.
        Ls : ``(*,3)`` array, optional
            The cartesian coordinates of the periodic images.
            Default is calculated by :func:`cell.get_lattice_Ls`.
        ish_rcut : (nish,) array, optional
            The cutoff radii of the shells for bra basis functions.
        jsh_rcut : (njsh,) array, optional
            The cutoff radii of the shells for ket basis functions.
        hermi : int, optional
            If :math:`hermi=1`, the task list is built only for
            the upper triangle of the matrix. Default is 0.
        precision : float, optional
            The integral precision. Default is :attr:`cell.precision`.
            If both ``ish_rcut`` and ``jsh_rcut`` are given,
            ``precision`` will be ignored.

    Returns: :class:`ctypes.POINTER`
        The C pointer of the :class:`NeighborList` structure.
    '''
    if cell1 is None:
        cell1 = cell
    if Ls is None:
        Ls = cell.get_lattice_Ls()
    Ls = np.asarray(Ls, order='C', dtype=float)
    nimgs = len(Ls)

    if hermi == 1 and cell1 is not cell:
        logger.warn(cell,
                    "Set hermi=0 because cell and cell1 are not the same.")
        hermi = 0

    ish_atm = np.asarray(cell._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell._env, order='C', dtype=float)
    nish = len(ish_bas)
    if ish_rcut is None:
        ish_rcut = cell.rcut_by_shells(precision=precision)
    assert nish == len(ish_rcut)

    if cell1 is cell:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
        if jsh_rcut is None:
            jsh_rcut = ish_rcut
    else:
        jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
        jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
        jsh_env = np.asarray(cell1._env, order='C', dtype=float)
        if jsh_rcut is None:
            jsh_rcut = cell1.rcut_by_shells(precision=precision)
    njsh = len(jsh_bas)
    assert njsh == len(jsh_rcut)

    nl = ctypes.POINTER(_CNeighborList)()
    libpbc.build_neighbor_list(
        ctypes.byref(nl),
        ish_atm.ctypes.data_as(ctypes.c_void_p),
        ish_bas.ctypes.data_as(ctypes.c_void_p),
        ish_env.ctypes.data_as(ctypes.c_void_p),
        ish_rcut.ctypes.data_as(ctypes.c_void_p),
        jsh_atm.ctypes.data_as(ctypes.c_void_p),
        jsh_bas.ctypes.data_as(ctypes.c_void_p),
        jsh_env.ctypes.data_as(ctypes.c_void_p),
        jsh_rcut.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nish),
        ctypes.c_int(njsh),
        Ls.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nimgs),
        ctypes.c_int(hermi),
    )
    return nl

def free_neighbor_list(nl):
    libpbc.del_neighbor_list(ctypes.byref(nl))

def neighbor_list_to_ndarray(cell, cell1, nl):
    '''
    Returns:
        Ls_list: (nLtot,) ndarray
            indices of Ls
        Ls_idx: (2 x nish x njsh,) ndarray
            starting and ending indices in Ls_list
    '''
    nish = cell.nbas
    njsh = cell1.nbas
    Ls_list = []
    Ls_idx = []
    nLtot = 0
    for i in range(nish):
        for j in range(njsh):
            pair = nl.contents.pairs[i*njsh+j]
            nL = pair.contents.nimgs
            nLtot += nL
            for iL in range(nL):
                idx = pair.contents.Ls_list[iL]
                Ls_list.append(idx)
            if nL > 0:
                Ls_idx.extend([nLtot-nL, nLtot])
            else:
                Ls_idx.extend([-1,-1])
    return np.asarray(Ls_list), np.asarray(Ls_idx)


class NeighborListOpt():
    def __init__(self, cell):
        self.cell = cell
        self.nl = None
        self._this = ctypes.POINTER(_CNeighborListOpt)()
        libpbc.NLOpt_init(ctypes.byref(self._this))

    def build(self, cell=None, cell1=None, Ls=None,
              ish_rcut=None, jsh_rcut=None,
              hermi=0, precision=None,
              set_nl=True, set_optimizer=True):
        if cell is None:
            cell = self.cell

        if (set_nl or set_optimizer) and self.nl is None:
            self.nl = build_neighbor_list_for_shlpairs(
                            cell, cell1=cell1, Ls=Ls,
                            ish_rcut=ish_rcut, jsh_rcut=jsh_rcut,
                            hermi=hermi, precision=precision)
            libpbc.NLOpt_set_nl(self._this, self.nl)

        if set_optimizer:
            libpbc.NLOpt_set_optimizer(self._this)

    def reset(self, free_nl=True):
        if self.nl is not None and free_nl:
            free_neighbor_list(self.nl)
        self.nl = None
        libpbc.NLOpt_reset(self._this)

    def __del__(self):
        self.reset()
        try:
            libpbc.NLOpt_del(ctypes.byref(self._this))
        except AttributeError:
            pass
