#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

"""
Hong-Zhou Ye and Timothy C. Berkelbach, to be published.
"""

import numpy as np

from pyscf.pbc import tools
from pyscf import gto as mol_gto
from pyscf import lib
from pyscf.lib.parameters import BOHR


def get_cell_id_in_cellplusimag(cell, nimgs):
    Nimgs = nimgs*2+1
    natm = cell.natm
    i0 = Nimgs[0]//2 * np.prod(Nimgs[1:]) + \
            Nimgs[1]//2 * Nimgs[2] + Nimgs[2]//2
    return i0 * natm
def get_dist_mat(rs1, rs2, dmin=1e-16):
    d = (np.linalg.norm(rs1,axis=1)**2.)[:,None] + \
         np.linalg.norm(rs2,axis=1)**2. - 2.*np.dot(rs1,rs2.T)
    np.clip(d, dmin**2., None, out=d)
    return d**0.5
def suplat_by_Rcut(cell, uniq_atms, Rcuts, nimgs=None):
    """ Determine the atoms within certain range (specified by "Rcuts") from the reference unit cell.

    Args:
        cell (Cell obj):
            Defines the reference unit cell.
        uniq_atms (list):
            A list of symbols for unique atoms in cell.
        Rcuts (np.ndarray):
            n-by-n array where n=len(uniq_atms). (i,j) element is the cutoff distance for i-th and j-th uniq atoms.

    Return:
        atms_sup, Rs_sup: list of atom symbols and positions, which can be used directly by :func:`_build_supmol_`.
    """
    natm_uniq = len(uniq_atms)
    Rcut = Rcuts.max()
# build cell plus imgs
    if nimgs is None:
        b = cell.reciprocal_vectors(norm_to=1)
        heights_inv = np.linalg.norm(b, axis=1)
        nimgs = np.ceil(Rcut*heights_inv + 1.1).astype(int)
    cell_all = tools.cell_plus_imgs(cell, nimgs)
    Rs_all = cell_all.atom_coords()
    natm_all = cell_all.natm
    atms_all = np.asarray([cell_all.atom_symbol(ia) for ia in range(natm_all)])
# find atoms from the ref cell
    iatm0_ref = get_cell_id_in_cellplusimag(cell, nimgs)
    natm_ref = cell.natm
    atms_ref = np.asarray([cell.atom_symbol(ia) for ia in range(natm_ref)])
    Rs_ref = Rs_all[iatm0_ref:iatm0_ref+natm_ref]
    mask_ref = np.zeros(natm_all, dtype=bool)
    mask_ref[iatm0_ref:iatm0_ref+natm_ref] = True
# find all atoms that (1) outside ref cell and (2) within Rcut
    uniq_atm_ids_ref = [np.where(atms_ref==atm)[0] for atm in uniq_atms]
    atm_ref_uniq_ids = np.empty(natm_ref, dtype=int)
    for iatm in range(natm_uniq):
        atm_ref_uniq_ids[uniq_atm_ids_ref[iatm]] = iatm
    atm_to_keep = []
    for iatm in range(natm_uniq):
        atm1 = uniq_atms[iatm]
        atm1_ids = np.where(atms_all==atm1)[0]
        d_atm1_ref = get_dist_mat(Rs_all[atm1_ids], Rs_ref)
        d_atm1_ref[mask_ref[atm1_ids]] = Rcut*100 # exclude atms from ref cell
        mask_ = np.zeros(d_atm1_ref.shape[0], dtype=bool)
        for jatm_ref in range(natm_ref):
            jatm_uniq = atm_ref_uniq_ids[jatm_ref]
            np.logical_or(mask_, d_atm1_ref[:,jatm_ref]<Rcuts[iatm,jatm_uniq],
                          out=mask_)
        atm_to_keep.append(atm1_ids[mask_])
    atm_to_keep = np.sort(np.concatenate(atm_to_keep))

    atms_sup = np.concatenate([atms_ref, atms_all[atm_to_keep]])
    Rs_sup = np.vstack([Rs_ref, Rs_all[atm_to_keep]])

    return atms_sup, Rs_sup

def _build_supmol_(cell, atms, Rs):
    """ Build a gto.Mole object with atoms type and positions specified by "atms" and "Rs", basis info by "cell._basis".

    [TODO] Like :func:`_build_supcell_` in pyscf.pbc.tools, this function constructs supmol._env directly without calling supmol.build() method. This reserves the basis contraction coefficients defined in "_basis".
    """
    Rs_Ang = Rs * BOHR
    atom = [[atm, R] for atm,R in zip(atms,Rs_Ang)]
    supmol = mol_gto.Mole(atom=atom, basis=cell._basis, spin=None)
    supmol.build()
# add a few useful attributes
    def map2loc(m, dtype=np.int32):
        loc = np.cumsum([0]+[len(m_) for m_ in m])
        mp = np.concatenate(m)
        return np.asarray(mp, dtype=dtype), np.asarray(loc, dtype=dtype)
# Ls --> lat shift vector by atm index in supmol
    latvec = cell.lattice_vectors()
    rs = cell.atom_coords()
    ds = (Rs[:,None,:] - rs).reshape(-1,3)
    ts = np.round( np.linalg.solve(latvec.T, ds.T).T, 4 )
    ids_keep = np.where(abs(np.rint(ts)-ts).sum(axis=1) < 1e-6)[0]
    assert(len(ids_keep) == supmol.natm)
    Ts = ts[ids_keep]
    supmol._Ls = lib.dot(Ts, latvec)
# refsupatm_map/loc --> list of supmol atms corresponding to ref atm
    taus_ref = cell.atom_coords()
    taus = Rs - lib.dot(Ts, latvec)
    dtaus = get_dist_mat(taus_ref,taus)
    refsupatm_map = [np.where(dtaus[Iatm]<1e-6)[0] for Iatm in range(cell.natm)]
    supmol._refsupatm_map, supmol._refsupatm_loc = map2loc(refsupatm_map)
# refsupshl_map/loc --> list of supmol shls corresponding to ref shl
    refshlstart = np.concatenate([cell.aoslice_nr_by_atom()[:,0], [cell.nbas]])
    shlstart = np.concatenate([supmol.aoslice_nr_by_atom()[:,0], [supmol.nbas]])
    refsupshl_map = [None] * cell.nbas
    for Iatm in range(cell.natm):
        supatms = refsupatm_map[Iatm]
        for Ishl in range(*refshlstart[Iatm:Iatm+2]):
            Ishlshift = Ishl - refshlstart[Iatm]
            refsupshl_map[Ishl] = shlstart[supatms] + Ishlshift
    supmol._refsupshl_map, supmol._refsupshl_loc = map2loc(refsupshl_map)

    return supmol

def get_refuniq_map(cell):
    """
    Return:
        refuniqshl_map[Ish] --> the uniq shl "ISH" that corresponds to ref shl "Ish".
        uniq_atms: a list of unique atom symbols.
        uniq_bas: concatenate basis for all uniq atomsm, i.e.,
                    [*cell._basis[atm] for atm in uniq_atms]
        uniq_bas_loc: uniq bas loc by uniq atoms (similar to cell.ao_loc)
    """
# get uniq atoms that respect the order it appears in cell
    n = len(cell._basis.keys())
    uniq_atms = []
    for i in range(cell.natm):
        atm = cell.atom_symbol(i)
        if not atm in uniq_atms:
            uniq_atms.append(atm)
        if len(uniq_atms) == n:
            break
    natm_uniq = len(uniq_atms)
# get uniq basis
    uniq_bas = [bas for ATM in uniq_atms for bas in cell._basis[ATM]]
    uniq_bas_loc = np.cumsum([0]+[len(cell._basis[ATM]) for ATM in uniq_atms])
    atms = np.array([cell.atom_symbol(i) for i in range(cell.natm)])
    shlstart = np.concatenate([cell.aoslice_nr_by_atom()[:,0], [cell.nbas]])
    refuniqshl_map = np.empty(cell.nbas, dtype=int)
    for IATM in range(natm_uniq):
        Iatms = np.where(atms==uniq_atms[IATM])[0]
        for ISHL in range(*uniq_bas_loc[IATM:IATM+2]):
            Ishlshift = ISHL - uniq_bas_loc[IATM]
            refuniqshl_map[shlstart[Iatms]+Ishlshift] = ISHL
# format to int32 (for interfacing C code)
    refuniqshl_map = np.asarray(refuniqshl_map, dtype=np.int32)
    return refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc

def binary_search(xlo, xhi, xtol, ret_bigger, fcheck, args=None,
                  MAX_RESCALE=5, MAX_CYCLE=20, early_exit=True):
    if args is None: args = tuple()
# rescale xlo/xhi if necessary
    first_time = True
    count = 0
    while True:
        ylo = fcheck(xlo, *args)
        if not ylo:
            xlo_rescaled = count > 0
            break
        if ylo and first_time and early_exit:
            return xlo
        if first_time: first_time = False
        xlo *= 0.5
        if count > MAX_RESCALE:
            if ERR_HANDLE == "raise":
                raise RuntimeError
            else:
                return xlo
        count += 1
    if xlo_rescaled and xlo*2 < xhi:
        xhi = xlo * 2
    else:
        count = 0
        while True:
            yhi = fcheck(xhi, *args)
            if yhi:
                xhi_rescaled = count > 0
                break
            xhi *= 1.5
            if count > MAX_RESCALE:
                raise RuntimeError
            count += 1
        if xhi_rescaled and xhi/1.5 > xlo:
            xlo = xhi / 1.5
# search
    cycle = 0
    while xhi-xlo > xtol:
        if cycle > MAX_CYCLE:
            raise RuntimeError
        cycle += 1
        xmi = 0.5*(xlo + xhi)
        fmi = fcheck(xmi, *args)
        if fmi:
            xhi = xmi
        else:
            xlo = xmi
    xret = xhi if ret_bigger else xlo
    return xret

def get_norm(a, axis=None):
    return np.linalg.norm(a, axis=axis)
