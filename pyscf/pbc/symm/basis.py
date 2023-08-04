#!/usr/bin/env python
# Copyright 2022-2023 The PySCF Developers. All Rights Reserved.
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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

'''
Symmetry adapted crystalline Gaussian orbitals for symmorphic space groups
'''
import numpy as np
from pyscf.pbc.symm import symmetry
from pyscf.pbc.symm.group import PGElement, PointGroup

def _symm_adapted_basis(cell, kpt_scaled, pg, spg_ops, Dmats, tol=1e-9):
    chartab = pg.character_table(return_full_table=True)
    dim = chartab[:,0]
    nirrep = len(chartab)
    nao = cell.nao
    atm_maps = []
    phases = []
    for op in spg_ops:
        atm_map, phase = symmetry._get_phase(cell, op, kpt_scaled)
        atm_maps.append(atm_map)
        phases.append(phase)
    atm_maps = np.asarray(atm_maps)
    tmp = np.unique(atm_maps, axis=0)
    tmp = np.sort(tmp, axis=0)
    tmp = np.unique(tmp, axis=1)
    eql_atom_ids = []
    for i in range(tmp.shape[-1]):
        eql_atom_ids.append(np.unique(tmp[:,i]))

    aoslice = cell.aoslice_by_atom()
    cbase = np.zeros((nirrep, nao, nao), dtype=np.complex128)
    for atom_ids in eql_atom_ids:
        for iatm in atom_ids:
            op_relate_idx = []
            for iop in range(pg.order):
                op_relate_idx.append(atm_maps[iop][iatm])
            ao_loc = np.asarray([aoslice[i,2] for i in op_relate_idx])

            b0, b1 = aoslice[iatm,:2]
            ioff = 0
            icol = aoslice[iatm, 2]
            for ib in range(b0, b1):
                nctr = cell.bas_nctr(ib)
                l = cell.bas_angular(ib)
                if cell.cart:
                    degen = (l+1) * (l+2) // 2
                else:
                    degen = l * 2 + 1

                for n in range(degen):
                    for iop in range(pg.order):
                        Dmat = Dmats[iop][l]
                        fac = dim/pg.order * chartab[:,iop].conj() * phases[iop][iatm]
                        tmp = np.einsum('x,y->xy', fac, Dmat[:,n])
                        idx = ao_loc[iop] + ioff
                        for ictr in range(nctr):
                            cbase[:, idx:idx+degen, icol+n+ictr*degen] += tmp
                            idx += degen
                ioff += degen * nctr
                icol += degen * nctr

    sos = []
    irrep_ids = []
    nso = 0
    for ir in range(nirrep):
        idx = np.where(np.sum(abs(cbase[ir]), axis=0) > tol)[0]
        so = cbase[ir][:,idx]
        if abs(so.imag).sum() < tol:
            so = so.real
        if so.shape[-1] > 0:
            so = _gram_schmidt(so)
            nso += so.shape[-1]
            sos.append(so)
            irrep_ids.append(ir)
    assert nso == cell.nao
    return sos, irrep_ids

def _gram_schmidt(v, tol=1e-9):
    ncol = v.shape[-1]
    u = np.zeros_like(v)
    u[:,0] = v[:,0] / np.linalg.norm(v[:,0])
    for k in range(1, ncol):
        uk = v[:,k]
        for j in range(k):
            uk = uk - np.dot(u[:,j].conj(), uk) * u[:,j]
        norm = np.linalg.norm(uk)
        if norm < tol:
            continue
        u[:,k] = uk / norm
    idx = np.where(np.sum(abs(u), axis=0) > tol)[0]
    u = u[:,idx]
    return u

def symm_adapted_basis(cell, kpts, tol=1e-9):
    sos_ks = []
    irrep_ids_ks = []
    Dmats = kpts.Dmats
    for i, ops in enumerate(kpts.little_cogroup_ops):
        kpt_scaled = kpts.kpts_scaled_ibz[i]
        elements = []
        for iop in ops:
            elements.append(kpts.ops[iop].rot)
        elements = np.asarray([PGElement(rot) for rot in elements])
        sort_idx = np.argsort(elements)
        Dmats_small = []
        spg_ops = []
        for iop in ops[sort_idx]:
            Dmats_small.append(Dmats[iop])
            spg_ops.append(kpts.ops[iop])
        elements = elements[sort_idx]
        pg = PointGroup(elements)
        sos, irrep_ids = _symm_adapted_basis(cell, kpt_scaled, pg, spg_ops, Dmats_small, tol)
        sos_ks.append(sos)
        irrep_ids_ks.append(irrep_ids)
    return sos_ks, irrep_ids_ks

if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = [['O' , (1. , 0.    , 0.   ,)],
                 ['H' , (0. , -.757 , 0.587,)],
                 ['H' , (0. , 0.757 , 0.587,)]]
    cell.a = [[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]
    cell.basis = 'ccpvdz'
    cell.verbose = 5
    cell.space_group_symmetry = True
    cell.symmorphic = True
    cell.build()
    kpts = cell.make_kpts([1,1,1], space_group_symmetry=True)
    so = symm_adapted_basis(cell, kpts)[0][0]

    from pyscf import gto as mol_gto
    from pyscf.symm import geom as mol_geom
    from pyscf.symm.basis import symm_adapted_basis as mol_symm_adapted_basis
    mol = cell.copy()
    gpname, origin, axes = mol_geom.detect_symm(mol._atom)
    atoms = mol_gto.format_atom(cell._atom, origin, axes)
    mol.build(False, False, atom=atoms)
    mol_so = mol_symm_adapted_basis(mol, gpname)[0]

    print(abs(so[0] - mol_so[0]).max())
    print(abs(so[1] - mol_so[1]).max())
    print(abs(so[2] - mol_so[3]).max())
    print(abs(so[3] - mol_so[2]).max())
