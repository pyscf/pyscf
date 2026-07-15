#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Analytic GTH-PP integrals for open boundary conditions.

See also pyscf/pbc/gto/pseudo/pp_int.py
'''

import numpy as np
from pyscf import lib
from pyscf.gto.mole import ATOM_OF, intor_cross

def get_gth_pp(mol):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore

    # Analytical integration for get_pp_loc_part1(cell).
    fakemol = pp_int.fake_cell_vloc(mol, 0)
    vpploc = 0
    if fakemol.nbas > 0:
        charges = fakemol.atom_charges()
        atmlst = fakemol._bas[:,ATOM_OF]
        v = incore.aux_e2(mol, fakemol, 'int3c2e', aosym='s2', comp=1)
        vpploc = np.einsum('pi,i->p', v, -charges[atmlst])

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    for cn in range(1, 5):
        fakemol = pp_int.fake_cell_vloc(mol, cn)
        if fakemol.nbas > 0:
            v = incore.aux_e2(mol, fakemol, intors[cn], aosym='s2', comp=1)
            vpploc += np.einsum('...i->...', v)

    if isinstance(vpploc, np.ndarray):
        vpploc = lib.unpack_tril(vpploc)

    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    intors = ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi')
    ppnl_half = _int_vnl(mol, fakemol, hl_blocks, intors)

    nao = mol.nao
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao))
        for i in range(hl_dim):
            p0 = offset[i]
            if ppnl_half[i] is None:
                ilp[i] = 0.
            else:
                ilp[i] = ppnl_half[i][p0:p0+nd]
            offset[i] = p0 + nd
        vpploc += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
    return vpploc

def _int_vnl(mol, fakemol, hl_blocks, intors, comp=None):
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas
    ppnl_half = []
    for i, intor in enumerate(intors):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            ppnl_half.append(intor_cross(intor, fakemol, mol, comp=comp))
        else:
            ppnl_half.append(None)
    fakemol._bas = _bas
    return ppnl_half

def vpploc_nuc_grad(mol, dm):
    '''
    Nuclear gradients of the local part (part2) of the GTH pseudo potential,
    contracted with the density matrix.
    '''
    from pyscf.pbc.gto import Cell
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list
    from pyscf.pbc.gto.cell import rcut_by_shells
    from pyscf.pbc.df.incore import int3c1e_nuc_grad

    # The gradients of get_pp_loc_part1(mol) are computed in grad.rhf.get_hcore
    # and grad.rhf.hcore_generator

    intors = ('int3c2e_ip1', 'int3c1e_ip1', 'int3c1e_ip1_r2_origk',
              'int3c1e_ip1_r4_origk', 'int3c1e_ip1_r6_origk')

    Ls = np.zeros((1, 3))
    as_cell = mol.view(Cell)
    as_cell.get_lattice_Ls = lambda *args: Ls
    as_cell.precision = 1e-12

    grad = np.zeros((mol.natm, 3))
    rcut = as_cell.rcut_by_shells()
    for cn in range(1, 5):
        fakecell = pp_int.fake_cell_vloc(mol, cn)
        fakecell.precision = 1e-8
        if fakecell.nbas > 0:
            neighbor_list = build_neighbor_list_for_shlpairs(
                as_cell, None, Ls, ish_rcut=rcut, jsh_rcut=rcut)
            grad += int3c1e_nuc_grad(
                as_cell, fakecell, dm, intors[cn], neighbor_list=neighbor_list)
            free_neighbor_list(neighbor_list)
    grad *= -2
    return grad

def vppnl_nuc_grad(mol, dm):
    '''
    Nuclear gradients of the non-local part of the GTH pseudo potential,
    contracted with the density matrix.
    '''
    from pyscf.pbc.gto.pseudo import pp_int
    assert dm.ndim == 2
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    ppnl_half = _int_vnl(mol, fakemol, hl_blocks,
                         ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi'))
    ppnl_half_ip2 = _int_vnl(
        mol, fakemol, hl_blocks,
        ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2'), comp=3)

    # int1e_ipovlp computes ip1 so multiply -1 to get ip2
    if len(ppnl_half_ip2[0]) > 0:
        ppnl_half_ip2[0] *= -1

    grad = np.zeros((mol.natm, 3))
    nao = dm.shape[-1]
    dppnl = np.zeros((3, nao, nao))
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]

        ilp = np.empty((hl_dim, nd, nao))
        dilp = np.empty((hl_dim, 3, nd, nao))
        for i in range(hl_dim):
            p0 = offset[i]
            if len(ppnl_half[i]) > 0:
                ilp[i] = ppnl_half[i][p0:p0+nd]
                dilp[i] = ppnl_half_ip2[i][:,p0:p0+nd]
            offset[i] = p0 + nd

        dppnl_i = lib.einsum('idlp,ij,jlq->dpq', dilp, hl, ilp)
        dppnl += dppnl_i

        i_pp_atom = fakemol._bas[ib, 0]
        grad[i_pp_atom] += np.einsum('dpq,qp->d', dppnl_i, dm)

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia,2:]
        grad[ia] -= np.einsum('dpq,qp->d', dppnl[:,p0:p1,:], dm[:,p0:p1])
    grad *= 2
    return grad
