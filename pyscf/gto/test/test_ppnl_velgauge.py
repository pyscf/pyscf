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

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.dft import radi

from pyscf.gto.ppnl_velgauge import get_gth_pp_nl_velgauge, get_gth_pp_nl_velgauge_commutator, _ft_ao_cross

import pytest

libpbc = lib.load_library('libpbc')

@pytest.mark.parametrize('q', [np.array([0.1, 0.2, 0.3]), np.array([0., 0., 0.]), np.array([-2., 1., 0.5])])
def test_ppnl_velgauge(q):
    mol = gto.Mole()
    mol.atom = 'S 1. .5 .5; C .1 1.3 2.1'
    mol.basis = 'gth-szv'
    mol.pseudo = 'gth-pade'
    mol.build()

    refvals = get_gth_pp_nl_velgauge_slow(mol, q)
    testvals = get_gth_pp_nl_velgauge(mol, q)
    assert np.allclose(refvals, testvals, atol=1e-8)

@pytest.mark.parametrize('q', [np.array([0.1, 0.2, 0.3]), np.array([0., 0., 0.]), np.array([-2., 1., 0.5])])
def test_ppnl_velgauge_commutator(q):
    mol = gto.Mole()
    mol.atom = 'C 1. .5 .5; C .1 1.3 2.1'
    mol.basis = 'gth-szv'
    mol.pseudo = 'gth-pade'
    mol.build()

    refvals = get_gth_pp_nl_velgauge_commutator_slow(mol, q)
    testvals = get_gth_pp_nl_velgauge_commutator(mol, q)

    assert np.allclose(refvals, testvals, atol=3e-3)

def rpow_cross_intor_slow(r_power, fakemol, mol, q=np.zeros(3), origin=(0,0,0), with_rc=False):
    quad_grid = gen_grid.Grids(mol)
    quad_grid.level = 6
    quad_grid.build()
    coords = quad_grid.coords
    weights = quad_grid.weights

    ng = coords.shape[0]

    e_minusiqr = np.exp(-1j * coords @ q).ravel()

    mol_aovals = mol.eval_gto('GTOval', coords)
    fakemol_aovals = fakemol.eval_gto('GTOval', coords)

    fakemol_ao_rn_exp_iqr = np.zeros((ng, fakemol.nao), dtype=np.complex128)

    ao_loc = fakemol.ao_loc

    for i in range(fakemol.nbas):
        center_i = fakemol.bas_coord(i)
        r2_from_i = np.sum((coords - center_i.reshape(1,3))**2, axis=1)
        rn_from_i = r2_from_i**(r_power/2)
        aoslice = slice(ao_loc[i], ao_loc[i+1])
        fakemol_ao_rn_exp_iqr[:, aoslice] = (fakemol_aovals[:, aoslice] * \
                                             e_minusiqr[:, None] \
                                             * rn_from_i[:, None])

    if with_rc:
        ints = lib.einsum('gi, gj, g, gx -> xij', fakemol_ao_rn_exp_iqr, mol_aovals, weights, coords - np.asarray(origin).reshape(1,3))

    else:
        ints = lib.einsum('gi, gj, g -> ij', fakemol_ao_rn_exp_iqr, mol_aovals, weights)

    return ints

def get_gth_pp_nl_velgauge_slow(mol, q):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas
    ppnl_half = []
    r_powers = (0, 2, 4)
    for i, rpow in enumerate(r_powers):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            ppnl_half.append(rpow_cross_intor_slow(rpow, fakemol, mol, q=q))
        else:
            ppnl_half.append(None)
    fakemol._bas = _bas

    nao = mol.nao
    vppnl = np.zeros((nao, nao), dtype=np.complex128)
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao), dtype=np.complex128)
        for i in range(hl_dim):
            p0 = offset[i]
            if ppnl_half[i] is None:
                ilp[i] = 0.
            else:
                ilp[i] = ppnl_half[i][p0:p0+nd]
            offset[i] = p0 + nd
        vppnl += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
    return vppnl

def get_gth_pp_nl_velgauge_commutator_slow(mol, q):
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas

    ppnl_half = []
    ppnl_rc_half = []

    r_powers = (0, 2, 4)

    for i, rpow in enumerate(r_powers):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            ppnl_half.append(rpow_cross_intor_slow(rpow, fakemol, mol, q=q))
            ppnl_rc_half.append(rpow_cross_intor_slow(rpow, fakemol, mol, q=q, with_rc=True))
        else:
            ppnl_half.append(None)
            ppnl_rc_half.append(None)
    fakemol._bas = _bas

    nao = mol.nao
    vppnl_commutator = np.zeros((3, nao, nao), dtype=np.complex128)
    offset = [0] * 3
    for ib, hl in enumerate(hl_blocks):
        l = fakemol.bas_angular(ib)
        nd = 2 * l + 1
        hl_dim = hl.shape[0]
        ilp = np.empty((hl_dim, nd, nao), dtype=np.complex128)
        rc_ilp = np.empty((3, hl_dim, nd, nao), dtype=np.complex128)
        for i in range(hl_dim):
            p0 = offset[i]
            if ppnl_half[i] is None:
                ilp[i] = 0.
                rc_ilp[:,i] = 0.
            else:
                ilp[i] = ppnl_half[i][p0:p0+nd]
                rc_ilp[:,i] = ppnl_rc_half[i][:, p0:p0+nd]
            offset[i] = p0 + nd
        vppnl_commutator += np.einsum('xilp,ij,jlq->xpq', rc_ilp.conj(), hl, ilp)
        vppnl_commutator -= np.einsum('ilp,ij,xjlq->xpq', ilp.conj(), hl, rc_ilp)
    return vppnl_commutator

@pytest.mark.parametrize('q', [np.array([0.1, 0.2, 0.3]), np.array([0., 0., 0.]), np.array([-2., 1., 0.5])])
def test_rpow_cross_intor_slow(q):
    mol = gto.Mole()
    mol.atom = 'S 1. .5 .5; C .1 1.3 2.1'
    mol.basis = 'gth-szv'
    mol.pseudo = 'gth-pade'
    mol.build()
    from pyscf.pbc.gto.pseudo import pp_int
    from pyscf.df import incore
    fakemol, hl_blocks = pp_int.fake_cell_vnl(mol)
    hl_dims = np.array([len(hl) for hl in hl_blocks])
    _bas = fakemol._bas

    r_powers = (0, 2, 4)
    intors = ('GTO_ft_ovlp', 'GTO_ft_r2_origi', 'GTO_ft_r4_origi')
    intors_rc = ('GTO_ft_rc', 'GTO_ft_rc_r2_origi', 'GTO_ft_rc_r4_origi')

    for i, (rpow, intor, intor_rc) in enumerate( zip(r_powers, intors, intors_rc) ):
        fakemol._bas = _bas[hl_dims>i]
        if fakemol.nbas > 0:
            assert np.allclose(rpow_cross_intor_slow(rpow, fakemol, mol, q=q), 
                               _ft_ao_cross(intor, fakemol, mol, Gv=q.reshape(1,3)),
                               atol=1e-8)
            assert np.allclose(rpow_cross_intor_slow(rpow, fakemol, mol, q=q, with_rc=True),
                               _ft_ao_cross(intor_rc, fakemol, mol, Gv=q.reshape(1,3), comp=3),
                               atol=1e-7)

