#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
#

'''Analytic PP integrals for GTH/HGH PPs in velocity gauge.

For GTH/HGH PPs, see:
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)

For the velocity gauge transformation, see:
[1] Comparison of Length, Velocity, and Symmetric Gauges for the Calculation
    of Absorption and Electric Circular Dichroism Spectra with Real-Time
    Time-Dependent Density Functional Theory,
    Johann Mattiat and Sandra Luber
    Journal of Chemical Theory and Computation 2022 18 (9), 5513-5526,
    DOI: 10.1021/acs.jctc.2c00644
'''

import numpy as np
from pyscf import lib, __config__
from pyscf.lib import logger
from pyscf.pbc import gto as pgto
from pyscf.pbc.df.ft_ao import estimate_rcut, ExtendedMole, _RangeSeparatedCell
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl
from pyscf.pbc.tools import k2gamma


RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 1.0)
# kecut=10 can roughly converge GTO with alpha=0.5
KECUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_kecut_threshold', 10.0)


libpbc = lib.load_library('libpbc')


def get_gth_pp_nl_velgauge(cell, q, kpts=None, vgppnl_helper=None):
    """Nonlocal part of GTH pseudopotential in velocity gauge.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        System cell
    A_over_c : np.ndarray
        Scaled magnetic vector potential. Shape is (3,)
    kpts : np.ndarray, optional
        k-point list.
    vgppnl_helper : VelGaugePPNLHelper, optional
        Helper object for velocity gauge PP integrals. By default None
        which causes a new helper object to be created and built.
    Returns
    -------
    tuple
        (ppnl, vgppnl_helper) where ppnl is an ndarray of shape (nkpts, nao, nao)
        and vgppnl_helper is the VelGaugePPNLHelper object used to compute the integrals.
    """
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)

    q = -q.reshape(1,3)

    if vgppnl_helper is None:
        vgppnl_helper = VelGaugePPNLHelper(cell, kpts=kpts_lst)
        vgppnl_helper.build()


    #ppnl_half = _int_vnl_ft(cell, fakecell, hl_blocks, kpts_lst, q)
    ppnl_half = vgppnl_helper.int_vnl_ft(q)
    nao = cell.nao_nr()

    # ppnl_half could be complex, so _contract_ppnl will not work.
    # if gamma_point(kpts_lst):
    #     return _contract_ppnl(cell, fakecell, hl_blocks, ppnl_half, kpts=kpts)

    buf = np.empty((3*9*nao), dtype=np.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    ppnl = np.zeros((nkpts,nao,nao), dtype=np.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = np.ndarray((hl_dim,nd,nao), dtype=np.complex128, buffer=buf)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                offset[i] = p0 + nd
            ppnl[k] += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)

    if kpts is None or np.shape(kpts) == (3,):
        ppnl = ppnl[0]
    return ppnl

def get_gth_pp_nl_velgauge_commutator(cell, q, kpts=None, vgppnl_helper=None):
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)

    q = -q.reshape(1,3)

    if vgppnl_helper is None:
        vgppnl_helper = VelGaugePPNLHelper(cell, kpts=kpts_lst)
        vgppnl_helper.build()

    ppnl_half = vgppnl_helper.int_vnl_ft(q)
    ppnl_rc_half = vgppnl_helper.int_vnl_ft(q, rc=True)

    nao = cell.nao_nr()

    buf = np.empty((3*9*nao), dtype=np.complex128)
    buf2 = np.empty((3*3*9*nao), dtype=np.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    vppnl_commutator = np.zeros((nkpts, 3, nao, nao), dtype=np.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = np.ndarray((hl_dim, nd, nao), dtype=np.complex128, buffer=buf)
            rc_ilp = np.ndarray((3, hl_dim, nd, nao), dtype=np.complex128, buffer=buf2)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]
                rc_ilp[:, i] = ppnl_rc_half[i][k][:, p0:p0+nd]
                offset[i] = p0 + nd
            vppnl_commutator[k] += np.einsum('xilp,ij,jlq->xpq', rc_ilp.conj(), hl, ilp)
            vppnl_commutator[k] -= np.einsum('ilp,ij,xjlq->xpq', ilp.conj(), hl, rc_ilp)
    if kpts is None or np.shape(kpts) == (3,):
        vppnl_commutator = vppnl_commutator[0]
    return vppnl_commutator



class VelGaugePPNLHelper:
    """Helper class for evaluating velocity gauge pseudopotential non-local integrals.
       Useful to avoid recomputing data that only depends on the cell and k-points.
    """
    def __init__(self, cell, kpts=None, intors=None, hl_max=3, origin=(0.0, 0.0, 0.0)):
        if kpts is None:
            kpts_lst = np.zeros((1,3))
        else:
            kpts_lst = np.reshape(kpts, (-1,3))
        nkpts = len(kpts_lst)
        self.kpts = kpts_lst
        self.nkpts = nkpts
        self.cell = cell
        self.origin = origin
        self.fakecell = None
        self.hl_blocks = None
        intors =  ['GTO_ft_ovlp', 'GTO_ft_r2_origi', 'GTO_ft_r4_origi']
        comm_intors = ['GTO_ft_rc', 'GTO_ft_rc_r2_origi', 'GTO_ft_rc_r4_origi']
        self.intors = intors
        self.comm_intors = comm_intors
        self.ft_data = {}
        self.hl_max = hl_max
        self.hl_dims = None

    def build(self):
        fakecell, hl_blocks = fake_cell_vnl(self.cell)
        self.fakecell = fakecell
        self.hl_blocks = hl_blocks
        hl_dims = np.asarray([len(hl) for hl in hl_blocks])
        self.hl_dims = hl_dims

        for hl_idx, intor_name in zip(range(self.hl_max), self.intors):
            shls_slice, ft_kern, cell_conc_fakecell = prepare_ppnl_ft_data(
                self.cell, self.fakecell, hl_idx, self.hl_blocks, self.kpts,
                intor=intor_name, origin=self.origin, comp=1)
            self.ft_data[intor_name] = (shls_slice, ft_kern, cell_conc_fakecell)

        for hl_idx, intor_name in zip(range(self.hl_max), self.comm_intors):
            shls_slice, ft_kern, cell_conc_fakecell = prepare_ppnl_ft_data(
                self.cell, self.fakecell, hl_idx, self.hl_blocks, self.kpts,
                intor=intor_name, origin=self.origin, comp=3)
            self.ft_data[intor_name] = (shls_slice, ft_kern, cell_conc_fakecell)

    def int_vnl_ft(self, Gv, q=np.zeros(3), rc=False):
        if rc:
            comp = 3
            intors = self.comm_intors
        else:
            comp = 1
            intors = self.intors
        ft_data = list(self.ft_data[intor_name] for intor_name in intors)

        # Normally you only need one point in reciprocal space at a time.
        # (this corresponds to one value of the vector potential A)
        assert Gv.shape[0] == 1, "Gv must be a single vector"

        def int_ket(ft_data_this_hl):
            shls_slice, ft_kern, cell_conc_fakecell = ft_data_this_hl
            retv = ft_kern(Gv, None, None, q, self.kpts, shls_slice)
            # Gv is a single vector
            if comp == 1:
                retv = retv[:, 0]
            else:
                retv = retv[:, :, 0]
            return retv

        out = (int_ket(ft_data[0]),
            int_ket(ft_data[1]),
            int_ket(ft_data[2]))
        return out


def prepare_ppnl_ft_data(cell, fakecell, hl_idx, hl_blocks, kpts, intor, origin=(0.0, 0.0, 0.0), comp=1):
    """Prepare ft_kernel methods for fast evaluation of velocity gauge ppnl integrals

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        System cell
    fakecell : pyscf.pbc.gto.Cell
        Fake cell containing GTH projectors
    hl_idx : int
        GTH projector angular momentum index
    hl_blocks : list
        GTH hl blocks
    kpts : np.ndarray
        k-points
    intor : str
        GTO ft-ao integral name
    comp : int, optional
        Size of each integral (eg scalar=1, vector=3), by default 1

    Returns
    -------
    tuple
        shls_slice, ft_kern, cell_conc_fakecell
    """
    hl_dims = np.asarray([len(hl) for hl in hl_blocks])
    fakecell_trunc = fakecell.copy(deep=True)
    fakecell_trunc._bas = fakecell_trunc._bas[hl_dims>hl_idx]
    cell_conc_fakecell = pgto.conc_cell(cell, fakecell_trunc)
    intor = cell_conc_fakecell._add_suffix(intor)
    nbas_conc = cell_conc_fakecell.nbas
    shls_slice = (cell.nbas, nbas_conc, 0, cell.nbas)

    # It's necessary to cache this because get_lattice_Ls is slow.
    ft_kern = ft_aopair_kpts_kern(cell_conc_fakecell, aosym='s1', kptjs=kpts,
                                    intor=intor, comp=comp, origin=origin)
    return shls_slice, ft_kern, cell_conc_fakecell


def ft_aopair_kpts_kern(cell,
                        aosym='s1',
                        kptjs=np.zeros((1,3)),
                        intor='GTO_ft_ovlp',
                        comp=1,
                        bvk_kmesh=None,
                        origin=(0.0, 0.0, 0.0)):
    r'''
    Fourier transform AO pair for a group of k-points
    \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3

    Modified version of pyscf.pbc.df.ft_ao.ft_aopair_kpts that returns
    the generated ft kernel.
    '''
    log = logger.new_logger(cell)
    kptjs = np.asarray(kptjs, order='C').reshape(-1,3)

    rs_cell = _RangeSeparatedCell.from_cell(cell, KECUT_THRESHOLD,
                                            RCUT_THRESHOLD, log)
    if bvk_kmesh is None:
        bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kptjs)
        log.debug2('Set bvk_kmesh = %s', bvk_kmesh)
    rcut = estimate_rcut(rs_cell)
    supmol = ExtendedMole.from_cell(rs_cell, bvk_kmesh, rcut.max(), log)
    supmol = supmol.strip_basis(rcut)

    supmol.set_common_orig(origin)

    ft_kern = supmol.gen_ft_kernel(aosym, intor=intor, comp=comp,
                                   return_complex=True, verbose=log)

    return ft_kern
