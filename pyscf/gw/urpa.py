#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

"""
Spin-unrestricted random phase approximation (direct RPA/dRPA in chemistry)
with N^4 scaling

Method:
    Main routines are based on GW-AC method described in:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    X. Ren et al., New J. Phys. 14, 053020 (2012)
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp import dfump2

from pyscf.gw.rpa import RPA

einsum = lib.einsum


def make_dielectric_matrix(omega, e_ov, f_ov, eris, blksize=None):
    '''
    Compute dielectric matrix at a given frequency omega

    Args:
        omega : float, frequency
        e_ov : 1D array (nocc * nvir), orbital energy differences
        eris : DF ERI object

    Returns:
        diel : 2D array (naux, naux), dielectric matrix
    '''
    assert blksize is not None

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux

    isreal = eris.dtype == np.float64

    diel = np.zeros((naux, naux), dtype=eris.dtype)

    for s in [0,1]:
        chi0 = (2.0 * e_ov[s] * f_ov[s] / (omega ** 2 + e_ov[s] ** 2)).ravel()
        for p0,p1 in lib.prange(0, nocc[s]*nvir[s], blksize):
            ovL = eris.get_ov_blk(s,p0,p1)
            ovL_chi = (ovL.T * chi0[p0:p1]).T
            if isreal:
                lib.ddot(ovL_chi.T, ovL, c=diel, beta=1)
            else:
                lib.dot(ovL_chi.T, ovL.conj(), c=diel, beta=1)
            ovL = ovL_chi = None

    return diel


class URPA(dfump2.DFUMP2):

    get_e_hf = RPA.get_e_hf
    kernel = RPA.kernel
    _finalize = RPA._finalize

    def make_e_ov(self):
        '''
        Compute orbital energy differences
        '''
        log = logger.new_logger(self)
        split_mo_energy = self.split_mo_energy()
        e_ov = [(split_mo_energy[s][1][:,None] - split_mo_energy[s][2]).ravel() for s in [0,1]]

        if self.nocc[1] > 0:
            gap = [-e_ov[s].max() for s in [0,1]]
            log.info('Lowest orbital energy difference: (% 6.4e, % 6.4e)', gap[0], gap[1])
        else:
            gap = (-e_ov[0].max(), )
            log.info('Lowest orbital energy difference: % 6.4e', np.min(gap))

        if (np.min(gap) < 1e-3):
            log.warn('RPA code is not well-defined for degenerate systems!')
            log.warn('Lowest orbital energy difference: % 6.4e', np.min(gap))

        return e_ov

    def make_f_ov(self):
        '''
        Compute orbital occupation number differences
        '''
        split_mo_occ = self.split_mo_occ()
        return [(split_mo_occ[s][1][:,None] - split_mo_occ[s][2]).ravel() for s in [0,1]]

    def make_dielectric_matrix(self, omega, e_ov=None, f_ov=None, eris=None,
                               max_memory=None, blksize=None):
        '''
        Args:
            omega : float, frequency
            e_ov : 1D array (nocc * nvir), orbital energy differences
            mo_coeff :  (nao, nmo), mean-field mo coefficient
            cderi_ov :  (naux, nocc, nvir), Cholesky decomposed ERI in OV subspace.

        Returns:
            diel : 2D array (naux, naux), dielectric matrix
        '''
        if e_ov is None: e_ov = self.make_e_ov()
        if f_ov is None: f_ov = self.make_f_ov()
        if eris is None: eris = self.ao2mo()
        if max_memory is None: max_memory = self.max_memory

        if blksize is None:
            mem_avail = max_memory - lib.current_memory()[0]
            nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
            dsize = eris.dsize
            mem_blk = 2*naux * dsize/1e6    # ovL and ovL*chi0
            blksize = max(1, min(max(nocc)*max(nvir), int(np.floor(mem_avail*0.7 / mem_blk))))
        else:
            blksize = min(blksize, e_ov.size)

        diel = make_dielectric_matrix(omega, e_ov, f_ov, eris, blksize=blksize)

        return diel


if __name__ == '__main__':
    from pyscf import gto, dft
    # Closed-shell unrestricted RPA
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2svp'
    mol.build()
    mol.verbose = 4

    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    # Shall be identical to the restricted RPA result
    rpa = URPA(mf)
    rpa.verbose = 5
    rpa.kernel()
    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr - -0.307830040357800) < 1e-6)
    assert (abs(rpa.e_tot  - -76.26651423730257) < 1e-6)

    # Open-shell RPA
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'F 0 0 0'
    mol.basis = 'def2-svp'
    mol.spin = 1
    mol.build()
    mol.verbose = 4

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    rpa = URPA(mf)
    rpa.verbose = 5
    rpa.kernel()
    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr - -0.20980646878974454) < 1e-6)
    assert (abs(rpa.e_tot  - -99.49455969299747) < 1e-6)
