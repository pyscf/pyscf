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
    Main routines are based on GW-AC method descirbed in:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    X. Ren et al., New J. Phys. 14, 053020 (2012)
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask

import pyscf.gw.rpa

einsum = lib.einsum

def _mo_energy_without_core(rpa, mo_energy):
    moidx = get_frozen_mask(rpa)
    mo_energy = (mo_energy[0][moidx[0]], mo_energy[1][moidx[1]])
    return np.asarray(mo_energy)

def _mo_without_core(rpa, mo):
    moidx = get_frozen_mask(rpa)
    mo = (mo[0][:,moidx[0]], mo[1][:,moidx[1]])
    return np.asarray(mo)

class URPA(pyscf.gw.rpa.RPA):
    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        log.info('RPA (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d)',
                 nocca, noccb, nvira, nvirb)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def make_e_ov(self, mo_energy=None):
        """
        Compute orbital energy differences
        """
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self.mo_energy)

        nocc_a, nocc_b = self.nocc
        e_ov_a = (mo_energy[0][:nocc_a, None] - mo_energy[0][None, nocc_a:]).ravel()
        e_ov_b = (mo_energy[1][:nocc_b, None] - mo_energy[1][None, nocc_b:]).ravel()

        gap = (-e_ov_a.max(), -e_ov_b.max())
        logger.info(self, 'Lowest orbital energy difference: (% 6.4e, % 6.4e)', gap[0], gap[1])

        if (np.min(gap) < 1e-3):
            logger.warn(self, 'RPA code not well-defined for degenerate systems!')
            logger.warn(self, 'Lowest orbital energy difference: % 6.4e', np.min(gap))

        return e_ov_a, e_ov_b

    def make_dielectric_matrix(self, omega, e_ov=None, cderi_ov=None, blksize=None):
        """
        Args:
            omega : float, frequency
            mo_energy : (2, nmo), mean-field mo energy
            mo_coeff :  (2, nao, nmo), mean-field mo coefficient
            cderi_ov :  (2, naux, nocc, nvir), Cholesky decomposed ERI in OV subspace.

        Returns:
            diel : 2D array (naux, naux), dielectric matrix
        """
        assert cderi_ov is not None
        assert e_ov is not None

        naux = self.with_df.get_naoaux()
        blksize = blksize or max(e_ov[0].size, e_ov[1].size)

        diel = np.zeros((naux, naux))
        for s, e_ov_s in enumerate((e_ov[0], e_ov[1])):
            cderi_ov_s = cderi_ov[s] if isinstance(cderi_ov, tuple) else cderi_ov["cderi_ov_%d" % s]
            diel += pyscf.gw.rpa.make_dielectric_matrix(omega, e_ov_s, cderi_ov_s, blksize=blksize)

        return diel

    def ao2mo(self, mo_coeff=None, blksize=None):
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self.mo_coeff)

        mo_coeff_a = mo_coeff[0]
        mo_coeff_b = mo_coeff[1]

        nocc_a, nocc_b = self.nocc
        norb_a, norb_b = self.nmo
        nvir_a, nvir_b = norb_a - nocc_a, norb_b - nocc_b

        naux = self.with_df.get_naoaux()
        sov_a = (0, nocc_a, nocc_a, norb_a)
        sov_b = (0, nocc_b, nocc_b, norb_b)

        blksize  = naux if blksize is None else blksize
        cderi_ov = None
        cderi_ov_a = None
        cderi_ov_b = None

        cput0 = (logger.process_clock(), logger.perf_counter())
        if blksize >= naux or self.mol.incore_anyway:
            assert isinstance(self.with_df._cderi, np.ndarray)
            cderi_ov_a = _ao2mo.nr_e2(
                self.with_df._cderi, mo_coeff_a,
                sov_a, aosym='s2', out=cderi_ov_a
                                    )

            cderi_ov_b = _ao2mo.nr_e2(
                self.with_df._cderi, mo_coeff_b,
                sov_b, aosym='s2', out=cderi_ov_b
                                    )
            cderi_ov = (cderi_ov_a, cderi_ov_b)

            logger.timer(self, 'incore ao2mo', *cput0)

        else:
            fswap = lib.H5TmpFile()
            fswap.create_dataset('cderi_ov_0', (naux, nocc_a * nvir_a), 'f8')
            fswap.create_dataset('cderi_ov_1', (naux, nocc_b * nvir_b), 'f8')

            q0 = 0
            for cderi in self.with_df.loop(blksize=blksize):
                q1 = q0 + cderi.shape[0]

                v_ov_a = _ao2mo.nr_e2(
                    cderi, mo_coeff_a,
                    sov_a, aosym='s2'
                                    )
                fswap['cderi_ov_0'][q0:q1] = v_ov_a
                v_ov_a = None

                v_ov_b = _ao2mo.nr_e2(
                    cderi, mo_coeff_b,
                    sov_b, aosym='s2'
                                    )
                fswap['cderi_ov_1'][q0:q1] = v_ov_b
                v_ov_b = None

                q0 = q1

            logger.timer(self, 'outcore ao2mo', *cput0)

            cderi_ov = fswap

        return cderi_ov


if __name__ == '__main__':
    from pyscf import gto, dft
    # Closed-shell unrestricted RPA
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2svp'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    # Shall be identical to the restricted RPA result
    rpa = URPA(mf)
    rpa.max_memory = 0
    rpa.verbose = 5
    rpa.kernel()
    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr - -0.307830040357800) < 1e-6)
    assert (abs(rpa.e_tot  - -76.26651423730257) < 1e-6)

    # Open-shell RPA
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = 'F 0 0 0'
    mol.basis = 'def2-svp'
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.max_memory = 0
    mf.kernel()

    rpa = URPA(mf)
    rpa.max_memory = 0
    rpa.verbose = 5
    rpa.kernel()
    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr - -0.20980646878974454) < 1e-6)
    assert (abs(rpa.e_tot  - -99.49455969299747) < 1e-6)




