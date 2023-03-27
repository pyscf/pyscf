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
from pyscf.gw.rpa import RPA, _get_scaled_legendre_roots

einsum = lib.einsum

# ****************************************************************************
# core routines, kernel, rpa_ecorr, rho_response
# ****************************************************************************

def kernel(rpa, mo_energy, mo_coeff, Lpq=None, nw=40, x0=0.5, verbose=logger.NOTE):
    """
    RPA correlation and total energy

    Args:
        Lpq : density fitting 3-center integral in MO basis.
        nw : number of frequency point on imaginary axis.
        x0: scaling factor for frequency grid.

    Returns:
        e_tot : RPA total energy
        e_hf : EXX energy
        e_corr : RPA correlation energy
    """
    mf = rpa._scf
    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)
        assert (rpa.frozen < rpa.nocc[0] and rpa.frozen < rpa.nocc[1])

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw, x0)

    # Compute HF exchange energy (EXX)
    dm = mf.make_rdm1()
    uhf = scf.UHF(rpa.mol)
    e_hf = uhf.energy_elec(dm=dm)[0]
    e_hf += mf.energy_nuc()

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, Lpq, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr

def get_rpa_ecorr(rpa, Lpq, freqs, wts):
    """
    Compute RPA correlation energy
    """
    mo_energy = _mo_energy_without_core(rpa, rpa._scf.mo_energy)
    nocca, noccb = rpa.nocc
    nw = len(freqs)
    naux = Lpq[0].shape[0]

    homo = max(mo_energy[0][nocca-1], mo_energy[1][noccb-1])
    lumo = min(mo_energy[0][nocca], mo_energy[1][noccb])
    if (lumo-homo) < 1e-3:
        logger.warn(rpa, 'Current RPA code not well-defined for degeneracy!')

    e_corr = 0.
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[0,:,:nocca,nocca:], Lpq[1,:,:noccb,noccb:])
        ec_w = np.log(np.linalg.det(np.eye(naux) - Pi))
        ec_w += np.trace(Pi)
        e_corr += 1./(2.*np.pi) * ec_w * wts[w]

    return e_corr

def get_rho_response(omega, mo_energy, Lpqa, Lpqb):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    naux, nocca, nvira = Lpqa.shape
    naux, noccb, nvirb = Lpqb.shape
    eia_a = mo_energy[0,:nocca,None] - mo_energy[0,None,nocca:]
    eia_b = mo_energy[1,:noccb,None] - mo_energy[1,None,noccb:]
    eia_a = eia_a / (omega**2 + eia_a*eia_a)
    eia_b = eia_b / (omega**2 + eia_b*eia_b)
    Pia_a = Lpqa * (eia_a * 2.0)
    Pia_b = Lpqb * (eia_b * 2.0)
    # Response from both spin-up and spin-down density
    Pi = einsum('Pia, Qia -> PQ', Pia_a, Lpqa) + einsum('Pia, Qia -> PQ', Pia_b, Lpqb)
    return Pi

def _mo_energy_without_core(rpa, mo_energy):
    moidx = get_frozen_mask(rpa)
    mo_energy = (mo_energy[0][moidx[0]], mo_energy[1][moidx[1]])
    return np.asarray(mo_energy)

def _mo_without_core(rpa, mo):
    moidx = get_frozen_mask(rpa)
    mo = (mo[0][:,moidx[0]], mo[1][:,moidx[1]])
    return np.asarray(mo)

class URPA(RPA):

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

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40, x0=0.5):
        """
        Args:
            mo_energy : 2D array (2, nmo), mean-field mo energy
            mo_coeff : 3D array (2, nmo, nmo), mean-field mo coefficient
            Lpq : 4D array (2, naux, nmo, nmo), 3-index ERI
            nw: interger, grid number
            x0: real, scaling factor for frequency grid

        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self._scf.mo_energy)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = \
                        kernel(self, mo_energy, mo_coeff, Lpq=Lpq, nw=nw, x0=x0, verbose=self.verbose)

        logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.nmo
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        mem_incore = (nmoa**2*naux + nmob**2*naux + nao**2*naux) * 8/1e6
        mem_now = lib.current_memory()[0]

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        ijslicea = (0, nmoa, 0, nmoa)
        ijsliceb = (0, nmob, 0, nmob)
        Lpqa = None
        Lpqb = None
        if (mem_incore + mem_now < 0.99*self.max_memory) or self.mol.incore_anyway:
            Lpqa = _ao2mo.nr_e2(self.with_df._cderi, moa, ijslicea, aosym='s2', out=Lpqa)
            Lpqb = _ao2mo.nr_e2(self.with_df._cderi, mob, ijsliceb, aosym='s2', out=Lpqb)
            return np.asarray((Lpqa.reshape(naux,nmoa,nmoa),Lpqb.reshape(naux,nmob,nmob)))
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = 'F 0 0 0'
    mol.basis = 'def2-svp'
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    rpa = URPA(mf)
    rpa.kernel()
    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr- -0.20980646878974454) < 1e-6)
    assert (abs(rpa.e_tot- -99.49292565821425) < 1e-6)
