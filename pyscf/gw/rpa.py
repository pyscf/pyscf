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
Spin-restricted random phase approximation (direct RPA/dRPA in chemistry)
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
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask

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
        assert rpa.frozen < rpa.nocc

    if Lpq is None:
        Lpq = rpa.ao2mo(mo_coeff)

    # Grids for integration on imaginary axis
    freqs, wts = _get_scaled_legendre_roots(nw, x0)

    # Compute HF exchange energy (EXX)
    dm = mf.make_rdm1()
    rhf = scf.RHF(rpa.mol)
    e_hf = rhf.energy_elec(dm=dm)[0]
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
    nocc = rpa.nocc
    nw = len(freqs)
    naux = Lpq.shape[0]

    if (mo_energy[nocc] - mo_energy[nocc-1]) < 1e-3:
        logger.warn(rpa, 'Current RPA code not well-defined for degeneracy!')

    e_corr = 0.
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:, :nocc, nocc:])
        ec_w = np.log(np.linalg.det(np.eye(naux) - Pi))
        ec_w += np.trace(Pi)
        e_corr += 1./(2.*np.pi) * ec_w * wts[w]

    return e_corr

def get_rho_response(omega, mo_energy, Lpq):
    """
    Compute density response function in auxiliary basis at freq iw.
    """
    naux, nocc, nvir = Lpq.shape
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia * eia)
    # Response from both spin-up and spin-down density
    Pia = Lpq * (eia * 4.0)
    Pi = einsum('Pia, Qia -> PQ', Pia, Lpq)
    return Pi

# ****************************************************************************
# frequency integral quadrature, legendre, clenshaw_curtis
# ****************************************************************************

def _get_scaled_legendre_roots(nw, x0=0.5):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [0, inf)
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D array
        wts : 1D array
    """
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    freqs_new = x0 * (1.0 + freqs) / (1.0 - freqs)
    wts = wts * 2.0 * x0 / (1.0 - freqs)**2
    return freqs_new, wts

def _get_clenshaw_curtis_roots(nw):
    """
    Clenshaw-Curtis qaudrature on [0,inf)
    Ref: J. Chem. Phys. 132, 234114 (2010)
    Returns:
        freqs : 1D array
        wts : 1D array
    """
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w + 1.0) / nw * np.pi * 0.5
        freqs[w] = a / np.tan(t)
        if w != nw - 1:
            wts[w] = a*np.pi * 0.5 / nw / (np.sin(t)**2)
        else:
            wts[w] = a*np.pi * 0.25 / nw / (np.sin(t)**2)
    return freqs[::-1], wts[::-1]

def _mo_energy_without_core(rpa, mo_energy):
    return mo_energy[get_frozen_mask(rpa)]

def _mo_without_core(rpa, mo):
    return mo[:,get_frozen_mask(rpa)]

class RPA(lib.StreamObject):

    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen

        # DF-RPA must use density fitting integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            if auxbasis:
                self.with_df.auxbasis = auxbasis
            else:
                self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.e_corr = None
        self.e_hf = None
        self.e_tot = None

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('RPA nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals = %d', self.frozen)
        return self

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, nw=40, x0=0.5):
        """
        Args:
            mo_energy : 1D array (nmo), mean-field mo energy
            mo_coeff : 2D array (nmo, nmo), mean-field mo coefficient
            Lpq : 3D array (naux, nmo, nmo), 3-index ERI
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
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * nmo**2*naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux, nmo, nmo)
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    rpa = RPA(mf)
    rpa.kernel()
    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr- -0.30783004035780076) < 1e-6)
    assert (abs(rpa.e_tot- -76.26428191794182) < 1e-6)
