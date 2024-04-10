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

import numpy as np, scipy

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask

einsum = lib.einsum

# ****************************************************************************
# core routines kernel
# ****************************************************************************

def kernel(rpa, mo_energy, mo_coeff, cderi_ov=None, nw=40, x0=0.5, verbose=logger.NOTE):
    """
    RPA correlation and total energy

    Args:
        cderi_ov:
            Array-like object, Cholesky decomposed ERI in OV subspace.
        nw:
            number of frequency point on imaginary axis.
        x0:
            scaling factor for frequency grid.

    Returns:
        e_tot:
            RPA total energy
        e_hf:
            EXX energy
        e_corr:
            RPA correlation energy
    """
    mf = rpa._scf

    # only support frozen core
    if rpa.frozen is not None:
        assert isinstance(rpa.frozen, int)
        assert rpa.frozen < np.min(rpa.nocc)

    # Get orbital number
    with_df = rpa.with_df
    naux = with_df.get_naoaux()
    norb = rpa._scf.mol.nao_nr()

    # Get memory information
    max_memory = max(0, rpa.max_memory * 0.9 - lib.current_memory()[0])
    if max_memory < naux ** 2 / 1e6:
        logger.warn(
            rpa, 'Memory may not be enough! Available memory %d MB < %d MB',
            max_memory, naux ** 2 / 1e6
                   )

    # AO -> MO transformation
    if cderi_ov is None:
        blksize = int(max_memory * 1e6 / (8 * norb ** 2))
        blksize = min(naux, blksize)
        blksize = max(1, blksize)

        # logger.debug(rpa, 'cderi    memory: %6d MB', naux * norb ** 2 * 8 / 1e6)
        # logger.debug(rpa, 'cderi_ov memory: %6d MB', naux * nocc * nvir * 8 / 1e6)
        logger.debug(rpa, 'ao2mo blksize = %d', blksize)
        if blksize == 1:
            logger.warn(rpa, 'Memory too small for ao2mo! blksize = 1')

        cderi_ov = rpa.ao2mo(mo_coeff, blksize=blksize)

    # Compute exact exchange energy (EXX)
    e_hf = _ene_hf(mf, with_df)
    e_ov = rpa.make_e_ov(mo_energy)

    # Compute RPA correlation energy
    e_corr = 0.0

    # Determine block size for dielectric matrix
    blksize = int(max_memory * 1e6 / 8 / naux)
    blksize = max(blksize, 1)

    if blksize == 1:
        logger.warn(rpa, 'Memory too small for dielectric matrix! blksize = 1')

    logger.debug(rpa, 'diel blksize = %d', blksize)

    # Grids for numerical integration on imaginary axis
    for omega, weigh in zip(*_get_scaled_legendre_roots(nw, x0)):
        diel = rpa.make_dielectric_matrix(omega, e_ov, cderi_ov, blksize=blksize)
        factor = weigh / (2.0 * np.pi)
        e_corr += factor * np.log(np.linalg.det(np.eye(naux) - diel))
        e_corr += factor * np.trace(diel)

    # Compute total energy
    e_tot = e_hf + e_corr
    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr

# ****************************************************************************
# frequency integral quadrature, legendre, clenshaw_curtis
# ****************************************************************************

def make_dielectric_matrix(omega, e_ov, cderi_ov, blksize=None):
    """
    Compute dielectric matrix at a given frequency omega

    Args:
        omega : float, frequency
        e_ov : 1D array (nocc * nvir), orbital energy differences
        cderi_ov : 2D array (naux, nocc * nvir), Cholesky decomposed ERI
                   in OV subspace.

    Returns:
        diel : 2D array (naux, naux), dielectric matrix
    """
    assert blksize is not None

    naux, nov = cderi_ov.shape

    chi0 = (2.0 * e_ov / (omega ** 2 + e_ov ** 2)).ravel()
    diel = np.zeros((naux, naux))

    for s in [slice(*x) for x in lib.prange(0, nov, blksize)]:
        v_ov = cderi_ov[:, s]
        diel += np.dot(v_ov * chi0[s], v_ov.T)
        v_ov = None

    return diel

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
            wts[w] = a * np.pi * 0.50 / nw / (np.sin(t)**2)
        else:
            wts[w] = a * np.pi * 0.25 / nw / (np.sin(t)**2)
    return freqs[::-1], wts[::-1]

def _ene_hf(mf=None, with_df=None):
    """
    Args:
        mf: converged mean-field object, can be either HF or KS
        with_df: density fitting object

    Returns:
        e_hf: float, total Hartree-Fock energy
    """
    assert mf.converged
    hf_obj = mf if not isinstance(mf, scf.hf.KohnShamDFT) else mf.to_hf()

    if not getattr(hf_obj, 'with_df', None):
        hf_obj = hf_obj.density_fit(with_df=with_df)
    dm = hf_obj.make_rdm1()

    e_hf  = hf_obj.energy_elec(dm=dm)[0]
    e_hf += hf_obj.energy_nuc()
    return e_hf

def _mo_energy_without_core(rpa, mo_energy):
    return mo_energy[get_frozen_mask(rpa)]

def _mo_without_core(rpa, mo):
    return mo[:,get_frozen_mask(rpa)]

class DirectRPA(lib.StreamObject):

    _keys = {
        'mol', 'frozen',
        'with_df', 'mo_energy',
        'mo_coeff', 'mo_occ',
        'e_corr', 'e_hf', 'e_tot',
    }

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

    def kernel(self, mo_energy=None, mo_coeff=None, cderi_ov=None, nw=40, x0=0.5):
        """
        The kernel function for direct RPA
        """

        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        res = kernel(
            self, mo_energy, mo_coeff,
            cderi_ov=cderi_ov, nw=nw, x0=x0,
            verbose=self.verbose
                    )
        self.e_tot, self.e_hf, self.e_corr = res

        logger.timer(self, 'RPA', *cput0)
        return self.e_corr

    def make_e_ov(self, mo_energy=None):
        """
        Compute orbital energy differences
        """
        if mo_energy is None:
            mo_energy = _mo_energy_without_core(self, self.mo_energy)

        nocc = self.nocc
        e_ov = (mo_energy[:nocc, None] - mo_energy[None, nocc:]).ravel()

        gap = (-e_ov.max(), )
        logger.info(self, 'Lowest orbital energy difference: % 6.4e', np.min(gap))

        if (np.min(gap) < 1e-3):
            logger.warn(rpa, 'RPA code not well-defined for degenerate systems!')
            logger.warn(rpa, 'Lowest orbital energy difference: % 6.4e', np.min(gap))

        return e_ov

    def make_dielectric_matrix(self, omega, e_ov=None, cderi_ov=None, blksize=None):
        """
        Args:
            omega : float, frequency
            e_ov : 1D array (nocc * nvir), orbital energy differences
            mo_coeff :  (nao, nmo), mean-field mo coefficient
            cderi_ov :  (naux, nocc, nvir), Cholesky decomposed ERI in OV subspace.

        Returns:
            diel : 2D array (naux, naux), dielectric matrix
        """

        assert e_ov is not None
        assert cderi_ov is not None

        blksize = blksize or max(e_ov.size)

        diel = 2.0 * make_dielectric_matrix(
            omega, e_ov,
            cderi_ov if isinstance(cderi_ov, np.ndarray) else cderi_ov["cderi_ov"],
            blksize=blksize
                                     )

        return diel

    def ao2mo(self, mo_coeff=None, blksize=None):
        if mo_coeff is None:
            mo_coeff = _mo_without_core(self, self.mo_coeff)

        nocc = self.nocc
        norb = self.nmo
        nvir = norb - nocc
        naux = self.with_df.get_naoaux()
        sov = (0, nocc, nocc, norb) # slice for OV block

        blksize  = naux if blksize is None else blksize
        cderi_ov = None

        cput0 = (logger.process_clock(), logger.perf_counter())
        if blksize >= naux or self.mol.incore_anyway:
            assert isinstance(self.with_df._cderi, np.ndarray)
            cderi_ov = _ao2mo.nr_e2(
                self.with_df._cderi, mo_coeff,
                sov, aosym='s2', out=cderi_ov
                                    )
            logger.timer(self, 'incore ao2mo', *cput0)

        else:
            fswap = lib.H5TmpFile()
            fswap.create_dataset('cderi_ov', (naux, nocc * nvir))

            q0 = 0
            for cderi in self.with_df.loop(blksize=blksize):
                q1 = q0 + cderi.shape[0]
                v_ov = _ao2mo.nr_e2(
                    cderi, mo_coeff,
                    sov, aosym='s2'
                                    )
                fswap['cderi_ov'][q0:q1] = v_ov
                v_ov = None
                q0 = q1

            logger.timer(self, 'outcore ao2mo', *cput0)
            cderi_ov = fswap

        return cderi_ov

RPA = dRPA = DirectRPA

if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    rpa = RPA(mf)
    rpa.verbose = 6

    nocc = rpa.nocc
    nvir = rpa.nmo - nocc
    norb = rpa.nmo
    e_ov = - (rpa.mo_energy[:nocc, None] - rpa.mo_energy[None, nocc:]).ravel()
    v_ov = rpa.ao2mo(rpa.mo_coeff, blksize=1)
    e_corr_0 = rpa.kernel(cderi_ov=v_ov)

    print ('RPA e_tot, e_hf, e_corr = ', rpa.e_tot, rpa.e_hf, rpa.e_corr)
    assert (abs(rpa.e_corr - -0.307830040357800) < 1e-6)
    assert (abs(rpa.e_tot  - -76.26651423730257) < 1e-6)

    # Another implementation of direct RPA N^6
    v_ov = np.array(v_ov["cderi_ov"])
    a = e_ov * np.eye(nocc * nvir) + 2 * np.dot(v_ov.T, v_ov)
    b = 2 * np.dot(v_ov.T, v_ov)
    apb = a + b
    amb = a - b
    c = np.dot(amb, apb)
    e_corr_1 = 0.5 * np.trace(
        scipy.linalg.sqrtm(c) - a
    )

    assert abs(e_corr_0 - e_corr_1) < 1e-8
